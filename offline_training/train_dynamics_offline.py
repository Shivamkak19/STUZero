"""
train_dynamics_offline.py

Offline dynamics network training using pre-collected latent state dataset.
Trains a DynamicsNetwork (or DynamicsNetworkWithSTU) to predict next latent
states, using both MSE loss and consistency loss through frozen projection networks.

Three phases:
  1. Verify: Load benchmark dynamics weights, confirm they reproduce stored predictions
  2. Train:  Train dynamics from scratch with MSE + consistency loss
  3. Eval:   Single-step metrics + multi-step autoregressive rollout

Usage:
    cd STUZero

    # Standard dynamics (validate pipeline):
    python train_dynamics_offline.py \
        --data_dir dynamics_dataset \
        --checkpoint model_100000.p \
        --epochs 50 --batch_size 256 --lr 1e-3

    # STU dynamics:
    python train_dynamics_offline.py \
        --data_dir dynamics_dataset \
        --checkpoint model_100000.p \
        --use_stu --epochs 100 --lr 5e-4

    # test with baseline dynamics network:
    python train_dynamics_offline.py \
        --data_dir dynamics_dataset \
        --checkpoint model_100000.p \
        --skip_verification --epochs 2 --eval_interval 1
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import json
import time
import argparse
from pathlib import Path
from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np

from ez.agents.models.base_model import (
    DynamicsNetwork,
    DynamicsNetworkWithSTU,
    ProjectionNetwork,
    ProjectionHeadNetwork,
)
from ez.utils.loss import cosine_similarity_loss


# ===========================================================================
# Datasets
# ===========================================================================

class DynamicsDataset(Dataset):
    """Flat dataset of (s_t, a_t, s_{t+1}) transitions from episode files.

    Loads all episodes into memory, filters by valid_next mask, and
    casts float16 -> float32.
    """

    def __init__(self, episode_paths):
        all_states, all_actions, all_next_states = [], [], []
        all_dynamics_preds, all_projections, all_dynamics_projs = [], [], []

        for ep_path in episode_paths:
            ep = torch.load(ep_path, map_location='cpu', weights_only=False)
            valid = ep['valid_next']

            all_states.append(ep['latent_states'][valid].float())
            all_actions.append(ep['actions'][valid])
            all_next_states.append(ep['next_latent_states'][valid].float())
            all_dynamics_preds.append(ep['dynamics_predictions'][valid].float())
            all_projections.append(ep['projections'][valid].float())
            all_dynamics_projs.append(ep['dynamics_projections'][valid].float())

        self.states = torch.cat(all_states)
        self.actions = torch.cat(all_actions)
        self.next_states = torch.cat(all_next_states)
        self.dynamics_preds = torch.cat(all_dynamics_preds)
        self.projections = torch.cat(all_projections)
        self.dynamics_projs = torch.cat(all_dynamics_projs)

        print(f"  Loaded {len(episode_paths)} episodes, "
              f"{len(self.states)} valid transitions")

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return {
            'state': self.states[idx],
            'action': self.actions[idx],
            'next_state': self.next_states[idx],
            'dynamics_pred': self.dynamics_preds[idx],
        }


class SequentialDynamicsDataset(Dataset):
    """Dataset of contiguous K-step windows for multi-step rollout evaluation.

    Each item contains K+1 ground-truth states and K actions from a single
    episode, with all transitions valid.
    """

    def __init__(self, episode_paths, rollout_len=10):
        self.rollout_len = rollout_len
        self.windows = []

        for ep_path in episode_paths:
            ep = torch.load(ep_path, map_location='cpu', weights_only=False)
            valid = ep['valid_next'].numpy()
            T = len(valid)

            states = ep['latent_states'].float()
            actions = ep['actions']
            next_states = ep['next_latent_states'].float()

            t = 0
            while t + rollout_len <= T:
                if valid[t:t + rollout_len].all():
                    # states: s_t through s_{t+K}
                    window_states = torch.cat([
                        states[t:t + rollout_len],
                        next_states[t + rollout_len - 1:t + rollout_len],
                    ], dim=0)
                    window_actions = actions[t:t + rollout_len]
                    self.windows.append((window_states, window_actions))
                    t += rollout_len  # non-overlapping
                else:
                    t += 1

        print(f"  Built {len(self.windows)} rollout windows of length {rollout_len}")

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, idx):
        states, actions = self.windows[idx]
        return {'states': states, 'actions': actions}


# ===========================================================================
# Model construction and weight loading
# ===========================================================================

def build_dynamics_network(action_space_size, use_stu=False,
                           num_blocks=1, num_channels=64,
                           action_embedding=True, action_embedding_dim=16,
                           dynamics_stu_seq_len=36,
                           dynamics_stu_num_filters=2):
    """Build standard or STU dynamics network."""
    if use_stu:
        return DynamicsNetworkWithSTU(
            num_blocks=num_blocks,
            num_channels=num_channels,
            action_space_size=action_space_size,
            action_embedding=action_embedding,
            action_embedding_dim=action_embedding_dim,
            dynamics_stu_seq_len=dynamics_stu_seq_len,
            dynamics_stu_num_filters=dynamics_stu_num_filters,
        )
    else:
        return DynamicsNetwork(
            num_blocks=num_blocks,
            num_channels=num_channels,
            action_space_size=action_space_size,
            action_embedding=action_embedding,
            action_embedding_dim=action_embedding_dim,
        )


def load_frozen_projection_networks(checkpoint_path, device):
    """Load ProjectionNetwork + ProjectionHeadNetwork from checkpoint, freeze them.

    These use BatchNorm1d internally, so we set eval mode to use
    running statistics from the checkpoint rather than batch statistics.
    """
    state_dim = 64 * 6 * 6  # 2304
    projection_model = ProjectionNetwork(
        input_dim=state_dim, hid_dim=1024, out_dim=1024,
    )
    projection_head_model = ProjectionHeadNetwork(
        input_dim=1024, hid_dim=256, out_dim=1024,
    )

    full_weights = torch.load(checkpoint_path, map_location='cpu',
                              weights_only=False)

    proj_weights = {
        k.replace('projection_model.', ''): v
        for k, v in full_weights.items()
        if k.startswith('projection_model.')
    }
    projection_model.load_state_dict(proj_weights)

    head_weights = {
        k.replace('projection_head_model.', ''): v
        for k, v in full_weights.items()
        if k.startswith('projection_head_model.')
    }
    projection_head_model.load_state_dict(head_weights)

    projection_model.to(device).eval()
    projection_head_model.to(device).eval()
    for p in projection_model.parameters():
        p.requires_grad_(False)
    for p in projection_head_model.parameters():
        p.requires_grad_(False)

    return projection_model, projection_head_model


def load_benchmark_dynamics(checkpoint_path, action_space_size, device):
    """Load the benchmark DynamicsNetwork weights from checkpoint for verification."""
    dynamics = DynamicsNetwork(
        num_blocks=1, num_channels=64,
        action_space_size=action_space_size,
        action_embedding=True, action_embedding_dim=16,
    )

    full_weights = torch.load(checkpoint_path, map_location='cpu',
                              weights_only=False)
    dyn_weights = {
        k.replace('dynamics_model.', ''): v
        for k, v in full_weights.items()
        if k.startswith('dynamics_model.')
    }
    dynamics.load_state_dict(dyn_weights)
    dynamics.to(device).eval()
    return dynamics


# ===========================================================================
# Verification
# ===========================================================================

@torch.no_grad()
def verify_benchmark_predictions(dynamics_model, dataset, device,
                                 num_samples=2000):
    """Verify loaded benchmark dynamics reproduces stored predictions.

    Confirms weight loading, action encoding, and forward pass are correct.
    Tolerance is relaxed to 0.01 because the dataset was stored in float16.
    """
    dynamics_model.eval()
    indices = torch.randperm(len(dataset))[:num_samples]

    max_errors, mean_errors = [], []
    batch_size = 256

    for start in range(0, num_samples, batch_size):
        end = min(start + batch_size, num_samples)
        batch_idx = indices[start:end]

        states = dataset.states[batch_idx].to(device)
        actions = dataset.actions[batch_idx].to(device).unsqueeze(-1)
        stored_preds = dataset.dynamics_preds[batch_idx].to(device)

        computed_preds = dynamics_model(states, actions)

        diff = (computed_preds - stored_preds).abs()
        max_errors.append(diff.max().item())
        mean_errors.append(diff.mean().item())

    results = {
        'max_error': max(max_errors),
        'mean_error': np.mean(mean_errors),
        'num_samples': num_samples,
    }
    passed = results['max_error'] < 0.01
    results['passed'] = passed

    print(f"\n{'='*60}")
    print(f"  BENCHMARK VERIFICATION")
    print(f"{'='*60}")
    print(f"  Samples checked:   {num_samples}")
    print(f"  Max abs error:     {results['max_error']:.6f}")
    print(f"  Mean abs error:    {results['mean_error']:.6f}")
    print(f"  Status:            {'PASSED' if passed else 'FAILED'}")
    if not passed:
        print(f"  NOTE: Error > 0.01 may be due to float16 dataset storage.")
    print(f"{'='*60}\n")

    return results


# ===========================================================================
# Training and evaluation
# ===========================================================================

def compute_losses(dynamics_model, projection_model, projection_head_model,
                   batch, device, consistency_weight=5.0):
    """Compute MSE + consistency loss for a batch.

    Replicates the online consistency loss from ez/agents/base.py lines 474-477:
      predicted branch: projection_head(projection(G(s_t, a_t)))  [with grad]
      target branch:    projection(s_{t+1}).detach()               [no grad]
    """
    states = batch['state'].to(device)
    actions = batch['action'].to(device).unsqueeze(-1)  # [B, 1]
    next_states = batch['next_state'].to(device)

    # Dynamics prediction
    pred_next = dynamics_model(states, actions)

    # MSE loss on raw latent states
    mse_loss = F.mse_loss(pred_next, next_states)

    # Consistency loss through frozen projection networks
    pred_proj = projection_model(pred_next)
    pred_proj_head = projection_head_model(pred_proj)

    with torch.no_grad():
        target_proj = projection_model(next_states)

    consistency = cosine_similarity_loss(pred_proj_head, target_proj).mean()

    total_loss = mse_loss + consistency_weight * consistency

    # Also compute MSE vs benchmark predictions (diagnostic only)
    benchmark_preds = batch['dynamics_pred'].to(device)
    benchmark_mse = F.mse_loss(pred_next, benchmark_preds)

    loss_dict = {
        'mse': mse_loss.item(),
        'consistency': consistency.item(),
        'total': total_loss.item(),
        'vs_benchmark_mse': benchmark_mse.item(),
    }
    return total_loss, loss_dict


def train_one_epoch(dynamics_model, projection_model, projection_head_model,
                    train_loader, optimizer, device, consistency_weight=5.0,
                    grad_clip=5.0):
    """Train for one epoch."""
    dynamics_model.train()

    epoch_metrics = defaultdict(float)
    n_batches = 0

    for batch in train_loader:
        loss, loss_dict = compute_losses(
            dynamics_model, projection_model, projection_head_model,
            batch, device, consistency_weight,
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(dynamics_model.parameters(), grad_clip)
        optimizer.step()

        for k, v in loss_dict.items():
            epoch_metrics[k] += v
        n_batches += 1

    return {k: v / n_batches for k, v in epoch_metrics.items()}


@torch.no_grad()
def evaluate(dynamics_model, projection_model, projection_head_model,
             eval_loader, device, consistency_weight=5.0):
    """Evaluate on held-out episodes."""
    dynamics_model.eval()

    epoch_metrics = defaultdict(float)
    n_batches = 0

    for batch in eval_loader:
        _, loss_dict = compute_losses(
            dynamics_model, projection_model, projection_head_model,
            batch, device, consistency_weight,
        )
        for k, v in loss_dict.items():
            epoch_metrics[k] += v
        n_batches += 1

    return {k: v / n_batches for k, v in epoch_metrics.items()}


@torch.no_grad()
def evaluate_multistep(dynamics_model, seq_dataset, device):
    """Evaluate multi-step autoregressive rollout error.

    Starting from ground-truth s_0, repeatedly applies the dynamics model:
        s_hat_{k+1} = G(s_hat_k, a_k)
    and measures MSE(s_hat_k, s_k) at each step.

    Returns dict mapping step number -> mean MSE.
    """
    dynamics_model.eval()
    loader = DataLoader(seq_dataset, batch_size=64, shuffle=False)

    step_errors = defaultdict(list)

    for batch in loader:
        gt_states = batch['states'].to(device)   # [B, K+1, C, H, W]
        actions = batch['actions'].to(device)     # [B, K]
        K = actions.shape[1]

        current = gt_states[:, 0]  # start from ground truth

        for step in range(K):
            action = actions[:, step].unsqueeze(-1)
            current = dynamics_model(current, action)

            gt = gt_states[:, step + 1]
            mse = F.mse_loss(current, gt, reduction='none')
            mse_per_sample = mse.reshape(current.shape[0], -1).mean(dim=1)
            step_errors[step + 1].extend(mse_per_sample.cpu().tolist())

    return {step: np.mean(errs) for step, errs in sorted(step_errors.items())}


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Offline dynamics network training for STUZero',
    )
    # Data
    p.add_argument('--data_dir', type=str, required=True,
                   help='Path to dynamics_dataset/ directory')
    p.add_argument('--checkpoint', type=str, required=True,
                   help='Path to EfficientZero checkpoint (model_100000.p)')
    p.add_argument('--train_split', type=float, default=0.8,
                   help='Fraction of episodes for training')

    # Architecture
    p.add_argument('--use_stu', action='store_true',
                   help='Use DynamicsNetworkWithSTU')
    p.add_argument('--action_space_size', type=int, default=None,
                   help='Override action space size (auto from metadata)')
    p.add_argument('--num_blocks', type=int, default=1)
    p.add_argument('--num_channels', type=int, default=64)
    p.add_argument('--action_embedding_dim', type=int, default=16)
    p.add_argument('--dynamics_stu_seq_len', type=int, default=36)
    p.add_argument('--dynamics_stu_num_filters', type=int, default=2)

    # Training
    p.add_argument('--epochs', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=256)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=1e-4)
    p.add_argument('--optimizer', type=str, default='Adam',
                   choices=['Adam', 'AdamW', 'SGD'])
    p.add_argument('--lr_schedule', type=str, default='cosine',
                   choices=['cosine', 'none'])
    p.add_argument('--consistency_weight', type=float, default=5.0)
    p.add_argument('--grad_clip', type=float, default=5.0)

    # Evaluation
    p.add_argument('--rollout_len', type=int, default=10)
    p.add_argument('--eval_interval', type=int, default=5)

    # Verification
    p.add_argument('--skip_verification', action='store_true')
    p.add_argument('--verify_samples', type=int, default=2000)

    # Logging
    p.add_argument('--use_wandb', action='store_true')
    p.add_argument('--wandb_project', type=str,
                   default='stuzero-dynamics-offline')
    p.add_argument('--wandb_run_name', type=str, default=None)

    # Output
    p.add_argument('--save_dir', type=str, default='dynamics_checkpoints')
    p.add_argument('--save_interval', type=int, default=10)

    # Misc
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=0,
                   help='DataLoader workers (0 = main process, avoids ray import issues)')

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("WARNING: CUDA not available. DynamicsNetwork requires CUDA.")
        print("The script will likely fail during the forward pass.")

    # --- Load metadata ---
    data_dir = Path(args.data_dir)
    metadata_path = data_dir / 'metadata.json'
    if metadata_path.exists():
        with open(metadata_path) as f:
            metadata = json.load(f)
        action_space_size = (
            args.action_space_size or metadata['config']['action_space_size']
        )
        print(f"Loaded metadata: {metadata.get('environment', '?')}, "
              f"|A|={action_space_size}")
    else:
        action_space_size = args.action_space_size
        if action_space_size is None:
            raise ValueError(
                "No metadata.json found. Specify --action_space_size."
            )

    # --- Split episodes ---
    all_episodes = sorted(data_dir.glob('episode_*.pt'))
    if len(all_episodes) == 0:
        raise FileNotFoundError(f"No episode_*.pt files in {data_dir}")

    n_train = int(len(all_episodes) * args.train_split)
    train_episodes = all_episodes[:n_train]
    eval_episodes = all_episodes[n_train:]
    print(f"Episodes: {len(all_episodes)} total, "
          f"{n_train} train, {len(eval_episodes)} eval")

    # --- Build datasets ---
    print("\nLoading training data...")
    train_dataset = DynamicsDataset(train_episodes)
    print("Loading eval data...")
    eval_dataset = DynamicsDataset(eval_episodes)

    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )
    eval_loader = DataLoader(
        eval_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.num_workers, pin_memory=True,
    )

    print("Building multi-step eval windows...")
    seq_eval_dataset = SequentialDynamicsDataset(
        eval_episodes, rollout_len=args.rollout_len,
    )

    # --- Phase 1: Verify benchmark ---
    if not args.skip_verification:
        print("\n--- Phase 1: Benchmark Verification ---")
        benchmark_dyn = load_benchmark_dynamics(
            args.checkpoint, action_space_size, device,
        )
        full_dataset = DynamicsDataset(all_episodes)
        verify_results = verify_benchmark_predictions(
            benchmark_dyn, full_dataset, device,
            num_samples=args.verify_samples,
        )
        del benchmark_dyn, full_dataset
        torch.cuda.empty_cache()

        if not verify_results['passed']:
            print("WARNING: Benchmark verification did not pass cleanly.")
            print("Likely due to float16 quantization in dataset. Continuing.\n")

    # --- Phase 2: Build trainable dynamics + frozen projections ---
    print("--- Phase 2: Training Setup ---")
    dynamics_model = build_dynamics_network(
        action_space_size=action_space_size,
        use_stu=args.use_stu,
        num_blocks=args.num_blocks,
        num_channels=args.num_channels,
        action_embedding_dim=args.action_embedding_dim,
        dynamics_stu_seq_len=args.dynamics_stu_seq_len,
        dynamics_stu_num_filters=args.dynamics_stu_num_filters,
    ).to(device)

    projection_model, projection_head_model = load_frozen_projection_networks(
        args.checkpoint, device,
    )

    n_params = sum(p.numel() for p in dynamics_model.parameters()
                   if p.requires_grad)
    print(f"Dynamics model: {type(dynamics_model).__name__}, "
          f"{n_params:,} trainable params")
    print(f"Model: {'STU' if args.use_stu else 'Standard'}")

    # Optimizer
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(
            dynamics_model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'AdamW':
        optimizer = optim.AdamW(
            dynamics_model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == 'SGD':
        optimizer = optim.SGD(
            dynamics_model.parameters(), lr=args.lr,
            weight_decay=args.weight_decay, momentum=0.9,
        )

    # LR scheduler
    scheduler = None
    if args.lr_schedule == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs,
        )

    # wandb
    if args.use_wandb:
        import wandb
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
        )

    # Save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    print(f"\n--- Training for {args.epochs} epochs ---\n")
    best_eval_mse = float('inf')

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            dynamics_model, projection_model, projection_head_model,
            train_loader, optimizer, device,
            args.consistency_weight, args.grad_clip,
        )

        if scheduler is not None:
            scheduler.step()

        elapsed = time.time() - t0
        lr = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"mse={train_metrics['mse']:.6f}  "
            f"consist={train_metrics['consistency']:.4f}  "
            f"vs_bench={train_metrics['vs_benchmark_mse']:.6f}  "
            f"lr={lr:.2e} | {elapsed:.1f}s"
        )

        # Evaluate
        if epoch % args.eval_interval == 0 or epoch == args.epochs:
            eval_metrics = evaluate(
                dynamics_model, projection_model, projection_head_model,
                eval_loader, device, args.consistency_weight,
            )
            multistep = evaluate_multistep(
                dynamics_model, seq_eval_dataset, device,
            )

            print(
                f"  [eval] mse={eval_metrics['mse']:.6f}  "
                f"consist={eval_metrics['consistency']:.4f}  "
                f"vs_bench={eval_metrics['vs_benchmark_mse']:.6f}"
            )
            rollout_str = "  ".join(
                f"s{k}={v:.5f}" for k, v in sorted(multistep.items())
            )
            print(f"  [rollout] {rollout_str}")

            if eval_metrics['mse'] < best_eval_mse:
                best_eval_mse = eval_metrics['mse']
                torch.save(dynamics_model.state_dict(),
                           save_dir / 'best_dynamics.pt')
                print(f"  ** New best eval MSE: {best_eval_mse:.6f}")

            if args.use_wandb:
                import wandb
                log = {f'train/{k}': v for k, v in train_metrics.items()}
                log.update({f'eval/{k}': v for k, v in eval_metrics.items()})
                log.update({f'rollout/step_{k}_mse': v
                            for k, v in multistep.items()})
                log['lr'] = lr
                wandb.log(log, step=epoch)

        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': dynamics_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'args': vars(args),
            }, save_dir / f'dynamics_epoch_{epoch:04d}.pt')

    # --- Summary ---
    print(f"\n{'='*60}")
    print(f"  Training complete!")
    print(f"  Best eval MSE:  {best_eval_mse:.6f}")
    print(f"  Checkpoints in: {save_dir}/")
    print(f"{'='*60}")

    if args.use_wandb:
        import wandb
        wandb.finish()


if __name__ == '__main__':
    main()
