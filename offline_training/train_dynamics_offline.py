"""
train_dynamics_offline.py

Offline dynamics network training using pre-collected latent state dataset.
Trains a DynamicsNetwork (or DynamicsNetworkWithSTU) to predict next latent
states, using both MSE loss and consistency loss through frozen projection networks.

Dataset: https://huggingface.co/datasets/Shivamkak/STUZero-Atari-Dynamics

Three phases:
  1. Verify: Load benchmark dynamics weights, confirm they reproduce stored predictions
  2. Train:  Train dynamics from scratch with MSE + consistency loss
  3. Eval:   Single-step metrics + multi-step autoregressive rollout

Usage:
    cd STUZero

    # Train on Pong (auto-resolves data_dir and checkpoint):
    python train_dynamics_offline.py --game pong --epochs 80 --batch_size 256

    # Train on Asterix with STU:
    python train_dynamics_offline.py --game asterix --model_type stu --epochs 80

    # Download dataset from HuggingFace first, then train:
    python train_dynamics_offline.py --game pong --download --epochs 80

    # Manual paths (override game defaults):
    python train_dynamics_offline.py \
        --data_dir dynamics_dataset \
        --checkpoint model_100000.p \
        --epochs 50 --batch_size 256 --lr 1e-3
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

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
    DynamicsNetworkResidualOnly,
    DynamicsNetworkWithMamba,
    DynamicsNetworkWithAttention,
    DynamicsNetworkTemporalBaseline,
    DynamicsNetworkSpatioTemporalSTU,
    DynamicsNetworkTemporalSTU,
    DynamicsNetworkTemporalMamba,
    DynamicsNetworkTemporalAttention,
    ProjectionNetwork,
    ProjectionHeadNetwork,
)
from ez.utils.loss import cosine_similarity_loss


# ===========================================================================
# Game configs — maps game name to dataset subfolder and checkpoint
# ===========================================================================

HF_REPO_ID = "Shivamkak/STUZero-Atari-Dynamics"

GAME_CONFIGS = {
    'pong': {
        'data_subdir': 'pong_100K',
        'checkpoint': 'pong_100K/pong_model_100000.p',
        'action_space_size': 6,
    },
    'asterix': {
        'data_subdir': 'asterix_110K',
        'checkpoint': 'asterix_110K/asterix_model_110000.p',
        'action_space_size': 9,
    },
}


def download_game_data(game, base_dir='.'):
    """Download a game's dataset from HuggingFace Hub if not already present."""
    from huggingface_hub import snapshot_download

    gc = GAME_CONFIGS[game]
    local_dir = Path(base_dir) / gc['data_subdir']

    if local_dir.exists() and any(local_dir.glob('episode_*.pt')):
        print(f"Dataset already exists at {local_dir}, skipping download.")
        return str(local_dir)

    print(f"Downloading {game} dataset from {HF_REPO_ID}...")
    snapshot_download(
        repo_id=HF_REPO_ID,
        repo_type="dataset",
        allow_patterns=f"{gc['data_subdir']}/*",
        local_dir=base_dir,
    )
    print(f"Downloaded to {local_dir}")
    return str(local_dir)


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


class TemporalDynamicsDataset(Dataset):
    """Dataset of (history, a_t, s_{t+1}) transitions with N-step history buffer.

    Each sample provides the N most recent states leading up to (and including)
    state s_t, plus the action a_t and the ground-truth next state s_{t+1}.
    For the first N-1 states of an episode, history is zero-padded on the left.
    """

    def __init__(self, episode_paths, buffer_size=10):
        self.buffer_size = buffer_size
        all_histories, all_actions, all_next_states = [], [], []
        all_dynamics_preds = []

        for ep_path in episode_paths:
            ep = torch.load(ep_path, map_location='cpu', weights_only=False)
            valid = ep['valid_next'].numpy()
            states = ep['latent_states'].float()       # [T, C, H, W]
            actions = ep['actions']                     # [T]
            next_states = ep['next_latent_states'].float()
            dynamics_preds = ep['dynamics_predictions'].float()

            T = len(valid)
            C, H, W = states.shape[1], states.shape[2], states.shape[3]

            for t in range(T):
                if not valid[t]:
                    continue

                # Build history buffer: states from t-N+1 to t (inclusive)
                start = t - buffer_size + 1
                if start >= 0:
                    history = states[start:t + 1]  # [N, C, H, W]
                else:
                    # Zero-pad on the left for early timesteps
                    pad_len = -start
                    pad = torch.zeros(pad_len, C, H, W)
                    history = torch.cat([pad, states[0:t + 1]], dim=0)  # [N, C, H, W]

                all_histories.append(history)
                all_actions.append(actions[t])
                all_next_states.append(next_states[t])
                all_dynamics_preds.append(dynamics_preds[t])

        self.histories = torch.stack(all_histories)       # [N_total, buffer_size, C, H, W]
        self.actions = torch.stack(all_actions)            # [N_total]
        self.next_states = torch.stack(all_next_states)    # [N_total, C, H, W]
        self.dynamics_preds = torch.stack(all_dynamics_preds)

        print(f"  Loaded {len(episode_paths)} episodes, "
              f"{len(self.histories)} valid transitions (buffer_size={buffer_size})")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, idx):
        return {
            'history': self.histories[idx],         # [N, C, H, W]
            'action': self.actions[idx],            # scalar
            'next_state': self.next_states[idx],    # [C, H, W]
            'dynamics_pred': self.dynamics_preds[idx],
        }


# ===========================================================================
# Model construction and weight loading
# ===========================================================================

def build_dynamics_network(action_space_size, model_type='baseline',
                           num_blocks=1, num_channels=64,
                           action_embedding=True, action_embedding_dim=16,
                           dynamics_stu_seq_len=36,
                           dynamics_stu_num_filters=2,
                           mamba_d_state=16, mamba_d_conv=4, mamba_expand=1,
                           attn_num_heads=4,
                           buffer_size=10):
    """Build dynamics network.

    model_type: 'baseline' | 'stu' | 'mamba' | 'attention' | 'residual_only'
                | 'temporal_stu' | 'temporal_mamba' | 'temporal_attention'
    """
    common = dict(
        num_blocks=num_blocks,
        num_channels=num_channels,
        action_space_size=action_space_size,
        action_embedding=action_embedding,
        action_embedding_dim=action_embedding_dim,
    )

    if model_type == 'stu':
        return DynamicsNetworkWithSTU(
            **common,
            dynamics_stu_seq_len=dynamics_stu_seq_len,
            dynamics_stu_num_filters=dynamics_stu_num_filters,
        )
    elif model_type == 'mamba':
        return DynamicsNetworkWithMamba(
            **common,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
    elif model_type == 'attention':
        return DynamicsNetworkWithAttention(
            **common,
            num_heads=attn_num_heads,
        )
    elif model_type == 'residual_only':
        return DynamicsNetworkResidualOnly(**common)
    elif model_type == 'temporal_baseline':
        return DynamicsNetworkTemporalBaseline(
            **common,
            buffer_size=buffer_size,
        )
    elif model_type == 'spatiotemporal_stu':
        return DynamicsNetworkSpatioTemporalSTU(
            **common,
            buffer_size=buffer_size,
            dynamics_stu_seq_len=dynamics_stu_seq_len,
            dynamics_stu_num_filters=dynamics_stu_num_filters,
        )
    elif model_type == 'temporal_stu':
        return DynamicsNetworkTemporalSTU(
            **common,
            buffer_size=buffer_size,
            dynamics_stu_num_filters=dynamics_stu_num_filters,
        )
    elif model_type == 'temporal_mamba':
        return DynamicsNetworkTemporalMamba(
            **common,
            buffer_size=buffer_size,
            d_state=mamba_d_state,
            d_conv=mamba_d_conv,
            expand=mamba_expand,
        )
    elif model_type == 'temporal_attention':
        return DynamicsNetworkTemporalAttention(
            **common,
            buffer_size=buffer_size,
            num_heads=attn_num_heads,
        )
    else:
        return DynamicsNetwork(**common)


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
                   batch, device, consistency_weight=5.0, is_temporal=False):
    """Compute MSE + consistency loss for a batch.

    Replicates the online consistency loss from ez/agents/base.py lines 474-477:
      predicted branch: projection_head(projection(G(s_t, a_t)))  [with grad]
      target branch:    projection(s_{t+1}).detach()               [no grad]
    """
    actions = batch['action'].to(device).unsqueeze(-1)  # [B, 1]
    next_states = batch['next_state'].to(device)

    # Dynamics prediction — temporal models take history, spatial take single state
    if is_temporal:
        history = batch['history'].to(device)  # [B, N, C, H, W]
        result = dynamics_model(history, actions)
        # Some temporal models return (output, buffer_state) tuple
        pred_next = result[0] if isinstance(result, tuple) else result
    else:
        states = batch['state'].to(device)
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
                    grad_clip=5.0, is_temporal=False):
    """Train for one epoch."""
    dynamics_model.train()

    epoch_metrics = defaultdict(float)
    n_batches = 0

    for batch in train_loader:
        loss, loss_dict = compute_losses(
            dynamics_model, projection_model, projection_head_model,
            batch, device, consistency_weight, is_temporal=is_temporal,
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
             eval_loader, device, consistency_weight=5.0, is_temporal=False):
    """Evaluate on held-out episodes."""
    dynamics_model.eval()

    epoch_metrics = defaultdict(float)
    n_batches = 0

    for batch in eval_loader:
        _, loss_dict = compute_losses(
            dynamics_model, projection_model, projection_head_model,
            batch, device, consistency_weight, is_temporal=is_temporal,
        )
        for k, v in loss_dict.items():
            epoch_metrics[k] += v
        n_batches += 1

    return {k: v / n_batches for k, v in epoch_metrics.items()}


@torch.no_grad()
def evaluate_multistep(dynamics_model, seq_dataset, device, is_temporal=False,
                       buffer_size=10):
    """Evaluate multi-step autoregressive rollout error.

    Starting from ground-truth s_0, repeatedly applies the dynamics model:
        s_hat_{k+1} = G(s_hat_k, a_k)
    and measures MSE(s_hat_k, s_k) at each step.

    For temporal models, maintains a FIFO buffer of size buffer_size.
    The buffer is warmed up with ground-truth states: the first
    min(buffer_size, K+1) states from the rollout window are used to fill
    the buffer. Autoregressive prediction starts after the warm-up period,
    and only post-warm-up steps are measured.

    Returns dict mapping step number -> mean MSE.
    """
    dynamics_model.eval()
    loader = DataLoader(seq_dataset, batch_size=64, shuffle=False)

    step_errors = defaultdict(list)

    for batch in loader:
        gt_states = batch['states'].to(device)   # [B, K+1, C, H, W]
        actions = batch['actions'].to(device)     # [B, K]
        K = actions.shape[1]
        B = gt_states.shape[0]

        current = gt_states[:, 0]  # start from ground truth

        if is_temporal:
            # Warm up buffer with ground-truth states
            warmup_len = min(buffer_size, K + 1)
            if warmup_len >= buffer_size:
                # Enough GT states to fill the buffer
                history = gt_states[:, :buffer_size]  # [B, N, C, H, W]
                current = gt_states[:, buffer_size - 1]
                start_step = buffer_size - 1  # first action to predict from
            else:
                # Not enough GT states — pad left with repeated first state
                pad = gt_states[:, 0:1].repeat(1, buffer_size - warmup_len, 1, 1, 1)
                history = torch.cat([pad, gt_states[:, :warmup_len]], dim=1)
                current = gt_states[:, warmup_len - 1]
                start_step = warmup_len - 1

            for step in range(start_step, K):
                action = actions[:, step].unsqueeze(-1)
                result = dynamics_model(history, action)
                # Some temporal models return (output, buffer_state) tuple
                if isinstance(result, tuple):
                    current, buffer_state = result
                else:
                    current = result
                    buffer_state = current
                # Shift buffer: drop oldest, append buffer_state (not final output)
                history = torch.cat([history[:, 1:], buffer_state.unsqueeze(1)], dim=1)

                gt = gt_states[:, step + 1]
                mse = F.mse_loss(current, gt, reduction='none')
                mse_per_sample = mse.reshape(B, -1).mean(dim=1)
                step_errors[step + 1].extend(mse_per_sample.cpu().tolist())
        else:
            for step in range(K):
                action = actions[:, step].unsqueeze(-1)
                current = dynamics_model(current, action)

                gt = gt_states[:, step + 1]
                mse = F.mse_loss(current, gt, reduction='none')
                mse_per_sample = mse.reshape(B, -1).mean(dim=1)
                step_errors[step + 1].extend(mse_per_sample.cpu().tolist())

    return {step: np.mean(errs) for step, errs in sorted(step_errors.items())}


# ===========================================================================
# Main
# ===========================================================================

def parse_args():
    p = argparse.ArgumentParser(
        description='Offline dynamics network training for STUZero',
    )
    # Game selection
    p.add_argument('--game', type=str, default=None,
                   choices=list(GAME_CONFIGS.keys()),
                   help='Game name — auto-resolves data_dir and checkpoint')
    p.add_argument('--download', action='store_true',
                   help='Download dataset from HuggingFace Hub before training')

    # Data (overrides --game defaults if specified)
    p.add_argument('--data_dir', type=str, default=None,
                   help='Path to dataset directory (auto from --game if omitted)')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Path to EfficientZero checkpoint (auto from --game if omitted)')
    p.add_argument('--train_split', type=float, default=0.8,
                   help='Fraction of episodes for training')

    # Architecture
    p.add_argument('--model_type', type=str, default='baseline',
                   choices=['baseline', 'stu', 'mamba', 'attention', 'residual_only',
                            'temporal_baseline', 'spatiotemporal_stu',
                            'temporal_stu', 'temporal_mamba', 'temporal_attention'],
                   help='Dynamics network variant')
    p.add_argument('--buffer_size', type=int, default=10,
                   help='History buffer size for temporal models (default: 10)')
    p.add_argument('--use_stu', action='store_true',
                   help='(Deprecated) Use --model_type=stu instead')
    p.add_argument('--action_space_size', type=int, default=None,
                   help='Override action space size (auto from metadata)')
    p.add_argument('--num_blocks', type=int, default=1)
    p.add_argument('--num_channels', type=int, default=64)
    p.add_argument('--action_embedding_dim', type=int, default=16)
    # STU-specific
    p.add_argument('--dynamics_stu_seq_len', type=int, default=36)
    p.add_argument('--dynamics_stu_num_filters', type=int, default=2)
    # Mamba-specific
    p.add_argument('--mamba_d_state', type=int, default=16)
    p.add_argument('--mamba_d_conv', type=int, default=4)
    p.add_argument('--mamba_expand', type=int, default=1)
    # Attention-specific
    p.add_argument('--attn_num_heads', type=int, default=4)

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

    # --- Resolve game defaults ---
    if args.game:
        gc = GAME_CONFIGS[args.game]
        if args.data_dir is None:
            args.data_dir = gc['data_subdir']
        if args.checkpoint is None:
            args.checkpoint = gc['checkpoint']
        if args.action_space_size is None:
            args.action_space_size = gc['action_space_size']
        if args.save_dir == 'dynamics_checkpoints':
            args.save_dir = f'dynamics_checkpoints/{args.game}'
        print(f"Game: {args.game} | data_dir={args.data_dir} | "
              f"checkpoint={args.checkpoint}")

    if args.data_dir is None or args.checkpoint is None:
        raise ValueError(
            "Specify --game or provide both --data_dir and --checkpoint"
        )

    # --- Optional HuggingFace download ---
    if args.download:
        if args.game is None:
            raise ValueError("--download requires --game to be specified")
        args.data_dir = download_game_data(args.game, base_dir='.')

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
    is_temporal = args.model_type.startswith('temporal_') or args.model_type == 'spatiotemporal_stu'

    if is_temporal:
        print(f"\nLoading temporal training data (buffer_size={args.buffer_size})...")
        train_dataset = TemporalDynamicsDataset(train_episodes, buffer_size=args.buffer_size)
        print("Loading temporal eval data...")
        eval_dataset = TemporalDynamicsDataset(eval_episodes, buffer_size=args.buffer_size)
    else:
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

    # Back-compat: --use_stu flag overrides model_type
    model_type = args.model_type
    if args.use_stu and model_type == 'baseline':
        model_type = 'stu'

    dynamics_model = build_dynamics_network(
        action_space_size=action_space_size,
        model_type=model_type,
        num_blocks=args.num_blocks,
        num_channels=args.num_channels,
        action_embedding_dim=args.action_embedding_dim,
        dynamics_stu_seq_len=args.dynamics_stu_seq_len,
        dynamics_stu_num_filters=args.dynamics_stu_num_filters,
        mamba_d_state=args.mamba_d_state,
        mamba_d_conv=args.mamba_d_conv,
        mamba_expand=args.mamba_expand,
        attn_num_heads=args.attn_num_heads,
        buffer_size=args.buffer_size,
    ).to(device)

    projection_model, projection_head_model = load_frozen_projection_networks(
        args.checkpoint, device,
    )

    n_params = sum(p.numel() for p in dynamics_model.parameters()
                   if p.requires_grad)
    print(f"Dynamics model: {type(dynamics_model).__name__}, "
          f"{n_params:,} trainable params")
    print(f"Model: {model_type}")

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
        wandb_tags = [model_type]
        if args.game:
            wandb_tags.insert(0, args.game)
        wandb.init(
            project=args.wandb_project,
            name=args.wandb_run_name,
            config=vars(args),
            tags=wandb_tags,
        )
        wandb.define_metric("rollout/best_final_step_mse_per_epoch", step_metric="epoch")
        wandb.define_metric("epoch")

    # Save directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # --- Training loop ---
    print(f"\n--- Training for {args.epochs} epochs ---\n")
    best_eval_mse = float('inf')
    best_rollout_final_mse = float('inf')
    best_rollout_data = None
    best_rollout_epoch = None
    all_rollout_rows = []  # accumulated across all eval epochs

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        train_metrics = train_one_epoch(
            dynamics_model, projection_model, projection_head_model,
            train_loader, optimizer, device,
            args.consistency_weight, args.grad_clip,
            is_temporal=is_temporal,
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
                is_temporal=is_temporal,
            )
            multistep = evaluate_multistep(
                dynamics_model, seq_eval_dataset, device,
                is_temporal=is_temporal, buffer_size=args.buffer_size,
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

            # Track best rollout by final-step MSE
            final_step = max(multistep.keys())
            final_step_mse = multistep[final_step]
            if final_step_mse < best_rollout_final_mse:
                best_rollout_final_mse = final_step_mse
                best_rollout_data = dict(multistep)
                best_rollout_epoch = epoch
                print(f"  ** New best rollout (step {final_step} MSE: "
                      f"{final_step_mse:.6f}, epoch {epoch})")

            if args.use_wandb:
                import wandb
                log = {f'train/{k}': v for k, v in train_metrics.items()}
                log.update({f'eval/{k}': v for k, v in eval_metrics.items()})
                log['lr'] = lr
                log['epoch'] = epoch
                log["rollout/best_final_step_mse_per_epoch"] = best_rollout_final_mse

                # Accumulate this eval's rollout data (logged once at end)
                for step_k, mse_v in sorted(multistep.items()):
                    all_rollout_rows.append([epoch, step_k, mse_v])

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
        # Log all rollout curves table once at end of training
        all_curves_table = wandb.Table(
            data=all_rollout_rows,
            columns=["epoch", "rollout_step", "mse"],
        )

        # Best rollout curve as graph + table
        best_table = wandb.Table(
            data=[[step_k, mse_v, best_rollout_epoch]
                  for step_k, mse_v in sorted(best_rollout_data.items())],
            columns=["rollout_step", "mse", "best_epoch"],
        )
        best_curve_plot = wandb.plot.line(
            best_table, "rollout_step", "mse",
            title="MSE over Best Rollout",
        )

        wandb.log({
            "rollout/all_curves_table": all_curves_table,
            "rollout/best_curve_table": best_table,
            "rollout/best_curve": best_curve_plot,
        })
        wandb.finish()


if __name__ == '__main__':
    main()
