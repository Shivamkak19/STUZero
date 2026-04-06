"""
train_dynamics_offline_dmc.py

Offline dynamics network training for DMC (DeepMind Control) state-based
environments using pre-collected latent state datasets.

Adapted from train_dynamics_offline.py (Atari version). Key differences:
  - Latent states are flat vectors [B, 128] not spatial [B, 64, 6, 6]
  - Actions are continuous [B, action_dim] (float32), not discrete [B] (long)
  - Projection networks are nn.Sequential (not custom classes)
  - Model variants from ez.agents.models.dmc_dynamics

Two phases:
  1. Train:  Train dynamics from scratch with MSE + consistency loss
  2. Eval:   Single-step metrics + multi-step autoregressive rollout

Usage:
    cd STUZero

    # Train on hopper_hop:
    python offline_training/train_dynamics_offline_dmc.py --game dmc_hopper_hop --epochs 80 --batch_size 256

    # Train with STU:
    python offline_training/train_dynamics_offline_dmc.py --game dmc_hopper_hop --model_type stu --epochs 80

    # Train with temporal STU:
    python offline_training/train_dynamics_offline_dmc.py --game dmc_hopper_hop --model_type stu_sequential --buffer_size 10 --epochs 80
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

from ez.agents.models.dmc_dynamics import (
    DMCDynamicsBaseline,
    DMCDynamicsBaselineNorm,
    DMCDynamicsWithSTU,
    DMCDynamicsSTUSequential,
)
from ez.utils.loss import cosine_similarity_loss


# ===========================================================================
# Game configs
# ===========================================================================

GAME_CONFIGS = {
    'dmc_hopper_hop': {
        'data_subdir': 'offline_training/dmc/dmc_hopper_hop_110K',
        'checkpoint': 'offline_training/dmc/dmc_hopper_hop_model_110000.p',
        'action_space_size': 4,
        'hidden_shape': 128,
        'dyn_shape': 256,
        'act_embed_shape': 64,
    },
    'dmc_walker_walk': {
        'data_subdir': 'offline_training/dmc/dmc_walker_walk_110K',
        'checkpoint': 'offline_training/dmc/dmc_walker_walk_model_110000.p',
        'action_space_size': 6,
        'hidden_shape': 128,
        'dyn_shape': 256,
        'act_embed_shape': 64,
    },
    'dmc_cheetah_run': {
        'data_subdir': 'offline_training/dmc/dmc_cheetah_run_110K',
        'checkpoint': 'offline_training/dmc/dmc_cheetah_run_model_110000.p',
        'action_space_size': 6,
        'hidden_shape': 128,
        'dyn_shape': 256,
        'act_embed_shape': 64,
    },
}


# ===========================================================================
# Datasets
# ===========================================================================

class DynamicsDataset(Dataset):
    """Flat dataset of (s_t, a_t, s_{t+1}) transitions from episode files.

    Loads all episodes into memory, filters by valid_next mask, and
    casts float16 -> float32. States are flat [T, hidden_dim].
    Actions are continuous [T, action_dim].
    """

    def __init__(self, episode_paths):
        all_states, all_actions, all_next_states = [], [], []
        all_dynamics_preds, all_projections, all_dynamics_projs = [], [], []

        for ep_path in episode_paths:
            ep = torch.load(ep_path, map_location='cpu', weights_only=False)
            valid = ep['valid_next']

            all_states.append(ep['latent_states'][valid].float())
            all_actions.append(ep['actions'][valid].float())
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
    episode, with all transitions valid. States are [T, hidden_dim],
    actions are [T, action_dim].
    """

    def __init__(self, episode_paths, rollout_len=10, stride=None):
        self.rollout_len = rollout_len
        self.windows = []
        step = stride if stride is not None else rollout_len

        for ep_path in episode_paths:
            ep = torch.load(ep_path, map_location='cpu', weights_only=False)
            valid = ep['valid_next'].numpy()
            T = len(valid)

            states = ep['latent_states'].float()
            actions = ep['actions'].float()
            next_states = ep['next_latent_states'].float()

            t = 0
            while t + rollout_len <= T:
                if valid[t:t + rollout_len].all():
                    window_states = torch.cat([
                        states[t:t + rollout_len],
                        next_states[t + rollout_len - 1:t + rollout_len],
                    ], dim=0)
                    window_actions = actions[t:t + rollout_len]
                    self.windows.append((window_states, window_actions))
                    t += step
                else:
                    t += 1

        print(f"  Built {len(self.windows)} rollout windows of length {rollout_len}"
              f" (stride={step})")

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

    States are flat [T, hidden_dim]. Actions are [T, action_dim].
    """

    def __init__(self, episode_paths, buffer_size=10):
        self.buffer_size = buffer_size
        all_histories, all_actions, all_next_states = [], [], []
        all_dynamics_preds = []

        for ep_path in episode_paths:
            ep = torch.load(ep_path, map_location='cpu', weights_only=False)
            valid = ep['valid_next'].numpy()
            states = ep['latent_states'].float()       # [T, hidden_dim]
            actions = ep['actions'].float()             # [T, action_dim]
            next_states = ep['next_latent_states'].float()
            dynamics_preds = ep['dynamics_predictions'].float()

            T = len(valid)
            hidden_dim = states.shape[1]

            for t in range(T):
                if not valid[t]:
                    continue

                # Build history buffer: states from t-N+1 to t (inclusive)
                start = t - buffer_size + 1
                if start >= 0:
                    history = states[start:t + 1]  # [N, hidden_dim]
                else:
                    # Zero-pad on the left for early timesteps
                    pad_len = -start
                    pad = torch.zeros(pad_len, hidden_dim)
                    history = torch.cat([pad, states[0:t + 1]], dim=0)  # [N, hidden_dim]

                all_histories.append(history)
                all_actions.append(actions[t])
                all_next_states.append(next_states[t])
                all_dynamics_preds.append(dynamics_preds[t])

        self.histories = torch.stack(all_histories)       # [N_total, buffer_size, hidden_dim]
        self.actions = torch.stack(all_actions)            # [N_total, action_dim]
        self.next_states = torch.stack(all_next_states)    # [N_total, hidden_dim]
        self.dynamics_preds = torch.stack(all_dynamics_preds)

        print(f"  Loaded {len(episode_paths)} episodes, "
              f"{len(self.histories)} valid transitions (buffer_size={buffer_size})")

    def __len__(self):
        return len(self.histories)

    def __getitem__(self, idx):
        return {
            'history': self.histories[idx],         # [N, hidden_dim]
            'action': self.actions[idx],            # [action_dim]
            'next_state': self.next_states[idx],    # [hidden_dim]
            'dynamics_pred': self.dynamics_preds[idx],
        }


# ===========================================================================
# Model construction and weight loading
# ===========================================================================

def build_dmc_dynamics_network(model_type, hidden_shape, action_shape,
                                num_blocks, dyn_shape, act_embed_shape,
                                **kwargs):
    """Build a DMC dynamics network variant."""
    if model_type == 'baseline':
        return DMCDynamicsBaseline(hidden_shape, action_shape, num_blocks,
                                   dyn_shape, act_embed_shape)
    elif model_type == 'baseline_norm':
        return DMCDynamicsBaselineNorm(hidden_shape, action_shape, num_blocks,
                                       dyn_shape, act_embed_shape)
    elif model_type == 'stu':
        return DMCDynamicsWithSTU(hidden_shape, action_shape, num_blocks,
                                  dyn_shape, act_embed_shape,
                                  stu_seq_len=kwargs.get('stu_seq_len', 16),
                                  stu_num_filters=kwargs.get('stu_num_filters', 2))
    elif model_type == 'stu_sequential':
        return DMCDynamicsSTUSequential(hidden_shape, action_shape,
                                         buffer_size=kwargs.get('buffer_size', 10),
                                         seq_hidden_dim=kwargs.get('seq_hidden_dim', 256),
                                         stu_num_filters=kwargs.get('stu_num_filters', 2),
                                         num_stu_layers=kwargs.get('num_stu_layers', 2))
    else:
        raise ValueError(f"Unknown model_type: {model_type}")


def load_frozen_projection_networks_dmc(checkpoint_path, hidden_shape,
                                         proj_hid_shape, proj_shape,
                                         pred_hid_shape, pred_shape, device):
    """Load projection + projection head networks from DMC checkpoint, freeze them.

    DMC projections are nn.Sequential with LayerNorm (not BatchNorm1d).
    Projection:      Linear(128->512) -> LN -> ReLU -> Linear(512->512) -> LN -> ReLU -> Linear(512->128) -> LN
    Projection head: Linear(128->512) -> LN -> ReLU -> Linear(512->128)
    """
    projection_model = nn.Sequential(
        nn.Linear(hidden_shape, proj_hid_shape),
        nn.LayerNorm(proj_hid_shape),
        nn.ReLU(),
        nn.Linear(proj_hid_shape, proj_hid_shape),
        nn.LayerNorm(proj_hid_shape),
        nn.ReLU(),
        nn.Linear(proj_hid_shape, proj_shape),
        nn.LayerNorm(proj_shape),
    )

    projection_head_model = nn.Sequential(
        nn.Linear(proj_shape, pred_hid_shape),
        nn.LayerNorm(pred_hid_shape),
        nn.ReLU(),
        nn.Linear(pred_hid_shape, pred_shape),
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


# ===========================================================================
# Training and evaluation
# ===========================================================================

def compute_losses(dynamics_model, projection_model, projection_head_model,
                   batch, device, consistency_weight=5.0, is_temporal=False):
    """Compute MSE + consistency loss for a batch.

    For DMC, actions are already [B, action_dim] -- no unsqueeze needed.
    """
    actions = batch['action'].to(device)  # [B, action_dim] -- continuous, already multi-dim
    next_states = batch['next_state'].to(device)

    # Dynamics prediction -- temporal models take history, flat models take single state
    if is_temporal:
        history = batch['history'].to(device)  # [B, N, hidden_dim]
        result = dynamics_model(history, actions)
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


def train_one_epoch_multistep(dynamics_model, seq_loader, optimizer, device,
                              grad_clip=5.0, is_temporal=False, buffer_size=10):
    """Train for one epoch with multi-step autoregressive loss.

    Unrolls the dynamics model K steps, computing MSE at each step against
    ground truth and backpropagating through the full rollout chain.

    For DMC, actions are [B, K, action_dim] -- no unsqueeze needed.
    """
    dynamics_model.train()

    epoch_metrics = defaultdict(float)
    n_batches = 0

    for batch in seq_loader:
        gt_states = batch['states'].to(device)   # [B, K+1, hidden_dim]
        actions = batch['actions'].to(device)     # [B, K, action_dim]
        K = actions.shape[1]
        B = gt_states.shape[0]

        total_mse = 0.0
        num_steps = 0

        if is_temporal:
            # Initialize history buffer with ground-truth states
            warmup_len = min(buffer_size, K + 1)
            if warmup_len >= buffer_size:
                history = gt_states[:, :buffer_size].clone()
                start_step = buffer_size - 1
            else:
                pad = gt_states[:, 0:1].expand(-1, buffer_size - warmup_len, -1)
                history = torch.cat([pad, gt_states[:, :warmup_len]], dim=1)
                start_step = warmup_len - 1

            for step in range(start_step, K):
                action = actions[:, step]  # [B, action_dim]
                result = dynamics_model(history, action)
                if isinstance(result, tuple):
                    current, buffer_state = result
                else:
                    current = result
                    buffer_state = current

                gt = gt_states[:, step + 1]
                step_mse = F.mse_loss(current, gt)
                total_mse = total_mse + step_mse
                num_steps += 1

                # Shift buffer -- keep computation graph for backprop
                history = torch.cat([history[:, 1:], buffer_state.unsqueeze(1)], dim=1)
        else:
            current = gt_states[:, 0]
            for step in range(K):
                action = actions[:, step]  # [B, action_dim]
                current = dynamics_model(current, action)

                gt = gt_states[:, step + 1]
                step_mse = F.mse_loss(current, gt)
                total_mse = total_mse + step_mse
                num_steps += 1

        if num_steps > 0:
            loss = total_mse / num_steps
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(dynamics_model.parameters(), grad_clip)
            optimizer.step()

            epoch_metrics['mse'] += loss.item()
        n_batches += 1

    return {k: v / max(n_batches, 1) for k, v in epoch_metrics.items()}


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

    Starting from ground-truth s_0, repeatedly applies the dynamics model
    and measures MSE(s_hat_k, s_k) at each step.

    For DMC, actions are [B, K, action_dim] -- no unsqueeze needed.
    """
    dynamics_model.eval()
    loader = DataLoader(seq_dataset, batch_size=64, shuffle=False)

    step_errors = defaultdict(list)

    for batch in loader:
        gt_states = batch['states'].to(device)   # [B, K+1, hidden_dim]
        actions = batch['actions'].to(device)     # [B, K, action_dim]
        K = actions.shape[1]
        B = gt_states.shape[0]

        current = gt_states[:, 0]

        if is_temporal:
            warmup_len = min(buffer_size, K + 1)
            if warmup_len >= buffer_size:
                history = gt_states[:, :buffer_size]
                current = gt_states[:, buffer_size - 1]
                start_step = buffer_size - 1
            else:
                pad = gt_states[:, 0:1].repeat(1, buffer_size - warmup_len, 1)
                history = torch.cat([pad, gt_states[:, :warmup_len]], dim=1)
                current = gt_states[:, warmup_len - 1]
                start_step = warmup_len - 1

            for step in range(start_step, K):
                action = actions[:, step]  # [B, action_dim]
                result = dynamics_model(history, action)
                if isinstance(result, tuple):
                    current, buffer_state = result
                else:
                    current = result
                    buffer_state = current
                history = torch.cat([history[:, 1:], buffer_state.unsqueeze(1)], dim=1)

                gt = gt_states[:, step + 1]
                mse = F.mse_loss(current, gt, reduction='none')
                mse_per_sample = mse.reshape(B, -1).mean(dim=1)
                step_errors[step + 1].extend(mse_per_sample.cpu().tolist())
        else:
            for step in range(K):
                action = actions[:, step]  # [B, action_dim]
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
        description='Offline dynamics network training for STUZero (DMC)',
    )
    # Game selection
    p.add_argument('--game', type=str, default=None,
                   choices=list(GAME_CONFIGS.keys()),
                   help='Game name -- auto-resolves data_dir, checkpoint, and model dims')

    # Data (overrides --game defaults if specified)
    p.add_argument('--data_dir', type=str, default=None,
                   help='Path to dataset directory (auto from --game if omitted)')
    p.add_argument('--checkpoint', type=str, default=None,
                   help='Path to EfficientZero checkpoint (auto from --game if omitted)')
    p.add_argument('--train_split', type=float, default=0.8,
                   help='Fraction of episodes for training')
    p.add_argument('--num_train_samples', type=int, default=None,
                   help='Subsample training transitions (for sample efficiency experiments)')

    # Architecture
    p.add_argument('--model_type', type=str, default='baseline',
                   choices=['baseline', 'baseline_norm', 'stu', 'stu_sequential'],
                   help='Dynamics network variant')
    p.add_argument('--hidden_shape', type=int, default=None,
                   help='Hidden state dimension (default from game config, typically 128)')
    p.add_argument('--dyn_shape', type=int, default=None,
                   help='Dynamics MLP hidden dimension (default from game config, typically 256)')
    p.add_argument('--act_embed_shape', type=int, default=None,
                   help='Action embedding dimension (default from game config, typically 64)')
    p.add_argument('--action_space_size', type=int, default=None,
                   help='Override action space size (auto from game config)')
    p.add_argument('--num_blocks', type=int, default=2,
                   help='Number of residual blocks in dynamics model')

    # STU-specific
    p.add_argument('--stu_seq_len', type=int, default=16,
                   help='Pseudo-sequence length for feature STU (hidden_shape must be divisible)')
    p.add_argument('--stu_num_filters', type=int, default=2,
                   help='Number of STU filters')

    # STU Sequential-specific
    p.add_argument('--buffer_size', type=int, default=10,
                   help='History buffer size for stu_sequential model')
    p.add_argument('--seq_hidden_dim', type=int, default=256,
                   help='Hidden dimension for stu_sequential encoder/decoder')
    p.add_argument('--num_stu_layers', type=int, default=2,
                   help='Number of stacked STU layers for stu_sequential model')

    # Projection network dimensions
    p.add_argument('--proj_hid_shape', type=int, default=512,
                   help='Projection network hidden dimension')
    p.add_argument('--proj_shape', type=int, default=128,
                   help='Projection network output dimension')
    p.add_argument('--pred_hid_shape', type=int, default=512,
                   help='Projection head hidden dimension')
    p.add_argument('--pred_shape', type=int, default=128,
                   help='Projection head output dimension')

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

    # Multi-step training
    p.add_argument('--multistep_train_len', type=int, default=0,
                   help='Multi-step training rollout length (0=disabled, use single-step)')
    p.add_argument('--multistep_stride', type=int, default=5,
                   help='Stride for overlapping multi-step training windows')

    # Evaluation
    p.add_argument('--rollout_len', type=int, default=10)
    p.add_argument('--eval_interval', type=int, default=5)

    # Logging
    p.add_argument('--use_wandb', action='store_true')
    p.add_argument('--wandb_project', type=str, default='stuzero-dmc-dynamics')
    p.add_argument('--wandb_run_name', type=str, default=None)

    # Output
    p.add_argument('--save_dir', type=str, default='dynamics_checkpoints')
    p.add_argument('--save_interval', type=int, default=10)

    # Misc
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--num_workers', type=int, default=0,
                   help='DataLoader workers (0 = main process)')

    return p.parse_args()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if device.type != 'cuda':
        print("WARNING: CUDA not available. Training will run on CPU (slow).")

    # --- Resolve game defaults ---
    if args.game:
        gc = GAME_CONFIGS[args.game]
        if args.data_dir is None:
            args.data_dir = gc['data_subdir']
        if args.checkpoint is None:
            args.checkpoint = gc['checkpoint']
        if args.action_space_size is None:
            args.action_space_size = gc['action_space_size']
        if args.hidden_shape is None:
            args.hidden_shape = gc['hidden_shape']
        if args.dyn_shape is None:
            args.dyn_shape = gc['dyn_shape']
        if args.act_embed_shape is None:
            args.act_embed_shape = gc['act_embed_shape']
        if args.save_dir == 'dynamics_checkpoints':
            args.save_dir = f'dynamics_checkpoints/{args.game}'
        print(f"Game: {args.game} | data_dir={args.data_dir} | "
              f"checkpoint={args.checkpoint}")

    # Apply defaults if not set by game config
    if args.hidden_shape is None:
        args.hidden_shape = 128
    if args.dyn_shape is None:
        args.dyn_shape = 256
    if args.act_embed_shape is None:
        args.act_embed_shape = 64

    if args.data_dir is None or args.checkpoint is None:
        raise ValueError(
            "Specify --game or provide both --data_dir and --checkpoint"
        )

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
    is_temporal = args.model_type == 'stu_sequential'

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

    # Optionally subsample training data for sample efficiency experiments
    if args.num_train_samples is not None and args.num_train_samples < len(train_dataset):
        indices = torch.randperm(len(train_dataset))[:args.num_train_samples]
        train_dataset = torch.utils.data.Subset(train_dataset, indices.tolist())
        print(f"  Subsampled to {len(train_dataset)} training transitions")

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

    # Build multi-step training dataset if requested
    seq_train_loader = None
    if args.multistep_train_len > 0:
        print(f"Building multi-step training windows (len={args.multistep_train_len}, "
              f"stride={args.multistep_stride})...")
        seq_train_dataset = SequentialDynamicsDataset(
            train_episodes, rollout_len=args.multistep_train_len,
            stride=args.multistep_stride,
        )
        seq_train_loader = DataLoader(
            seq_train_dataset, batch_size=min(args.batch_size, 64),
            shuffle=True, num_workers=args.num_workers, pin_memory=True,
            drop_last=True,
        )

    # --- Build trainable dynamics + frozen projections ---
    print("\n--- Training Setup ---")

    dynamics_model = build_dmc_dynamics_network(
        model_type=args.model_type,
        hidden_shape=args.hidden_shape,
        action_shape=action_space_size,
        num_blocks=args.num_blocks,
        dyn_shape=args.dyn_shape,
        act_embed_shape=args.act_embed_shape,
        stu_seq_len=args.stu_seq_len,
        stu_num_filters=args.stu_num_filters,
        buffer_size=args.buffer_size,
        seq_hidden_dim=args.seq_hidden_dim,
        num_stu_layers=args.num_stu_layers,
    ).to(device)

    projection_model, projection_head_model = load_frozen_projection_networks_dmc(
        args.checkpoint,
        hidden_shape=args.hidden_shape,
        proj_hid_shape=args.proj_hid_shape,
        proj_shape=args.proj_shape,
        pred_hid_shape=args.pred_hid_shape,
        pred_shape=args.pred_shape,
        device=device,
    )

    n_params = sum(p.numel() for p in dynamics_model.parameters()
                   if p.requires_grad)
    print(f"Dynamics model: {type(dynamics_model).__name__}, "
          f"{n_params:,} trainable params")
    print(f"Model type: {args.model_type}")

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
        wandb_tags = [args.model_type]
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
    all_rollout_rows = []

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()

        if seq_train_loader is not None:
            train_metrics = train_one_epoch_multistep(
                dynamics_model, seq_train_loader, optimizer, device,
                grad_clip=args.grad_clip,
                is_temporal=is_temporal, buffer_size=args.buffer_size,
            )
        else:
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

        consist_str = (f"  consist={train_metrics['consistency']:.4f}"
                       if 'consistency' in train_metrics else "")
        bench_str = (f"  vs_bench={train_metrics['vs_benchmark_mse']:.6f}"
                     if 'vs_benchmark_mse' in train_metrics else "")
        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"mse={train_metrics['mse']:.6f}{consist_str}{bench_str}  "
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
                try:
                    torch.save(dynamics_model.state_dict(),
                               save_dir / 'best_dynamics.pt')
                except Exception as e:
                    print(f"  WARNING: Failed to save best checkpoint: {e}")
                print(f"  ** New best eval MSE: {best_eval_mse:.6f}")

            # Track best rollout by final-step MSE
            if multistep:
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

                for step_k, mse_v in sorted(multistep.items()):
                    all_rollout_rows.append([epoch, step_k, mse_v])

                wandb.log(log, step=epoch)

        # Save periodic checkpoint
        if epoch % args.save_interval == 0:
            try:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': dynamics_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'args': vars(args),
                }, save_dir / f'dynamics_epoch_{epoch:04d}.pt')
            except Exception as e:
                print(f"  WARNING: Failed to save checkpoint: {e}")

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
        if best_rollout_data:
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
        else:
            wandb.log({
                "rollout/all_curves_table": all_curves_table,
            })
        wandb.finish()


if __name__ == '__main__':
    main()
