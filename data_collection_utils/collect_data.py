"""
collect_data.py

Collect a dynamics training dataset from a trained EfficientZero benchmark checkpoint.
Runs self-play episodes with MCTS and records transitions including latent states,
dynamics predictions, and projections for offline dynamics network training.

IMPORTANT: The model architecture in ez/agents/ez_atari.py must match the checkpoint
you are loading. If your benchmark was trained with the standard DynamicsNetwork +
ValuePolicyNetwork (no STU), uncomment those options and comment out the STU variants
in ez_atari.py before running this script.

Usage:
    cd STUZero
python collect_data.py \
    exp_config=ez/config/exp/atari.yaml \
    eval.model_path=model_100000.p \
    +collect.n_episodes=50 \
    +collect.max_steps=27000 \
    +collect.output_dir=dynamics_dataset

Output structure:
    dynamics_dataset/
    ├── README.md           # HuggingFace dataset card
    ├── metadata.json       # Dataset config, shapes, statistics
    ├── episode_0000.pt     # Per-episode data (PyTorch tensors)
    ├── episode_0001.pt
    └── ...

Each episode_XXXX.pt contains a dict with keys:
    frames:                [T, H, W, C] uint8   - raw observation frame per step
    actions:               [T]          long     - discrete action taken
    rewards:               [T]          float32  - reward received
    latent_states:         [T, 64, 6, 6] float32 - H(stacked_obs_t)
    next_latent_states:    [T, 64, 6, 6] float32 - H(stacked_obs_{t+1}) ground truth
    dynamics_predictions:  [T, 64, 6, 6] float32 - G(s_t, a_t) benchmark prediction
    projections:           [T, D]        float32 - projection of s_t
    dynamics_projections:  [T, D]        float32 - projection of G(s_t, a_t)
    dones:                 [T]          bool     - episode termination flags
    valid_next:            [T]          bool     - whether next_latent_state is valid

Upload to HuggingFace:
    pip install huggingface_hub
    huggingface-cli login
    from huggingface_hub import HfApi
    api = HfApi()
    api.create_repo('YOUR_USERNAME/stuzero-pong-dynamics', repo_type='dataset')
    api.upload_folder(
        folder_path='dynamics_dataset',
        repo_id='YOUR_USERNAME/stuzero-pong-dynamics',
        repo_type='dataset',
    )
"""

import os
import sys
sys.path.append(os.getcwd())

import json
import torch
import ray
import numpy as np
import multiprocessing
from pathlib import Path
from tqdm.auto import tqdm
from torch.cuda.amp import autocast
from omegaconf import OmegaConf, open_dict
import hydra

from ez import agents, mcts
from ez.envs import make_envs
from ez.utils.format import formalize_obs_lst


@torch.no_grad()
def collect_dataset(agent, model, config, n_episodes, max_steps, output_dir):
    """Run self-play episodes and collect dynamics training data.

    Adapted from ez/eval.py. At each step, records the latent state, action,
    dynamics prediction, and projections. Ground-truth next latent states are
    computed by shifting the latent_states array by one timestep.

    Args:
        agent: EZAtariAgent instance (with updated config)
        model: Loaded EfficientZero model on CUDA
        config: Hydra config (must match checkpoint architecture)
        n_episodes: Number of episodes to collect
        max_steps: Maximum steps per episode before forced termination
        output_dir: Directory to write episode .pt files

    Returns:
        dict with collection statistics
    """
    model.cuda()
    model.eval()

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stats = {
        'total_transitions': 0,
        'episode_lengths': [],
        'episode_rewards': [],
    }

    # Process episodes in batches (multiple envs in parallel)
    batch_size = min(n_episodes, 8)
    episodes_collected = 0

    while episodes_collected < n_episodes:
        current_batch = min(batch_size, n_episodes - episodes_collected)

        # Create environments
        envs = make_envs(
            config.env.env, config.env.game, current_batch,
            config.env.base_seed + episodes_collected,
            save_path=None, episodic_life=False, **config.env
        )

        # Initialize observation stacks and trajectories
        stack_obs_windows, game_trajs = agent.init_envs(envs, max_steps)
        [traj.set_inf_len() for traj in game_trajs]

        dones = np.array([False] * current_batch)

        # Per-env accumulators
        accum = [{
            'frames': [],
            'actions': [],
            'rewards': [],
            'latent_states': [],
            'dynamics_predictions': [],
            'projections': [],
            'dynamics_projections': [],
            'dones': [],
        } for _ in range(current_batch)]

        step = 0
        pb = tqdm(
            total=max_steps,
            desc=f"Episodes {episodes_collected+1}-{episodes_collected+current_batch}",
            leave=False,
        )

        while not dones.all() and step < max_steps:
            # Format stacked observations -> CUDA tensor [B, n_stack*C, H, W]
            stacked_obs = formalize_obs_lst(
                stack_obs_windows, image_based=config.env.image_based
            )

            # Forward pass: representation -> value/policy
            # training=False returns processed values for MCTS;
            # `states` is the same raw latent tensor in both modes
            with autocast():
                states, values, policies = model.initial_inference(stacked_obs)

            # Projections of current latent states
            with autocast():
                projections = model.do_projection(states, with_grad=False)

            # Detach to CPU for storage
            latent_cpu = states.detach().cpu()
            proj_cpu = projections.detach().cpu()
            values_np = values.detach().cpu().numpy().flatten()

            # MCTS action selection
            tree = mcts.names[config.mcts.language](
                num_actions=config.env.action_space_size,
                discount=config.rl.discount,
                env=config.env.env,
                **config.mcts,
                **config.model,
            )

            if config.mcts.use_gumbel:
                _, _, best_actions, _ = tree.search(
                    model, current_batch, states, values_np, policies,
                    use_gumble_noise=False,
                )
            else:
                _, _, best_actions, _ = tree.search_ori_mcts(
                    model, current_batch, states, values_np, policies,
                    use_noise=False,
                )

            # Per-env: compute dynamics prediction, step environment, store data
            for i in range(current_batch):
                if dones[i]:
                    continue

                action = best_actions[i]

                # Dynamics prediction: G(s_t, a_t) -> s_hat_{t+1}
                action_tensor = torch.tensor([[action]]).cuda().long()
                with autocast():
                    dyn_pred = model.do_dynamics(states[i:i+1], action_tensor)
                    dyn_proj = model.do_projection(dyn_pred, with_grad=False)

                # Step environment
                obs, reward, done, info = envs[i].step(action)

                # Store transition data
                a = accum[i]
                # Raw frame: last frame in the observation stack (before stepping)
                a['frames'].append(
                    np.array(stack_obs_windows[i][-1], dtype=np.uint8)
                )
                a['actions'].append(int(action))
                a['rewards'].append(float(info['raw_reward']))
                a['latent_states'].append(latent_cpu[i].numpy())
                a['dynamics_predictions'].append(
                    dyn_pred.detach().cpu().squeeze(0).numpy()
                )
                a['projections'].append(proj_cpu[i].numpy())
                a['dynamics_projections'].append(
                    dyn_proj.detach().cpu().squeeze(0).numpy()
                )
                a['dones'].append(bool(done))

                dones[i] = done

                # Update observation stack for next step
                del stack_obs_windows[i][0]
                stack_obs_windows[i].append(obs)

                # Feed trajectory (needed for init_envs bookkeeping)
                game_trajs[i].store_search_results(values_np[i], 0.0, np.zeros(config.env.action_space_size))
                game_trajs[i].append(action, obs, reward)

            step += 1
            pb.update(1)

        pb.close()

        # Save each completed episode as a .pt file
        for i in range(current_batch):
            a = accum[i]
            T = len(a['actions'])
            if T == 0:
                continue

            # Ground-truth next latent states via shift:
            #   next_latent_state[t] = latent_state[t+1]
            latent_arr = np.stack(a['latent_states'])       # [T, C, H, W]
            next_latent = np.zeros_like(latent_arr)
            next_latent[:-1] = latent_arr[1:]
            next_latent[-1] = latent_arr[-1]                # placeholder

            # Mark which transitions have a valid ground-truth next state
            valid_next = np.ones(T, dtype=bool)
            valid_next[-1] = False
            for t in range(T - 1):
                if a['dones'][t]:
                    valid_next[t] = False

            episode_data = {
                'frames': torch.from_numpy(np.stack(a['frames'])),
                'actions': torch.tensor(a['actions'], dtype=torch.long),
                'rewards': torch.tensor(a['rewards'], dtype=torch.float32),
                'latent_states': torch.from_numpy(latent_arr),
                'next_latent_states': torch.from_numpy(next_latent),
                'dynamics_predictions': torch.from_numpy(
                    np.stack(a['dynamics_predictions'])
                ),
                'projections': torch.from_numpy(np.stack(a['projections'])),
                'dynamics_projections': torch.from_numpy(
                    np.stack(a['dynamics_projections'])
                ),
                'dones': torch.tensor(a['dones'], dtype=torch.bool),
                'valid_next': torch.from_numpy(valid_next),
            }

            ep_idx = episodes_collected + i
            torch.save(episode_data, output_path / f'episode_{ep_idx:04d}.pt')

            ep_reward = sum(a['rewards'])
            stats['episode_lengths'].append(T)
            stats['episode_rewards'].append(ep_reward)
            stats['total_transitions'] += T

            print(f"  Episode {ep_idx}: {T} steps, reward={ep_reward:.1f}")

        episodes_collected += current_batch
        for env in envs:
            env.close()

    return stats


def save_metadata(output_dir, config, checkpoint_path, n_episodes, stats):
    """Save dataset metadata as JSON for reproducibility."""
    metadata = {
        'description': (
            'Dynamics training dataset collected from a trained EfficientZero '
            'benchmark on Atari. Contains latent states, actions, ground-truth '
            'next states, and benchmark dynamics predictions for offline '
            'dynamics network development.'
        ),
        'environment': config.env.game,
        'checkpoint': str(checkpoint_path),
        'n_episodes': n_episodes,
        'config': {
            'n_stack': config.env.n_stack,
            'obs_shape': list(config.env.obs_shape),
            'action_space_size': config.env.action_space_size,
            'gray_scale': config.env.gray_scale,
            'num_channels': config.model.num_channels,
            'down_sample': config.model.down_sample,
            'state_norm': config.model.state_norm,
            'discount': config.rl.discount,
        },
        'shapes': {
            'frames': f'[T, {config.env.obs_shape[1]}, {config.env.obs_shape[2]}, {config.env.obs_shape[0]}]',
            'latent_states': f'[T, {config.model.num_channels}, 6, 6]',
            'projections': f'[T, {config.model.projection_layers[1]}]',
        },
        'stats': {
            'total_transitions': stats['total_transitions'],
            'n_episodes_collected': len(stats['episode_lengths']),
            'mean_episode_length': float(np.mean(stats['episode_lengths'])),
            'std_episode_length': float(np.std(stats['episode_lengths'])),
            'mean_episode_reward': float(np.mean(stats['episode_rewards'])),
            'min_episode_reward': float(np.min(stats['episode_rewards'])),
            'max_episode_reward': float(np.max(stats['episode_rewards'])),
        },
        'fields': {
            'frames': '[T, H, W, C] uint8 - raw observation frame at each step',
            'actions': '[T] long - discrete action taken',
            'rewards': '[T] float32 - reward received',
            'latent_states': '[T, C, 6, 6] float32 - s_t = H(stacked_obs_t)',
            'next_latent_states': '[T, C, 6, 6] float32 - s_{t+1} ground truth',
            'dynamics_predictions': '[T, C, 6, 6] float32 - G(s_t, a_t) benchmark',
            'projections': '[T, D] float32 - projection of s_t',
            'dynamics_projections': '[T, D] float32 - projection of G(s_t, a_t)',
            'dones': '[T] bool - episode termination',
            'valid_next': '[T] bool - True if next_latent_state is a real next state',
        },
        'notes': {
            'stacking': (
                f'Latent states are computed from {config.env.n_stack} stacked '
                f'frames. At step t, stacked_obs = [frame_{{t-{config.env.n_stack-1}}}'
                f', ..., frame_t]. At episode start (t < {config.env.n_stack}), '
                f'earlier positions are filled with copies of the initial frame.'
            ),
            'normalization': (
                'Frames are stored as uint8 [0-255]. The representation network '
                'expects float32 inputs normalized by dividing by 255.'
            ),
            'valid_next': (
                'valid_next[t]=False at the last step of an episode or at terminal '
                'states. next_latent_states[t] is meaningless when valid_next[t]=False.'
            ),
        },
    }

    with open(Path(output_dir) / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


def save_dataset_card(output_dir, config, stats):
    """Create a HuggingFace dataset card (README.md)."""
    n_eps = len(stats['episode_lengths'])
    mean_len = np.mean(stats['episode_lengths'])
    mean_rew = np.mean(stats['episode_rewards'])

    card = f"""---
license: mit
task_categories:
  - reinforcement-learning
tags:
  - world-models
  - dynamics-prediction
  - atari
  - efficientzero
  - spectral-transfer-units
size_categories:
  - {"1K<n<10K" if stats['total_transitions'] < 10000
     else "10K<n<100K" if stats['total_transitions'] < 100000
     else "100K<n<1M"}
---

# STUZero {config.env.game} Dynamics Dataset

Offline dynamics training dataset collected from a trained EfficientZero
benchmark model on Atari {config.env.game}.

## Purpose

Train and evaluate alternative dynamics network architectures (e.g., STU-based)
**offline**, without running full online training (~15 hours per experiment).
Provides a direct dynamics loss signal via ground-truth next latent states.

## Statistics

| Metric | Value |
|--------|-------|
| Episodes | {n_eps} |
| Total transitions | {stats['total_transitions']:,} |
| Mean episode length | {mean_len:.0f} steps |
| Mean episode reward | {mean_rew:.1f} |

## Data Fields

Each `episode_XXXX.pt` file is a PyTorch dict:

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `frames` | `[T, H, W, C]` | uint8 | Raw observation frame |
| `actions` | `[T]` | long | Discrete action |
| `rewards` | `[T]` | float32 | Reward |
| `latent_states` | `[T, {config.model.num_channels}, 6, 6]` | float32 | H(stacked_obs_t) |
| `next_latent_states` | `[T, {config.model.num_channels}, 6, 6]` | float32 | H(stacked_obs_{{t+1}}) ground truth |
| `dynamics_predictions` | `[T, {config.model.num_channels}, 6, 6]` | float32 | G(s_t, a_t) benchmark |
| `projections` | `[T, {config.model.projection_layers[1]}]` | float32 | Projected s_t |
| `dynamics_projections` | `[T, {config.model.projection_layers[1]}]` | float32 | Projected G(s_t, a_t) |
| `dones` | `[T]` | bool | Terminal flag |
| `valid_next` | `[T]` | bool | Whether next state is valid |

## Quick Start

```python
import torch
from pathlib import Path

# Load one episode
ep = torch.load('episode_0000.pt')

# Extract valid transitions for dynamics training
valid = ep['valid_next']
s_t    = ep['latent_states'][valid]          # [N, {config.model.num_channels}, 6, 6]
a_t    = ep['actions'][valid]                # [N]
s_next = ep['next_latent_states'][valid]     # [N, {config.model.num_channels}, 6, 6] target

# Benchmark dynamics predictions for comparison
s_pred = ep['dynamics_predictions'][valid]   # [N, {config.model.num_channels}, 6, 6]

# Load all episodes into a single dataset
episodes = sorted(Path('.').glob('episode_*.pt'))
all_s, all_a, all_s_next = [], [], []
for ep_path in episodes:
    ep = torch.load(ep_path)
    v = ep['valid_next']
    all_s.append(ep['latent_states'][v])
    all_a.append(ep['actions'][v])
    all_s_next.append(ep['next_latent_states'][v])

s_t    = torch.cat(all_s)
a_t    = torch.cat(all_a)
s_next = torch.cat(all_s_next)
print(f"Dataset: {{s_t.shape[0]}} transitions")
```

## Configuration

| Parameter | Value |
|-----------|-------|
| Environment | Atari {config.env.game} |
| Observation | {list(config.env.obs_shape)} ({'grayscale' if config.env.gray_scale else 'RGB'}) |
| Frame stacking | {config.env.n_stack} frames |
| Action space | {config.env.action_space_size} discrete actions |
| Latent shape | [{config.model.num_channels}, 6, 6] (16x spatial downsampling) |
| Projection dim | {config.model.projection_layers[1]} |

## Notes

- **Frame stacking:** latent state s_t encodes {config.env.n_stack} stacked frames.
  At episode start, the stack is padded with copies of the initial frame.
- **Normalization:** frames are uint8 [0-255]; the representation network
  expects float32 / 255.
- **valid_next:** always check this mask before using `next_latent_states`.
  Invalid entries occur at episode boundaries and terminal states.
"""

    with open(Path(output_dir) / 'README.md', 'w') as f:
        f.write(card)


def verify_dataset(output_dir):
    """Quick sanity check on the saved dataset."""
    output_path = Path(output_dir)
    episodes = sorted(output_path.glob('episode_*.pt'))

    if not episodes:
        print("WARNING: No episode files found!")
        return False

    print(f"\nVerification: found {len(episodes)} episode files")

    ep = torch.load(episodes[0])
    print(f"  Sample episode: {episodes[0].name}")
    for key, val in ep.items():
        print(f"    {key:25s} {str(val.shape):20s} {val.dtype}")

    T = ep['actions'].shape[0]
    n_valid = ep['valid_next'].sum().item()
    print(f"  Steps: {T}, valid transitions: {n_valid}")

    # Check shapes are consistent
    assert ep['latent_states'].shape == ep['next_latent_states'].shape
    assert ep['latent_states'].shape == ep['dynamics_predictions'].shape
    assert ep['projections'].shape == ep['dynamics_projections'].shape
    assert ep['actions'].shape[0] == T
    assert ep['rewards'].shape[0] == T
    assert ep['dones'].shape[0] == T
    assert ep['valid_next'].shape[0] == T

    print("  All shape checks passed.")
    return True


@hydra.main(config_path="ez/config", config_name='config', version_base='1.1')
def main(config):
    if config.exp_config is not None:
        exp_config = OmegaConf.load(config.exp_config)
        config = OmegaConf.merge(config, exp_config)

    # Collection parameters (pass via +collect.key=value on command line)
    if hasattr(config, 'collect'):
        n_episodes = config.collect.get('n_episodes', 50)
        max_steps = config.collect.get('max_steps', 27000)
        output_dir = config.collect.get('output_dir', 'dynamics_dataset')
    else:
        n_episodes = 50
        max_steps = 27000
        output_dir = 'dynamics_dataset'

    checkpoint_path = config.eval.model_path

    print(f"\n{'='*60}")
    print(f"  Dynamics Dataset Collection")
    print(f"{'='*60}")
    print(f"  Checkpoint:  {checkpoint_path}")
    print(f"  Game:        {config.env.game}")
    print(f"  Episodes:    {n_episodes}")
    print(f"  Max steps:   {max_steps}")
    print(f"  Output:      {output_dir}")
    print(f"{'='*60}\n")

    # Validate checkpoint exists
    if not os.path.exists(checkpoint_path):
        print(f"ERROR: Checkpoint not found at: {checkpoint_path}")
        print(f"Update eval.model_path to point to your trained model_XXXXX.p file.")
        print(f"Example: eval.model_path=results/Atari/Pong/.../models/model_100000.p")
        return

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required. Run this script on a machine with a GPU.")
        return

    # Override max_episode_steps to match collection setting
    with open_dict(config):
        config.env.max_episode_steps = max_steps

    # Initialize agent (updates config with env info like action_space_size)
    agent = agents.names[config.agent_name](config)

    # Initialize Ray (needed for MCTS and trajectory internals)
    num_gpus = torch.cuda.device_count()
    num_cpus = multiprocessing.cpu_count()
    ray.init(
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        object_store_memory=(
            150 * 1024 * 1024 * 1024 if config.env.image_based
            else 100 * 1024 * 1024 * 1024
        ),
    )

    # Build model and load checkpoint
    model = agent.build_model()
    print(f"Loading checkpoint: {checkpoint_path}")
    weights = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(weights)
    print("Checkpoint loaded successfully.")

    if int(torch.__version__[0]) == 2:
        model = torch.compile(model)

    # Collect dataset
    stats = collect_dataset(agent, model, config, n_episodes, max_steps, output_dir)

    # Save metadata and HuggingFace dataset card
    save_metadata(output_dir, config, checkpoint_path, n_episodes, stats)
    save_dataset_card(output_dir, config, stats)

    # Verify output
    verify_dataset(output_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"  Collection complete!")
    print(f"  Episodes:    {len(stats['episode_lengths'])}")
    print(f"  Transitions: {stats['total_transitions']:,}")
    print(f"  Mean reward: {np.mean(stats['episode_rewards']):.1f}")
    print(f"  Output dir:  {output_dir}/")
    print(f"{'='*60}")
    print(f"\n  To upload to HuggingFace Hub:")
    print(f"    pip install huggingface_hub")
    print(f"    huggingface-cli login")
    print(f"    python -c \"")
    print(f"    from huggingface_hub import HfApi")
    print(f"    api = HfApi()")
    print(f"    api.create_repo('YOUR_USER/stuzero-{config.env.game.lower()}-dynamics', repo_type='dataset')")
    print(f"    api.upload_folder(")
    print(f"        folder_path='{output_dir}',")
    print(f"        repo_id='YOUR_USER/stuzero-{config.env.game.lower()}-dynamics',")
    print(f"        repo_type='dataset',")
    print(f"    )\"")
    print()


if __name__ == '__main__':
    main()
