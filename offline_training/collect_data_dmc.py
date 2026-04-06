"""
collect_data_dmc.py

Collect a dynamics training dataset from a trained EfficientZero benchmark checkpoint
for DMC (DeepMind Control) state-based environments.

Adapted from data_collection_utils/collect_data.py for Atari. Key differences:
  - Uses dmc_state_agent (EZDMCStateAgent) instead of Atari agent
  - Uses make_dmc (single env) instead of make_envs (batched)
  - Observations are state vectors (float64), not images
  - Actions are continuous (float32, [action_dim]), not discrete
  - Latent states are flat vectors [B, 128], not spatial grids [B, 64, 6, 6]
  - n_stack=1 for state-based DMC
  - Saves raw state observations instead of image frames

Usage:
    cd STUZero
    conda run -n ezv2 python offline_training/collect_data_dmc.py \
        exp_config=ez/config/exp/dmc_state.yaml \
        eval.model_path=offline_training/dmc/dmc_hopper_hop_model_110000.p \
        +collect.n_episodes=50 \
        +collect.max_steps=1000 \
        +collect.output_dir=offline_training/dmc/dmc_hopper_hop_110K

Output structure:
    output_dir/
    ├── metadata.json       # Dataset config, shapes, statistics
    ├── episode_0000.pt     # Per-episode data (PyTorch tensors)
    ├── episode_0001.pt
    └── ...

Each episode_XXXX.pt contains a dict with keys:
    observations:          [T, obs_dim]    float64  - raw state observation per step
    actions:               [T, action_dim] float32  - continuous action taken
    rewards:               [T]             float32  - reward received
    latent_states:         [T, hidden_dim] float32  - H(obs_t)
    next_latent_states:    [T, hidden_dim] float32  - H(obs_{t+1}) ground truth
    dynamics_predictions:  [T, hidden_dim] float32  - G(s_t, a_t) benchmark prediction
    projections:           [T, proj_dim]   float32  - projection of s_t
    dynamics_projections:  [T, proj_dim]   float32  - projection of G(s_t, a_t)
    dones:                 [T]             bool     - episode termination flags
    valid_next:            [T]             bool     - whether next_latent_state is valid
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
from ez.envs import make_dmc
from ez.utils.format import formalize_obs_lst


@torch.no_grad()
def collect_dataset(agent, model, config, n_episodes, max_steps, output_dir):
    """Run self-play episodes and collect dynamics training data for DMC.

    At each step, records the latent state, continuous action, dynamics prediction,
    and projections. Ground-truth next latent states are computed by shifting the
    latent_states array by one timestep.

    Args:
        agent: EZDMCStateAgent instance (with updated config)
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

        # Create environments individually (make_dmc returns a single env)
        envs = []
        for i in range(current_batch):
            env = make_dmc(
                config.env.game,
                seed=config.env.base_seed + episodes_collected + i,
                save_path=None,
                **config.env,
            )
            envs.append(env)

        # Initialize observation stacks and trajectories
        stack_obs_windows, game_trajs = agent.init_envs(envs, max_steps)
        [traj.set_inf_len() for traj in game_trajs]

        dones = np.array([False] * current_batch)

        # Per-env accumulators
        accum = [{
            'observations': [],
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
            # Format stacked observations -> CUDA tensor
            # For state-based DMC: [B, n_stack, obs_dim] -> [B, n_stack*obs_dim]
            stacked_obs = formalize_obs_lst(
                stack_obs_windows, image_based=config.env.image_based
            )

            # Forward pass: representation -> value/policy
            with autocast():
                states, values, policies = model.initial_inference(stacked_obs)

            # Projections of current latent states
            with autocast():
                projections = model.do_projection(states, with_grad=False)

            # Detach to CPU for storage
            latent_cpu = states.detach().cpu()
            proj_cpu = projections.detach().cpu()
            values_np = values.detach().cpu().numpy().flatten()

            # MCTS action selection (continuous for DMC)
            # For continuous control, num_actions = num_sampled_actions (not action_space_size)
            # because MCTS samples N candidate actions and treats them as discrete choices
            tree = mcts.names[config.mcts.language](
                num_actions=config.mcts.num_sampled_actions,
                discount=config.rl.discount,
                env=config.env.env,
                **config.mcts,
                **config.model,
            )

            # DMC uses search_continuous for continuous action spaces
            r_values, r_policies, best_actions, _, _, _ = tree.search_continuous(
                model, current_batch, states, values_np, policies,
                use_gumble_noise=False, add_noise=False,
            )

            # Per-env: compute dynamics prediction, step environment, store data
            for i in range(current_batch):
                if dones[i]:
                    continue

                action = best_actions[i]  # continuous action, shape [action_dim]

                # Dynamics prediction: G(s_t, a_t) -> s_hat_{t+1}
                # For DMC, action is a continuous float tensor [1, action_dim]
                action_tensor = torch.from_numpy(
                    np.array(action, dtype=np.float32)
                ).unsqueeze(0).cuda()
                with autocast():
                    dyn_pred = model.do_dynamics(states[i:i+1], action_tensor)
                    dyn_proj = model.do_projection(dyn_pred, with_grad=False)

                # Step environment with continuous action
                obs, reward, done, info = envs[i].step(action)

                # Store transition data
                a = accum[i]
                # Raw state observation (last obs in the stack, before stepping)
                a['observations'].append(
                    np.array(stack_obs_windows[i][-1], dtype=np.float64)
                )
                a['actions'].append(np.array(action, dtype=np.float32))
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
                # For continuous actions, policy is the raw policy output
                game_trajs[i].store_search_results(
                    values_np[i], 0.0, r_policies[i]
                )
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
            latent_arr = np.stack(a['latent_states'])       # [T, hidden_dim]
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
                'observations': torch.from_numpy(np.stack(a['observations'])),
                'actions': torch.from_numpy(np.stack(a['actions'])),
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
            'benchmark on DMC (DeepMind Control) state-based environment. '
            'Contains latent states, continuous actions, ground-truth next states, '
            'and benchmark dynamics predictions for offline dynamics network training.'
        ),
        'environment': config.env.game,
        'env_type': 'DMC',
        'checkpoint': str(checkpoint_path),
        'n_episodes': n_episodes,
        'config': {
            'n_stack': config.env.n_stack,
            'obs_shape': list(config.env.obs_shape) if isinstance(config.env.obs_shape, (list, tuple)) else int(config.env.obs_shape),
            'action_space_size': int(config.env.action_space_size),
            'image_based': config.env.image_based,
            'n_skip': config.env.n_skip,
            'max_episode_steps': config.env.max_episode_steps,
            'hidden_shape': config.model.hidden_shape,
            'proj_shape': config.model.proj_shape,
            'state_norm': config.model.state_norm,
            'discount': config.rl.discount,
        },
        'shapes': {
            'observations': f'[T, {config.env.obs_shape}]',
            'latent_states': f'[T, {config.model.hidden_shape}]',
            'projections': f'[T, {config.model.proj_shape}]',
            'actions': f'[T, {config.env.action_space_size}]',
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
            'observations': '[T, obs_dim] float64 - raw state observation at each step',
            'actions': '[T, action_dim] float32 - continuous action taken',
            'rewards': '[T] float32 - reward received',
            'latent_states': '[T, hidden_dim] float32 - s_t = H(obs_t)',
            'next_latent_states': '[T, hidden_dim] float32 - s_{t+1} ground truth',
            'dynamics_predictions': '[T, hidden_dim] float32 - G(s_t, a_t) benchmark',
            'projections': '[T, proj_dim] float32 - projection of s_t',
            'dynamics_projections': '[T, proj_dim] float32 - projection of G(s_t, a_t)',
            'dones': '[T] bool - episode termination',
            'valid_next': '[T] bool - True if next_latent_state is a real next state',
        },
        'notes': {
            'stacking': (
                f'n_stack={config.env.n_stack}. For state-based DMC this is typically 1, '
                f'so the representation network receives the raw state vector directly.'
            ),
            'actions': (
                'Actions are continuous float32 vectors. The dynamics network '
                'processes them through a linear embedding layer.'
            ),
            'valid_next': (
                'valid_next[t]=False at the last step of an episode or at terminal '
                'states. next_latent_states[t] is meaningless when valid_next[t]=False.'
            ),
        },
    }

    with open(Path(output_dir) / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)


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


@hydra.main(config_path="../ez/config", config_name='config', version_base='1.1')
def main(config):
    if config.exp_config is not None:
        exp_config = OmegaConf.load(config.exp_config)
        config = OmegaConf.merge(config, exp_config)

    # Collection parameters (pass via +collect.key=value on command line)
    if hasattr(config, 'collect'):
        n_episodes = config.collect.get('n_episodes', 50)
        max_steps = config.collect.get('max_steps', 1000)
        output_dir = config.collect.get('output_dir', 'dmc_dynamics_dataset')
    else:
        n_episodes = 50
        max_steps = 1000
        output_dir = 'dmc_dynamics_dataset'

    checkpoint_path = config.eval.model_path

    print(f"\n{'='*60}")
    print(f"  DMC Dynamics Dataset Collection")
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
        print(f"Example: eval.model_path=offline_training/dmc/dmc_hopper_hop_model_110000.p")
        return

    if not torch.cuda.is_available():
        print("ERROR: CUDA is required. Run this script on a machine with a GPU.")
        return

    # Override max_episode_steps to match collection setting
    with open_dict(config):
        config.env.max_episode_steps = max_steps

    # Initialize agent (updates config with env info like action_space_size, obs_shape)
    agent = agents.names[config.agent_name](config)

    # Initialize Ray (needed for MCTS and trajectory internals)
    num_gpus = torch.cuda.device_count()
    num_cpus = multiprocessing.cpu_count()
    ray.init(
        num_gpus=num_gpus,
        num_cpus=num_cpus,
        object_store_memory=100 * 1024 * 1024 * 1024,
    )

    # Build model and load checkpoint
    model = agent.build_model()
    print(f"Loading checkpoint: {checkpoint_path}")
    weights = torch.load(checkpoint_path, map_location='cpu')
    model.load_state_dict(weights)
    print("Checkpoint loaded successfully.")

    print(f"  obs_shape:         {config.env.obs_shape}")
    print(f"  action_space_size: {config.env.action_space_size}")
    print(f"  hidden_shape:      {config.model.hidden_shape}")
    print(f"  proj_shape:        {config.model.proj_shape}")
    print(f"  n_stack:           {config.env.n_stack}")
    print(f"  n_skip:            {config.env.n_skip}")

    if int(torch.__version__[0]) == 2:
        model = torch.compile(model)

    # Collect dataset
    stats = collect_dataset(agent, model, config, n_episodes, max_steps, output_dir)

    # Save metadata
    save_metadata(output_dir, config, checkpoint_path, n_episodes, stats)

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
    print(f"    api.create_repo('YOUR_USER/stuzero-dmc-{config.env.game.lower()}-dynamics', repo_type='dataset')")
    print(f"    api.upload_folder(")
    print(f"        folder_path='{output_dir}',")
    print(f"        repo_id='YOUR_USER/stuzero-dmc-{config.env.game.lower()}-dynamics',")
    print(f"        repo_type='dataset',")
    print(f"    )\"")
    print()


if __name__ == '__main__':
    main()
