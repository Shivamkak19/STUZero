---
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
  - 100K<n<1M
---

# STUZero Atari Dynamics Dataset

Offline dynamics training datasets collected from trained EfficientZero V2 (EZv2) benchmark models on Atari games. Each game's data is stored in a subfolder named `{game}_{steps}` indicating the game and the number of training steps of the source checkpoint. While all models were trained for 120K steps, best results in some games were attained at earlier checkpoints. The model with best eval scores was used to curate data for each game.

## Purpose

Train and evaluate alternative dynamics network architectures **offline**, without running full online EZv2 training. Each dataset provides ground-truth next latent states, benchmark dynamics predictions, and projections, enabling direct dynamics loss computation and multi-step rollout evaluation.

## Available Games

| Game | Checkpoint Steps | Episodes | Total Transitions | Mean Ep Length | Mean Reward | Actions | Subfolder |
|------|-----------------|----------|-------------------|----------------|-------------|---------|-----------|
| Pong | 100K | 50 | 87,548 | 1,751 | 20.6 | 6 | `pong_100K/` |
| Asterix | 110K | 50 | 184,876 | 3,697 | 8,510 | 9 | `asterix_110K/` |
| Alien | 110K | 50 | 29,687 | 594 | 725 | 18 | `alien_110K/` |
| Seaquest | 110K | 50 | 113,381 | 2,268 | 1,775 | 18 | `seaquest_110K/` |
| KungFuMaster | 110K | 50 | 173,287 | 3,466 | 25,684 | 14 | `kungfumaster_110K/` |
| MsPacman | 90K | 50 | 40,357 | 807 | 2,439 | 9 | `mspacman_90K/` |
| RoadRunner | 110K | 50 | 48,696 | 974 | 29,102 | 18 | `roadrunner_110K/` |

**Total: 7 games, 350 episodes, 677,832 transitions**

## Data Collection Procedure

Data is collected by running self-play episodes using a fully trained EZv2 checkpoint with MCTS action selection. The collection process works as follows:

1. **Environment setup:** Atari environments are created in parallel batches (up to 8 concurrent envs). Each environment uses the standard EZv2 preprocessing: frame resizing to 96x96, optional grayscale conversion, and stacking of the most recent 4 frames.

2. **Action selection:** At each step, the stacked observation is passed through the trained representation network to produce a latent state `s_t = H(stacked_obs_t)`. MCTS (Gumbel or standard, per config) is then run using the trained model to select the best action — no exploration noise is added, so actions reflect the learned policy.

3. **Dynamics recording:** For each selected action `a_t`, the trained dynamics network produces a one-step prediction `s_hat_{t+1} = G(s_t, a_t)`. Both the latent state and dynamics prediction are projected through the trained projection head. All of these are recorded per timestep.

4. **Ground-truth next states:** After collection, ground-truth next latent states are computed by shifting the latent state array by one timestep: `next_latent_state[t] = latent_state[t+1]`. A `valid_next` mask marks transitions where this shift is meaningful (i.e., not at episode boundaries or terminal states).

5. **Storage:** Each episode is saved as a separate `.pt` file containing PyTorch tensors. A `metadata.json` file in each subfolder records the environment config, checkpoint path, tensor shapes, and collection statistics.

## Data Fields

Each `episode_XXXX.pt` file contains a dict with the following keys:

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `frames` | `[T, 96, 96, 3]` | uint8 | Raw observation frame at each step |
| `actions` | `[T]` | long | Discrete action taken (selected by MCTS) |
| `rewards` | `[T]` | float32 | Reward received |
| `latent_states` | `[T, 64, 6, 6]` | float16 | `s_t = H(stacked_obs_t)` from representation network |
| `next_latent_states` | `[T, 64, 6, 6]` | float16 | `s_{t+1} = H(stacked_obs_{t+1})` ground-truth next state |
| `dynamics_predictions` | `[T, 64, 6, 6]` | float16 | `G(s_t, a_t)` benchmark dynamics prediction |
| `projections` | `[T, 1024]` | float16 | Projection of `s_t` |
| `dynamics_projections` | `[T, 1024]` | float16 | Projection of `G(s_t, a_t)` |
| `dones` | `[T]` | bool | Episode termination flag |
| `valid_next` | `[T]` | bool | Whether `next_latent_states[t]` is a real next state |

## Quick Start

```python
import torch
from pathlib import Path

# Load all episodes from a game
game_dir = Path("pong_100K")
episodes = sorted(game_dir.glob("episode_*.pt"))

all_s, all_a, all_s_next = [], [], []
for ep_path in episodes:
    ep = torch.load(ep_path)
    valid = ep["valid_next"]
    all_s.append(ep["latent_states"][valid])
    all_a.append(ep["actions"][valid])
    all_s_next.append(ep["next_latent_states"][valid])

s_t = torch.cat(all_s)
a_t = torch.cat(all_a)
s_next = torch.cat(all_s_next)
print(f"Dataset: {s_t.shape[0]} valid transitions")
```

## Notes

- **Frame stacking:** Each latent state encodes 4 stacked frames. At episode start, the stack is padded with copies of the initial frame.
- **Normalization:** Frames are stored as uint8 [0-255]. The representation network expects float32 inputs divided by 255.
- **valid_next:** Always check this mask before using `next_latent_states`. Invalid entries occur at the last step of each episode and at terminal states.
- **Action spaces vary by game:** Check `metadata.json` in each subfolder for game-specific config.
