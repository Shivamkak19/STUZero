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
  - 10K<n<100K
---

# STUZero UpNDown Dynamics Dataset

Offline dynamics training dataset collected from a trained EfficientZero
benchmark model on Atari UpNDown.

## Purpose

Train and evaluate alternative dynamics network architectures (e.g., STU-based)
**offline**, without running full online training (~15 hours per experiment).
Provides a direct dynamics loss signal via ground-truth next latent states.

## Statistics

| Metric | Value |
|--------|-------|
| Episodes | 50 |
| Total transitions | 68,997 |
| Mean episode length | 1380 steps |
| Mean episode reward | 3721.4 |

## Data Fields

Each `episode_XXXX.pt` file is a PyTorch dict:

| Field | Shape | Type | Description |
|-------|-------|------|-------------|
| `frames` | `[T, H, W, C]` | uint8 | Raw observation frame |
| `actions` | `[T]` | long | Discrete action |
| `rewards` | `[T]` | float32 | Reward |
| `latent_states` | `[T, 64, 6, 6]` | float16 | H(stacked_obs_t) |
| `next_latent_states` | `[T, 64, 6, 6]` | float16 | H(stacked_obs_{t+1}) ground truth |
| `dynamics_predictions` | `[T, 64, 6, 6]` | float16 | G(s_t, a_t) benchmark |
| `projections` | `[T, 1024]` | float16 | Projected s_t |
| `dynamics_projections` | `[T, 1024]` | float16 | Projected G(s_t, a_t) |
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
s_t    = ep['latent_states'][valid]          # [N, 64, 6, 6]
a_t    = ep['actions'][valid]                # [N]
s_next = ep['next_latent_states'][valid]     # [N, 64, 6, 6] target

# Benchmark dynamics predictions for comparison
s_pred = ep['dynamics_predictions'][valid]   # [N, 64, 6, 6]

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
print(f"Dataset: {s_t.shape[0]} transitions")
```

## Configuration

| Parameter | Value |
|-----------|-------|
| Environment | Atari UpNDown |
| Observation | [3, 96, 96] (RGB) |
| Frame stacking | 4 frames |
| Action space | 6 discrete actions |
| Latent shape | [64, 6, 6] (16x spatial downsampling) |
| Projection dim | 1024 |

## Notes

- **Frame stacking:** latent state s_t encodes 4 stacked frames.
  At episode start, the stack is padded with copies of the initial frame.
- **Normalization:** frames are uint8 [0-255]; the representation network
  expects float32 / 255.
- **valid_next:** always check this mask before using `next_latent_states`.
  Invalid entries occur at episode boundaries and terminal states.
