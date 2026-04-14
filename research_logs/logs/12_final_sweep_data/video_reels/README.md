# Training-Progression Video Reels

Film-strip visualizations of STUZero EZ-V2 Atari policy evolution. Inspired by DreamerV3 Figure 4 (open-loop prediction reels) but adapted for "how does the trained policy change across training steps."

## Layout

For each reel PNG:

- **Rows** — training checkpoints (step_0 → step_120k → final), one row per checkpoint
- **Columns** — 10 frames sampled at fixed percentages (0%, 11%, 22%, …, 100%) of the first evaluation episode at that checkpoint
- **Left label** — training step
- **Top label** — within-episode frame percentage (f0%, f11%, …, f100%)
- **Frames** — 96×96 native, upscaled 2× to 192×192 nearest-neighbor for readability

## Files

### STUZero EZ-V2 Atari reels (all baseline, no STU)

| Reel | Game | Evals | Size |
|------|------|-------|------|
| `Pong_EZ-V2-seed=0-2025-11-21_03-45-30.png` | Pong | 14 checkpoints | 217 KB |
| `KungFuMaster_EZ-V2-seed=0-2026-04-04_21-44-11.png` | KungFuMaster | 14 | 1.0 MB |
| `Boxing_EZ-V2-seed=0-2026-04-06_00-24-13.png` | Boxing | 13 | 618 KB |
| `Assault_EZ-V2-seed=0-2026-04-05_13-08-09.png` | Assault | 14 | 476 KB |
| `Gopher_EZ-V2-seed=0-2026-04-06_15-25-32.png` | Gopher | 14 | 551 KB |

### stu-dreamer-jax reels (STU vs baseline comparison — log #8)

| Reel | Description | Size |
|------|-------------|------|
| `dreamer_walker_walk_methods_at_400k.png` | **5-method comparison at step ~400k** — baseline / stuB (K=24) / stuRand / stuPlus / stufix (orig fail). Most useful plot: shows baseline walking, stuB/rand/plus all walking similarly, and stufix (the original broken integration) flat on the ground. | 337 KB |
| `dreamer_walker_walk_stuPlus_progression.png` | stuPlus progression across training: 7 checkpoints showing walker learn to walk | 475 KB |
| `dreamer_walker_walk_baseline_progression.png` | baseline progression across training (same task, for comparison) | 479 KB |
| `dreamer_hopper_stand_stuPlus_progression.png` | stuPlus size1m on hopper_stand: 10 checkpoints showing hopper learn to stand and balance | 533 KB |

### ReZero Pong reels (STU vs baseline — log #9)

Extracted from TensorBoard `eval_video` image summaries (animated GIFs) embedded in the tfevents files. The PyTorch R2-Dreamer logger stores eval rollouts as GIFs at each eval step.

| Reel | Description | Size |
|------|-------------|------|
| `rezero_pong_baseline_vs_stu_at_matched_steps.png` | **5 training steps × 2 methods** (baseline vs stu-hankelscaled). Shows that STU variant's play quality at step 400k is similar to baseline's play at step 200k — about 2× slower convergence (consistent with the eval scores in log #9: baseline +4.3 vs STU -12.8 at step 400k). | 64 KB |
| `rezero_pong_baseline_progression.png` | baseline progression across 10 training steps (0 → 400k) | 63 KB |
| `rezero_pong_stu_hankelscaled_progression.png` | stu-hankelscaled progression | 68 KB |

Note on visual artifacts: the ReZero reels occasionally show a colored splash-screen frame (green with yellow/blue rectangles) — this is from the tensorboard-GIF format, not a rendering bug. The actual gameplay is visible in the majority of cells showing the standard Atari Pong arena with paddles and ball.

## Not included

- **Asterix** — only has step_0 evaluation (training didn't complete with video recording enabled)
- **Pong spectral/STU variants** (`EZ-V2-spectral-seed=0-2025-11-14 *`) — only step_0 available, no training completed
- **Non-chosen Pong runs** — 36 Pong runs exist in total; picked the one with most eval checkpoints (14) for the reel

## How to read

Each reel shows a single eval episode, so the column progression (left→right) is the game unfolding over ~5 minutes of play. The row progression (top→bottom) is the policy maturing over training. Patterns to look for:

- **Early training (top rows)** — policy often stuck in corners, spasmodic movements, episodes end quickly or with no score
- **Mid training** — exploratory behaviors, occasional good plays
- **Late training (bottom rows)** — consistent positioning, purposeful movement, sustained scoring

Boxing is particularly clear: the two fighters' positions across columns show the engagement pattern, and the trained policy (bottom) approaches the opponent more decisively than the untrained policy (top).

## How to regenerate

```bash
# STUZero EZ-V2 Atari reels + stu-dreamer-jax reels
cd /lambda/nfs/hazan/stu-dreamer-jax
.venv/bin/python /lambda/nfs/hazan/STUZero/research_logs/logs/12_final_sweep_data/video_reels/make_reels.py
.venv/bin/python /lambda/nfs/hazan/STUZero/research_logs/logs/12_final_sweep_data/video_reels/make_dreamer_reels.py

# ReZero reels (need the ReZero venv since it has tensorboard/PyTorch)
/lambda/nfs/hazan/ReZero/.venv/bin/python /lambda/nfs/hazan/STUZero/research_logs/logs/12_final_sweep_data/video_reels/make_rezero_reels.py
```

Both scripts use `ffprobe` + `ffmpeg` to extract frames and PIL to tile. Parameters inside the scripts:
- `N_COLS = 10` — frames per row
- `SCALE = 2` or `3` — upscale factor for readability

### Video source paths

- **STUZero EZ-V2** (96×96, 10 fps, ~3100 frames per episode): `/lambda/nfs/hazan/STUZero/results/Atari/<Game>/<run>/evaluation/step_*/recordings/epi_0_27000.mp4`
- **stu-dreamer-jax** (64×64, 10 fps, 1001 frames per episode): `/home/ubuntu/dreamer_runs/<run>/scope/epstats-policy_log-image.mp4/<step>-<hash>.mp4`. Auto-logged by DreamerV3 embodied.run logger. The step count is the leading integer in the mp4 filename.
- **ReZero** (64×64 GIF, variable length): **embedded in tfevents** at `/lambda/nfs/hazan/ReZero/runs/<run>/events.out.tfevents.*`. Extracted via `tensorboard.backend.event_processing.event_accumulator` → `a.Images('eval_video')` → each entry has `.step` (training step) and `.encoded_image_string` (GIF bytes).
