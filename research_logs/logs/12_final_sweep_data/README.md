# Final Sweep Data

Catch-all for experimental data discovered during a final sweep before offloading from the GPU machine. Contents from across all four repos:

## Contents

| Archive | Size | Origin | Description |
|---------|------|--------|-------------|
| `stuzero_atari_online.tar.gz` | 213 KB | `STUZero/results/Atari/` | **42 EZ-V2 online MCTS training runs** across 6 Atari games (Pong: 36 runs, Asterix: 2, KungFuMaster/Assault/Boxing/Gopher: 1 each). For each run: `logs/Train.log`, `logs/Eval.log`, all `evaluation/step_*/` JSON files. Model `.p` checkpoints (30 MB + 385 MB/run) excluded. |
| `stuzero_wandb.tar.gz` | 723 KB | `STUZero/wandb/` | Per-run `files/` directories from 55 wandb runs: `config.yaml`, `wandb-metadata.json`, `wandb-summary.json`, `output.log`, `requirements.txt`, `conda-environment.yaml`. Binary wandb internals excluded. |
| `stuzero_toplevel_train.log` | 223 KB | `STUZero/train.log` | Top-level train.log from the STUZero repo root |
| `stuzero_toplevel_hydra/` | — | `STUZero/.hydra/` | Hydra config from the most recent STUZero train invocation (config.yaml, hydra.yaml, overrides.yaml) |
| `dreamer_runs_extra/` | 225 KB | `/home/ubuntu/dreamer_runs/` + `/home/ubuntu/stufix_proprio_run.tar.gz` | **The original failing stufix run** (pre-this-work, motivated the STU integration investigation in log #8) plus the stuA post-hoc failure (Run A from log #8). metrics+scores+config only. |
| `rezero_logdir_extra/` | 89 KB | `ReZero/logdir/` + `ReZero/runs/queue/` | Short debug/setup runs in `ReZero/logdir/` (mostly 1-201 metric entries), plus the queue scripts `pong_baseline_vs_stu_seed0.sh` and `pong_slurm.sh` that launched the main Pong experiments. |
| `video_reels/` | 10 MB | Derived from video sources across all three repos | **Training-progression film strips** (DreamerV3 Fig 4 style). **12 reels total:** 5 STUZero EZ-V2 Atari (baseline), 4 stu-dreamer-jax (walker_walk and hopper_stand, with key `methods_at_400k` showing baseline/stuB/stuRand/stuPlus/stufix side-by-side), 3 ReZero Pong (baseline vs STU + progression). See `video_reels/README.md`. |

**Total: 12 MB**

## What's in `stuzero_atari_online.tar.gz`

EZ-V2 is the main online MCTS training stack of STUZero (separate from the offline dynamics experiments in log #11). Each run directory contains:

```
Atari/<game>/EZ-V2-seed=0-<timestamp>/
├── logs/
│   ├── Train.log          # training stdout (17 KB per run)
│   └── Eval.log           # evaluation stdout
└── evaluation/
    ├── step_0/            # initial eval
    ├── step_10011/        # periodic evals every ~10k steps
    ├── step_20033/
    ├── ...
    └── final/             # final eval at 120k-140k steps
        └── recordings/
            ├── openaigym.episode_batch.*.stats.json   # per-episode scores
            └── openaigym.manifest.*.manifest.json
```

### Game coverage and timestamps

| Game | Runs | Most recent |
|------|------|-------------|
| Pong | 36 | 2025-12-02 (most recent), earliest 2025-11-14 |
| Asterix | 2 | 2026-02-11 |
| KungFuMaster | 1 | 2026-04-04 |
| Assault | 1 | 2026-04-05 |
| Boxing | 1 | 2026-04-06 |
| Gopher | 1 | 2026-04-06 |

Pong has many runs because it was the primary development/debugging target. The single-run games (KungFu, Assault, Boxing, Gopher) are from the April 2026 expansion described in research logs #4, #5, #7.

## What's in `dreamer_runs_extra/`

The **original failing stufix run** (`stufix_proprio_original_failed_run.tar.gz`) is the result that started this entire investigation. It shows episode scores stuck at ~29 on dmc_walker_walk despite 500K+ steps of training — the catastrophic failure that motivated all the STU-Dreamer work in log #8.

`stufix_proprio_20260411T073936.tar.gz` is the same run but extracted from `/home/ubuntu/dreamer_runs/` (live on disk, not the tar.gz snapshot). The 3 smaller tar.gz files (~1.8 KB each) are empty placeholder directories from failed starts.

`stuA_proprio_20260411T230338.tar.gz` is the **Run A post-hoc STU** integration I tested early in the stu-dreamer-jax work — the one that tracked baseline initially but wasn't the principled fix. Referenced in log #8 but its data wasn't previously archived.

## What's in `rezero_logdir_extra/`

Short test runs in `ReZero/logdir/` before the main `pong_baseline_seed0` / `pong_stu_hankelscaled_seed0` experiments in log #9. The `2026-04-11_05-19-16.tar.gz` run is more substantial (201 metric entries) and appears to be an intermediate test.

`queue_scripts/pong_baseline_vs_stu_seed0.sh` is the launch script that started the main comparison.

## Cross-references

- Main narrative logs that reference this data:
  - Log #4, #5, #7 — claim EZ-V2 online Atari experiments were run (archived here)
  - Log #8 — references the original stufix failure and Run A (archived here)
  - Log #9 — references ReZero logdir runs (archived here)
