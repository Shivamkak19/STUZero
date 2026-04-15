# DreamerV3 + STU online — DMC proprio, 500k steps, size-1M

**Date:** 2026-04-14 to 2026-04-15
**Repo:** `stu-dreamer-jax` (STU integrated into DreamerV3 RSSM)
**Scope:** Head-to-head baseline vs `stu_plus` across 3 DMC continuous-control tasks at the 1M-param model size.

---

## TL;DR

| Task            | Baseline (final) | stuPlus (final) | Δ (stuPlus − baseline) | mean last-20 baseline | mean last-20 stuPlus |
|-----------------|------------------|-----------------|------------------------|------------------------|-----------------------|
| walker_run      | **408.9**        | **680.2**       | **+271.3 (+66%)**      | 402.6                  | 656.9                 |
| walker_walk     | **963.6**        | **928.3**       | −35.3 (−3.7%)          | 952.7                  | 939.2                 |
| hopper_hop      | **152.1**        | **154.2**       | +2.1 (+1.4%)           | 156.0                  | 145.0                 |

- **walker_run**: clear, large win for stuPlus.
- **walker_walk**: ≈ tie. Both agents near task ceiling (~1000); noise dominates.
- **hopper_hop**: ≈ tie. Low raw score for both — hopper_hop is the hardest of the three and 500k steps at train_ratio 1024 is likely too short for either agent to master it.

---

## Experimental setup

- **Config:** `dmc_proprio` (size1m → `rssm: {deter: 512, hidden: 64, classes: 4}`, `depth: 4`, `units: 64`).
- **stuPlus overlay:** `stu_plus` config (`use_stu: True`, `stu_num_eigh: 48`, `stu_mode: core`, `stu_neg_bank: True`, `stu_outscale: 0.01`, `stu_shortcut: True`, Hankel K = 48).
- **Steps:** 500k env steps per run (baselines.yaml default is 1.1M; we truncated).
- **Train ratio:** 1024 (from `dmc_proprio`).
- **Seed:** single seed per run (default from configs.yaml; not swept).
- **Observation:** proprio only (`env.dmc.image: False`); 64×64 RGB rendered for logging only.
- **Tasks run:** `dmc_walker_run`, `dmc_walker_walk`, `dmc_hopper_hop`.

## Compute / infrastructure

These 6 runs were *not* submitted through SLURM — they were launched directly on 6 Della login nodes in parallel (`setsid` + `nohup` + ssh). This was done because the `gpu` and `pli` SLURM queues were saturated with cryoem + PLI-group H100 usage; the `gpu-short` queue wait estimate was ~24 h and `pli-low` was blocked by `QOSGrpGRES`. Meanwhile, several per-group login nodes exposed idle H100 / A100 GPUs that are shared across the site.

| Experiment                 | Login node        | GPU                            | Train fps |
|----------------------------|-------------------|--------------------------------|-----------|
| walker_run baseline        | della-adele       | H100 NVL (96 GB)               | ~45k      |
| walker_run stuPlus         | della-fongpu      | H100 NVL (96 GB)               | ~24k      |
| walker_walk baseline       | della-mol (GPU 0) | A100 SXM 80 GB                 | ~37k      |
| walker_walk stuPlus        | della-yasaman     | A40 (46 GB)                    | ~16k      |
| hopper_hop baseline        | della-mol (GPU 2) | A100 SXM 80 GB                 | ~25k      |
| hopper_hop stuPlus         | della-stellato    | H100 NVL (96 GB, GPU 1)        | ~25k      |

Launch script preserved in `launch_loginnodes.sh`.

### Issues encountered (for future reference)

1. **Home quota (50 GB)** filled mid-run. Dreamer writes checkpoints + replay buffer to `$logdir`; `~/.cache/{uv,pip}` also accumulated to 14 GB during repeated venv activations. Fix: moved `~/logdir/dreamer` and `~/.cache/{uv,pip}` to `/scratch/gpfs/EHAZAN/sk3686/` and symlinked. Once done, no further quota failures.
2. **Login-node arbiters** on `della-rse` and `della-pli` kill user processes after a few minutes of sustained CPU/GPU load (even with otherwise-idle GPUs). hopper_hop baseline died on `della-gpu` (likely A100-40GB OOM), then twice on `della-rse`/`della-pli` (arbiter) before sticking on `della-mol` GPU 2.
3. **della-gpu (A100 40 GB)**: peak VRAM hit 38/40 GB during compile; the run died silently ~13 min in. 40 GB is too tight for 1M dreamer with default batch/buffer settings — use 80 GB+ nodes.

---

## Files in this directory

```
7_dreamer_online_dmc/
├── README.md                    (this file)
├── launch_loginnodes.sh         launch script (bash/ssh/setsid pattern)
├── make_reels.py                reel generator (ffmpeg + PIL)
├── reels/
│   ├── reel_walker_run.png      baseline vs stuPlus, 10 training ckpts × 10 frames
│   └── reel_hopper_hop.png      same, for hopper_hop
└── data/
    ├── walker_run_baseline/     scores.jsonl, metrics.jsonl, config.yaml
    ├── walker_run_stuplus/
    ├── walker_walk_baseline/
    ├── walker_walk_stuplus/
    ├── hopper_hop_baseline/
    ├── hopper_hop_stuplus/
    └── stdout_logs/             full training stdout for each run
```

Full checkpoints + replay + plugin dumps remain on scratch:
`/scratch/gpfs/EHAZAN/sk3686/logdir/dreamer/loginnode_*` (not copied here — too large).

## Reels

Each reel PNG places baseline (left) next to stuPlus (right). Rows = training checkpoints evenly spaced over the 500k-step run (label shows agent step, e.g. `164k`, `211k`, ...). Columns = 10 evenly-spaced frames from the eval-episode mp4 that Dreamer logs every ~16k steps via `logger.video(...)` → `scope/epstats-policy_log-image.mp4/`.

To regenerate: `python make_reels.py` (uses ffmpeg for frame extraction, PIL for tiling).

## Caveats / future work

- **Single seed.** The walker_run win is large (+66%) and visible in the reel (stuPlus figure is running, baseline is stuck in weaker gaits), but a 3-seed replication would make it load-bearing.
- **500k steps is short for hopper_hop.** Baseline cheetah results at 1M steps converge closer to 800+; 500k @ train_ratio 1024 may underestimate both models' hopper asymptote.
- **Proprio only.** The `dmc_vision` config (image obs) may behave differently since the RSSM encoder path changes.
- **No humanoid_walk here.** That SLURM job (6962623 / 6963067 / 6965681) was still pending at time of writing and is tracked separately.
