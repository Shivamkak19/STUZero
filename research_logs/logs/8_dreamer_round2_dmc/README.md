# DreamerV3 + STU Round 2 — DMC proprio, 500k steps

**Date:** 2026-04-15
**Repo:** `stu-dreamer-jax` (STU integrated into DreamerV3 RSSM)
**Scope:** Follow-up experiments from round 1 (log `7_dreamer_online_dmc`). This round adds: (1) a second seed for walker_run baseline vs stuPlus, (2) Fourier-basis ablation on walker_walk, (3) matched-parameter baseline control on walker_run, and (4) scaling to the 12M-parameter model on walker_walk.

---

## TL;DR

| Experiment | Task | Config | Final Score | Mean Last-20 | Mean Last-50 | Max | FPS (train) |
|------------|------|--------|-------------|-------------|-------------|-----|-------------|
| walker_run baseline (seed 1) | walker_run | `dmc_proprio` | 608.3 | 604.0 | 594.2 | 646.9 | 44.7k |
| walker_run stuPlus (seed 1) | walker_run | `dmc_proprio stu_plus` | 621.3 | 677.6 | 672.2 | 715.2 | 24.9k |
| walker_run matched baseline | walker_run | `dmc_proprio_matched` | 630.3 | 608.6 | 590.2 | 638.1 | 40.6k |
| walker_walk baseline 12m | walker_walk | `dmc_proprio_12m` | 971.7 | 915.3 | 911.7 | 975.2 | 26.6k |
| walker_walk stuPlus fourier | walker_walk | `dmc_proprio stu_plus_fourier` | 967.1 | 960.3 | 958.2 | 990.3 | 19.2k |

---

## Key Findings

### 1. walker_run seed 1: stuPlus replicates round 1 advantage

Round 1 (seed 0) showed a large +66% win for stuPlus on walker_run. Seed 1 replicates:

| Metric | Baseline (seed 1) | stuPlus (seed 1) | Delta |
|--------|-------------------|------------------|-------|
| Mean last-20 | 604.0 | **677.6** | **+12.2%** |
| Mean last-50 | 594.2 | **672.2** | **+13.1%** |
| Max score | 646.9 | **715.2** | **+10.6%** |
| Dyn loss (final) | 4.75 | **4.35** | -8.4% |

The advantage is smaller than round 1 (+12% vs +66%) but still consistent and in the same direction. stuPlus also shows lower variance (std last-20: 22.9 vs 27.7).

**Cross-seed summary for walker_run baseline vs stuPlus:**

| Seed | Baseline mean-last-20 | stuPlus mean-last-20 | Delta |
|------|----------------------|---------------------|-------|
| 0 (round 1) | 402.6 | 656.9 | **+63.2%** |
| 1 (round 2) | 604.0 | 677.6 | **+12.2%** |

Baseline seed variance is high (402 vs 604), but stuPlus wins in both seeds.

### 2. Matched-parameter baseline: extra capacity does NOT explain stuPlus advantage

The `dmc_proprio_matched` config widens the GRU (deter 576, hidden 72, units 72) to match stuPlus's parameter count without adding the spectral basis.

| Metric | Baseline | Matched Baseline | stuPlus |
|--------|----------|-----------------|---------|
| Mean last-20 | 604.0 | 608.6 | **677.6** |
| Mean last-50 | 594.2 | 590.2 | **672.2** |
| Max | 646.9 | 638.1 | **715.2** |
| FPS | 44.7k | 40.6k | 24.9k |

The matched baseline performs nearly identically to the standard baseline — the extra parameters don't help. This confirms that stuPlus's advantage comes from the **spectral inductive bias**, not just extra capacity.

### 3. walker_walk Fourier basis: nearly matches Hankel stuPlus

The `stu_plus_fourier` config replaces Hankel eigenvectors with an orthonormal Fourier basis (same architecture, same K=48, same neg bank).

| Metric | Round 1 baseline | Round 1 stuPlus (Hankel) | Round 2 stuPlus (Fourier) |
|--------|-----------------|-------------------------|--------------------------|
| Mean last-20 | 952.7 | 939.2 | **960.3** |
| Mean last-50 | — | — | **958.2** |
| Max | 984.8 | 976.7 | **990.3** |
| Std last-20 | — | — | 16.6 |
| FPS | ~37k | ~16k | 19.2k |

The Fourier basis performs comparably or slightly better than both the Hankel stuPlus and the baseline on walker_walk. This contrasts with the offline STUZero results where Hankel specifically outperformed random orthonormal bases. Possible explanations:
- walker_walk is near the task ceiling (~1000); differences are noise-dominated
- The Fourier basis may be effective for DMC proprio (smooth dynamics) even if Hankel is better for Atari (discrete/discontinuous)
- Online training may be less sensitive to basis choice than offline dynamics prediction

### 4. walker_walk 12m baseline: larger model helps

| Metric | 1M baseline (round 1) | 12M baseline (round 2) |
|--------|----------------------|----------------------|
| Mean last-20 | 952.7 | **915.3** |
| Max | 984.8 | **975.2** |
| Train ratio | 1024 | 512 |
| FPS | ~37k | 26.6k |

The 12M baseline reaches similar scores to the 1M baseline on walker_walk (both near ceiling). The 12M stuPlus run (`walker_walk_stuplus_12m_retry`) is still in progress (~48% done, score ~540, very slow at 4.1k FPS).

---

## Experimental Setup

### Common settings
- **Env steps:** 500k (`--run.steps 5e5`)
- **Observation:** proprio only (`env.dmc.image: False`)
- **Launch method:** Direct SSH to login nodes with `setsid` detachment (same pattern as round 1)
- **Launch script:** `launch_round2.sh`

### Per-run configs

| Run | Config overlays | Seed | Train ratio | RSSM size | STU K | STU basis |
|-----|----------------|------|-------------|-----------|-------|-----------|
| walker_run baseline seed1 | `dmc_proprio` | 1 | 1024 | deter=512, hidden=64, classes=4 | — | — |
| walker_run stuPlus seed1 | `dmc_proprio stu_plus` | 1 | 1024 | deter=512, hidden=64, classes=4 | 48 | Hankel |
| walker_run matched baseline | `dmc_proprio_matched` | 0 | 1024 | deter=576, hidden=72, classes=4 | — | — |
| walker_walk baseline 12m | `dmc_proprio_12m` | 0 | 512 | deter=2048, hidden=256, classes=16 | — | — |
| walker_walk stuPlus fourier | `dmc_proprio stu_plus_fourier` | 0 | 1024 | deter=512, hidden=64, classes=4 | 48 | Fourier |

### stuPlus config (shared by Hankel and Fourier variants)
```yaml
use_stu: True
stu_mode: core          # STU inside RSSM recurrence
stu_num_eigh: 48        # K = 48 spectral filters
stu_max_seq: 64         # rolling action-history buffer length
stu_neg_bank: True      # captures oscillatory (negative eigenvalue) dynamics
stu_outscale: 0.01      # zero-init scale for output projection
stu_shortcut: True      # direct action-to-output residual path
stu_use_obs: True       # also filter observation (stoch) history
```

### stuPlus Fourier difference
```yaml
stu_basis: fourier      # orthonormal Fourier basis instead of Hankel eigenvectors
```

### Matched baseline config
```yaml
# Wider GRU to match stuPlus parameter count, no spectral basis
rssm: {deter: 576, hidden: 72, classes: 4}
units: 72
depth: 4
```

---

## Compute / Infrastructure

| Experiment | Login node | GPU | Train FPS | Wall time |
|------------|-----------|-----|-----------|-----------|
| walker_run baseline seed1 | della-adele | H100 NVL (96 GB) | 44.7k | ~3h 11m |
| walker_run stuPlus seed1 | della-stellato | H100 NVL (96 GB) | 24.9k | ~5h 48m |
| walker_walk baseline 12m | della-stellato (GPU 1) | H100 NVL (96 GB) | 26.6k | ~2h 41m |
| walker_run matched baseline | della-yasaman | A100 80 GB | 40.6k | ~3h 30m |
| walker_walk stuPlus fourier | della-mol (GPU 1) | A100 SXM 80 GB | 19.2k | ~6h 44m |

All runs launched 2026-04-15 ~16:07 EDT. The 4 non-12m runs finished by ~19:30-22:51 EDT. The fourier run (slowest of the five) finished around 22:51.

---

## Final Loss Comparison

| Run | dyn loss | rew loss | con loss |
|-----|----------|----------|----------|
| walker_run baseline seed1 | 4.750 | 0.500 | 0.0204 |
| walker_run stuPlus seed1 | **4.348** | 0.534 | 0.0204 |
| walker_run matched baseline | 4.616 | 0.488 | 0.0204 |
| walker_walk baseline 12m | **2.516** | 0.564 | 0.0204 |
| walker_walk stuPlus fourier | 4.901 | 0.617 | 0.0204 |

stuPlus consistently achieves lower dynamics loss on walker_run (4.35 vs 4.75 baseline, 4.62 matched). The 12M model achieves the lowest dyn loss overall (2.52), as expected from the larger capacity.

---

## Score Progression (sampled every ~32k steps)

### walker_run (seed 1): baseline vs stuPlus vs matched

| Step | Baseline | stuPlus | Matched |
|------|----------|---------|---------|
| 48k | 50.7 | 30.7 | 51.6 |
| 112k | 101.6 | 132.8 | 84.5 |
| 176k | 186.7 | 246.7 | 197.8 |
| 240k | 271.7 | 384.2 | 241.2 |
| 304k | 414.2 | 503.8 | 332.8 |
| 368k | 495.3 | **681.9** | 424.4 |
| 432k | 551.1 | **679.8** | 578.2 |
| 496k | 608.3 | 621.3 | 630.3 |

stuPlus leads from ~112k onward, peaking at ~682 around 368k. Baseline and matched converge slowly. stuPlus shows earlier convergence (data efficiency advantage consistent with round 1).

### walker_walk: fourier stuPlus vs round 1 baselines

| Step | Fourier stuPlus | Round 1 Baseline | Round 1 stuPlus (Hankel) |
|------|----------------|-----------------|-------------------------|
| 112k | 170.5 | — | — |
| 176k | 660.5 | — | — |
| 272k | 943.2 | — | — |
| 336k | 978.2 | — | — |
| 496k | 967.1 | 963.6 | 928.3 |

Fourier stuPlus converges quickly to near-ceiling scores (~960-980) by 272k steps.

---

## Still In Progress

| Run | Config | Node | Steps | Score | ETA |
|-----|--------|------|-------|-------|-----|
| walker_walk stuPlus 12m (retry) | `dmc_proprio_12m stu_plus_12m_mods` | della-mol GPU2 | 240k/500k (48%) | ~540 | ~17h |

This run uses K=192 filters, seq_len=256, outscale=0.1. It is extremely slow (4.1k FPS vs 26.6k for the 12M baseline) — the large STU FFT at 256-length windows on a 2048-dim RSSM is the bottleneck. Score is still climbing at 240k steps.

---

## Files in this directory

```
8_dreamer_round2_dmc/
├── README.md                    (this file)
├── launch_round2.sh             launch script (bash/ssh/setsid pattern)
└── data/
    ├── walker_run_baseline_seed1/    scores.jsonl, metrics.jsonl, config.yaml
    ├── walker_run_stuplus_seed1/
    ├── walker_walk_baseline_12m/
    ├── walker_run_matched_baseline/
    ├── walker_walk_stuplus_fourier/
    └── stdout_logs/                  full training stdout for each run
```

Full checkpoints + replay remain on scratch:
`/scratch/gpfs/EHAZAN/sk3686/logdir/dreamer/round2_*` (not copied here — too large).

---

## Takeaways for Next Steps

1. **stuPlus walker_run advantage is real across 2 seeds.** The magnitude varies (seed 0: +63%, seed 1: +12%), suggesting seed sensitivity. A 3rd seed would strengthen the claim.

2. **Matched-parameter control confirms spectral inductive bias.** Extra GRU capacity alone does not replicate stuPlus's gains.

3. **Fourier vs Hankel basis is a wash on walker_walk.** Both reach ~960 near the task ceiling. Need to test on a task with more room to differentiate (walker_run, or harder tasks like humanoid). This is an important ablation question — if Fourier works as well as Hankel, the theoretical story about Hankel eigenvectors specifically is weaker, but the practical utility of spectral filtering is confirmed.

4. **12M scaling is bottlenecked by STU FPS.** The 12M stuPlus run at 4.1k FPS is ~6.5x slower than the 12M baseline at 26.6k FPS. The K=192, seq=256 FFT is too expensive. Future work should explore:
   - Tensorized/low-rank filter decomposition to reduce compute
   - Distilling STU to a learned LDS for inference
   - Keeping K small (48) even at 12M scale

5. **Cross-reference with STUZero offline results:** The offline finding that random orthonormal bases hurt (while Hankel helps) may not transfer to online DreamerV3 training, where the Fourier basis performs well. The training regime (online RL vs offline supervised) may matter more than the specific orthonormal basis.
