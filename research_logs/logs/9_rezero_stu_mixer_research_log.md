# Research Log — 2026-04-09 to 2026-04-11
## Goal: Test STU as a *pre-RSSM encoder mixer* in R2-Dreamer (PyTorch)

---

## Executive Summary

Three findings from integrating STU as a sequence mixer between the encoder and RSSM in the R2-Dreamer PyTorch codebase (`ReZero/`):

1. **Pre-RSSM STU mixer did NOT help on Pong.** At 400K env steps, baseline R2-Dreamer reached eval score **+4.3** (starting to win) while `STU-hankelscaled` was stuck at **-12.8** (still losing by 12 points). The mixer is a stack of `STUSandwichBlock` modules applying causal STU over encoder outputs `[B, T, E]` before the RSSM consumes them.

2. **The failure contrasts with our stu-dreamer-jax result.** In the JAX integration (log #8), STU-inside-`_core` (fed as a 4th input branch into the BlockGRU recurrence) produced +20-36% convergence speedup. The same spectral filtering principle applied *before* the RSSM (pre-encoder mixer, as in spectral-transformer-jax's successful usage) did NOT transfer to the Dreamer setting.

3. **Hypothesis for the difference.** In spectral-transformer-jax (language modeling), STU mixes tokens at the sequence-modeling level where spectral filtering directly helps state tracking. In Dreamer, the RSSM's GRU already handles temporal integration — adding STU *before* the RSSM gives it a modified input stream that the GRU must then learn to re-interpret. In contrast, STU *inside* the recurrence directly shapes how past actions influence the deter update, which is where spectral filtering's LDS-approximation theorem actually applies.

**Key insight for the paper:** Placement of STU within world-model architectures matters. Mixing embeds before the RSSM (ReZero approach) does not obviously improve learning; integrating STU as a direct input to the recurrence (stu-dreamer-jax approach) does.

---

## Setup

### Repository
- **Repo**: `/lambda/nfs/hazan/ReZero` (fork of `NM512/r2dreamer`)
- **Backend**: PyTorch (Torch DreamerV3/R2-Dreamer). Distinct from stu-dreamer-jax which is JAX.
- **GPU**: Single A100 40GB (later della-gpu Princeton)
- **Task**: Atari Pong (100k-style, ~400K env steps)
- **Seed**: 0 for both runs

### Integration Design
From `REZERO_SETUP.md` (copied into this directory):

```
Vanilla R2-Dreamer flow:
embed = encoder(data)              # [B, T, E]
post_stoch, post_deter, _ = rssm.observe(embed, ...)

With STU mixer (this work):
embed = encoder(data)              # [B, T, E]
embed = stu_mixer(embed)           # [B, T, E]      *  causal Hankel_L
post_stoch, post_deter, _ = rssm.observe(embed, ...)
```

The mixer is a stack of `STUSandwichBlock` modules (mirroring `spectral-transformer-jax/src/stj/models/blocks.py::STUSandwichBlock`):

```
sublayer 1:  x = x + LayerScale * FFN_in(LN(x))    # per-position
sublayer 2:  x = x + LayerScale * STU(LN(x))       # sequence mixing
sublayer 3:  x = x + LayerScale * FFN_out(LN(x))   # per-position
```

- `LayerScale` initialized to `1e-4` → mixer ≈ identity at init
- `use_hankel_L=True` → causal Hankel basis Z_L (causality enforced)
- Training: mixer runs over full `[B, T=64, E]` batch in parallel
- Inference: rolling embed buffer of length `seq_len=64` in agent state; each step appends new embed, runs mixer over window, takes last position

### Filter Variant Tested
- `hankelscaled` — Hankel eigenvectors with σ^{1/4} energy scaling (same as our stu-dreamer-jax stuB default). Rationale from offline study: `0409026.md` found `hankel_scaled` was best at Pong 5-ep multistep dynamics MSE vs `hankel`, `dct`, `random_normalized`.

---

## Experiments

### Table: Pong runs (both at seed 0, 400K env steps)

| Run | Config | Final eval score (step 400K) | Last 10 train mean | Trajectory shape |
|-----|--------|------------------------------|-------------------|------------------|
| `pong_baseline_seed0` | vanilla R2-Dreamer | **+4.3** | -7.8 | Climbs steadily from -21 → +4 |
| `pong_stu_hankelscaled_seed0` | encoder + STUSandwichBlock + RSSM | **-12.8** | -12.8 | Stalls around -12 to -17 from step 100K onwards |

### Eval score trajectory (baseline vs STU)

| step | baseline eval | STU eval |
|------|--------------|----------|
| 50K  | -19.8 | -21.0 |
| 100K | -17.6 | -19.3 |
| 150K | -18.5 | -18.5 |
| 200K | -16.9 | -18.3 |
| 250K | -8.9  | -15.8 |
| 300K | -7.4  | -14.8 |
| 340K | **+1.5** | -14.3 |
| 400K | **+4.3** | -12.8 |

The baseline made a clear breakthrough around step 250K (-17 → -8), while STU lagged and never recovered.

Plot: `plots/pong_comparison.png`

### Known limitations (from REZERO_SETUP.md)

- **Episode warm-up**: first 63 steps of each episode run with partial (zero-padded) buffer. Train and inference not bit-equivalent during warm-up.
- **Cross-episode contamination at training time**: if a batch row spans an episode reset (replay buffer does not currently prevent), mixer's conv mixes pre-reset embeds into post-reset positions for up to `seq_len` steps.
- `_video_pred` does NOT apply the mixer (slices `embed[:, :5]` but mixer expects `seq_len=64`).
- Causality enforced via `use_hankel_L=True`; non-causal basis not exposed.

---

## Connection to Other Experiments

### vs stu-dreamer-jax (log #8)

| dimension | ReZero (this work) | stu-dreamer-jax |
|-----------|-------------------|-----------------|
| STU placement | **before** RSSM (encoder → STU → RSSM) | **inside** RSSM (GRU 4th input branch) |
| What STU mixes | encoder outputs `[B, T, E]` | action history `[B, W, action_dim]` |
| Result | Stalls below baseline on Pong | +20-36% convergence speedup on walker_walk |
| Theoretical fit | Unclear — GRU re-interprets mixed embeds | Direct: STU approximates the action→state LDS per Thm 3.1 |

### vs spectral-transformer-jax (log #10, planned)

The ReZero design *copies* the STUSandwichBlock from spectral-transformer-jax exactly. In language modeling that block works (STU is competitive with attention). In Dreamer, it doesn't transfer. This suggests:
- STU as a transformer-block token mixer is a solved/working pattern
- STU as a world-model encoder mixer needs different theory
- STU as a world-model recurrence input (stu-dreamer-jax) does work

### vs STUZero offline dynamics (logs #1-7, #0409026)

The offline study found `hankel_scaled` was the best filter for Pong dynamics prediction at 5 episodes. In ReZero we used the same filter choice but in a very different integration (as forward-path mixer, not as open-loop dynamics decoder). The offline ranking did not transfer, consistent with the "placement matters" finding.

---

## Open Questions / Next Steps

1. **Move STU inside the ReZero RSSM** as we did in stu-dreamer-jax. Same integration but in PyTorch would provide a direct cross-framework validation of the "STU inside recurrence" finding.

2. **Ablate the mixer placement**: instead of before the RSSM, try after (post-RSSM) or as a parallel pathway. Would isolate whether the issue is the mixer itself or its position in the pipeline.

3. **Test on DMC proprio** instead of Pong. Pong's discrete 6-action space and highly-strategic value function may be a poor fit for a spectral-filtering inductive bias. The stu-dreamer-jax success was on continuous-control DMC — the domain may matter.

4. **Multi-seed validation.** Pong is notoriously seed-sensitive; the -12.8 vs +4.3 gap could be within seed noise, though 17 points is large for the Dreamer Pong regime.

---

## Artifacts

All artifacts in `9_rezero_stu_mixer_data/`:

- `REZERO_SETUP.md` — original integration design doc (copied from ReZero repo)
- `code/filter_factory.py`, `code/stu_dynamics.py`, `code/stu_layer.py` — STU implementation files from `ReZero/stu_dynamics/`
- `runs/pong_baseline_seed0.tar.gz` — baseline metrics + console log
- `runs/pong_stu_hankelscaled_seed0.tar.gz` — STU-mixer run metrics + console log
- `plots/pong_comparison.png` — eval score trajectory, baseline vs STU-mixer
- `tfevents/pong_*_tfevents.tar.gz` — compressed tensorboard event files (~15 MB each). Contains scalars AND image summaries (`eval_video`, `train_video` as animated GIFs, 41 eval + 52 train per run). Used to generate the ReZero Pong reels in `12_final_sweep_data/video_reels/`. Load via `tensorboard.backend.event_processing.event_accumulator`.
- `configs/` — R2-Dreamer hydra configs (env + model sizes from 12M to 400M)

---

## References

- Agarwal, N., Suo, D., Chen, X., & Hazan, E. (2024). Spectral State Space Models. arXiv:2312.06837.
- NM512 R2-Dreamer implementation (PyTorch): base codebase this work forks.
- `STUZero/research_logs/logs/0409026.md` — offline filter ranking that motivated `hankelscaled` choice.
- `STUZero/research_logs/logs/8_stu_dreamer_research_log.md` — parallel JAX experiment with different integration point.
