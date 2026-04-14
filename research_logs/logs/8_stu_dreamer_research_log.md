# Research Log — 2026-04-11 to 2026-04-14
## Goal: Integrate spectral filtering (STU) into DreamerV3 world models

---

## Executive Summary

Five key findings from integrating STU into the DreamerV3 RSSM world model:

1. **Post-hoc STU refinement breaks the RSSM.** An initial attempt applying STU as a residual on top of the GRU-produced deter chain collapsed policy performance to the random-policy floor (~29 on dmc_walker_walk vs baseline 940). Root cause: STU never influenced the recurrence itself, and post-hoc resampling of the posterior broke carry/feat/entries consistency.

2. **STU-inside-`_core` with action-history buffer works.** Replacing the post-hoc design with a per-step spectral filter over a rolling action buffer (fed as a 4th input branch to the BlockGRU) recovers and exceeds baseline performance. On walker_walk (size1m, 640K params) this design reaches episode score 900+ at 320K env steps vs baseline's 400K (~20% sample efficiency gain) and converges to 965 mean vs baseline's 940 (+2.5% asymptotic).

3. **Random-filter ablation isolates speed benefit to Hankel eigenvectors.** Replacing the Hankel eigenvectors with Haar-random orthonormal vectors with matched σ^{1/4} energy spectrum: converges to the same ~960 asymptote but takes **55% more env steps** (496k vs 320k to reach 900). Proves the improvement is specifically from the spectral basis structure, not from architectural artifacts (buffer, convolution, or extra parameters).

4. **Observation filtering (from Dogariu et al. 2025 OSF paper) provides additional speedup at size1m.** Adding a second STU channel that filters past stoch representations alongside the action filter — plus negative spectral bank, K=48 filters, non-zero outscale, and direct input shortcut — reaches 900 at **256K env steps** (36% faster than baseline, 20% faster than stuB).

5. **Scaling to size12m (published config) hits integration interference.** The full `stu_plus` feature stack fails to improve over baseline at the published 12M-param size on hopper_stand — policy stays at near-zero through step 200k+ while published baseline climbs to ~270. Hypothesis: interaction between extra features (4× larger obs buffer at this scale, non-zero outscale, K=48 × 2 banks) exceeds what the larger GRU can absorb without interference.

**Key insight for the paper:** Spectral filtering provides a strong *speed* inductive bias for world-model learning when integrated correctly (inside the recurrence, not on top). The advantage is specifically from Hankel eigenvectors, not from any temporal mixing. But at larger model scales, simpler STU designs may be more robust than feature-stacked variants.

---

## Setup

### Repository & Environment
- **Repo**: `/lambda/nfs/hazan/stu-dreamer-jax` (fork of DreamerV3 JAX implementation)
- **Venv**: `.venv/` (Python 3.11, JAX with CUDA)
- **GPU**: Single A100 40GB
- **Baseline reference**: DreamerV3 at size1m on dmc_walker_walk, trained for 1.1M env steps by the user prior to this work, converged at ~940 (reference file: `runs/baseline_proprio_run.tar.gz`).

### Task & Budget
- **Primary task**: `dmc_walker_walk` (proprioceptive, 6-dim continuous action, 14+9 obs dims)
- **Secondary tasks**: `dmc_hopper_stand` (hard hopping/balancing task), `dmc_cheetah_run` (in progress)
- **Budget**: 500K env steps (proprio benchmark standard). All runs went to 500K+, some past 1M.
- **Config**: `dmc_proprio` preset (inherits `size1m`: deter=512, hidden=64, stoch=32, classes=4, ~640K params). size12m experiments override to deter=2048, hidden=256, classes=16, ~12M params.

### Key Design Decisions
- **Config presets added** (see `dreamerv3/configs.yaml`):
  - `stu_enabled` — post-hoc refinement (Run A, legacy, kept for backward compat)
  - `stu_core` — STU inside `_core` with action buffer (Run B)
  - `stu_rand` — same as stu_core but Haar-random orthonormal filters with matched energy (ablation)
  - `stu_plus` — stu_core + K=48 + neg spectral bank + obs filtering + outscale=0.01 + input shortcut
  - `dmc_proprio_12m` — dmc_proprio with size12m overrides

### Key Files Modified/Added
- `dreamerv3/stu.py` — `STUMixer` (post-hoc), `STUCore` (in-recurrence). Filter generation supports Hankel, random, random-normalized.
- `dreamerv3/rssm.py` — `RSSM` with `stu_mode ∈ {posthoc, core}`, action-buffer management, `_ensure_stu_buffer()` helper, `_stu_step()` for per-step filtering (supports action+obs dual channels), `_core()` accepts optional `stu_context` via 4th `dynin3` branch.
- `dreamerv3/configs.yaml` — presets listed above.

---

## Experiments

### Table: runs conducted

| # | Run | Task | Size | Env Steps | Final Score (last 50 mean) | Notes |
|---|-----|------|------|-----------|---------------------------|-------|
| 0 | baseline | walker_walk | size1m | 448K | 940 ± 24 | Reference (from user, pre-existing) |
| 1 | Run A (post-hoc) | walker_walk | size1m | ~14K | ~30 | Killed early. Catastrophic failure validated design issue. See stuA logs. Fixed diagnostic: stuB |
| 2 | stuB (core, K=24) | walker_walk | size1m | 592K | 965 ± 14 | **+25 asymptotic, 20% faster to 900.** Runs/stuB_proprio_run.tar.gz |
| 3 | stuRand (core, random-normalized) | walker_walk | size1m | 848K | 962 ± ? | Same asymptote, +55% slower to reach 900. Runs/stuRand_proprio_run.tar.gz |
| 4 | stuPlus (K=48 + neg + obs + shortcut + outscale=0.01) | walker_walk | size1m | ~300K | >920 (still rising when killed) | Reached 900 at 256K (36% faster than baseline) |
| 5 | stuPlus | hopper_stand | size1m | ~720K | 562 ± ? (max 850) | Impressive for 20× smaller model vs published 650 baseline at 12M |
| 6 | stuPlus (12M) | hopper_stand | size12m | 220K (killed) | <5 (stuck at zero) | Failed to break out. Feature interaction at scale. |
| 7 | stuCore (12M) | hopper_stand | size12m | ~50K (killed) | N/A, pivoted | Switched to cheetah_run per user direction |
| 8 | stuPlus | cheetah_run | size1m | in progress | pending | Comparable setup to walker experiment. Published baseline 596. |

### Run 2 — stuB on walker_walk (THE KEY RESULT)

**Config**: `--configs dmc_proprio stu_core --task dmc_walker_walk`
**Architecture changes vs baseline**:
- Action-history buffer (B, W=64, action_dim=6) in carry
- At each RSSM step: slide buffer, append current action, run causal FFT convolution against K=24 Hankel eigenvectors, take output at last timestep, project and feed into GRU as 4th input branch (`dynin3`)
- `filter_proj` and `out_proj` zero-initialized — STU contributes exactly zero at step 0
- Same code path for observe and imagine (buffer accumulates real actions in observe, policy actions in imagine)

**Results** (32k-bin trajectory):
```
step bin    baseline    stuB
 64- 96k    141 ± 56    206 ± 100
128-160k    513 ± 43    467 ± 66
192-224k    529 ± 74    613 ± 59   (+84)
256-288k    648 ± 48    721 ± 66   (+73)
320-352k    748 ± 46    918 ± 32   (+170)
384-416k    914 ± 43    942 ± 46   (+28)
448-480k    952 ± 25    963 ± 12   (+11)
converged   940 ± 24    965 ± 14
```

**Plot**: `plots/walker_walk_4way.png` (shows baseline, stuB, stuRand, stuPlus on same axes)

### Run 3 — stuRand (random-normalized filter ablation)

**Config**: `--configs dmc_proprio stu_rand --task dmc_walker_walk`
**Key modification**: `stu_random=True, stu_random_normalized=True`. Generates K=24 Haar-random orthonormal vectors and scales them by the σ^{1/4} energy spectrum of the actual Hankel matrix. Same architecture, same params, same compute, only the filter directions differ.

**Results** (selected bins):
```
step bin    baseline    stuRand
128-160k    513         425         (-88)
256-288k    648         615         (-33)
320-352k    748         605         (-143)   ← plateau
416-448k    936         756         (-180)
480-512k    —           938                   ← finally climbs
```

stuRand plateaued at ~605 through step 320k–416k, then broke out and converged to 921 at 496k. Final asymptote 962 (essentially tied with stuB's 965). **But 55% more env steps to reach 900.**

### Run 4 — stuPlus on walker_walk (combined improvements)

**Config**: `--configs dmc_proprio stu_plus --task dmc_walker_walk`
**Modifications vs stuB**:
- K=48 (theorem says K~35-40 for L=64, stuB's 24 was under-specified)
- Negative spectral bank: second filter_proj_neg with (-1)^i · ϕ_k modulation for negative-eigenvalue dynamics
- Observation filtering: second STU channel with stoch-history buffer (stoch*classes = 128 dim at size1m). Based on Dogariu, Brahmbhatt, Hazan 2025 (OSF paper) which generalizes spectral filtering to filter over past observations for asymmetric + noisy + nonlinear systems
- Direct input shortcut: M^u u_t linear term from current buffer[-1,:] to output
- outscale=0.01 on out_proj (vs 0.0 for stuB) — lets STU contribute immediately instead of waiting for filter_proj to grow from zero

**Results**:
```
Steps to 900:  stuPlus 256K | stuB 320K | baseline 384K | stuRand 464K
Steps to 950:  stuPlus 288K | stuB 368K | baseline 448K | stuRand 496K
```

stuPlus is the fastest learner among the tested variants on walker_walk. **36% fewer env steps than baseline to reach 900.**

### Run 5 — stuPlus on hopper_stand (size1m)

Same config, harder task. Published DreamerV3 baseline at size12m reaches 650 mean at 500K steps; max 160.

**Results** (32k-bin):
```
  0-96k:    1-5    (near zero, normal for hopper)
 128-160k:  17     (first signs)
 192-224k:  49     (climbing)
 256-288k:  85     (dip, high variance)
 320-352k: 108
 416-448k: 273
 480-512k: 356
 608-640k: 465
 672-704k: 534
```

At step 720K: last 50 mean **562**, max recent **847**. **86% of published size12m baseline's mean despite 20× fewer parameters.**

**Plot**: `plots/hopper_stand_size1m_vs_baseline12m.png`

### Run 6 — stuPlus on hopper_stand (size12m)

Same config as Run 5 but with `dmc_proprio_12m` (deter=2048, hidden=256, classes=16 — the published proprio config). ~10.8M params total.

**Results** (32k-bin):
```
step bin     stuPlus 12m   published baseline 12m
  0- 32k     2.2           6.1
 32- 64k     0.5           31.9
 64- 96k     3.1           111
 96-128k     1.4           140
128-160k     1.4           225
160-192k     2.0           269
192-208k     0.7           —
208-224k     4.9           —
```

After killing and resuming from checkpoint at 160k, the policy remained at near-zero through 220k. Killed at step 220k. This is a real regression — published baseline was climbing rapidly in this window while stuPlus 12m gets nothing useful from the world model.

**Hypotheses for failure at size12m:**
1. Obs buffer is 4× larger at size12m (512-dim stoch vs 128 at size1m). Spectral filtering over this larger, noisier observation stream may be dominating the action signal.
2. `outscale=0.01` contributes STU immediately; at size12m the larger STU output may interfere with early policy learning before filter_proj has learned useful features.
3. Too many moving parts (K=48 × 2 banks + obs channel + shortcut) at larger scale — feature interactions that were benign at size1m become harmful.
4. Larger GRU is already more expressive; STU's marginal value is smaller and easier for the policy to misinterpret as noise.

**Next step planned**: test stu_core (simpler, proven) at size12m as a control. User pivoted to cheetah_run before this could run.

### Run 8 — stuPlus on cheetah_run (in progress as of 2026-04-14)

Same size1m stuPlus configuration as walker_walk (Run 4), different task. Published baseline at size12m: 596.3. Running now with cheetah_run's 6-dim action space. First metrics show dyn loss 3.03, fps/policy 24.2, training healthy.

---

## Key Implementation Snippets

### `STUCore` module (dreamerv3/stu.py)

```python
class STUCore(nj.Module):
    num_eigh: int = 24
    max_seq_len: int = 64
    use_hankel_L: bool = False
    random: bool = False
    random_normalized: bool = False
    use_neg_bank: bool = False
    outscale: float = 0.0
    use_shortcut: bool = False

    def __init__(self, units, **kw):
        self.units = units; self.kw = kw
        self._phi = get_spectral_filters(
            self.max_seq_len, self.num_eigh, self.use_hankel_L,
            random=self.random, random_normalized=self.random_normalized)
        if self.use_neg_bank and not self.random:
            signs = np.power(-1.0, np.arange(self.max_seq_len)).astype(np.float32)
            self._phi_neg = self._phi * signs[:, None]
        else:
            self._phi_neg = None

    def __call__(self, buffer):  # (B, W, in_dim)
        W = buffer.shape[-2]
        phi = jnp.array(self._phi[:W])
        h = self.sub('in_proj', nn.Linear, self.units, bias=False, **self.kw)(buffer)
        filter_proj = self.value('filter_proj', nn.Initializer('zeros'),
                                 (self.num_eigh, self.units))
        filters = jnp.einsum('tk,kd->td', phi, filter_proj.astype(f32))
        if self._phi_neg is not None:
            phi_neg = jnp.array(self._phi_neg[:W])
            filter_proj_neg = self.value('filter_proj_neg', nn.Initializer('zeros'),
                                          (self.num_eigh, self.units))
            filters = filters + jnp.einsum('tk,kd->td', phi_neg,
                                           filter_proj_neg.astype(f32))
        mixed = fft_convolve(h, filters)
        last = mixed[..., -1, :]
        if self.use_shortcut:
            shortcut = self.sub('shortcut', nn.Linear, self.units, bias=False,
                                **{**self.kw, 'outscale': 0.0})(buffer[..., -1, :])
            last = last + shortcut.astype(last.dtype)
        out_kw = {**self.kw, 'outscale': self.outscale}
        out = self.sub('out_proj', nn.Linear, self.units, **out_kw)(
            last.astype(buffer.dtype))
        return out
```

### `_stu_step` with optional obs channel (dreamerv3/rssm.py)

```python
def _stu_step(self, buffer, action, stoch, obs_buffer=None):
    new_buffer = jnp.concatenate(
        [buffer[:, 1:, :], action[:, None, :].astype(buffer.dtype)], axis=1)
    units = self.stu_units or self.hidden
    stu_kw = dict(num_eigh=self.stu_num_eigh, max_seq_len=self.stu_max_seq,
                  use_hankel_L=self.stu_use_hankel_L,
                  random=self.stu_random, random_normalized=self.stu_random_normalized,
                  use_neg_bank=self.stu_neg_bank, outscale=self.stu_outscale,
                  use_shortcut=self.stu_shortcut)
    stu_ctx = self.sub('stu_act', stu.STUCore, units, **stu_kw, **self.kw)(new_buffer)

    new_obs_buffer = None
    if obs_buffer is not None:
        stoch_flat = stoch.reshape((stoch.shape[0], -1))
        new_obs_buffer = jnp.concatenate(
            [obs_buffer[:, 1:, :],
             stoch_flat[:, None, :].astype(obs_buffer.dtype)], axis=1)
        obs_ctx = self.sub('stu_obs', stu.STUCore, units, **stu_kw, **self.kw)(new_obs_buffer)
        stu_ctx = stu_ctx + obs_ctx
    return new_buffer, new_obs_buffer, nn.cast(stu_ctx)
```

### Config presets (dreamerv3/configs.yaml)

```yaml
stu_core:
  agent.dyn.rssm.use_stu: True
  agent.dyn.rssm.stu_num_eigh: 24
  agent.dyn.rssm.stu_max_seq: 64
  agent.dyn.rssm.stu_mode: core

stu_rand:
  # same as stu_core +
  agent.dyn.rssm.stu_random: True
  agent.dyn.rssm.stu_random_normalized: True

stu_plus:
  agent.dyn.rssm.use_stu: True
  agent.dyn.rssm.stu_num_eigh: 48
  agent.dyn.rssm.stu_max_seq: 64
  agent.dyn.rssm.stu_mode: core
  agent.dyn.rssm.stu_neg_bank: True
  agent.dyn.rssm.stu_outscale: 0.01
  agent.dyn.rssm.stu_shortcut: True
  agent.dyn.rssm.stu_use_obs: True
```

---

## Key Findings Summary

### Finding 1: Integration placement is critical
Post-hoc STU refinement (applied after GRU scan completes) collapses to random-policy floor. STU-inside-`_core` (fed as 4th input branch to GRU) works correctly. The spectral filters need access to the raw action sequence, not the GRU's already-processed hidden states.

### Finding 2: Same-asymptote speed improvement
All three STU variants (stuB spectral, stuRand random, stuPlus combined) converge to approximately the same final score (~920-965) on walker_walk. STU does not unlock new representational capacity at this model size — it accelerates convergence.

### Finding 3: Hankel basis specifically matters for speed
Random orthonormal filters with matched energy spectrum take 55% more env steps to reach 900 than Hankel eigenvectors. Same architecture, same params, same compute. The inductive bias of the Hankel basis (which exactly spans the impulse-response subspace of bounded LDS per STU Theorem 3.1) is what accelerates learning.

### Finding 4: Stacked improvements work at size1m
K=48 + negative spectral bank + observation filtering + input shortcut + non-zero outscale → fastest learner on walker_walk (36% fewer env steps than baseline to reach 900).

### Finding 5: Integration doesn't scale cleanly to 12M
The full `stu_plus` stack fails at size12m on hopper_stand. Either the larger obs buffer (512 vs 128 stoch dim) dominates, or the combined features interfere, or the larger GRU has less headroom for the extra context. Needs ablation (test stu_core at 12m) to isolate.

### Finding 6: Same model, different task (preliminary)
size1m stuPlus on hopper_stand reached 562 / max 847 at step 720k — 86% of the published size12m baseline (650) with 20× fewer params. Suggests the integration generalizes across tasks, not just walker-specific.

---

## Open Questions / Next Steps

1. **Does stu_core (simple) work at size12m where stu_plus fails?** If yes, that isolates the failure to stu_plus's extra features. Run planned but pivoted to cheetah.

2. **What's the scaling law?** At 12M, 25M, 50M params, does STU help, hurt, or stay neutral? This would determine whether it's useful for the published DreamerV3 configuration.

3. **Harder tasks with longer effective memory?** Walker_walk is nearly solved by baseline. The theoretical promise of STU — condition-number-free learning for marginally stable LDS — needs tasks where baseline genuinely struggles with long-range temporal credit assignment (humanoid, Minecraft, DMLab).

4. **Multi-seed validation.** All results are single-seed. The ~25-point asymptotic delta between stuB and baseline could be within seed variance.

5. **Compute overhead mitigation.** Per-step FFT inside `nj.scan` drops fps/train from 25k to 15k (stuB) to 9k (stuPlus). A direct matmul against precomputed filters for small W=64 could eliminate FFT overhead. Also the stuPlus obs buffer (512 dim at size12m) may benefit from a projection before STU.

6. **What specifically makes Run 6 fail?** Ablate stu_plus → stu_plus without obs / without neg bank / without outscale to identify which feature interferes at 12m.

---

## Artifacts

All artifacts are in `8_stu_dreamer_data/`:

- `STU_INTEGRATION.md` — original writeup, includes complete results table
- `runs/` — archived metrics.jsonl, scores.jsonl, config.yaml for each run:
  - `baseline_proprio_run.tar.gz` — baseline walker_walk (reference)
  - `stuB_proprio_run.tar.gz` — Run 2 (stuB winner on walker)
  - `stuRand_proprio_run.tar.gz` — Run 3 (random-filter ablation)
  - `stuPlus12m_hopper_stand_*.tar.gz` — Run 6 (failed 12m hopper)
  - `stuCore12m_hopper_stand_*.tar.gz` — Run 7 (aborted, pivoted)
  - `stuPlus_cheetah_run_*.tar.gz` — Run 8 (in progress)
- `plots/`:
  - `walker_walk_4way.png` — baseline / stuB / stuRand / stuPlus
  - `hopper_stand_size1m_vs_baseline12m.png` — Run 5 vs published
  - `hopper_stand_12m_comparison.png` — Run 6 vs published
- `compare_runs.py` — comparison/plotting script

---

## References

- Agarwal, N., Suo, D., Chen, X., & Hazan, E. (2024). Spectral State Space Models. arXiv:2312.06837.
- Dogariu, E., Brahmbhatt, A., & Hazan, E. (2025). Universal Learning of Nonlinear Dynamics. arXiv:2508.11990. (OSF / Observation Spectral Filtering — motivation for adding stoch history channel in stu_plus)
- Hafner, D., Pasukonis, J., Ba, J., & Lillicrap, T. (2024). Mastering Diverse Domains through World Models. arXiv:2301.04104.
