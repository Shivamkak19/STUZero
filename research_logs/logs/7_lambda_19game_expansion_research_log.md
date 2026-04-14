# 19-Game Expansion on Lambda A100 — Research Log
## Dates: 2026-04-07 → 2026-04-09 (paused)

---

## Goal

Expand the offline dynamics sample-efficiency study from the original 7-game cohort (pong, asterix, alien, seaquest, kungfumaster, mspacman, roadrunner) to a 26-game cohort by collecting and training on 19 additional Atari games on the Lambda A100 box. The 19 new games:

```
assault bankheist battlezone amidar boxing breakout choppercommand crazyclimber
demonattack freeway frostbite gopher hero jamesbond kangaroo krull privateeye
qbert upndown
```

---

## Status (paused 2026-04-09)

Pipeline killed cleanly so the box is free for other work. Everything is **resumable** — sentinels (`collect.done`, `upload.done`, per-run `training_complete.flag`) are in place and the new pipeline scripts will skip any work already done.

### Data collection — 100% done

All 19 new games collected (50 episodes each, max_steps=27000) and uploaded to HuggingFace:
`Shivamkak/STUZero-Atari-Dynamics`

| Game | Disk | Mean ep length | Notes |
|------|-----:|----------------:|-------|
| amidar | 2.2 GB | 978 | |
| assault | 2.8 GB | 1,281 | |
| bankheist | 2.0 GB | 915 | |
| battlezone | 4.2 GB | 1,921 | |
| boxing | 1.6 GB | 686 | |
| breakout | **58 GB** | **27,000** | expert never dies, hits cap |
| choppercommand | 1 GB | 245 | |
| crazyclimber | 7.4 GB | 3,436 | |
| demonattack | 10 GB | 4,623 | |
| freeway | 5 GB | 2,045 | mean reward 0 (expert hadn't learned to cross) |
| frostbite | 1.7 GB | 800 | |
| gopher | **58 GB** | **27,000** | expert never dies, hits cap |
| hero | 13 GB | 5,958 | |
| jamesbond | 3.1 GB | 1,487 | |
| kangaroo | 1.5 GB | 709 | |
| krull | 2.8 GB | 1,255 | |
| privateeye | 6 GB | 2,695 | |
| qbert | 2 GB | 901 | |
| upndown | 3 GB | 1,380 | |

**Total expanded cohort**: 26 games, 1,300 episodes, 4,943,581 transitions on HF.

The HF root README and all 19 per-game READMEs were updated on 2026-04-08:
- Single combined 26-game alphabetical table
- Long-episode caveat noted for breakout/gopher
- Per-game READMEs corrected from `float32` → `float16` for the latent fields (the actual stored dtype is float16; the original per-game READMEs were wrong about this)

### Training runs — partial

Completed sample-efficiency suites (full 30 runs each — 4 splits × 3 models × 2 train modes + multi-seed + rl100):

```
amidar         30/30
assault        30/30
bankheist      30/30
battlezone     30/30
boxing         30/30
choppercommand 30/30
crazyclimber   30/30
```

Partial:
```
demonattack    10/30  (5ep × 3 models × 2 train modes done; rest pending)
breakout       1/30   (5ep multi-step STU only)
```

5ep multi-step only (priority pass for STU/baseline/attention, no other splits):
```
freeway        3/3 ✓
frostbite      3/3 ✓
hero           3/3 ✓
jamesbond      3/3 ✓
kangaroo       3/3 ✓
krull          3/3 ✓
privateeye     3/3 ✓
qbert          3/3 ✓
upndown        3/3 ✓
breakout       1/3 (STU only — baseline + attention pending)
gopher         0/3 (queued)
```

The 7 original-cohort games have 5ep multi-step results in `master_summary.md` from prior Della-cluster runs; not re-run on Lambda.

---

## Key findings so far — 5ep multi-step rollout MSE (step-50)

Combining the 24 games available: 17 from this Lambda pass + 7 from `master_summary.md`. Bold = winner per row.

| Game | Baseline+GN | STU | Attention | STU vs BL | Best |
|------|------------:|----:|----------:|----------:|------|
| Freeway | 0.01705 | **0.00026** | 0.00509 | **65.6x** | STU |
| Seaquest | 0.01365 | **0.00070** | 0.00228 | **19.5x** | STU |
| PrivateEye | 0.00775 | **0.00042** | 0.00133 | **18.4x** | STU |
| BattleZone | 0.01822 | **0.00147** | 0.00290 | **12.4x** | STU |
| Pong | 0.01970 | **0.00170** | 0.00290 | **11.6x** | STU |
| Jamesbond | 0.02348 | **0.00250** | 0.00411 | **9.4x** | STU |
| CrazyClimber | 0.00554 | **0.00099** | 0.00245 | **5.6x** | STU |
| Assault | 0.03945 | 0.01021 | **0.00851** | 3.9x | Attn |
| Frostbite | 0.26428 | 0.07156 | **0.00430** | 3.7x | Attn |
| KungFuMaster | 0.00537 | **0.00146** | 0.00183 | **3.7x** | STU |
| Hero | 0.00111 | **0.00037** | 0.00258 | **3.0x** | STU |
| Krull | 0.01866 | 0.00641 | **0.00476** | 2.9x | Attn |
| RoadRunner | 0.07117 | 0.02599 | **0.00404** | 2.7x | Attn |
| Amidar | 0.06909 | 0.02581 | **0.00211** | 2.7x | Attn |
| BankHeist | 0.07270 | 0.03237 | **0.00197** | 2.3x | Attn |
| Boxing | 0.22611 | 0.10677 | **0.00422** | 2.1x | Attn |
| Asterix | 0.00297 | **0.00157** | 0.00293 | **1.9x** | STU |
| Kangaroo | 0.16313 | 0.10455 | **0.00129** | 1.6x | Attn |
| Qbert | 0.04379 | 0.02918 | **0.00913** | 1.5x | Attn |
| DemonAttack | 0.00392 | **0.00284** | 0.00387 | 1.4x | STU |
| UpNDown | 0.03386 | 0.02708 | **0.00814** | 1.3x | Attn |
| Alien | 0.14769 | 0.12732 | **0.00649** | 1.2x | Attn |
| ChopperCommand | 0.65769 | 0.57524 | **0.00892** | 1.1x | Attn |
| MsPacman | 0.05331 | 0.04796 | **0.00487** | 1.1x | Attn |

Breakout STU 5ep multi-step: **0.00104** (no BL/Attn yet).
Gopher: not yet run.

### What this table changes about the story

**STU vs baseline.** STU beats baseline in 24/24 games. Range 1.1x–65.6x. Rock-solid claim.

**STU vs attention.** This is the new tension. With 24 games:
- STU wins outright in **10/24** games (Freeway, Seaquest, PrivateEye, BattleZone, Pong, Jamesbond, CrazyClimber, KungFuMaster, Hero, Asterix, DemonAttack — actually 11)
- Attention wins in **13/24** games

So attention is the best architecture on a slight majority of games. The "STU is best" framing the prior cohort suggested no longer holds — it's "STU is the best low-parameter inductive bias against the EZv2 baseline" but attention is competitive and sometimes better. This needs to be reported honestly.

**Per-game patterns.** STU dominant games tend to have small action spaces and structured dynamics (Freeway 3 actions, Pong 6, Jamesbond 18 but very repetitive). Attention dominant games tend to be high-action, visually diverse (Frostbite, Boxing, Bankheist, Amidar — most with 18 actions).

**Headline number candidates.** Freeway 65.6x is the largest single-game STU advantage in the project — nearly 4x larger than anything in the prior cohort. Worth eyeballing for sanity but if real it's a paper-quality result.

---

## Bugs found and fixed during this expansion

These are pre-existing bugs in the pipeline scripts that bit us hard. All fixed in the current code.

### 1. Checkpoint resolution bug
`train_dynamics_offline.py:514` loads the expert checkpoint from a path relative to the data dir (e.g. `assault_90K/atari_assault_model_90000.p`), but the actual file lives in `offline_training/atari_models/`. Every fresh game's training jobs failed at "Phase 2: Training Setup" with `FileNotFoundError`.

**Fix:** Pipeline script `run_remaining_games.sh` and `run_5ep_priority.sh` now symlink the expert checkpoint into the data dir as a setup step, after collection but before training. Backfilled the symlink for all already-collected new-cohort dirs.

### 2. False-completion sentinel bug
`run_sample_efficiency_parallel.sh` exited 0 even when every child training job failed. Combined with `tee` swallowing the exit code in the parent pipeline, the parent wrote `sample_eff.done` and reported "FULLY COMPLETE" for games where 0/30 runs had actually succeeded. The first 5 "completed" games (assault, bankheist, battlezone, amidar, boxing) initially had this fake-complete state.

**Fix:** `run_sample_efficiency_parallel.sh` now records failures to `.failed_runs.$$` and exits non-zero if any are present. `run_remaining_games.sh` checks this exit code via `${PIPESTATUS[0]}` before writing `sample_eff.done`. The 5 false sentinels were scrubbed and the games re-run.

### 3. `wait` blocking on `tee` subshell
`exec > >(tee -a "$GLOBAL_LOG")` in `run_5ep_priority.sh` created a tee subshell as a bash background job. A subsequent `wait` (no args) at the end of each iteration would block forever waiting on tee, since tee never exits.

**Fix:** Removed `exec >>(tee)` from the script. Caller redirects stdout/stderr externally instead.

### 4. `wait $oldest_pid` lockstep batching
First version of the global pool tracker waited on the *oldest* tracked pid when no others were obviously dead. This blocked the script from launching new jobs even when younger siblings finished, because reap_one was parked inside `wait $oldest`. Pool ran in lock-step batches of 8 instead of streaming.

**Fix:** Replaced custom pid tracking with bash's built-in `jobs -rp` and `wait -n` (wait for *any* child). `wait -n` returns as soon as any background job finishes, so the pool can immediately fire a new job. This is the version currently in `run_5ep_priority.sh`.

### 5. Auto-PARALLEL_JOBS from dataset size
Added to `run_remaining_games.sh` after the 2026-04-07 OOM event where breakout's 58 GB dataset × 4 parallel jobs blew through 216 GB system RAM, killed the python processes, the parent shell, and the prior `claude` session. Now auto-picks: <10 GB → 4, 10–30 GB → 2, ≥30 GB → 1.

### 6. Original 7 cohort experts have non-standard naming
Pong's expert is `pong_model_100000.p`, asterix's is `asterix_model_110000.p` (no `atari_` prefix). The 19 new-cohort experts use the `atari_<game>_model_<steps>.p` convention. This matters when looking for the expert file or computing symlinks. Both old-style files are present on HF.

---

## What's still left to do

### High-priority (research story)

1. **Finish 5ep multistep priority pass** — 5 runs left:
   - breakout_baseline_only_5ep_multistep
   - breakout_attention_5ep_multistep
   - gopher_stu_5ep_multistep
   - gopher_baseline_only_5ep_multistep
   - gopher_attention_5ep_multistep
   - All 5 must run with PARALLEL_JOBS=1 (memory cliff).
   - Estimated wall time: ~70 min/run × 5 = ~6 hours

2. **Multi-seed validation for the priority cohort** — repeat the 5ep multistep STU and baseline runs at seeds 123, 456 for the 12 games that only got the priority single-seed pass. ~24 runs.

3. **100-step rollout for the priority cohort** — repeat 5ep multi-step STU and baseline with `--rollout_len 100` for those 12 games. Critical for the "STU gets better with horizon" story. ~24 runs.

4. **Investigate the Freeway 65.6x advantage** — biggest number in the project. Look at the per-epoch curves and the rollout trajectory to make sure it's not an artifact (e.g., baseline diverged late in training, or freeway's near-zero-MSE absolute scale is misleading).

### Medium-priority

5. **Full sample-efficiency suites for the 12 priority-only games** — they only have 5ep multistep so far. To complete the cross-game story, each needs the full 30-run suite (4 splits × 3 models × 2 train modes + multi-seed + rl100). ~25 runs per game × 12 games = ~300 runs. Use `run_remaining_games.sh` which is fully resumable.

6. **Demonattack remainder** — has 10/30 done, needs 20 more.

### Lower-priority / open questions

7. **STU vs attention is the new headline tension.** Decide how to frame it in the paper. Current evidence: attention wins more games (13/24 vs STU's 11/24) but has known instability issues (single-step training diverges). Need a clean-up of attention single-step before drawing conclusions.

8. **Episode-length non-comparability.** "5 episodes" means 1.7K transitions for pong, 135K for breakout. Cross-game comparisons should be plotted in transitions, not episodes. Not yet done.

9. **No causal story.** Hankel ablation (Direction B in TODO.md) — random orthogonal basis vs DCT vs learned basis vs Hankel — to argue it's the spectral structure specifically that helps.

10. **No downstream RL link.** Sample-limited online training (Direction A in TODO.md) — does the better dynamics MSE translate to better RL agent? Untouched on Lambda.

---

## Files of interest for resuming

### Pipeline scripts (all in `offline_training/`)
- `run_remaining_games.sh` — full pipeline (collect → upload → sample-eff suite per game). Resumable. Auto-picks PARALLEL_JOBS from dataset size.
- `run_5ep_priority.sh` — priority pass: 5ep multi-step STU/baseline/attention only. Global job pool with `wait -n`. Pool cap = 8 (env var `POOL_N`).
- `run_sample_efficiency_parallel.sh` — old per-game parallel runner used by `run_remaining_games.sh`. Now correctly propagates failures.
- `collect_remaining_only.sh` — collect+upload only, no training. Used during the 2026-04-08 collection-only phase.
- `train_dynamics_offline.py` — main training script. The actual training; unchanged.

### Sentinel directories
- `offline_training/results/pipeline_logs/<game>/{collect,upload,sample_eff}.done`
- `offline_training/results/sample_efficiency/<game>_<model>_<split>_<train_mode>/training_complete.flag`

### Logs (most recent on top)
- `/tmp/stuzero_5ep_priority.out` — last priority-pass log
- `/tmp/stuzero_collect.out` — collection-only phase log (2026-04-08)
- `/tmp/stuzero_pipeline.out` — main pipeline log (2026-04-07)
- `offline_training/results/pipeline_logs/run_5ep_priority_*.log`
- `offline_training/results/pipeline_logs/collect_only_*.log`
- `offline_training/results/pipeline_logs/run_remaining_games_*.log`

### How to resume

Killed everything cleanly on 2026-04-09. To resume:

```bash
cd /lambda/nfs/hazan/STUZero
source /home/ubuntu/miniconda3/etc/profile.d/conda.sh
conda activate ezv2

# To finish the 5ep priority pass (breakout baseline+attention, gopher all 3):
setsid nohup bash offline_training/run_5ep_priority.sh > /tmp/stuzero_5ep_priority.out 2>&1 < /dev/null & disown

# To run the full sample-efficiency suite for everything (resumes all sentinels):
setsid nohup bash offline_training/run_remaining_games.sh > /tmp/stuzero_pipeline.out 2>&1 < /dev/null & disown
```

Both scripts are fully detached via `setsid` so an OOM or terminal disconnect cannot drag down the calling shell (lesson learned the hard way on 2026-04-07).

---

## Cohort summary at pause

- **Datasets**: 26/26 on HF, fully documented
- **Sample-eff training**: 7 fully complete games (amidar, assault, bankheist, battlezone, boxing, choppercommand, crazyclimber) + 12 partial (mostly 5ep multistep only)
- **Findings table**: 24/26 games have 5ep multi-step results in some form (17 fresh from Lambda + 7 from master_summary)
- **Missing for the headline 5ep multistep table**: breakout BL+Attn, gopher all 3
