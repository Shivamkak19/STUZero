# STUZero Offline Experiment Logs

This directory contains the archived logs (stdout/stderr from training + pipeline scripts) from the STUZero offline dynamics experiments. Checkpoints (`.pt` files, ~340 MB total) are NOT included — only logs and experimental output.

## Archives

| File | Size | Contents |
|------|------|----------|
| `sample_efficiency_logs.tar.gz` | 1.3 MB | 286 training-stdout `.log` files from `offline_training/results/sample_efficiency/`, plus `completed_runs.txt` listing all runs that produced `training_complete.flag` |
| `filter_ablation_logs.tar.gz` | 86 KB | 33 filter-ablation training logs (different filter bases: hankel, hankel_scaled, dct, dft, random, random_normalized on Pong/Breakout 5-episode) |
| `pipeline_logs.tar.gz` | 5.8 MB | 10 pipeline-runner logs (orchestration scripts — HuggingFace dataset collection, run dispatching, etc.) |

**Total: ~7.2 MB (compressed from ~154 MB raw)**

## Game coverage for 5ep multistep (the key experiment)

248 total runs completed across all (game, method, ep, step) combinations.

For **5ep_multistep** specifically (the "low-data multi-step rollout" flagship experiment):

### Fully complete (STU + baseline + attention, with seed variants)
8 games — amidar, assault, bankheist, battlezone, boxing, choppercommand, crazyclimber, and the single-seed versions listed below:

### Single-run only (STU + baseline + attention at seed 0)
amidar, assault, bankheist, battlezone, boxing, choppercommand, crazyclimber, demonattack, freeway, frostbite, hero, jamesbond, kangaroo, krull, privateeye, qbert, upndown

### Partial / STU-only
- breakout (STU only, no baseline/attention)

### MISSING from disk entirely (7 games)
**pong, mspacman, seaquest, roadrunner, kungfumaster, gopher, alien**

Logs 4 and 5 in the research_logs describe these games being run (Alien+Seaquest batch, then KungFu+MsPacman+RoadRunner batch) and reference an `offline_training/results/batch_04052026/` directory. That directory does not exist on this machine. The data may be on Princeton della's `/scratch/gpfs/EHAZAN/stuzero_data/` but is not locally available.

- `alien` has log files in `sample_efficiency_logs/` but they're all from failed runs (HuggingFace dataset download 503 error) — no training actually happened.

### Summary
- **26 games** claimed (by logs 4, 5, 7)
- **19 games** with actual 5ep-multistep logs that trained successfully
- **7 games** missing: alien, pong, mspacman, seaquest, roadrunner, kungfumaster, gopher

## How to extract metrics from the logs

Each successful run's `.log` file contains the training stdout. The numeric metrics (dynamics MSE, rollout MSE) are printed as progress lines during training. The script to parse these is in `offline_training/` — example grep patterns:

```bash
# Final rollout MSE:
tar -xzf sample_efficiency_logs.tar.gz -O sample_efficiency_logs/amidar_stu_5ep_multistep.log \
  | grep -E "step 50 rollout|test rollout|dynamics_mse"
```

## Related logs

- `../1_pong_spatial_stu_research_log.md` through `../7_lambda_19game_expansion_research_log.md` — narrative research logs that reference this data
- `../0409026.md` — filter ablation (uses `filter_ablation_logs.tar.gz`)
- `../6_master_summary.md` — chronological index

## Original canonical location

`/lambda/nfs/hazan/STUZero/offline_training/results/` — the repo location these logs were copied from. That directory also has the `.pt` checkpoints (not included here).
