# STUZero Offline Dynamics Research — Complete Summary
## Last updated: 2026-04-06

---

## Project Overview

Investigating whether spectral filtering (STU — Spectral Transform Unit) improves world model dynamics prediction in RL, with focus on sample efficiency. The STU uses Hankel eigenvectors as a spatial filter over latent state representations.

**Core finding**: STU improves dynamics prediction in all 7 tested Atari games at low data (5 episodes), with advantages ranging from 1.1x to 19.6x.

**Open question**: Does better dynamics MSE translate to better RL agent performance?

---

## Experiments Completed

### Total: ~300 experiments across 7 Atari + 4 DMC environments

### Chronological Log

| Date | Log File | What was run |
|------|----------|-------------|
| 2026-03-18 | `research_log.md` | Pong: multi-step training, spatial STU, STU sequential, sample efficiency (5/10/20/40 ep) |
| 2026-04-02 | `research_log_04022026.md` | DMC Hopper Hop: Feature STU, STU Sequential, baseline+norm, single-step + multi-step |
| 2026-04-04 | `research_log_04042026.md` | Walker Walk, Cheetah Run, Pong extended (100-step, very low data), multi-seed validation, self-attention comparison, DMC Image Hopper |
| 2026-04-05 | `research_log_04052026.md` | Alien + Seaquest: full sample efficiency (60 experiments) |
| 2026-04-05 | `research_log_04052026_part2.md` | KungFuMaster + MsPacman + RoadRunner: full sample efficiency (90 experiments) |
| 2026-04-11 to 04-14 | `8_stu_dreamer_research_log.md` | STU integration into DreamerV3 (JAX): walker_walk (baseline / stuB / stuRand / stuPlus), hopper_stand (size1m + size12m), cheetah_run. Finding: STU-inside-`_core` with Hankel eigenvectors provides 20-36% faster convergence to baseline-level performance; random filters match asymptote but take 55% more steps; stu_plus scales poorly to 12m. Data in `8_stu_dreamer_data/`. |
| 2026-04-09 to 04-11 | `9_rezero_stu_mixer_research_log.md` | STU as pre-RSSM encoder mixer in R2-Dreamer (PyTorch). Pong: baseline reaches eval +4.3 at step 400K; STU-hankelscaled stalls at -12.8. **Finding: STU-mixer placed before RSSM does NOT help on Pong** — contrasts with stu-dreamer-jax result where STU inside the recurrence does help. Suggests placement within world-model architectures matters. Data in `9_rezero_stu_mixer_data/`. |
| 2026-04-11 (approx) | `10_spectral_transformer_jax_research_log.md` | Broad JAX transformer stack with Attention/STU/Mamba mixers. Random-filter ablation on state tracking: Hankel eval_loss 0.047 vs random-normalized 0.556 (**12× gap** — much larger than Dreamer ablation). Data in `10_spectral_transformer_jax_data/`. |
| consolidation | `11_stuzero_offline_data/` | Archived training + pipeline logs (no checkpoints) from the STUZero offline experiments referenced by logs #0-#7. 248 completed runs across 19 games (5ep/10ep/20ep/40ep × single/multistep × stu/baseline/attention). **7 games missing from disk** despite logs 4-5 claiming them run: alien, pong, mspacman, seaquest, roadrunner, kungfumaster, gopher. 7.2 MB compressed. See README. |
| final sweep | `12_final_sweep_data/` | Catch-all for experimental data discovered during final pre-offload sweep: **42 EZ-V2 online Atari runs** (Pong×36, Asterix×2, KungFuMaster/Assault/Boxing/Gopher×1 — logs + eval JSONs), 55 wandb run metadata bundles, the original failing stufix run that motivated log #8 (pre-existing, not produced by this work), Run A (post-hoc STU) data referenced in log #8, ReZero logdir debug runs referenced in log #9. **12 video reels** (STUZero EZ-V2 Atari + stu-dreamer-jax baseline/stuB/stuRand/stuPlus/stufix + ReZero Pong baseline-vs-STU). 12 MB total. See README. |

## Sibling directories (peer to `logs/`)

`research_logs/logs/` contains the chronological research notes. Two sibling directories hold processed results and writeup materials:

| Dir | Size | Contents |
|-----|------|----------|
| `research_logs/experiments/` | 2.7 MB | **Processed experimental data for thesis plots.** 11 CSV files with step-50 MSE, sample efficiency, multiseed validation, attention comparison, DMC/Atari, pong rollout divergence. 13 plots (PNG + PDF) generated from those CSVs: pong rollout divergence, cross-game 5ep bar chart, sample efficiency curves, STU advantage vs episodes, multiseed validation. `experiment_descriptions.md` explains methodology. `generate_plots.py` regenerates plots. |
| `research_logs/draft_paper/` | 9.7 MB | **Draft thesis writeup.** `draft_report.tex` + `references.bib` + `graphs/` + compiled `draft_report.pdf` + `Thesis Draft Report - Shivam Kak.pdf`. |
| `research_logs/notes_della/` | 642 KB | Informal working notes from Princeton della runs: `offline_runs/` (11 txt files with baseline/attention/mamba/STU run notes), `plans/` (midterm progress, spectral explanation, temporal setup). |

**Note:** `research_logs/experiments/` files were in git but deleted from filesystem as of this sweep. Restored via `git checkout HEAD -- research_logs/experiments/`.

All logs are in: `offline_training/results/` (except `8_stu_dreamer_data/` which is co-located with its log file).

### Results Directories

| Directory | Contents |
|-----------|----------|
| `results/sample_efficiency/` | Early Pong single-step experiments |
| `results/multistep/` | Early Pong multi-step experiments |
| `results/ablations/` | Early ablation experiments |
| `results/batch_04042026/` | Walker/Cheetah/Pong extended + multi-seed |
| `results/batch_04052026/` | Alien, Seaquest, KungFuMaster, MsPacman, RoadRunner (~150 experiments) |
| `results/dmc_image_hopper/` | DMC Image Hopper experiments |

### Datasets

All datasets stored on scratch: `/scratch/gpfs/EHAZAN/stuzero_data/`
Symlinked from `offline_training/{game}_*/`

| Game | Location | Episodes | Transitions | Expert Steps |
|------|----------|----------|-------------|-------------|
| Pong | `pong_100K/` | 50 | ~17.5K | 100K |
| Asterix | `asterix_110K/` | 50 | ~35K | 110K |
| Alien | `alien_110K/` | 50 | 29,687 | 110K |
| Seaquest | `seaquest_110K/` | 50 | 113,381 | 110K |
| KungFuMaster | `kungfumaster_110K/` | 50 | ~170K | 110K |
| MsPacman | `mspacman_90K/` | 50 | ~44K | 90K |
| RoadRunner | `roadrunner_110K/` | 50 | ~53K | 110K |
| DMC Hopper (state) | `dmc/dmc_hopper_hop_110K/` | 50 | 25,000 | 110K |
| DMC Walker (state) | `dmc/dmc_walker_walk_110K/` | 50 | 25,000 | 110K |
| DMC Cheetah (state) | `dmc/dmc_cheetah_run_110K/` | 50 | 25,000 | 110K |
| DMC Hopper (image) | `dmc/dmc_image_hopper_hop_210K/` | 37 | 18,500 | 210K |

---

## Master Results Table: Atari Multi-step Training (step-50 MSE)

### 5 episodes

| Game | Actions | Baseline | STU | Attention | STU Advantage |
|------|---------|----------|-----|-----------|---------------|
| Seaquest | 18 | 0.01365 | **0.00070** | 0.00228 | **19.6x** |
| Pong | 6 | 0.01970 | **0.00170** | 0.00290 | **11.7x** |
| KungFuMaster | 14 | 0.00537 | **0.00146** | 0.00183 | **3.7x** |
| RoadRunner | 18 | 0.07117 | **0.02599** | 0.00404 | **2.7x** |
| Asterix | 9 | 0.00297 | **0.00157** | 0.00293 | **1.9x** |
| Alien | 18 | 0.14769 | **0.12732** | 0.00649 | **1.2x** |
| MsPacman | 9 | 0.05331 | **0.04796** | 0.00487 | **1.1x** |

### 10 episodes

| Game | Baseline | STU | STU Advantage |
|------|----------|-----|---------------|
| MsPacman | 0.01753 | **0.00166** | **10.6x** |
| RoadRunner | 0.01885 | **0.00276** | **6.8x** |
| Seaquest | 0.00278 | **0.00052** | **5.4x** |
| Alien | 0.02862 | **0.00786** | **3.6x** |
| Pong | 0.00520 | **0.00170** | **3.1x** |
| Asterix | 0.00184 | **0.00158** | **1.2x** |
| KungFuMaster | **0.00127** | 0.00135 | Baseline 1.06x |

### 20 episodes

| Game | Baseline | STU | STU Advantage |
|------|----------|-----|---------------|
| MsPacman | 0.00731 | **0.00095** | **7.7x** |
| Alien | 0.01050 | **0.00241** | **4.4x** |
| Seaquest | 0.00116 | **0.00065** | **1.8x** |
| Pong | 0.00123 | **0.00072** | **1.7x** |
| RoadRunner | 0.00344 | **0.00231** | **1.5x** |
| KungFuMaster | **0.00080** | 0.00084 | Baseline 1.05x |

### 40 episodes

| Game | Baseline | STU | Winner |
|------|----------|-----|--------|
| MsPacman | 0.00248 | **0.00057** | **STU 4.3x** |
| Alien | 0.00425 | **0.00151** | **STU 2.8x** |
| Seaquest | **0.00020** | 0.00038 | Baseline 1.9x |
| Pong | **0.00020** | 0.00030 | Baseline 1.4x |
| RoadRunner | 0.00185 | **0.00160** | **STU 1.2x** |
| KungFuMaster | 0.00053 | **0.00051** | ~Even |

### 100-step Rollout (5ep, multi-step)

| Game | STU step-50 | STU step-100 | BL step-50 | BL step-100 | STU Adv@100 |
|------|-------------|--------------|------------|-------------|-------------|
| RoadRunner | 0.026 | 0.026 | 0.165 | 0.235 | **9.0x** |
| Seaquest | 0.0007 | 0.0008 | 0.014 | 0.014 | **17.2x** |
| Pong | 0.002 | 0.002 | 0.022 | 0.022 | **9.1x** |
| KungFuMaster | 0.002 | 0.002 | 0.005 | 0.005 | **3.2x** |
| Alien | 0.129 | 0.127 | 0.188 | 0.216 | **1.7x** |
| MsPacman | 0.048 | 0.048 | 0.052 | 0.053 | **1.1x** |

### Multi-Seed Validation (5ep, multi-step, seeds 42/123/456)

| Game | Baseline Mean +/- Std | STU Mean +/- Std | Advantage |
|------|----------------------|------------------|-----------|
| Seaquest | 0.01211 +/- 0.0014 | 0.00069 +/- 0.00002 | 17.5x |
| Pong | 0.0197 +/- 0.0013 | 0.0017 +/- 0.0005 | 11.7x |
| KungFuMaster | 0.00474 +/- 0.0009 | 0.00143 +/- 0.0003 | 3.3x |
| RoadRunner | 0.05491 +/- 0.0143 | 0.02342 +/- 0.0073 | 2.3x |
| MsPacman | 0.05779 +/- 0.0085 | 0.03930 +/- 0.0076 | 1.5x |
| Alien | 0.1484 +/- 0.0013 | 0.1225 +/- 0.0043 | 1.2x |

---

## DMC Results Summary

### Single-step Training: STU Sequential dominates (all 3 tasks)

| Env | 5ep BL+Norm | 5ep STU Seq | Advantage |
|-----|-------------|-------------|-----------|
| Cheetah | 0.490 | **0.209** | **2.3x** |
| Hopper | 0.271 | **0.153** | **1.8x** |
| Walker | 0.680 | **0.567** | **1.2x** |

### Multi-step Training: Feature STU wins at 5ep

| Env | 5ep Baseline | 5ep Feature STU | Advantage |
|-----|-------------|-----------------|-----------|
| Hopper | 0.026 | **0.019** | **1.4x** |
| Walker | 0.013 | **0.011** | **1.2x** |
| Cheetah | 0.022 | **0.020** | **1.1x** |

---

## Key Technical Decisions & Notes

### Architecture
- **Spatial STU (Atari)**: Operates on [B, 64, 6, 6] latent grid, seq_len=36, K=2 filters, 138K params
- **Baseline (Atari)**: DynamicsNetwork with GroupNorm output, 121K params
- **Self-Attention**: Multi-head attention (4 heads) over spatial positions, 138K params
- **Feature STU (DMC)**: Reshapes [B, 128] → [B, 16, 8], applies STU, 150K params
- **STU Sequential**: Temporal STU over history buffer (buffer_size=10), 693K params

### Training Setup
- Multi-step: 20-step autoregressive unroll, stride 5
- Single-step: MSE + consistency loss (weight 5.0)
- Optimizer: Adam, lr=1e-3, cosine schedule, 80 epochs
- Batch size: 256

### Known Issues
- **Mamba is broken**: NaN from epoch 1. Root cause: missing GroupNorm + conflicting pre-LayerNorm. Fix attempted (added GroupNorm, removed pre-norm) but not validated — Mamba process hung consuming no GPU.
- **Disk quota**: Recurring issue on Della home (50GB limit). Solved by moving datasets to scratch. Periodic checkpoints disabled in training script.
- **MsPacman/Battlezone checkpoints**: Have `_orig_mod.` prefix from torch.compile. Auto-stripped in collect_data.py and train_dynamics_offline.py.

---

## TODO: Next Experiments

### Immediate (before GPU transfer)
- [ ] Collect data for Assault, BankHeist, Battlezone (commands below)
- [ ] Run sample efficiency experiments for these 3 games (`run_assault_bankheist_battlezone.sh`)

### After GPU Transfer — High Priority
- [ ] **Sample-limited online training**: Run full EfficientZero training loop but cap at 10K/20K/50K/100K steps. Compare STU vs baseline eval scores. This is the make-or-break experiment for NeurIPS.
- [ ] **Fewer MCTS simulations**: Can STU dynamics enable 4 simulations instead of 16 while maintaining score?

### After GPU Transfer — Medium Priority
- [ ] **Ablation: Hankel vs random orthogonal basis** — prove the spectral part specifically matters
- [ ] **Fix Mamba comparison** — validate the GroupNorm fix, run on all games
- [ ] **Theory sketch** — connect sample efficiency to spectral properties of dynamics operator

### Lower Priority
- [ ] More Atari games beyond 10 (diminishing returns)
- [ ] DMC image experiments on more environments
- [ ] Formal NeurIPS paper writing

---

## Data Collection Commands (Assault, BankHeist, Battlezone)

Run these from the STUZero repo root:

```bash
# Assault (action_space=7, ~90K step model)
python data_collection_utils/collect_data.py \
    exp_config=ez/config/exp/atari_assault.yaml \
    eval.model_path=offline_training/atari_models/atari_assault_model_90000.p \
    +collect.n_episodes=50 \
    +collect.max_steps=27000 \
    +collect.output_dir=offline_training/assault_90K

# BankHeist (action_space=18, ~110K step model)
python data_collection_utils/collect_data.py \
    exp_config=ez/config/exp/atari_bankheist.yaml \
    eval.model_path=offline_training/atari_models/atari_bankheist_model_110000.p \
    +collect.n_episodes=50 \
    +collect.max_steps=27000 \
    +collect.output_dir=offline_training/bankheist_110K

# Battlezone (action_space=18, ~40K step model)
python data_collection_utils/collect_data.py \
    exp_config=ez/config/exp/atari_battlezone.yaml \
    eval.model_path=offline_training/atari_models/atari_battlezone_model_40000.p \
    +collect.n_episodes=50 \
    +collect.max_steps=27000 \
    +collect.output_dir=offline_training/battlezone_40K
```

After data collection, run experiments:
```bash
bash offline_training/run_assault_bankheist_battlezone.sh
```

---

## File Index

### Config files
- `ez/config/exp/atari_{pong,asterix,alien,seaquest,kungfumaster,mspacman,roadrunner,assault,bankheist,battlezone}.yaml`
- `ez/config/exp/dmc_state{,2,_cheetah}.yaml`, `dmc_image.yaml`

### Model files
- `ez/agents/models/base_model.py` — All Atari dynamics model variants (STU, baseline, attention, mamba)
- `ez/agents/models/dmc_dynamics.py` — DMC dynamics model variants
- `ez/agents/models/minimal.py` — MiniSTU implementation

### Training scripts
- `offline_training/train_dynamics_offline.py` — Main Atari offline training
- `offline_training/train_dynamics_offline_dmc.py` — DMC offline training

### Experiment scripts
- `offline_training/run_alien_seaquest_sample_efficiency.sh`
- `offline_training/collect_and_run_new_games.sh` (KungFuMaster, MsPacman, RoadRunner)
- `offline_training/run_assault_bankheist_battlezone.sh` (ready, needs data)

### Data collection
- `data_collection_utils/collect_data.py` — Main collection script
- `offline_training/collect_data_dmc.py` — DMC collection
