# Experiment Descriptions and Results

## For Thesis: "Spectral Filtering as an Inductive Bias for Sample-Efficient World Model Dynamics"

---

## 1. Experimental Setup

### 1.1 Offline Dynamics Training Framework

All experiments use an offline dynamics training pipeline. A pre-trained EfficientZero V2 (EZv2) benchmark agent is used to collect a fixed dataset of latent state transitions via MCTS self-play. The dynamics network is then trained offline on this dataset, with the representation and projection networks frozen from the benchmark checkpoint.

**Training objective:** MSE between predicted and ground-truth next latent states, plus a consistency loss in projection space (weight 5.0).

**Evaluation metric:** Multi-step rollout MSE at step 50 (or step 100 for extended horizon experiments). The trained dynamics model is unrolled autoregressively for 50 steps using ground-truth actions, and the MSE between predicted and actual latent states is measured at each step. The step-50 MSE captures long-horizon composability — how well the model's predictions compound over many steps.

### 1.2 Model Architectures

| Model | Description | Params (Atari) | Params (DMC) |
|-------|-------------|----------------|--------------|
| **Baseline** | Standard EZv2 DynamicsNetwork with GroupNorm output. Conv3x3 + BN + residual + ResBlock + GroupNorm. | 121K | 216K |
| **Spatial STU** | Baseline architecture with Hankel spectral filter inserted as a residual block operating on the [B, 64, 6, 6] spatial grid (seq_len=36, K=2 filters). GroupNorm output. | 138K | N/A |
| **Self-Attention** | Baseline architecture with multi-head self-attention (4 heads) replacing the spectral filter. Same parameter count as STU. | 138K | N/A |
| **Feature STU** | For DMC flat states: reshapes [B, 128] to [B, 16, 8] and applies spectral filtering over 16 pseudo-positions. | N/A | 150K |
| **STU Sequential** | Temporal model: maintains a buffer of 10 previous states, applies spectral filtering over the time dimension. | N/A | 693K |
| **Baseline+Norm** | DMC baseline with output LayerNorm (no spectral filtering). | N/A | 216K |

### 1.3 Training Configurations

**Single-step training:** The model predicts one step ahead. Loss = MSE(G(s_t, a_t), s_{t+1}) + 5.0 * consistency_loss. This tests whether the architectural prior alone provides stability.

**Multi-step training:** The model is unrolled autoregressively for 20 steps (stride 5 for overlapping windows). The loss is summed over all 20 steps, providing gradient signal for composable predictions. This explicitly teaches the model to make predictions that compound well.

**Common hyperparameters:** Adam optimizer, lr=1e-3, cosine schedule, 80 epochs, batch size 256, gradient clipping at 5.0.

### 1.4 Sample Efficiency Protocol

Each game's dataset contains 50 episodes. The `train_split` parameter controls how many episodes are used for training:
- 5 episodes (train_split=0.1): ~low data regime
- 10 episodes (train_split=0.2)
- 20 episodes (train_split=0.4)
- 40 episodes (train_split=0.8): ~full data regime

For key comparisons (5 episodes, multi-step training), results are validated across 3 random seeds (42, 123, 456).

---

## 2. Experiment 1: Single-Step vs Multi-Step Training (Pong)

### Description
The foundational experiment comparing single-step and multi-step training on Pong, establishing that (a) multi-step training is essential for long-horizon prediction and (b) the spatial STU provides additional benefit on top of multi-step training.

### Key Finding
Multi-step training improves all models by 100-4000x on step-50 rollout prediction. Without multi-step training, the baseline diverges catastrophically (step-50 MSE = 2.96e15), while the spatial STU stays bounded (step-50 MSE = 0.82). With multi-step training, the spatial STU achieves the best result (step-50 MSE = 0.000196), beating the baseline (0.000246) by 1.25x.

### Data
- CSV: `csv_data/pong_multistep_improvement.csv`
- CSV: `csv_data/pong_rollout_error_by_step_singlestep.csv`

### Suggested Figures
- **Figure: Rollout error by step (single-step training, Pong).** Log-scale y-axis. Shows baseline diverging exponentially while STU stays bounded. This is the most visually dramatic result.
- **Table: Single-step vs multi-step MSE for all model variants.** Shows the improvement factor from multi-step training.

---

## 3. Experiment 2: Atari Sample Efficiency (7 Games)

### Description
The core experiment: how does STU compare to the baseline and self-attention across varying amounts of training data (5, 10, 20, 40 episodes) on 7 Atari games with multi-step training?

### Key Findings
1. **STU wins at 5 episodes in all 7 games** with advantages ranging from 1.1x (MsPacman) to 19.6x (Seaquest). Geometric mean advantage: 2.9x.
2. **Two distinct patterns emerge:**
   - *Immediate dominance* (Pong, Seaquest, KungFuMaster): Large advantage at 5 episodes that shrinks with more data. Baseline catches up at 20-40 episodes.
   - *Delayed advantage* (Alien, MsPacman, RoadRunner): Small advantage at 5 episodes, growing to peak at 10-20 episodes. STU maintains advantage even at 40 episodes.
3. **Self-attention plateaus** at a fixed MSE (~0.002-0.005) regardless of data size across all games, confirming it lacks the right inductive bias for composable dynamics.

### Data
- CSV: `csv_data/atari_multistep_step50_mse.csv`
- CSV: `csv_data/atari_multiseed_5ep_multistep.csv`
- CSV: `csv_data/attention_comparison_multistep.csv`

### Suggested Figures
- **Figure: Sample efficiency curves (multi-step, step-50 MSE vs episodes) for all 7 games.** 2x4 subplot grid, log-scale y-axis. Each subplot shows baseline, STU, and attention lines.
- **Figure: Cross-game bar chart at 5 episodes.** Bar chart comparing baseline vs STU MSE across all 7 games with error bars from multi-seed validation. Sorted by STU advantage.
- **Figure: STU advantage factor vs episodes.** Line plot for each game showing how the advantage changes with data quantity.
- **Table: Complete results at all episode counts for all 7 games.** The master table.

---

## 4. Experiment 3: Atari Single-Step Training Stability (7 Games)

### Description
Testing whether the spatial STU provides rollout stability without multi-step training on Atari games. In single-step training, the model only receives one-step prediction loss — any long-horizon stability must come from the architectural inductive bias.

### Key Findings
1. Single-step results are mixed across games. STU wins at 5 episodes in 6/7 games (all except none — STU wins everywhere at 5ep single-step).
2. At higher episode counts, baseline often overtakes STU in single-step mode — the baseline learns effective single-step dynamics that happen to compose reasonably with enough data.
3. RoadRunner is the standout: STU wins at ALL data sizes (1.43-2.11x) with single-step training.
4. Self-attention diverges catastrophically (10^24 to 10^32) with single-step training on all games.

### Data
- CSV: `csv_data/atari_singlestep_step50_mse.csv`

### Suggested Figures
- **Table: Single-step vs multi-step comparison across all games at 5 episodes.** Shows that multi-step training is the primary driver of composability, but STU provides additional architectural stability.

---

## 5. Experiment 4: Extended Horizon Rollouts (100 Steps)

### Description
Testing whether the STU advantage persists or grows at longer prediction horizons (100 steps instead of 50) using multi-step training with 5 episodes.

### Key Findings
1. **STU advantage grows with horizon on RoadRunner**: 2.7x at step-50 to **9.0x at step-100**. The baseline degrades 43% (0.165 → 0.235) while STU stays flat (0.026 → 0.026).
2. **STU advantage grows on Alien**: 1.46x at step-50 to 1.70x at step-100. Baseline degrades 15%.
3. **STU error is stable or decreasing** from step-50 to step-100 on all games. The spectral basis acts as a stabilizer preventing error compounding.
4. Seaquest maintains its massive advantage (17.2x at step-100).

### Data
- CSV: `csv_data/atari_100step_rollout_5ep_multistep.csv`

### Suggested Figures
- **Figure: Rollout error curves from step 1 to step 100 for RoadRunner.** Shows baseline climbing while STU stays flat — most visually compelling long-horizon result.
- **Table: Step-50 vs step-100 comparison across all tested games.**

---

## 6. Experiment 5: DMC State-Based Environments (3 Tasks)

### Description
Extending the STU to continuous control tasks in DeepMind Control Suite. The key architectural difference: DMC latent states are flat vectors [B, 128] rather than spatial grids [B, 64, 6, 6]. Two STU variants are tested:
- **Feature STU**: Reshapes [128] → [16, 8] and applies spectral filtering over 16 pseudo-positions.
- **STU Sequential**: Maintains a temporal buffer of 10 states and applies spectral filtering over time.

### Key Findings
1. **Single-step training**: STU Sequential dominates across all 3 DMC tasks (Hopper, Walker, Cheetah), achieving 1.2-2.3x better step-50 MSE than baseline+LayerNorm at 5 episodes. The baseline without normalization diverges catastrophically (MSE 22-188).
2. **Multi-step training**: Feature STU wins at 5 episodes on all 3 tasks (1.1-1.4x), but margins are smaller than Atari. Multi-step training provides the composability signal that makes the spectral prior less necessary.
3. **Generalization**: The stability effect generalizes from spatial grids (Atari) to flat vectors (DMC), confirming it's an architecture-general property of spectral filtering.

### Data
- CSV: `csv_data/dmc_singlestep_step50_mse.csv`
- CSV: `csv_data/dmc_multistep_step50_mse.csv`

### Suggested Figures
- **Figure: DMC single-step training comparison.** Bar chart or table showing bare baseline diverging, baseline+norm being stable, and STU Sequential being best across all 3 tasks.
- **Figure: DMC multi-step training sample efficiency.** Similar to Atari but showing the smaller margins.
- **Table: Comparison of STU variants on DMC.** Feature STU vs STU Sequential in both training modes.

---

## 7. Experiment 6: DMC Image-Based Hopper (Bridging Domain)

### Description
Testing on DMC Hopper Hop with image observations, producing [64, 6, 6] spatial latent states — the same representation as Atari. This bridges the gap between Atari (spatial, discrete actions) and DMC state (flat, continuous actions).

### Key Findings
1. **STU advantage grows with data** on DMC image (1.1x at 5ep → 6.4x at 40ep in multi-step). This is the opposite of Atari, suggesting image-based continuous control dynamics benefit more from the spectral prior as the model capacity is utilized.
2. **Spatial structure matters**: Same Hopper Hop task shows 1.35-6.4x advantage with spatial states (image) vs 1.1-1.2x with flat states (state). The Hankel eigenvector basis is most effective for spatial feature mixing.
3. **Attention anomaly**: Self-attention achieves 0.006 MSE at 5ep multi-step (82x better than baseline), but only on this task. Needs investigation.

### Data
- CSV: `csv_data/dmc_image_hopper_step50_mse.csv`

### Suggested Figures
- **Figure: DMC image vs state comparison on Hopper Hop.** Shows the spatial structure amplifies the STU advantage.

---

## 8. Experiment 7: Self-Attention Comparison

### Description
Comparing STU against multi-head self-attention (4 heads) as an alternative spatial mixing mechanism. Both have the same parameter count (138K on Atari). This tests whether the spectral filtering advantage is specific to the Hankel basis or whether any spatial mixing improves dynamics.

### Key Findings
1. **Attention plateaus regardless of data**: On all 7 Atari games, attention achieves a fixed MSE (~0.002-0.005) that doesn't improve with more training episodes. It overfits to the training horizon (20-step unrolls) and doesn't generalize to 50-step rollouts.
2. **Attention diverges with single-step training**: On all games, attention produces NaN or values in the range 10^24-10^32 with single-step training. It is fundamentally unstable without multi-step gradient signal.
3. **STU beats attention at 20+ episodes**: While attention sometimes wins at 5 episodes on complex games (Alien, MsPacman, RoadRunner), STU overtakes it with more data and ultimately achieves better performance.

### Data
- CSV: `csv_data/attention_comparison_multistep.csv`

### Suggested Figures
- **Figure: Attention plateau visualization.** For 2-3 games, show attention MSE as a flat line while STU and baseline curves change with data.

---

## 9. Experiment 8: TDMPC2 Dynamics (3 DMC Tasks)

### Description
Testing spectral filtering on a different world model architecture: TD-MPC2. Unlike EZv2 which uses a CNN-based dynamics network, TD-MPC2 uses an MLP-based latent dynamics model with a different training objective (joint reward + value + dynamics). The spectral STU filter is inserted as a residual block in the TDMPC2 dynamics network. Experiments run on 3 DMC tasks (Hopper Hop, Walker Walk, Cheetah Run) with full data (40 episodes) and a Walker Walk sample efficiency sweep (5/10/20/40 episodes).

**Note:** Temporal STU crashed on all 3 tasks due to a dimension mismatch in the FFT convolution when seq_len=10 and d_in=512. Only spectral (spatial-style) STU results are available.

### Key Findings
1. **Multi-step training**: Spectral STU improves over baseline on Cheetah (1.56x) and Hopper (1.24x), but slightly worse on Walker (-1.06x). Smaller margins than EZv2 experiments.
2. **Single-step training**: Mixed results. STU helps on Hopper (1.27x) and Walker (1.12x) but hurts on Cheetah (-1.42x).
3. **Walker sample efficiency**: Very small margins (1.01-1.10x). The TDMPC2 architecture appears to be more robust to low data than EZv2's CNN dynamics, leaving less room for the spectral prior to help.
4. **Overall**: The STU advantage is smaller on TDMPC2 than on EZv2. This may be because TDMPC2's MLP dynamics already have implicit regularization from the joint training objective, or because the flat MLP representation doesn't benefit from spatial spectral filtering as much as CNN feature grids.

### Results Table

**Full Data (40 episodes), Multi-step Training**

| Environment | Baseline (step-50) | STU Spectral (step-50) | STU Advantage |
|-------------|-------------------|----------------------|---------------|
| Cheetah Run | 0.004958 | **0.003177** | **1.56x** |
| Hopper Hop | 0.004673 | **0.003757** | **1.24x** |
| Walker Walk | **0.002617** | 0.002772 | Baseline 1.06x |

**Full Data (40 episodes), Single-step Training**

| Environment | Baseline (step-50) | STU Spectral (step-50) | STU Advantage |
|-------------|-------------------|----------------------|---------------|
| Hopper Hop | 0.012017 | **0.009467** | **1.27x** |
| Walker Walk | 0.005101 | **0.004546** | **1.12x** |
| Cheetah Run | **0.004384** | 0.006237 | Baseline 1.42x |

**Walker Walk Sample Efficiency (Multi-step Training)**

| Episodes | Baseline (step-50) | STU Spectral (step-50) | STU Advantage |
|----------|-------------------|----------------------|---------------|
| 5 | 0.002128 | **0.002115** | **1.01x** |
| 10 | 0.002660 | **0.002424** | **1.10x** |
| 20 | 0.002816 | **0.002665** | **1.06x** |
| 40 | **0.002617** | 0.002772 | Baseline 1.06x |

### Data
- CSV: `csv_data/tdmpc2_results.csv`

### Suggested Figures
- **Table: TDMPC2 vs EZv2 comparison.** Shows that STU's advantage is architecture-dependent — larger on CNN-based EZv2 dynamics than MLP-based TDMPC2.

---

## 10. Summary Statistics

### Overall Results Across All Environments

| Metric | Value |
|--------|-------|
| Total experiments run | ~300 |
| Atari games tested | 7 (Pong, Asterix, Alien, Seaquest, KungFuMaster, MsPacman, RoadRunner) |
| DMC tasks tested | 4 (Hopper state, Walker state, Cheetah state, Hopper image) |
| Games where STU wins at 5ep multi-step | 7/7 Atari, 3/3 DMC state, 1/1 DMC image |
| Geometric mean STU advantage (Atari, 5ep) | 2.9x |
| Maximum STU advantage | 19.6x (Seaquest, 5ep multi-step) |
| Maximum long-horizon advantage | 9.0x (RoadRunner, step-100) |
| Multi-seed validated | 6 Atari games (3 seeds each) |

### Model Parameter Counts

| Architecture | Atari Params | DMC State Params |
|-------------|-------------|-----------------|
| Baseline (GroupNorm) | 121,000 | 216,000 |
| Spatial STU (K=2) | 138,000 | N/A |
| Self-Attention (4 heads) | 138,000 | N/A |
| Feature STU | N/A | 150,336 |
| STU Sequential (buffer=10) | N/A | 692,864 |
| Baseline+LayerNorm | N/A | 216,256 |
