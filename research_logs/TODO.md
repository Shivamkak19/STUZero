# STUZero Research TODO
## Last updated: 2026-04-06

---

## Current Status

- **7/26 Atari games** have offline dynamics sample efficiency results (Pong, Asterix, Alien, Seaquest, KungFuMaster, MsPacman, RoadRunner)
- **7/26 games** have generated datasets on HuggingFace (`Shivamkak/STUZero-Atari-Dynamics`)
- **Core finding**: STU wins at 5 episodes in all 7 games (1.1x-19.6x advantage)
- **Core weakness**: No evidence yet that better dynamics MSE leads to better RL performance
- Expert model files available for all 26 Atari games in `offline_training/atari_models/`

---

## Immediate Next Steps (Lambda A100)

### Step 1: Generate datasets for remaining 19 Atari games
- Run `collect_data.py` for each game using expert model checkpoints
- Upload to HuggingFace repo
- Games needed: Assault, BankHeist, Battlezone, Breakout, CrazyClimber, DemonAttack, Freeway, Frostbite, Gopher, Hero, Jamesbond, Kangaroo, Krull, PrivateEye, Qbert, UpNDown, (check atari_models/ for full list)

### Step 2: Run sample efficiency experiments for all remaining games
- Same protocol as prior games: 3 models (STU, baseline, attention) x 4 splits (5/10/20/40 ep) x 2 training modes (single-step, multi-step) + multi-seed validation at 5ep + 100-step rollout
- ~30 experiments per game, ~570 experiments total

### Step 3: Compile results into charts/graphs
- Cross-game bar charts: STU advantage at 5ep, 10ep, 20ep, 40ep
- Sample efficiency curves per game
- 100-step rollout stability comparisons
- Statistical significance tables (multi-seed)

### Step 4: Reflect and decide direction

---

## Research Directions to Consider (Post Step 4)

### Direction A: Downstream RL Performance (highest impact, highest risk)
**Question**: Does STU's better dynamics MSE actually produce a better RL agent?

Options:
1. **Sample-limited online training**: Run full EfficientZero loop capped at 10K/20K/50K/100K steps. Compare STU vs baseline eval game scores at each checkpoint. If STU reaches score X at 20K steps that baseline needs 100K for — that's the paper.
2. **Fewer MCTS simulations**: Can STU dynamics enable 4 MCTS simulations instead of 16 while maintaining game score? This would show the dynamics model quality directly translates to computational savings.
3. **Plug-in dynamics swap**: Take a trained agent, swap its dynamics network with the offline-trained STU dynamics, evaluate. Simpler but less convincing.

**Risk**: Prior online training experiments on Pong showed STU didn't clearly help (consistency loss similar, eval scores slightly worse). But this was NOT tested in the sample-limited regime.

### Direction B: Ablation of Spectral Basis (medium impact, low risk)
**Question**: Is it the Hankel eigenvectors specifically, or would any orthogonal basis regularize similarly?

Experiments:
- Replace Hankel eigenvectors with random orthogonal basis
- Replace with DCT basis
- Replace with learned basis (just an MLP of same parameter count)
- If Hankel specifically outperforms → strong theoretical story
- If any orthogonal basis works → still useful but weaker claim

### Direction C: Proper SSM/Transformer Baselines (medium impact, medium risk)
**Question**: How does STU compare to published sequence models?

- Fix Mamba integration (GroupNorm fix attempted but not validated)
- Add S4/S4D baseline
- These are important for reviewer satisfaction but unlikely to change the core story

### Direction D: Theoretical Contribution (high impact, high risk)
**Question**: Can we formally connect sample efficiency to spectral properties?

- Connect to Hazan et al. 2017 regret bounds for online learning with spectral filtering
- Show that the Hankel eigenvector basis provides condition-number-free representation of dynamics operators
- Prove sample complexity reduction for dynamics learning with spectral prior
- This is hard but would make the paper much stronger

---

## Known Weaknesses to Address

1. **No downstream RL impact** — the "so what?" problem. Better MSE that doesn't change agent behavior is not impactful.

2. **Baseline catches up with data** — at 40 episodes, baseline ties or wins in 4/7 games. Need to motivate the low-data regime (sim-to-real, offline RL, expensive environments, computational cost of multi-step training).

3. **Comparing against vanilla CNN only** — not DreamerV3, IRIS, or published world models. Mitigated by: we're studying an architectural component (spectral filtering), not a full system. The CNN baseline IS the standard EfficientZero dynamics network.

4. **Seaquest 19.6x may be inflated** — episodes are nearly identical. STU might exploit homogeneity rather than learning generalizable dynamics. Test: evaluate on held-out episodes with different initial conditions.

5. **No ablation of WHY it works** — correlational, not causal. Direction B above.

6. **No theory** — without theoretical grounding, this is a purely empirical paper, and the empirics alone may not be sufficient for NeurIPS.

---

## Technical Notes for Lambda Setup

- Datasets download via `--download` flag: `python train_dynamics_offline.py --game pong --download`
- MsPacman and Battlezone models have `_orig_mod.` prefix (torch.compile) — auto-handled
- Ray object store: use 20GB (`ray.init(object_store_memory=20 * 1024 * 1024 * 1024)`) — 150GB default causes OOM
- Periodic checkpoint saving is disabled in training script to save disk
- All game configs exist in `ez/config/exp/atari_{game}.yaml`
