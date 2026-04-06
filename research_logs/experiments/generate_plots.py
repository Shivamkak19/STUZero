"""
Generate all thesis plots from CSV data.
Run: python research_logs/experiments/generate_plots.py
"""
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import numpy as np
from pathlib import Path

CSV_DIR = Path(__file__).parent / "csv_data"
PLOT_DIR = Path(__file__).parent / "plots"
PLOT_DIR.mkdir(exist_ok=True)

# Style
sns.set_theme(style="whitegrid", font_scale=1.2)
COLORS = {
    'baseline': '#2196F3',
    'spatial_stu': '#E91E63',
    'attention': '#9C27B0',
    'feature_stu': '#FF9800',
    'stu_sequential': '#4CAF50',
    'baseline_norm': '#03A9F4',
}

def save(fig, name):
    fig.savefig(PLOT_DIR / f"{name}.png", dpi=200, bbox_inches='tight')
    fig.savefig(PLOT_DIR / f"{name}.pdf", bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {name}")


# =========================================================================
# Plot 1: Pong rollout error by step (single-step training) — the dramatic one
# =========================================================================
print("Plot 1: Pong rollout divergence (single-step)")
df = pd.read_csv(CSV_DIR / "pong_rollout_error_by_step_singlestep.csv")
fig, ax = plt.subplots(figsize=(8, 5))
ax.semilogy(df['rollout_step'], df['baseline_mse'], 'o-', color=COLORS['baseline'],
            label='Baseline+GroupNorm', linewidth=2, markersize=6)
ax.semilogy(df['rollout_step'], df['stu_mse'], 's-', color=COLORS['spatial_stu'],
            label='Spatial STU (K=2)', linewidth=2, markersize=6)
ax.set_xlabel('Rollout Step')
ax.set_ylabel('MSE (log scale)')
ax.set_title('Rollout Error Divergence — Pong, Single-Step Training')
ax.legend()
ax.set_xlim(0, 52)
save(fig, "01_pong_rollout_divergence_singlestep")


# =========================================================================
# Plot 2: Cross-game bar chart at 5 episodes (multi-step)
# =========================================================================
print("Plot 2: Cross-game 5ep comparison")
df = pd.read_csv(CSV_DIR / "atari_multistep_step50_mse.csv")
df5 = df[df['episodes'] == 5].copy()
df5 = df5.sort_values('stu_advantage', ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(df5))
w = 0.3
bars1 = ax.bar(x - w, df5['baseline'], w, label='Baseline+GroupNorm', color=COLORS['baseline'])
bars2 = ax.bar(x, df5['spatial_stu'], w, label='Spatial STU', color=COLORS['spatial_stu'])
bars3 = ax.bar(x + w, df5['attention'], w, label='Self-Attention', color=COLORS['attention'])
ax.set_xticks(x)
ax.set_xticklabels([g.title() for g in df5['game']], rotation=30, ha='right')
ax.set_ylabel('Step-50 MSE')
ax.set_title('Sample Efficiency at 5 Episodes — Multi-Step Training')
ax.legend()
ax.set_yscale('log')
# Add advantage labels
for i, (_, row) in enumerate(df5.iterrows()):
    adv = row['stu_advantage']
    if adv > 0:
        ax.text(i, row['spatial_stu'] * 0.5, f'{adv:.1f}x', ha='center', va='top',
                fontsize=9, fontweight='bold', color=COLORS['spatial_stu'])
save(fig, "02_crossgame_5ep_bar_chart")


# =========================================================================
# Plot 3: Sample efficiency curves per game (multi-step)
# =========================================================================
print("Plot 3: Sample efficiency curves")
df = pd.read_csv(CSV_DIR / "atari_multistep_step50_mse.csv")
games = df['game'].unique()
ncols = 4
nrows = 2
fig, axes = plt.subplots(nrows, ncols, figsize=(16, 8))
axes = axes.flatten()

for i, game in enumerate(games):
    ax = axes[i]
    gdf = df[df['game'] == game]
    ax.semilogy(gdf['episodes'], gdf['baseline'], 'o-', color=COLORS['baseline'],
                label='Baseline', linewidth=2, markersize=5)
    ax.semilogy(gdf['episodes'], gdf['spatial_stu'], 's-', color=COLORS['spatial_stu'],
                label='STU', linewidth=2, markersize=5)
    if gdf['attention'].notna().any():
        ax.semilogy(gdf['episodes'], gdf['attention'], '^-', color=COLORS['attention'],
                    label='Attention', linewidth=1.5, markersize=5, alpha=0.7)
    ax.set_title(game.title(), fontsize=11)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Step-50 MSE')
    ax.set_xticks([5, 10, 20, 40])
    if i == 0:
        ax.legend(fontsize=8)

# Hide empty subplot
for j in range(len(games), len(axes)):
    axes[j].set_visible(False)

fig.suptitle('Atari Sample Efficiency — Multi-Step Training', fontsize=14, y=1.02)
fig.tight_layout()
save(fig, "03_atari_sample_efficiency_curves")


# =========================================================================
# Plot 4: STU advantage factor vs episodes
# =========================================================================
print("Plot 4: STU advantage curves")
df = pd.read_csv(CSV_DIR / "atari_multistep_step50_mse.csv")
fig, ax = plt.subplots(figsize=(9, 5))
for game in df['game'].unique():
    gdf = df[df['game'] == game]
    adv = gdf['stu_advantage'].values
    # Negative values mean baseline won; plot as 1/abs for uniform scale
    display_adv = [a if a > 0 else -1/a for a in adv]
    ax.plot(gdf['episodes'], display_adv, 'o-', label=game.title(), linewidth=2, markersize=5)

ax.axhline(y=1, color='gray', linestyle='--', alpha=0.5, label='Equal')
ax.set_xlabel('Training Episodes')
ax.set_ylabel('STU Advantage (x)')
ax.set_title('STU Advantage Over Baseline vs Data Quantity')
ax.set_xticks([2, 3, 5, 10, 20, 40])
ax.set_xscale('log')
ax.get_xaxis().set_major_formatter(ticker.ScalarFormatter())
ax.legend(fontsize=8, ncol=2)
ax.set_ylim(0, 22)
save(fig, "04_stu_advantage_vs_episodes")


# =========================================================================
# Plot 5: Multi-seed validation bar chart
# =========================================================================
print("Plot 5: Multi-seed validation")
df = pd.read_csv(CSV_DIR / "atari_multiseed_5ep_multistep.csv")
bl = df[df['model'] == 'baseline'].sort_values('mean')
stu = df[df['model'] == 'spatial_stu']
# Merge
merged = bl[['game', 'mean', 'std']].rename(columns={'mean': 'bl_mean', 'std': 'bl_std'})
stu_data = stu[['game', 'mean', 'std']].rename(columns={'mean': 'stu_mean', 'std': 'stu_std'})
merged = merged.merge(stu_data, on='game')
merged = merged.sort_values('bl_mean', ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(merged))
w = 0.35
ax.bar(x - w/2, merged['bl_mean'], w, yerr=merged['bl_std'], label='Baseline',
       color=COLORS['baseline'], capsize=4)
ax.bar(x + w/2, merged['stu_mean'], w, yerr=merged['stu_std'], label='Spatial STU',
       color=COLORS['spatial_stu'], capsize=4)
ax.set_xticks(x)
ax.set_xticklabels([g.title() for g in merged['game']], rotation=30, ha='right')
ax.set_ylabel('Step-50 MSE (mean ± std, 3 seeds)')
ax.set_title('Multi-Seed Validation — 5 Episodes, Multi-Step Training')
ax.legend()
ax.set_yscale('log')
save(fig, "05_multiseed_validation")


# =========================================================================
# Plot 6: 100-step rollout comparison
# =========================================================================
print("Plot 6: 100-step rollout")
df = pd.read_csv(CSV_DIR / "atari_100step_rollout_5ep_multistep.csv")
df = df.sort_values('stu_advantage_step100', ascending=False)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: step-50 vs step-100 MSE
x = np.arange(len(df))
w = 0.2
ax1.bar(x - 1.5*w, df['baseline_step50'], w, label='BL step-50', color=COLORS['baseline'], alpha=0.7)
ax1.bar(x - 0.5*w, df['baseline_step100'], w, label='BL step-100', color=COLORS['baseline'])
ax1.bar(x + 0.5*w, df['stu_step50'], w, label='STU step-50', color=COLORS['spatial_stu'], alpha=0.7)
ax1.bar(x + 1.5*w, df['stu_step100'], w, label='STU step-100', color=COLORS['spatial_stu'])
ax1.set_xticks(x)
ax1.set_xticklabels([g.title() for g in df['game']], rotation=30, ha='right')
ax1.set_ylabel('MSE')
ax1.set_title('Step-50 vs Step-100 MSE')
ax1.legend(fontsize=7)
ax1.set_yscale('log')

# Right: advantage at step-50 vs step-100
ax2.bar(x - w/2, df['stu_advantage_step50'], w, label='Advantage @ step-50',
        color=COLORS['spatial_stu'], alpha=0.7)
ax2.bar(x + w/2, df['stu_advantage_step100'], w, label='Advantage @ step-100',
        color=COLORS['spatial_stu'])
ax2.set_xticks(x)
ax2.set_xticklabels([g.title() for g in df['game']], rotation=30, ha='right')
ax2.set_ylabel('STU Advantage (x)')
ax2.set_title('STU Advantage Grows with Horizon')
ax2.legend(fontsize=9)
ax2.axhline(y=1, color='gray', linestyle='--', alpha=0.5)

fig.suptitle('Extended Horizon (100-Step Rollout) — 5 Episodes, Multi-Step', fontsize=13)
fig.tight_layout()
save(fig, "06_100step_rollout")


# =========================================================================
# Plot 7: DMC single-step — baseline divergence vs STU stability
# =========================================================================
print("Plot 7: DMC single-step stability")
df = pd.read_csv(CSV_DIR / "dmc_singlestep_step50_mse.csv")
envs = df['environment'].unique()

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for i, env in enumerate(envs):
    ax = axes[i]
    edf = df[df['environment'] == env]
    ax.semilogy(edf['episodes'], edf['baseline'], 'o-', color='gray',
                label='Baseline (no norm)', linewidth=2, markersize=5)
    ax.semilogy(edf['episodes'], edf['baseline_norm'], 'd-', color=COLORS['baseline'],
                label='Baseline+Norm', linewidth=2, markersize=5)
    ax.semilogy(edf['episodes'], edf['feature_stu'], 'v-', color=COLORS['feature_stu'],
                label='Feature STU', linewidth=2, markersize=5)
    ax.semilogy(edf['episodes'], edf['stu_sequential'], 's-', color=COLORS['stu_sequential'],
                label='STU Sequential', linewidth=2, markersize=5)
    ax.set_title(env.replace('_', ' ').title())
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Step-50 MSE')
    ax.set_xticks([5, 10, 20, 40])
    if i == 0:
        ax.legend(fontsize=7)

fig.suptitle('DMC Single-Step Training — Rollout Stability', fontsize=13)
fig.tight_layout()
save(fig, "07_dmc_singlestep_stability")


# =========================================================================
# Plot 8: DMC multi-step sample efficiency
# =========================================================================
print("Plot 8: DMC multi-step")
df = pd.read_csv(CSV_DIR / "dmc_multistep_step50_mse.csv")
envs = df['environment'].unique()

fig, axes = plt.subplots(1, 3, figsize=(14, 4.5))
for i, env in enumerate(envs):
    ax = axes[i]
    edf = df[df['environment'] == env]
    ax.plot(edf['episodes'], edf['baseline'], 'o-', color=COLORS['baseline'],
            label='Baseline', linewidth=2, markersize=5)
    ax.plot(edf['episodes'], edf['feature_stu'], 's-', color=COLORS['spatial_stu'],
            label='Feature STU', linewidth=2, markersize=5)
    ax.set_title(env.replace('_', ' ').title())
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Step-50 MSE')
    ax.set_xticks([5, 10, 20, 40])
    if i == 0:
        ax.legend(fontsize=9)

fig.suptitle('DMC Multi-Step Training — Sample Efficiency', fontsize=13)
fig.tight_layout()
save(fig, "08_dmc_multistep_sample_efficiency")


# =========================================================================
# Plot 9: Attention plateau visualization
# =========================================================================
print("Plot 9: Attention plateau")
df = pd.read_csv(CSV_DIR / "attention_comparison_multistep.csv")
games_to_show = ['pong', 'seaquest', 'alien', 'roadrunner']

fig, axes = plt.subplots(1, 4, figsize=(16, 4))
for i, game in enumerate(games_to_show):
    ax = axes[i]
    gdf = df[df['game'] == game]
    ax.semilogy(gdf['episodes'], gdf['baseline'], 'o-', color=COLORS['baseline'],
                label='Baseline', linewidth=2, markersize=5)
    ax.semilogy(gdf['episodes'], gdf['spatial_stu'], 's-', color=COLORS['spatial_stu'],
                label='STU', linewidth=2, markersize=5)
    ax.semilogy(gdf['episodes'], gdf['attention'], '^-', color=COLORS['attention'],
                label='Attention', linewidth=2, markersize=5)
    ax.set_title(game.title())
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Step-50 MSE')
    ax.set_xticks([5, 10, 20, 40])
    if i == 0:
        ax.legend(fontsize=7)

fig.suptitle('Self-Attention Plateau — Multi-Step Training', fontsize=13)
fig.tight_layout()
save(fig, "09_attention_plateau")


# =========================================================================
# Plot 10: Single-step vs multi-step improvement (Pong)
# =========================================================================
print("Plot 10: Single-step vs multi-step (Pong)")
df = pd.read_csv(CSV_DIR / "pong_multistep_improvement.csv")
df = df.dropna(subset=['singlestep_step50_mse', 'multistep_step50_mse'])

fig, ax = plt.subplots(figsize=(8, 5))
x = np.arange(len(df))
w = 0.35
ax.bar(x - w/2, df['singlestep_step50_mse'], w, label='Single-step', color='#EF5350')
ax.bar(x + w/2, df['multistep_step50_mse'], w, label='Multi-step', color='#66BB6A')
ax.set_xticks(x)
labels = [str(m).replace('_', '\n') for m in df['model']]
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel('Step-50 MSE (log scale)')
ax.set_title('Single-Step vs Multi-Step Training — Pong')
ax.set_yscale('log')
ax.legend()
# Add improvement factor labels
for i, (_, row) in enumerate(df.iterrows()):
    factor = row['improvement_factor']
    if not np.isnan(factor):
        ax.text(i, row['multistep_step50_mse'] * 0.3, f'{factor:.0f}x',
                ha='center', fontsize=9, fontweight='bold')
save(fig, "10_singlestep_vs_multistep_pong")


# =========================================================================
# Plot 11: TDMPC2 results
# =========================================================================
print("Plot 11: TDMPC2 comparison")
df = pd.read_csv(CSV_DIR / "tdmpc2_results.csv")
# Full data multi-step
ms40 = df[(df['training'] == 'multistep') & (df['episodes'] == 40)]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

# Left: TDMPC2 multi-step results
envs = ms40['environment'].unique()
x = np.arange(len(envs))
w = 0.3
bl_vals = ms40[ms40['model'] == 'baseline']['step50_mse'].values
stu_vals = ms40[ms40['model'] == 'stu_spectral']['step50_mse'].values
ax1.bar(x - w/2, bl_vals, w, label='Baseline', color=COLORS['baseline'])
ax1.bar(x + w/2, stu_vals, w, label='STU Spectral', color=COLORS['spatial_stu'])
ax1.set_xticks(x)
ax1.set_xticklabels([e.replace('-', ' ').title() for e in envs], rotation=15)
ax1.set_ylabel('Step-50 MSE')
ax1.set_title('TDMPC2 — Full Data, Multi-Step')
ax1.legend()

# Right: Walker sample efficiency
walker = df[(df['environment'] == 'walker-walk') & (df['training'] == 'multistep')]
bl_w = walker[walker['model'] == 'baseline']
stu_w = walker[walker['model'] == 'stu_spectral']
ax2.plot(bl_w['episodes'], bl_w['step50_mse'], 'o-', color=COLORS['baseline'],
         label='Baseline', linewidth=2)
ax2.plot(stu_w['episodes'], stu_w['step50_mse'], 's-', color=COLORS['spatial_stu'],
         label='STU Spectral', linewidth=2)
ax2.set_xlabel('Episodes')
ax2.set_ylabel('Step-50 MSE')
ax2.set_title('TDMPC2 Walker — Sample Efficiency')
ax2.set_xticks([5, 10, 20, 40])
ax2.legend()

fig.suptitle('TDMPC2 Dynamics Experiments', fontsize=13)
fig.tight_layout()
save(fig, "11_tdmpc2_results")


# =========================================================================
# Plot 12: DMC image hopper
# =========================================================================
print("Plot 12: DMC Image Hopper")
df = pd.read_csv(CSV_DIR / "dmc_image_hopper_step50_mse.csv")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.5))
for training, ax, title in [('singlestep', ax1, 'Single-Step'), ('multistep', ax2, 'Multi-Step')]:
    tdf = df[df['training'] == training]
    ax.semilogy(tdf['episodes'], tdf['baseline'], 'o-', color=COLORS['baseline'],
                label='Baseline', linewidth=2)
    ax.semilogy(tdf['episodes'], tdf['spatial_stu'], 's-', color=COLORS['spatial_stu'],
                label='Spatial STU', linewidth=2)
    if tdf['attention'].notna().any():
        ax.semilogy(tdf['episodes'], tdf['attention'], '^-', color=COLORS['attention'],
                    label='Attention', linewidth=2)
    ax.set_xlabel('Episodes')
    ax.set_ylabel('Step-50 MSE')
    ax.set_title(f'{title} Training')
    ax.set_xticks([5, 10, 20, 40])
    ax.legend(fontsize=8)

fig.suptitle('DMC Image Hopper Hop — Spatial [64,6,6] Latent States', fontsize=13)
fig.tight_layout()
save(fig, "12_dmc_image_hopper")


# =========================================================================
# Plot 13: Heatmap — STU advantage across games and episodes
# =========================================================================
print("Plot 13: Advantage heatmap")
df = pd.read_csv(CSV_DIR / "atari_multistep_step50_mse.csv")
pivot = df.pivot_table(index='game', columns='episodes', values='stu_advantage')
# Order games by 5-ep advantage
game_order = pivot[5].sort_values(ascending=False).index if 5 in pivot.columns else pivot.index
pivot = pivot.loc[game_order]

fig, ax = plt.subplots(figsize=(8, 5))
# For display, cap negative values and show as text
display = pivot.copy()
display = display.apply(lambda col: col.map(lambda x: x if x > 0 else -1/x if x < 0 else 1))

sns.heatmap(display, annot=True, fmt='.1f', cmap='RdYlGn', center=1,
            ax=ax, linewidths=0.5, cbar_kws={'label': 'STU Advantage (x)'},
            xticklabels=[str(int(c)) for c in display.columns],
            yticklabels=[g.title() for g in display.index])
ax.set_xlabel('Training Episodes')
ax.set_ylabel('')
ax.set_title('STU Advantage Heatmap — Atari Multi-Step Training')
save(fig, "13_advantage_heatmap")


print(f"\nAll plots saved to {PLOT_DIR}/")
print(f"Total: {len(list(PLOT_DIR.glob('*.png')))} PNG + {len(list(PLOT_DIR.glob('*.pdf')))} PDF files")
