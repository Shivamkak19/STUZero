"""Four-way comparison: baseline vs stuB vs stuRand vs stuPlus."""
import json, glob, sys, statistics
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

def load(p):
    xs, ys = [], []
    if not p or not Path(p).exists(): return xs, ys
    for line in Path(p).read_text().splitlines():
        if not line.strip(): continue
        try: r = json.loads(line)
        except: continue
        if 'episode/score' in r and 'step' in r:
            xs.append(r['step']); ys.append(r['episode/score'])
    return np.array(xs), np.array(ys)

def latest(prefix):
    matches = sorted(Path('/home/ubuntu/dreamer_runs').glob(prefix + '*/scores.jsonl'))
    return str(matches[-1]) if matches else None

def smooth(ys, k=20):
    if len(ys) < k: return ys
    return np.convolve(ys, np.ones(k)/k, mode='valid')

RUNS = {
    'baseline': '/home/ubuntu/dreamer_runs/baseline_proprio_20260411T173021/scores.jsonl',
    'stuB (spectral K=24)': latest('stuB_proprio_'),
    'stuRand (random norm)': latest('stuRand_proprio_'),
    'stuPlus (K=48+neg+obs)': latest('stuPlus_proprio_'),
}

fig, ax = plt.subplots(figsize=(10, 6))
colors = ['#0022ff', '#33aa00', '#ff0011', '#dd8800']
for (label, path), color in zip(RUNS.items(), colors):
    xs, ys = load(path)
    if len(ys) == 0:
        print(f'  [skip] {label}')
        continue
    print(f'  [{label}]: {len(ys)} eps, last 50 mean {ys[-50:].mean():.1f}, max {ys.max():.1f}')
    ax.plot(xs, ys, alpha=0.15, color=color)
    sm = smooth(ys)
    if len(sm):
        ax.plot(xs[-len(sm):], sm, label=label, linewidth=2, color=color)

ax.set_xlabel('env step')
ax.set_ylabel('episode score')
ax.set_title('dmc_walker_walk: STU integration comparison')
ax.legend(loc='lower right')
ax.grid(True, alpha=0.3)

out = sys.argv[1] if len(sys.argv) > 1 else '/tmp/compare_all.png'
fig.savefig(out, dpi=120, bbox_inches='tight')
print(f'wrote {out}')
