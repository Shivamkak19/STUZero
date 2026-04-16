"""Extract eval_video GIFs from ReZero tensorboard tfevents and build training-progression reels.

One reel per run (baseline, stu_hankelscaled) showing the policy evolution.
Plus a comparison reel at matched training step.
"""
import glob
import io
import os
from pathlib import Path

from tensorboard.backend.event_processing import event_accumulator as ea
import numpy as np
from PIL import Image, ImageDraw, ImageSequence

OUT = Path('/lambda/nfs/hazan/STUZero/research_logs/logs/12_final_sweep_data/video_reels')
OUT.mkdir(parents=True, exist_ok=True)

N_COLS = 10
SCALE = 3


def load_all_eval_videos(run_dir):
    """Return [(step, PIL.Image GIF), ...] for eval_video tag across all tfevent shards."""
    files = sorted(glob.glob(f'{run_dir}/events.out.tfevents*'))
    out = []
    seen = set()
    for f in files:
        a = ea.EventAccumulator(f, size_guidance={'images': 0})
        a.Reload()
        if 'eval_video' not in a.Tags().get('images', []):
            continue
        for img in a.Images('eval_video'):
            if img.step in seen:
                continue
            seen.add(img.step)
            gif = Image.open(io.BytesIO(img.encoded_image_string))
            out.append((img.step, gif))
    return sorted(out, key=lambda x: x[0])


def sample_frames(gif, n_frames):
    frames = [f.convert('RGB').copy() for f in ImageSequence.Iterator(gif)]
    if len(frames) == 0:
        return []
    if len(frames) <= n_frames:
        return frames
    indices = np.linspace(0, len(frames) - 1, n_frames).astype(int)
    return [frames[i] for i in indices]


def build_grid(rows, labels, out_path, scale=SCALE, pad=3, label_w=140, header_h=30):
    scaled = [[f.resize((f.width * scale, f.height * scale), Image.NEAREST) for f in row]
              for row in rows]
    n_rows = len(scaled); n_cols = max(len(r) for r in scaled)
    fw, fh = scaled[0][0].size
    W = label_w + n_cols * fw + (n_cols + 1) * pad
    H = header_h + n_rows * fh + (n_rows + 1) * pad
    img = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(img)
    for c in range(n_cols):
        pct = int(c * 100 / max(1, n_cols - 1))
        draw.text((label_w + pad + c * (fw + pad) + 4, 8), f'{pct:>3}%', fill='black')
    for r, (frames, label) in enumerate(zip(scaled, labels)):
        y = header_h + pad + r * (fh + pad)
        for i, line in enumerate(label.split('\n')):
            draw.text((4, y + 4 + i * 14), line, fill='black')
        for c, f in enumerate(frames):
            x = label_w + pad + c * (fw + pad)
            img.paste(f, (x, y))
    img.save(out_path)
    print(f'  wrote {out_path}')


runs = {
    'baseline': '/lambda/nfs/hazan/ReZero/runs/pong_baseline_seed0',
    'stu-hankelscaled': '/lambda/nfs/hazan/ReZero/runs/pong_stu_hankelscaled_seed0',
}

# =============================================================================
# Reel 1: progression per run
# =============================================================================
run_videos = {}
for name, rd in runs.items():
    print(f'loading {name}...')
    vids = load_all_eval_videos(rd)
    print(f'  got {len(vids)} eval videos (steps {vids[0][0]} to {vids[-1][0]})')
    run_videos[name] = vids

for name, vids in run_videos.items():
    if not vids:
        continue
    # Pick ~10 evenly spaced training steps
    target_steps = np.linspace(vids[0][0], vids[-1][0], 10).astype(int)
    chosen = []
    used = set()
    for t in target_steps:
        best = min(vids, key=lambda sv: abs(sv[0] - t))
        if best[0] not in used:
            chosen.append(best); used.add(best[0])
    rows, labels = [], []
    for step, gif in chosen:
        frames = sample_frames(gif, N_COLS)
        if frames:
            rows.append(frames)
            labels.append(f'{name}\nstep {step//1000}k')
    build_grid(rows, labels, OUT / f'rezero_pong_{name.replace("-","_")}_progression.png')

# =============================================================================
# Reel 2: comparison at matched step
# =============================================================================
# Find steps both runs have — pick 5 evenly spaced
common_steps = sorted(set(s for s, _ in run_videos['baseline']) &
                      set(s for s, _ in run_videos['stu-hankelscaled']))
if common_steps:
    picks = [common_steps[i] for i in np.linspace(0, len(common_steps) - 1, 5).astype(int)]
    rows, labels = [], []
    lookups = {name: dict(vids) for name, vids in run_videos.items()}
    for step in picks:
        for name in ['baseline', 'stu-hankelscaled']:
            gif = lookups[name][step]
            frames = sample_frames(gif, N_COLS)
            if frames:
                rows.append(frames)
                labels.append(f'{name}\nstep {step//1000}k')
    build_grid(rows, labels, OUT / 'rezero_pong_baseline_vs_stu_at_matched_steps.png')

print('done')
