"""Build reels from stu-dreamer-jax auto-saved policy videos.

Two outputs:
1. Method comparison at matched training step (walker_walk: 5 methods)
2. Training progression for one method (walker_walk stuPlus)
"""
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

ROOT = Path('/home/ubuntu/dreamer_runs')
OUT = Path('/lambda/nfs/hazan/STUZero/research_logs/logs/12_final_sweep_data/video_reels')
OUT.mkdir(parents=True, exist_ok=True)

N_COLS = 10
SCALE = 3
FRAME_SIZE = 64


def extract_frames(mp4, n):
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v',
         '-show_entries', 'stream=nb_frames', '-of', 'default=nw=1:nk=1', str(mp4)],
        capture_output=True, text=True)
    try:
        total = int(r.stdout.strip())
    except ValueError:
        return []
    indices = np.linspace(0, total - 1, n).astype(int)
    frames = []
    for idx in indices:
        tmp = f'/tmp/_dframe_{os.getpid()}.png'
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error', '-i', str(mp4),
             '-vf', f"select='eq(n,{idx})'", '-vframes', '1', tmp],
            check=True)
        if os.path.exists(tmp):
            frames.append(Image.open(tmp).copy())
            os.remove(tmp)
    return frames


def list_videos(run_dir):
    """Return [(step_int, mp4_path), ...] sorted by step."""
    vids = list((run_dir / 'scope' / 'epstats-policy_log-image.mp4').glob('*.mp4'))
    out = []
    for v in vids:
        m = re.match(r'(\d+)-', v.name)
        if m:
            out.append((int(m.group(1)), v))
    return sorted(out)


def build_grid(rows_data, row_labels, out_path, scale=SCALE, pad=3, label_w=140, header_h=30):
    """rows_data: list of lists of PIL frames. row_labels: one label per row."""
    assert len(rows_data) == len(row_labels)
    # Scale up frames
    scaled = [[f.resize((f.width * scale, f.height * scale), Image.NEAREST)
               if scale != 1 else f for f in row] for row in rows_data]
    n_rows = len(scaled)
    n_cols = max(len(r) for r in scaled)
    fw, fh = scaled[0][0].size
    W = label_w + n_cols * fw + (n_cols + 1) * pad
    H = header_h + n_rows * fh + (n_rows + 1) * pad
    img = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(img)

    # Header (column labels)
    for c in range(n_cols):
        pct = int(c * 100 / max(1, n_cols - 1))
        x = label_w + pad + c * (fw + pad)
        draw.text((x + 4, 8), f'{pct:>3}%', fill='black')

    for r, (frames, label) in enumerate(zip(scaled, row_labels)):
        y = header_h + pad + r * (fh + pad)
        # Multi-line label
        for i, line in enumerate(label.split('\n')):
            draw.text((4, y + 4 + i * 14), line, fill='black')
        for c, f in enumerate(frames):
            x = label_w + pad + c * (fw + pad)
            img.paste(f, (x, y))

    img.save(out_path)
    print(f'  wrote {out_path}')


# =============================================================================
# Reel 1: walker_walk 5-method comparison — pick video closest to step ~400k
# =============================================================================
methods = [
    ('baseline', 'baseline_proprio_20260411T173021'),
    ('stuB (K=24)', 'stuB_proprio_20260411T231533'),
    ('stuRand', 'stuRand_proprio_20260412T080926'),
    ('stuPlus', 'stuPlus_proprio_20260412T205437'),
    ('stufix (orig fail)', 'stufix_proprio_20260411T073936'),
]
TARGET_STEP = 400_000

rows, labels = [], []
for name, dname in methods:
    rundir = ROOT / dname
    vids = list_videos(rundir)
    if not vids:
        print(f'  skip {name}: no videos')
        continue
    # Pick video with step closest to TARGET_STEP
    best = min(vids, key=lambda sv: abs(sv[0] - TARGET_STEP))
    step, video = best
    print(f'{name}: using step {step:>6} from {video.name}')
    frames = extract_frames(video, N_COLS)
    if frames:
        rows.append(frames)
        labels.append(f'{name}\nstep {step//1000}k')

build_grid(rows, labels, OUT / 'dreamer_walker_walk_methods_at_400k.png')

# =============================================================================
# Reel 2: walker_walk progression for stuPlus — sample at ~every 50k
# =============================================================================
rundir = ROOT / 'stuPlus_proprio_20260412T205437'
vids = list_videos(rundir)
# Pick videos evenly spaced in step
if vids:
    max_step = vids[-1][0]
    target_steps = np.linspace(vids[0][0], max_step, 7).astype(int)
    chosen = []
    for t in target_steps:
        best = min(vids, key=lambda sv: abs(sv[0] - t))
        if best not in chosen:
            chosen.append(best)
    rows, labels = [], []
    for step, video in chosen:
        frames = extract_frames(video, N_COLS)
        if frames:
            rows.append(frames)
            labels.append(f'stuPlus\nstep {step//1000}k')
    build_grid(rows, labels, OUT / 'dreamer_walker_walk_stuPlus_progression.png')

# =============================================================================
# Reel 3: walker_walk progression for baseline — for comparison
# =============================================================================
rundir = ROOT / 'baseline_proprio_20260411T173021'
vids = list_videos(rundir)
if vids:
    target_steps = np.linspace(vids[0][0], vids[-1][0], 7).astype(int)
    chosen = []
    for t in target_steps:
        best = min(vids, key=lambda sv: abs(sv[0] - t))
        if best not in chosen:
            chosen.append(best)
    rows, labels = [], []
    for step, video in chosen:
        frames = extract_frames(video, N_COLS)
        if frames:
            rows.append(frames)
            labels.append(f'baseline\nstep {step//1000}k')
    build_grid(rows, labels, OUT / 'dreamer_walker_walk_baseline_progression.png')

# =============================================================================
# Reel 4: hopper_stand progression for stuPlus size1m
# =============================================================================
rundir = ROOT / 'stuPlus_hopper_stand_20260413T023902'
vids = list_videos(rundir)
if vids:
    target_steps = np.linspace(vids[0][0], vids[-1][0], 10).astype(int)
    chosen = []
    for t in target_steps:
        best = min(vids, key=lambda sv: abs(sv[0] - t))
        if best not in chosen:
            chosen.append(best)
    rows, labels = [], []
    for step, video in chosen:
        frames = extract_frames(video, N_COLS)
        if frames:
            rows.append(frames)
            labels.append(f'stuPlus\nstep {step//1000}k')
    build_grid(rows, labels, OUT / 'dreamer_hopper_stand_stuPlus_progression.png')

print('done')
