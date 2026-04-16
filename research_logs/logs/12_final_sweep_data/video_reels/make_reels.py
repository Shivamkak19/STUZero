"""Build training-progression filmstrips for STUZero EZ-V2 Atari runs.

For each run:
  rows    = evaluation checkpoints (step_0, ..., final)
  columns = frames sampled at fixed intervals within epi_0 at that checkpoint
"""
import os
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path('/lambda/nfs/hazan/STUZero/results/Atari')
OUT_DIR = Path('/lambda/nfs/hazan/STUZero/research_logs/logs/12_final_sweep_data/video_reels')
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_COLS = 10  # number of frames per row


def extract_frames(mp4_path, n_frames):
    """Extract n_frames evenly spaced frames from an mp4 as PIL Images."""
    # First get nb_frames
    r = subprocess.run(
        ['ffprobe', '-v', 'error', '-select_streams', 'v',
         '-show_entries', 'stream=nb_frames', '-of', 'default=nw=1:nk=1',
         str(mp4_path)],
        capture_output=True, text=True)
    try:
        total = int(r.stdout.strip())
    except ValueError:
        return []
    if total <= 0:
        return []

    # Pick evenly spaced frame indices (first frame, ..., near-last)
    indices = np.linspace(0, total - 1, n_frames).astype(int)

    # Use ffmpeg to extract each frame as png — fast enough for our needs
    frames = []
    for idx in indices:
        tmp = f'/tmp/_reel_frame_{os.getpid()}.png'
        subprocess.run(
            ['ffmpeg', '-y', '-loglevel', 'error',
             '-i', str(mp4_path),
             '-vf', f"select='eq(n,{idx})'",
             '-vframes', '1', tmp],
            check=True)
        if os.path.exists(tmp):
            frames.append(Image.open(tmp).copy())
            os.remove(tmp)
    return frames


def step_sort_key(step_name):
    if step_name == 'final':
        return float('inf')
    m = re.match(r'step_(\d+)', step_name)
    return int(m.group(1)) if m else -1


def build_reel(run_dir: Path, out_path: Path, row_height=96, pad=2, label_w=90, scale=2):
    eval_dir = run_dir / 'evaluation'
    if not eval_dir.is_dir():
        print(f'  no evaluation/ in {run_dir.name}')
        return False

    step_names = sorted(
        [d.name for d in eval_dir.iterdir() if d.is_dir()],
        key=step_sort_key)
    if not step_names:
        return False

    rows = []
    labels = []
    for step in step_names:
        video = eval_dir / step / 'recordings' / 'epi_0_27000.mp4'
        if not video.exists():
            continue
        frames = extract_frames(video, N_COLS)
        if not frames:
            continue
        rows.append(frames)
        labels.append(step.replace('step_', '').rjust(7))

    if not rows:
        print(f'  no usable videos in {run_dir.name}')
        return False

    # Scale up frames for readability
    if scale != 1:
        rows = [[f.resize((f.width * scale, f.height * scale), Image.NEAREST) for f in frames]
                for frames in rows]

    # Compose the grid
    frame_w, frame_h = rows[0][0].size
    W = label_w + N_COLS * frame_w + (N_COLS + 1) * pad
    H = len(rows) * frame_h + (len(rows) + 1) * pad + 30  # header
    img = Image.new('RGB', (W, H), 'white')
    draw = ImageDraw.Draw(img)

    # Header: column indices (frame numbers)
    for c in range(N_COLS):
        x = label_w + pad + c * (frame_w + pad)
        col_label = f'f{int(c * 99 / (N_COLS - 1)):>2d}%' if N_COLS > 1 else 'f0%'
        draw.text((x + 4, 6), col_label, fill='black')

    for r, (frames, label) in enumerate(zip(rows, labels)):
        y = 30 + pad + r * (frame_h + pad)
        draw.text((4, y + frame_h // 2 - 6), f'step {label}', fill='black')
        for c, f in enumerate(frames):
            x = label_w + pad + c * (frame_w + pad)
            img.paste(f, (x, y))

    img.save(out_path)
    return True


def main():
    games = ['KungFuMaster', 'Assault', 'Boxing', 'Gopher', 'Asterix']
    # Skip Pong (36 runs, pick only the latest for size)
    for game in games:
        game_dir = ROOT / game
        if not game_dir.is_dir():
            continue
        runs = sorted([d for d in game_dir.iterdir() if d.is_dir()])
        # For games with multiple runs, use the latest (most recent)
        runs_to_process = runs[-1:] if len(runs) > 1 else runs
        for run in runs_to_process:
            out = OUT_DIR / f'{game}_{run.name.replace(" ", "_").replace(":", "-")}.png'
            print(f'{game}/{run.name} -> {out.name}')
            try:
                ok = build_reel(run, out)
                if ok:
                    print(f'  wrote {out}')
            except Exception as e:
                print(f'  ERROR: {e}')

    # Pong: pick the run with most eval checkpoints (not the most recent)
    pong_dir = ROOT / 'Pong'
    if pong_dir.is_dir():
        best_run, best_count = None, 0
        for d in pong_dir.iterdir():
            if not d.is_dir(): continue
            eval_d = d / 'evaluation'
            if not eval_d.is_dir(): continue
            cnt = sum(1 for x in eval_d.iterdir() if x.is_dir())
            if cnt > best_count:
                best_count, best_run = cnt, d
        if best_run:
            out = OUT_DIR / f'Pong_{best_run.name.replace(" ", "_").replace(":", "-")}.png'
            print(f'Pong/{best_run.name} ({best_count} evals) -> {out.name}')
            try:
                ok = build_reel(best_run, out)
                if ok:
                    print(f'  wrote {out}')
            except Exception as e:
                print(f'  ERROR: {e}')


if __name__ == '__main__':
    main()
