"""Generate comparison reels (baseline vs stuPlus) for walker_run and hopper_hop.

For each run, find mp4s in scope/epstats-policy_log-image.mp4/, pick N evenly
spaced training steps (rows), extract M evenly spaced frames per video (cols),
tile into a grid. Then put baseline and stuPlus grids side-by-side.
"""
import re
import subprocess
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

LOGROOT = Path("/scratch/gpfs/EHAZAN/sk3686/logdir/dreamer")
OUT = Path("/home/sk3686/hazan/reels")
OUT.mkdir(exist_ok=True)

RUNS = {
    "walker_run": {
        "baseline": LOGROOT / "loginnode_walker_run_baseline_20260414_200239_della-adele",
        "stuPlus":  LOGROOT / "loginnode_walker_run_stuplus_20260414_200239_della-fongpu",
    },
    "hopper_hop": {
        "baseline": LOGROOT / "loginnode_hopper_hop_baseline_20260414_213513_della-mol",
        "stuPlus":  LOGROOT / "loginnode_hopper_hop_stuplus_20260414_200239_della-stellato",
    },
}

N_ROWS = 10       # training checkpoints
N_COLS = 10       # frames per video
CELL = 64         # frame size (native dmc_proprio image is 64x64)
PAD = 2
LABEL_W = 80      # left label column width
HEADER_H = 24     # top title height


def list_videos(run_dir):
    vdir = run_dir / "scope" / "epstats-policy_log-image.mp4"
    mp4s = []
    for f in vdir.iterdir():
        m = re.match(r"(\d+)-", f.name)
        if m:
            mp4s.append((int(m.group(1)), f))
    mp4s.sort()
    return mp4s


def pick_rows(mp4s, n):
    # Evenly spaced across available training steps.
    if len(mp4s) <= n:
        return mp4s
    idxs = np.linspace(0, len(mp4s) - 1, n).round().astype(int)
    return [mp4s[i] for i in idxs]


def extract_frames(mp4, n):
    # Probe frame count.
    out = subprocess.check_output([
        "ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)
    ]).decode().strip()
    total = int(out)
    idxs = np.linspace(0, total - 1, n).round().astype(int)
    # Use select filter to pull specific frames.
    expr = "+".join(f"eq(n\\,{i})" for i in idxs)
    tmp = Path(f"/tmp/reel_{mp4.stem}_%03d.png")
    # Clean any prior.
    for p in Path("/tmp").glob(f"reel_{mp4.stem}_*.png"):
        p.unlink()
    subprocess.run([
        "ffmpeg", "-v", "error", "-y", "-i", str(mp4),
        "-vf", f"select='{expr}'", "-vsync", "0",
        str(tmp),
    ], check=True)
    frames = sorted(Path("/tmp").glob(f"reel_{mp4.stem}_*.png"))
    imgs = [Image.open(f).convert("RGB").resize((CELL, CELL)) for f in frames]
    for f in frames:
        f.unlink()
    # Make sure we return exactly n (pad last if ffmpeg merged duplicates).
    while len(imgs) < n:
        imgs.append(imgs[-1])
    return imgs[:n]


def build_grid(mp4s, title):
    rows = pick_rows(mp4s, N_ROWS)
    W = LABEL_W + N_COLS * (CELL + PAD) + PAD
    H = HEADER_H + len(rows) * (CELL + PAD) + PAD
    img = Image.new("RGB", (W, H), "white")
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf", 14)
        small = ImageFont.truetype("/usr/share/fonts/dejavu/DejaVuSans.ttf", 10)
    except Exception:
        font = ImageFont.load_default()
        small = font
    draw.text((LABEL_W + 4, 4), title, fill="black", font=font)
    for r, (step, mp4) in enumerate(rows):
        y = HEADER_H + r * (CELL + PAD)
        label = f"{step/1000:.0f}k"
        draw.text((4, y + CELL // 2 - 6), label, fill="black", font=small)
        frames = extract_frames(mp4, N_COLS)
        for c, fr in enumerate(frames):
            x = LABEL_W + c * (CELL + PAD)
            img.paste(fr, (x, y))
    return img


def compose_comparison(task):
    runs = RUNS[task]
    print(f"[{task}] baseline ...")
    gb = build_grid(list_videos(runs["baseline"]), "(a) Baseline")
    print(f"[{task}] stuPlus ...")
    gs = build_grid(list_videos(runs["stuPlus"]), "(b) stuPlus")
    GAP = 20
    W = gb.width + GAP + gs.width
    H = max(gb.height, gs.height)
    out = Image.new("RGB", (W, H), "white")
    out.paste(gb, (0, 0))
    out.paste(gs, (gb.width + GAP, 0))
    path = OUT / f"reel_{task}.png"
    out.save(path)
    print(f"  wrote {path}")
    return path


if __name__ == "__main__":
    for task in RUNS:
        compose_comparison(task)
