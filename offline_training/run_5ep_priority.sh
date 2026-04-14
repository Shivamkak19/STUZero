#!/bin/bash
# run_5ep_priority.sh
# Priority pass: 5-episode multi-step (STU/baseline/attention) for every game
# that doesn't already have training_complete.flag.
#
# Uses a GLOBAL job pool (not per-game throttling) so jobs from multiple games
# interleave on the GPU and we hide non-GPU phases across games.
#
# Cap is set to N=8 by default:
#   - small games (~8 GB RES per process)  -> 8 jobs ~ 64 GB RAM, safe
#   - hero (~25 GB RES per process)        -> 8 jobs ~ 200 GB RAM, fits in 216 GB
#   - breakout/gopher (~60 GB per process) -> excluded from the pool, run last with N=1
#
# Override via env var POOL_N.

set -u
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

POOL_N=${POOL_N:-8}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/offline_training"

SCRIPT="train_dynamics_offline.py"
RESULTS_DIR="results/sample_efficiency"
mkdir -p "$RESULTS_DIR" results/pipeline_logs

# Game -> data subdir for symlink/setup
declare -A DATA_DIRS=(
    [alien]=alien_110K [amidar]=amidar_110K [assault]=assault_90K [asterix]=asterix_110K
    [bankheist]=bankheist_110K [battlezone]=battlezone_40K [boxing]=boxing_110K
    [breakout]=breakout_110K [choppercommand]=choppercommand_50K [crazyclimber]=crazyclimber_90K
    [demonattack]=demonattack_110K [freeway]=freeway_110K [frostbite]=frostbite_100K
    [gopher]=gopher_100K [hero]=hero_60K [jamesbond]=jamesbond_90K [kangaroo]=kangaroo_110K
    [krull]=krull_110K [kungfumaster]=kungfumaster_110K [mspacman]=mspacman_90K
    [pong]=pong_100K [privateeye]=privateeye_110K [qbert]=qbert_80K
    [roadrunner]=roadrunner_110K [seaquest]=seaquest_110K [upndown]=upndown_60K
)

# Pool games: only the new-cohort games that need 5ep multistep runs.
# (Original 7 cohort already has results in master_summary.md; pong skipped per user.)
POOL_GAMES=(
    freeway frostbite hero jamesbond kangaroo krull privateeye qbert upndown
)
# Run last, one job at a time, due to ~60 GB RES each
SOLO_GAMES=(breakout gopher)

echo "============================================"
echo "  5ep priority pass start: $(date)"
echo "  POOL_N=$POOL_N (global cap, applies to small/medium pool only)"
echo "  Pool games: ${#POOL_GAMES[@]}, solo games: ${#SOLO_GAMES[@]}"
echo "============================================"

# Symlink expert checkpoint into existing data dirs (no-op for fresh games — they'll download)
for GAME in "${!DATA_DIRS[@]}"; do
    DD=${DATA_DIRS[$GAME]}
    [ -d "$DD" ] || continue
    has_model=0
    for f in "$DD"/atari_${GAME}_model_*.p "$DD"/${GAME}_model_*.p; do
        [ -e "$f" ] && { has_model=1; break; }
    done
    if [ $has_model -eq 0 ]; then
        for src in atari_models/atari_${GAME}_model_*.p; do
            [ -f "$src" ] || continue
            fn=$(basename "$src")
            ln -sf "../atari_models/$fn" "$DD/$fn"
            echo ">>> [setup] symlinked $fn into $DD/"
        done
    fi
done

# Use bash's job table directly. `jobs -rp` lists running background jobs (pids);
# `wait -n` blocks until ANY one of them terminates and reaps it. This avoids
# the trap of waiting on a specific pid (which blocks even if a younger sibling
# finishes first). No tracked array needed.

throttle_to() {
    local cap=$1
    while [ "$(jobs -rp | wc -l)" -ge "$cap" ]; do
        wait -n 2>/dev/null || true
    done
}

launch_one() {
    local GAME=$1 MODEL=$2
    local RUN_NAME="${GAME}_${MODEL}_5ep_multistep"
    local SAVE_DIR="$RESULTS_DIR/$RUN_NAME"
    if [ -f "$SAVE_DIR/training_complete.flag" ]; then
        echo ">>> SKIP $RUN_NAME (already complete)"
        return
    fi
    local MARG
    case $MODEL in
        stu)           MARG="--model_type stu --num_blocks 1 --dynamics_stu_num_filters 2" ;;
        baseline_only) MARG="--model_type baseline_only --num_blocks 1" ;;
        attention)     MARG="--model_type attention --num_blocks 1 --attn_num_heads 4" ;;
    esac
    echo ">>> [$(date +%H:%M:%S)] LAUNCH $RUN_NAME (in-flight=$(jobs -rp | wc -l))"
    (
        python $SCRIPT \
            --game $GAME --download \
            --train_split 0.1 --seed 42 \
            --epochs 80 --rollout_len 50 --eval_interval 5 \
            --skip_verification \
            $MARG \
            --multistep_train_len 20 --multistep_stride 5 \
            --save_dir "$SAVE_DIR" \
            > "$RESULTS_DIR/${RUN_NAME}.log" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            touch "$SAVE_DIR/training_complete.flag"
            echo ">>> [$(date +%H:%M:%S)] DONE  $RUN_NAME"
        else
            echo ">>> [$(date +%H:%M:%S)] FAIL  $RUN_NAME (rc=$rc)"
        fi
    ) &
}

echo ""
echo "===== Phase 1: pool games (cap=$POOL_N) ====="
for GAME in "${POOL_GAMES[@]}"; do
    for MODEL in stu baseline_only attention; do
        throttle_to "$POOL_N"
        launch_one "$GAME" "$MODEL"
    done
done
# Drain pool
throttle_to 1

echo ""
echo "===== Phase 2: solo games (cap=1) ====="
for GAME in "${SOLO_GAMES[@]}"; do
    for MODEL in stu baseline_only attention; do
        throttle_to 1
        launch_one "$GAME" "$MODEL"
    done
done
throttle_to 1

# Final drain
while [ "$(jobs -rp | wc -l)" -gt 0 ]; do
    wait -n 2>/dev/null || true
done

echo ""
echo "============================================"
echo "  5ep priority pass complete: $(date)"
echo "============================================"
