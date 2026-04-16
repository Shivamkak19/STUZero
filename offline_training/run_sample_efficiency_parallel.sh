#!/bin/bash
# run_sample_efficiency_parallel.sh
#
# Same protocol as run_sample_efficiency.sh, but runs up to N training jobs in
# parallel on a single GPU. Each dynamics-only training run is small (~1-3 GB
# GPU mem), so on a 40GB A100 we can comfortably run 4 at a time.
#
# Usage:
#   bash offline_training/run_sample_efficiency_parallel.sh <game1> [game2] ...
# Env:
#   PARALLEL_JOBS=4   number of concurrent training processes (default 4)
#
# Skips runs whose training_complete.flag already exists.

set -u
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

PARALLEL_JOBS=${PARALLEL_JOBS:-4}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT/offline_training"

SCRIPT="train_dynamics_offline.py"
RESULTS_DIR="results/sample_efficiency"
mkdir -p "$RESULTS_DIR"

EPOCHS=80
ROLLOUT_LEN=50
EVAL_INTERVAL=5
MULTISTEP="--multistep_train_len 20 --multistep_stride 5"

GAMES=("$@")
if [ ${#GAMES[@]} -eq 0 ]; then
    echo "Usage: $0 <game1> [game2] ..."
    exit 1
fi

echo "Parallel jobs: $PARALLEL_JOBS"
echo "Games: ${GAMES[*]}"

# Bash semaphore via background-job slot waiting
wait_for_slot() {
    while [ "$(jobs -rp | wc -l)" -ge "$PARALLEL_JOBS" ]; do
        wait -n 2>/dev/null || sleep 1
    done
}

FAIL_COUNT=0
launch_run() {
    local GAME=$1
    local MODEL=$2
    local SPLIT=$3
    local SNAME=$4
    local TRAINING=$5
    local SEED=${6:-42}
    local ROLLOUT_OVERRIDE=${7:-}

    local SEED_SUFFIX=""
    if [ "$SEED" != "42" ]; then
        SEED_SUFFIX="_seed${SEED}"
    fi

    local RUN_NAME="${GAME}_${MODEL}_${SNAME}_${TRAINING}${SEED_SUFFIX}"
    if [ -n "$ROLLOUT_OVERRIDE" ]; then
        RUN_NAME="${GAME}_${MODEL}_${SNAME}_${TRAINING}_rl${ROLLOUT_OVERRIDE}"
    fi
    local SAVE_DIR="$RESULTS_DIR/${RUN_NAME}"

    if [ -f "$SAVE_DIR/training_complete.flag" ]; then
        echo ">>> SKIP $RUN_NAME (already complete)"
        return 0
    fi

    local RL=${ROLLOUT_OVERRIDE:-$ROLLOUT_LEN}
    local COMMON="--game $GAME --download --epochs $EPOCHS --rollout_len $RL --eval_interval $EVAL_INTERVAL --skip_verification --train_split $SPLIT --seed $SEED"

    local MODEL_ARGS=""
    case $MODEL in
        stu) MODEL_ARGS="--model_type stu --num_blocks 1 --dynamics_stu_num_filters 2" ;;
        baseline_only) MODEL_ARGS="--model_type baseline_only --num_blocks 1" ;;
        attention) MODEL_ARGS="--model_type attention --num_blocks 1 --attn_num_heads 4" ;;
    esac

    local TRAIN_ARGS=""
    if [ "$TRAINING" = "multistep" ]; then
        TRAIN_ARGS="$MULTISTEP"
    fi

    wait_for_slot
    echo ">>> [$(date +%H:%M:%S)] LAUNCH $RUN_NAME"
    (
        python $SCRIPT $COMMON $MODEL_ARGS $TRAIN_ARGS \
            --save_dir "$SAVE_DIR" \
            > "$RESULTS_DIR/${RUN_NAME}.log" 2>&1
        rc=$?
        if [ $rc -eq 0 ]; then
            touch "$SAVE_DIR/training_complete.flag"
            echo ">>> [$(date +%H:%M:%S)] DONE  $RUN_NAME"
        else
            echo ">>> [$(date +%H:%M:%S)] FAIL  $RUN_NAME (rc=$rc) — see ${RESULTS_DIR}/${RUN_NAME}.log"
            echo "$RUN_NAME rc=$rc" >> "$RESULTS_DIR/.failed_runs.$$"
        fi
    ) &
}

SPLITS=(0.1 0.2 0.4 0.8)
SPLIT_NAMES=(5ep 10ep 20ep 40ep)

for GAME in "${GAMES[@]}"; do
    echo ""
    echo "============================================"
    echo "  GAME: $GAME"
    echo "============================================"

    # Phase 1: Multi-step training
    for MODEL in stu baseline_only attention; do
        for i in "${!SPLITS[@]}"; do
            launch_run "$GAME" "$MODEL" "${SPLITS[$i]}" "${SPLIT_NAMES[$i]}" "multistep"
        done
    done

    # Phase 2: Single-step training
    for MODEL in stu baseline_only attention; do
        for i in "${!SPLITS[@]}"; do
            launch_run "$GAME" "$MODEL" "${SPLITS[$i]}" "${SPLIT_NAMES[$i]}" "singlestep"
        done
    done

    # Phase 3: Multi-seed validation (5ep multi-step)
    for MODEL in stu baseline_only; do
        for SEED in 123 456; do
            launch_run "$GAME" "$MODEL" "0.1" "5ep" "multistep" "$SEED"
        done
    done

    # Phase 4: Extended horizon (100-step rollout)
    for MODEL in stu baseline_only; do
        launch_run "$GAME" "$MODEL" "0.1" "5ep" "multistep" "42" "100"
    done
done

# Wait for all remaining background jobs
wait
echo ""
echo "============================================"
echo "  All sample-efficiency runs complete: $(date)"
echo "============================================"

# Surface failures so the parent pipeline can detect them via exit code.
FAIL_FILE="$RESULTS_DIR/.failed_runs.$$"
if [ -f "$FAIL_FILE" ]; then
    NFAIL=$(wc -l < "$FAIL_FILE")
    echo ""
    echo "FAILED RUNS ($NFAIL):"
    cat "$FAIL_FILE"
    rm -f "$FAIL_FILE"
    exit 1
fi
exit 0
