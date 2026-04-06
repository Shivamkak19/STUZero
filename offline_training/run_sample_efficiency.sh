#!/bin/bash
# Portable sample efficiency experiment runner
# Usage: bash offline_training/run_sample_efficiency.sh <game1> [game2] [game3] ...
# Example: bash offline_training/run_sample_efficiency.sh assault bankheist battlezone
#
# Runs from repo root. Downloads data from HuggingFace if not present locally.

set -e
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=0

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
    echo "Available games: pong asterix alien seaquest kungfumaster mspacman roadrunner assault bankheist battlezone"
    exit 1
fi

run_experiment() {
    local GAME=$1
    local MODEL=$2
    local SPLIT=$3
    local SNAME=$4
    local TRAINING=$5
    local SEED=${6:-42}

    local SEED_SUFFIX=""
    if [ "$SEED" != "42" ]; then
        SEED_SUFFIX="_seed${SEED}"
    fi

    local RUN_NAME="${GAME}_${MODEL}_${SNAME}_${TRAINING}${SEED_SUFFIX}"
    local SAVE_DIR="$RESULTS_DIR/${RUN_NAME}"

    if [ -f "$SAVE_DIR/training_complete.flag" ]; then
        echo ">>> [$(date +%H:%M:%S)] SKIP $RUN_NAME (already complete)"
        return 0
    fi

    local COMMON="--game $GAME --download --epochs $EPOCHS --rollout_len $ROLLOUT_LEN --eval_interval $EVAL_INTERVAL --skip_verification --train_split $SPLIT --seed $SEED"

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

    echo ""
    echo ">>> [$(date +%H:%M:%S)] $RUN_NAME"
    python $SCRIPT $COMMON $MODEL_ARGS $TRAIN_ARGS \
        --save_dir "$SAVE_DIR" \
        2>&1 | tee "$RESULTS_DIR/${RUN_NAME}.log"

    touch "$SAVE_DIR/training_complete.flag"
}

SPLITS=(0.1 0.2 0.4 0.8)
SPLIT_NAMES=(5ep 10ep 20ep 40ep)

for GAME in "${GAMES[@]}"; do
    echo ""
    echo "============================================"
    echo "  GAME: $GAME"
    echo "============================================"

    # Phase 1: Multi-step training
    echo "=== Phase 1: Multi-step training ==="
    for MODEL in stu baseline_only attention; do
        for i in "${!SPLITS[@]}"; do
            run_experiment "$GAME" "$MODEL" "${SPLITS[$i]}" "${SPLIT_NAMES[$i]}" "multistep"
        done
    done

    # Phase 2: Single-step training
    echo "=== Phase 2: Single-step training ==="
    for MODEL in stu baseline_only attention; do
        for i in "${!SPLITS[@]}"; do
            run_experiment "$GAME" "$MODEL" "${SPLITS[$i]}" "${SPLIT_NAMES[$i]}" "singlestep"
        done
    done

    # Phase 3: Multi-seed validation (5ep multi-step)
    echo "=== Phase 3: Multi-seed (5ep, multi-step) ==="
    for MODEL in stu baseline_only; do
        for SEED in 123 456; do
            run_experiment "$GAME" "$MODEL" "0.1" "5ep" "multistep" "$SEED"
        done
    done
done

# Phase 4: Extended horizons (100-step rollout, 5ep multi-step)
echo "=== Phase 4: Extended horizons (100-step) ==="
for GAME in "${GAMES[@]}"; do
    for MODEL in stu baseline_only; do
        RUN_NAME="${GAME}_${MODEL}_5ep_multistep_rl100"
        SAVE_DIR="$RESULTS_DIR/${RUN_NAME}"

        if [ -f "$SAVE_DIR/training_complete.flag" ]; then
            echo ">>> [$(date +%H:%M:%S)] SKIP $RUN_NAME (already complete)"
            continue
        fi

        echo ">>> [$(date +%H:%M:%S)] $RUN_NAME"
        MODEL_ARGS=""
        case $MODEL in
            stu) MODEL_ARGS="--model_type stu --num_blocks 1 --dynamics_stu_num_filters 2" ;;
            baseline_only) MODEL_ARGS="--model_type baseline_only --num_blocks 1" ;;
        esac

        python $SCRIPT \
            --game $GAME --download --epochs $EPOCHS --rollout_len 100 --eval_interval $EVAL_INTERVAL \
            --skip_verification --train_split 0.1 --seed 42 \
            $MODEL_ARGS $MULTISTEP \
            --save_dir "$SAVE_DIR" \
            2>&1 | tee "$RESULTS_DIR/${RUN_NAME}.log"

        touch "$SAVE_DIR/training_complete.flag"
    done
done

echo ""
echo "============================================"
echo "  All experiments complete! $(date)"
echo "============================================"
