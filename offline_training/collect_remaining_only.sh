#!/bin/bash
# collect_remaining_only.sh
# Collect + upload (no training) for the games that still need data.
# Uses the same conventions as run_remaining_games.sh so the resulting
# sentinels (collect.done, upload.done) are picked up cleanly when the
# main pipeline is relaunched.

set -u
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

HF_REPO_ID="Shivamkak/STUZero-Atari-Dynamics"

GAMES=(
    "freeway:110000:Freeway"
    "frostbite:100000:Frostbite"
    "gopher:100000:Gopher"
    "hero:60000:Hero"
    "jamesbond:90000:Jamesbond"
    "kangaroo:110000:Kangaroo"
    "krull:110000:Krull"
    "privateeye:110000:PrivateEye"
    "qbert:80000:Qbert"
    "upndown:60000:UpNDown"
)

mkdir -p offline_training/results/pipeline_logs
GLOBAL_LOG="offline_training/results/pipeline_logs/collect_only_$(date +%Y%m%d_%H%M%S).log"
echo "Global log: $GLOBAL_LOG"
exec > >(tee -a "$GLOBAL_LOG") 2>&1

echo "============================================"
echo "  Collect-only pipeline start: $(date)"
echo "  Games: ${#GAMES[@]}"
echo "============================================"

for entry in "${GAMES[@]}"; do
    SHORT="${entry%%:*}"
    rest="${entry#*:}"
    STEPS="${rest%%:*}"
    PROPER="${rest##*:}"
    STEPS_K=$((STEPS / 1000))
    SUBDIR="${SHORT}_${STEPS_K}K"
    DATA_DIR="offline_training/${SUBDIR}"
    MODEL_FILE="offline_training/atari_models/atari_${SHORT}_model_${STEPS}.p"
    HF_PATH="${SUBDIR}"
    HF_MODEL_NAME="atari_${SHORT}_model_${STEPS}.p"
    SENTINEL_DIR="offline_training/results/pipeline_logs/${SHORT}"
    mkdir -p "$SENTINEL_DIR"

    echo ""
    echo "############################################"
    echo "# COLLECT: $SHORT  ($PROPER)"
    echo "############################################"

    if [ ! -f "$MODEL_FILE" ]; then
        echo "ERROR: expert model missing: $MODEL_FILE"
        continue
    fi

    # ---- collect ----
    if [ -f "$SENTINEL_DIR/collect.done" ]; then
        echo ">>> [collect] SKIP (already done)"
    else
        echo ">>> [collect] $(date +%H:%M:%S) starting collection..."
        python data_collection_utils/collect_data.py \
            exp_config="ez/config/exp/atari_${SHORT}.yaml" \
            eval.model_path="$MODEL_FILE" \
            +collect.n_episodes=50 \
            +collect.max_steps=27000 \
            +collect.output_dir="$DATA_DIR" \
            2>&1 | tee "$SENTINEL_DIR/collect.log"
        rc=${PIPESTATUS[0]}
        if [ $rc -ne 0 ] || [ ! -f "$DATA_DIR/metadata.json" ]; then
            echo "ERROR: collect failed for $SHORT (rc=$rc)"
            continue
        fi
        touch "$SENTINEL_DIR/collect.done"
        echo ">>> [collect] DONE $(date +%H:%M:%S)"
    fi

    # ---- symlink expert into data dir for training later ----
    if [ ! -e "$DATA_DIR/$HF_MODEL_NAME" ]; then
        ln -sf "../atari_models/$HF_MODEL_NAME" "$DATA_DIR/$HF_MODEL_NAME"
        echo ">>> [setup] symlinked expert checkpoint into $DATA_DIR/"
    fi

    # ---- upload to HF ----
    if [ -f "$SENTINEL_DIR/upload.done" ]; then
        echo ">>> [upload] SKIP (already done)"
    else
        echo ">>> [upload] $(date +%H:%M:%S) uploading dataset + model..."
        python - <<PYEOF
from huggingface_hub import HfApi
api = HfApi()
api.upload_folder(
    folder_path="${DATA_DIR}",
    path_in_repo="${HF_PATH}",
    repo_id="${HF_REPO_ID}",
    repo_type="dataset",
)
api.upload_file(
    path_or_fileobj="${MODEL_FILE}",
    path_in_repo="${HF_PATH}/${HF_MODEL_NAME}",
    repo_id="${HF_REPO_ID}",
    repo_type="dataset",
)
print("Uploaded ${SHORT} -> ${HF_REPO_ID}/${HF_PATH}")
PYEOF
        rc=$?
        if [ $rc -ne 0 ]; then
            echo "ERROR: upload failed for $SHORT (rc=$rc)"
            continue
        fi
        touch "$SENTINEL_DIR/upload.done"
        echo ">>> [upload] DONE $(date +%H:%M:%S)"
    fi

    echo ">>> $SHORT collect+upload complete $(date +%H:%M:%S)"
done

echo ""
echo "============================================"
echo "  Collect-only pipeline finished: $(date)"
echo "============================================"
