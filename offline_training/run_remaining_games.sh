#!/bin/bash
# run_remaining_games.sh
# End-to-end pipeline for the remaining 19 Atari games:
#   1) collect dynamics dataset from expert checkpoint
#   2) upload dataset + model to HuggingFace under <game>_<steps>K/
#   3) run sample efficiency experiments
#
# Resumable: each step is skipped if its sentinel already exists.
#
# Usage:
#   bash offline_training/run_remaining_games.sh                # all 19 games
#   bash offline_training/run_remaining_games.sh breakout       # one game
#   bash offline_training/run_remaining_games.sh assault bankheist battlezone

set -u
export OMP_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO_ROOT"

HF_REPO_ID="Shivamkak/STUZero-Atari-Dynamics"

# game:steps:atari_game_name (atari_game_name = exact ALE casing)
ALL_GAMES=(
    "assault:90000:Assault"
    "bankheist:110000:BankHeist"
    "battlezone:40000:BattleZone"
    "amidar:110000:Amidar"
    "boxing:110000:Boxing"
    "choppercommand:50000:ChopperCommand"
    "crazyclimber:90000:CrazyClimber"
    "demonattack:110000:DemonAttack"
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
    # breakout last: 58 GB dataset, runs serially with PARALLEL_JOBS=1
    "breakout:110000:Breakout"
)

# Filter if user passed specific games
if [ $# -gt 0 ]; then
    REQUESTED=("$@")
    FILTERED=()
    for entry in "${ALL_GAMES[@]}"; do
        short="${entry%%:*}"
        for r in "${REQUESTED[@]}"; do
            if [ "$short" = "$r" ]; then
                FILTERED+=("$entry")
            fi
        done
    done
    GAMES=("${FILTERED[@]}")
else
    GAMES=("${ALL_GAMES[@]}")
fi

if [ ${#GAMES[@]} -eq 0 ]; then
    echo "No matching games. Valid: assault bankheist battlezone amidar boxing breakout choppercommand crazyclimber demonattack freeway frostbite gopher hero jamesbond kangaroo krull privateeye qbert upndown"
    exit 1
fi

mkdir -p offline_training/results/pipeline_logs
GLOBAL_LOG="offline_training/results/pipeline_logs/run_remaining_games_$(date +%Y%m%d_%H%M%S).log"
echo "Global log: $GLOBAL_LOG"
exec > >(tee -a "$GLOBAL_LOG") 2>&1

echo "============================================"
echo "  Pipeline start: $(date)"
echo "  Games: ${GAMES[*]}"
echo "============================================"

run_one_game() {
    local entry=$1
    local SHORT="${entry%%:*}"
    local rest="${entry#*:}"
    local STEPS="${rest%%:*}"
    local PROPER="${rest##*:}"
    local STEPS_K=$((STEPS / 1000))
    local SUBDIR="${SHORT}_${STEPS_K}K"
    local DATA_DIR="offline_training/${SUBDIR}"
    local MODEL_FILE="offline_training/atari_models/atari_${SHORT}_model_${STEPS}.p"
    local HF_PATH="${SUBDIR}"
    local HF_MODEL_NAME="atari_${SHORT}_model_${STEPS}.p"
    local SENTINEL_DIR="offline_training/results/pipeline_logs/${SHORT}"
    mkdir -p "$SENTINEL_DIR"

    echo ""
    echo "############################################"
    echo "# GAME: $SHORT  ($PROPER, ${STEPS} steps, action_space queried at runtime)"
    echo "# data_dir: $DATA_DIR"
    echo "# model:    $MODEL_FILE"
    echo "# HF path:  $HF_REPO_ID/$HF_PATH"
    echo "############################################"

    if [ ! -f "$MODEL_FILE" ]; then
        echo "ERROR: expert model missing: $MODEL_FILE"
        return 1
    fi


    # ---- Step 1: collect ----
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
        if [ $rc -ne 0 ]; then
            echo "ERROR: collect failed for $SHORT (rc=$rc)"
            return 1
        fi
        if [ ! -f "$DATA_DIR/metadata.json" ]; then
            echo "ERROR: $DATA_DIR/metadata.json missing after collection"
            return 1
        fi
        touch "$SENTINEL_DIR/collect.done"
        echo ">>> [collect] DONE $(date +%H:%M:%S)"
    fi

    # ---- Step 2: upload to HF ----
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
            return 1
        fi
        touch "$SENTINEL_DIR/upload.done"
        echo ">>> [upload] DONE $(date +%H:%M:%S)"
    fi

    # ---- Step 2.5: ensure expert checkpoint is reachable from data dir ----
    # train_dynamics_offline.py resolves the checkpoint relative to the data
    # dir (e.g. choppercommand_50K/atari_choppercommand_model_50000.p), but
    # the actual file lives in atari_models/. Symlink it now (after collect).
    if [ -d "$DATA_DIR" ] && [ ! -e "$DATA_DIR/$HF_MODEL_NAME" ]; then
        ln -sf "../atari_models/$HF_MODEL_NAME" "$DATA_DIR/$HF_MODEL_NAME"
        echo ">>> [setup] symlinked expert checkpoint into $DATA_DIR/"
    fi

    # ---- Step 3: sample efficiency ----
    if [ -f "$SENTINEL_DIR/sample_eff.done" ]; then
        echo ">>> [sample_eff] SKIP (already done)"
    else
        # Auto-pick PARALLEL_JOBS based on dataset size to avoid OOMs.
        # train_dynamics_offline.py loads ALL train+eval episodes into RAM,
        # so big datasets (e.g. breakout 58 GB) require fewer concurrent jobs.
        # Override by exporting PARALLEL_JOBS before invoking this script.
        if [ -z "${PARALLEL_JOBS:-}" ]; then
            DATA_GB=$(du -sBG "$DATA_DIR" 2>/dev/null | awk '{print $1}' | tr -d 'G')
            if [ -z "$DATA_GB" ]; then DATA_GB=0; fi
            if   [ "$DATA_GB" -ge 30 ]; then GAME_PARALLEL=1
            elif [ "$DATA_GB" -ge 10 ]; then GAME_PARALLEL=2
            else                             GAME_PARALLEL=4
            fi
            echo ">>> [sample_eff] dataset size ${DATA_GB} GB -> PARALLEL_JOBS=$GAME_PARALLEL"
        else
            GAME_PARALLEL=$PARALLEL_JOBS
            echo ">>> [sample_eff] PARALLEL_JOBS=$GAME_PARALLEL (override)"
        fi
        echo ">>> [sample_eff] $(date +%H:%M:%S) running sample efficiency suite..."
        PARALLEL_JOBS=$GAME_PARALLEL bash offline_training/run_sample_efficiency_parallel.sh "$SHORT" \
            2>&1 | tee "$SENTINEL_DIR/sample_eff.log"
        rc=${PIPESTATUS[0]}
        if [ $rc -ne 0 ]; then
            echo "ERROR: sample_eff failed for $SHORT (rc=$rc)"
            return 1
        fi
        touch "$SENTINEL_DIR/sample_eff.done"
        echo ">>> [sample_eff] DONE $(date +%H:%M:%S)"
    fi

    echo ">>> $SHORT FULLY COMPLETE $(date +%H:%M:%S)"
    return 0
}

FAILED=()
for entry in "${GAMES[@]}"; do
    if ! run_one_game "$entry"; then
        FAILED+=("${entry%%:*}")
    fi
done

echo ""
echo "============================================"
echo "  Pipeline finished: $(date)"
if [ ${#FAILED[@]} -gt 0 ]; then
    echo "  FAILED games: ${FAILED[*]}"
    exit 1
else
    echo "  All games completed successfully."
fi
echo "============================================"
