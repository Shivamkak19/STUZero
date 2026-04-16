#!/bin/bash
set -u
TS=$(date +%Y%m%d_%H%M%S)
LOGROOT=/home/sk3686/hazan/stu-dreamer-jax/logs/loginnode
mkdir -p "$LOGROOT"

# launch NAME NODE CUDA TASK SEED CONFIGS [EXTRA_FLAGS]
launch() {
  local NAME=$1 NODE=$2 CUDA=$3 TASK=$4 SEED=$5 CONFIGS=$6
  local EXTRA=${7:-}
  local LOGDIR="$HOME/logdir/dreamer/round2_${NAME}_${TS}_${NODE}"
  local OUT="$LOGROOT/${NAME}_${TS}_${NODE}.out"
  local PIDFILE="$LOGROOT/${NAME}_${TS}_${NODE}.pid"

  echo "Launching $NAME on $NODE GPU=$CUDA cfg=[$CONFIGS] extra=[$EXTRA]"
  ssh -o BatchMode=yes -o StrictHostKeyChecking=no "$NODE" bash -s <<REMOTE &
set -e
cd /home/sk3686/hazan/stu-dreamer-jax
mkdir -p '$LOGDIR'
export MUJOCO_GL=egl
export XLA_PYTHON_CLIENT_MEM_FRACTION=.80
export CUDA_VISIBLE_DEVICES=$CUDA
setsid bash -c '
  source /home/sk3686/hazan/stu-dreamer-jax/.venv/bin/activate
  exec python dreamerv3/main.py \
    --logdir "$LOGDIR" \
    --configs $CONFIGS \
    --task dmc_$TASK \
    --seed $SEED \
    --run.steps 5e5 $EXTRA
' </dev/null >'$OUT' 2>&1 &
echo \$! > '$PIDFILE'
sleep 0.3
REMOTE
}

launch walker_run_baseline_seed1    della-adele    0 walker_run  1 "dmc_proprio"
launch walker_run_stuplus_seed1     della-stellato 0 walker_run  1 "dmc_proprio stu_plus"
launch walker_walk_stuplus_fourier  della-mol      1 walker_walk 0 "dmc_proprio stu_plus_fourier"
launch walker_walk_baseline_12m     della-stellato 1 walker_walk 0 "dmc_proprio_12m" "--run.train_ratio 512"
launch walker_walk_stuplus_12m      della-mol      2 walker_walk 0 "dmc_proprio_12m stu_plus_12m_mods" "--run.train_ratio 512"
launch walker_run_matched_baseline  della-yasaman  0 walker_run  0 "dmc_proprio_matched"

wait
echo "All ssh launches returned. Logs: $LOGROOT"
