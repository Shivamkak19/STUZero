#!/bin/bash
# Launch 6 dreamer experiments across 6 della login nodes.
# 500k env steps each. Uses setsid to fully detach from ssh channel.

set -u
TS=$(date +%Y%m%d_%H%M%S)
LOGROOT=/home/sk3686/hazan/stu-dreamer-jax/logs/loginnode
mkdir -p "$LOGROOT"

launch() {
  local NODE=$1 CUDA=$2 TASK=$3 VARIANT=$4
  local NAME="${TASK}_${VARIANT}_${TS}"
  local CONFIGS="dmc_proprio"
  [ "$VARIANT" = "stuplus" ] && CONFIGS="dmc_proprio stu_plus"
  local LOGDIR="$HOME/logdir/dreamer/loginnode_${NAME}_${NODE}"
  local OUT="$LOGROOT/${NAME}_${NODE}.out"
  local PIDFILE="$LOGROOT/${NAME}_${NODE}.pid"

  echo "Launching $NAME on $NODE (GPU $CUDA)"
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
    --run.steps 5e5
' </dev/null >'$OUT' 2>&1 &
echo \$! > '$PIDFILE'
sleep 0.3
REMOTE
}

# della-gpu/rse/pli all killed us (arbiters). Use della-mol GPU2 — 80GB A100, fully idle, node already hosts another run.
launch della-mol       2 hopper_hop   baseline

wait
echo
echo "All ssh launches returned. Logs: $LOGROOT"
