#!/bin/sh

cd $(dirname $(dirname "$0")) || exit
ROOT_DIR=$(pwd)
PYTHON=python

TRAIN_CODE=${TRAIN_CODE:-train.py}
POINTCEPT_TRAIN_LAUNCHER=${POINTCEPT_TRAIN_LAUNCHER:-pointcept}

DATASET=scannet
CONFIG="None"
EXP_NAME=debug
WEIGHT="None"
RESUME=false
NUM_GPU=None
NUM_MACHINE=1
MACHINE_RANK=${MACHINE_RANK:-${SLURM_NODEID:-0}}
DIST_URL=${DIST_URL:-auto}


while getopts "p:d:c:n:w:g:m:r:" opt; do
  case $opt in
    p)
      PYTHON=$OPTARG
      ;;
    d)
      DATASET=$OPTARG
      ;;
    c)
      CONFIG=$OPTARG
      ;;
    n)
      EXP_NAME=$OPTARG
      ;;
    w)
      WEIGHT=$OPTARG
      ;;
    r)
      RESUME=$OPTARG
      ;;
    g)
      NUM_GPU=$OPTARG
      ;;
    m)
      NUM_MACHINE=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG"
      ;;
  esac
done

if [ "${NUM_GPU}" = 'None' ]
then
  NUM_GPU=`$PYTHON -c 'import torch; print(torch.cuda.device_count())'`
fi

echo "Experiment name: $EXP_NAME"
echo "Python interpreter dir: $PYTHON"
echo "Dataset: $DATASET"
echo "Config: $CONFIG"
echo "GPU Num: $NUM_GPU"
echo "Machine Num: $NUM_MACHINE"
echo "Machine Rank: $MACHINE_RANK"

if [ -n "${MASTER_ADDR:-}" ] && [ -n "${MASTER_PORT:-}" ]; then
  DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT
elif [ -n "$SLURM_NODELIST" ]; then
  MASTER_HOSTNAME=$(scontrol show hostname "$SLURM_NODELIST" | head -n 1)
  MASTER_ADDR=$(getent hosts "$MASTER_HOSTNAME" | awk '{ print $1 }')
  MASTER_PORT=$((10000 + 0x$(echo -n "${DATASET}/${EXP_NAME}" | md5sum | cut -c 1-4 | awk '{print $1}') % 20000))
  DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT
elif [ "$POINTCEPT_TRAIN_LAUNCHER" = "torchrun" ] && [ "$DIST_URL" = "auto" ]; then
  MASTER_ADDR=127.0.0.1
  MASTER_PORT=$((10000 + 0x$(echo -n "${DATASET}/${EXP_NAME}" | md5sum | cut -c 1-4 | awk '{print $1}') % 20000))
  DIST_URL=tcp://$MASTER_ADDR:$MASTER_PORT
fi

echo "Dist URL: $DIST_URL"

EXP_DIR=exp/${DATASET}/${EXP_NAME}
MODEL_DIR=${EXP_DIR}/model
CODE_DIR=${EXP_DIR}/code
CONFIG_DIR=configs/${DATASET}/${CONFIG}.py


echo " =========> CREATE EXP DIR <========="
echo "Experiment dir: $ROOT_DIR/$EXP_DIR"
if [ "${RESUME}" = true ] && [ -d "$EXP_DIR" ]
then
  CONFIG_DIR=${EXP_DIR}/config.py
  WEIGHT=$MODEL_DIR/model_last.pth
else
  RESUME=false
  mkdir -p "$MODEL_DIR" "$CODE_DIR"
  cp -r scripts tools pointcept "$CODE_DIR"
fi

echo "Loading config in:" $CONFIG_DIR
export PYTHONPATH=./$CODE_DIR
echo "Running code in: $CODE_DIR"


echo " =========> RUN TASK <========="
ulimit -n 65536

RUN_ARGS="--config-file $CONFIG_DIR"
RUN_OPTIONS="save_path=$EXP_DIR"
if [ "${WEIGHT}" != "None" ]; then
  RUN_OPTIONS="$RUN_OPTIONS resume=$RESUME weight=$WEIGHT"
fi

run_pointcept_launcher() {
  if [ "${WEIGHT}" = "None" ]
  then
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$NUM_GPU" \
    --num-machines "$NUM_MACHINE" \
    --machine-rank "$MACHINE_RANK" \
    --dist-url ${DIST_URL} \
    --options save_path="$EXP_DIR"
  else
    $PYTHON "$CODE_DIR"/tools/$TRAIN_CODE \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$NUM_GPU" \
    --num-machines "$NUM_MACHINE" \
    --machine-rank "$MACHINE_RANK" \
    --dist-url ${DIST_URL} \
    --options save_path="$EXP_DIR" resume="$RESUME" weight="$WEIGHT"
  fi
}

run_torchrun_launcher() {
  DIST_HOST=$(printf '%s' "$DIST_URL" | sed -n 's#^tcp://\([^:]*\):\([0-9][0-9]*\)$#\1#p')
  DIST_PORT=$(printf '%s' "$DIST_URL" | sed -n 's#^tcp://\([^:]*\):\([0-9][0-9]*\)$#\2#p')
  DIST_HOST=${DIST_HOST:-127.0.0.1}
  DIST_PORT=${DIST_PORT:-29500}
  $PYTHON -m torch.distributed.run \
    --nproc_per_node="$NUM_GPU" \
    --nnodes="$NUM_MACHINE" \
    --node_rank="$MACHINE_RANK" \
    --master_addr="$DIST_HOST" \
    --master_port="$DIST_PORT" \
    "$CODE_DIR"/tools/ddp_train.py \
    --config-file "$CONFIG_DIR" \
    --num-gpus "$NUM_GPU" \
    --num-machines "$NUM_MACHINE" \
    --machine-rank "$MACHINE_RANK" \
    --dist-url ${DIST_URL} \
    --options ${RUN_OPTIONS}
}

case "$POINTCEPT_TRAIN_LAUNCHER" in
  pointcept)
    run_pointcept_launcher
    ;;
  torchrun)
    run_torchrun_launcher
    ;;
  *)
    echo "Unknown POINTCEPT_TRAIN_LAUNCHER=$POINTCEPT_TRAIN_LAUNCHER" >&2
    exit 2
    ;;
esac
