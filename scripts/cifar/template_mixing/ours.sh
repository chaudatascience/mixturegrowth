#!/bin/bash

export PYTHONUNBUFFERED="True"

NUM_GPUS=$1
DATASET=$2
TAG=$3

array=($@)
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

SCHEDULE="30 60 90"
GAMMA="0.2 0.2 0.2"
DECAY=5e-4
BATCH_SIZE=128
DEPTH=28
WIDTH=10

case ${DATASET} in
cifar10)
  python main.py data --dataset ${DATASET} \
    --depth ${DEPTH} --wide ${WIDTH} \
    --cutout --tag ${TAG} \
    --batch_size ${BATCH_SIZE} \
    --decay ${DECAY} --schedule ${SCHEDULE} \
    --gammas ${GAMMA} --ngpu ${NUM_GPUS} \
    --reset_scheduler --no_wandb \
    ${EXTRA_ARGS}
  ;;
cifar100)
  python main.py data --dataset ${DATASET} \
    --depth ${DEPTH} --wide ${WIDTH} \
    --cutout --tag ${TAG} \
    --batch_size ${BATCH_SIZE} \
    --decay ${DECAY} --schedule ${SCHEDULE} \
    --gammas ${GAMMA} --ngpu ${NUM_GPUS} \
    --reset_scheduler --no_wandb \
    ${EXTRA_ARGS}
  ;;
*)
  echo "No dataset given"
  exit
  ;;
esac
