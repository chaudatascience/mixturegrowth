#!/bin/bash

export PYTHONUNBUFFERED="True"

NUM_GPUS=$1
DATASET=$2
TAG=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

ARCH="normalwrn"
DECAY=5e-4
BATCH_SIZE=128
DEPTH=28
WIDTH=10


case ${DATASET} in
    cifar10)
	    python main.py data --dataset ${DATASET} \
	    --arch ${ARCH} \
	    --growth_epochs -1 \
	    --depth ${DEPTH} --wide ${WIDTH} \
	    --cutout --tag ${TAG} \
	    --batch_size ${BATCH_SIZE} \
	    --decay ${DECAY} \
	    --ngpu ${NUM_GPUS} \
	    ${EXTRA_ARGS}
	;;
    cifar100)
      python main.py data --dataset ${DATASET} \
      --arch ${ARCH} \
      --growth_epochs -1 \
      --depth ${DEPTH} --wide ${WIDTH} \
      --cutout --tag ${TAG} \
      --batch_size ${BATCH_SIZE} \
      --decay ${DECAY} \
      --ngpu ${NUM_GPUS} \
      ${EXTRA_ARGS}
        ;;

    *)
        echo "No dataset given"
        exit
        ;;
esac

