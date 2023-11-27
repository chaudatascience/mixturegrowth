#!/bin/bash

export PYTHONUNBUFFERED="True"

NUM_GPUS=$1
DATASET=$2
TAG=$3

array=( $@ )
len=${#array[@]}
EXTRA_ARGS=${array[@]:3:$len}

SCHEDULER_TYPE="cosine"


DECAY=0.0001
BATCH_SIZE=512
DEPTH=56
WIDTH=4

case ${DATASET} in
    imagenet)
            python main.py /projectnb/ivc-ml/dbash/data/imagenet/ILSVRC/Data/CLS-LOC --dataset ${DATASET} \
            --arch wrn_imagenet \
            --depth ${DEPTH} --wide ${WIDTH} \
            --tag ${TAG} \
            --batch_size ${BATCH_SIZE} \
            --decay ${DECAY} --scheduler_type ${SCHEDULER_TYPE} \
            --no_nesterov \
            --ngpu ${NUM_GPUS} \
            ${EXTRA_ARGS}
        ;;
    *)
        echo "No dataset given"
        exit
        ;;
esac

