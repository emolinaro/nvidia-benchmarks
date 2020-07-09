#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#8 GPUS x 4 batch/GPU

BASEDIR=${BASEDIR:-'../../../Datasets'}
DATASET=${DATASET:-UNet_data}
DATADIR=${BASEDIR}/${DATASET}
BS=${BS:-8}
RESULTS='./results'
LOGFILE="$RESULTS/joblog.log"
CHECKPOINTS=${CHECKPOINTS:-'./checkpoints'}
AMP=${AMP:-''} # --use_amp for mixed precision

num_gpus=${1:-1}

mode=${2:-train}

if ! [ -d "$RESULTS" ]; then mkdir $RESULTS; fi

## start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"
echo ""

## run program

if [ $mode = "train" ]; then

    ## training benchmark
    horovodrun -np $num_gpus  python main.py \
               --data_dir $DATADIR \
               --model_dir $CHECKPOINTS \
               --batch_size $BS \
               --exec_mode train \
               --augment \
               --benchmark \
               --warmup_steps 200 \
               --max_steps 1000 \
               --use_xla \
               $AMP \
               | tee $LOGFILE

else
    ## inference benchmark
    horovodrun -np 1 python main.py \
               --data_dir $DATADIR \
               --model_dir $CHECKPOINTS \
               --batch_size $BS \
               --exec_mode predict \
               --benchmark \
               --warmup_steps 200 \
               --max_steps 600 \
               --xla \
               $AMP \
               | tee $LOGFILE

fi

## end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo ""
echo "ENDING TIMING RUN AT $end_fmt"

## report result
result=$(( $end - $start ))
result_name="segmentation"

echo "RESULT,$result_name,$seed,$result,$start_fmt"
