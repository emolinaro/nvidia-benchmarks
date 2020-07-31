#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#8 GPUS x 4 batch/GPU

NAME='MRCNN_TRAIN'
BS=${BS:-2}
RESULTS='./results'
DATA_DIR=${DATA_DIR:-'./data_dir'}
MODEL_DIR=${MODEL_DIR:-'./weights'}
LOGFILE="$RESULTS/joblog.log"
AMP=${AMP:-''}  # it can be '--amp' for mixed precision 

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
    python scripts/benchmark_training.py \
           --gpus $num_gpus \
           --batch_size $BS \
           --data_dir $DATA_DIR \
           --model_dir $MODEL_DIR \
           --weights_dir $MODEL_DIR \
           $AMP \
           | tee $LOGFILE

else
    ## inference benchmark
    echo "TODO: Add inference command." 
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
