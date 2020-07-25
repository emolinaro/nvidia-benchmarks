#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#8 GPUS x 4 batch/GPU

RESULTS='./results'
LOGFILE="joblog.log"
BS=${BS:-16} # batch size
AMP=${AMP:-''}  # --fp16 for mixed precision 

num_gpus=${1:-1}

mode=${2:-train}


if [ $num_gpus == 1 ]; then
    CMD='python'
else
    CMD="python -m torch.distributed.launch --nproc_per_node=$num_gpus" 
fi

if ! [ -d "$RESULTS" ]; then mkdir $RESULTS; fi

## start timing
start=$(date +%s)
start_fmt=$(date +%Y-%m-%d\ %r)
echo "STARTING TIMING RUN AT $start_fmt"
echo ""

## run program

if [ $mode = "train" ]; then

    ## training benchmark
    $CMD    train.py \
            --config_file wt103_base.yaml \
            --config trainbench \
            --local_batch_size $BS \
            --max_step 40000 \
            --work_dir $RESULTS \
            --txtlog_file $LOGFILE \
            --save_all \
            ${AMP} 
else
    ## inference benchmark
    exit 0
fi

## end timing
end=$(date +%s)
end_fmt=$(date +%Y-%m-%d\ %r)
echo ""
echo "ENDING TIMING RUN AT $end_fmt"

## report result
result=$(( $end - $start ))
result_name="language modeling"

echo "RESULT,$result_name,$seed,$result,$start_fmt"
