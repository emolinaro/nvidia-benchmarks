#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#8 GPUS x 4 batch/GPU

RESULTS='./results'
EPOCHS=1501
#DATA='LJSpeech-1.1'
LOGFILE="$RESULTS/joblog.log"
AMP=${AMP:-'FP32'}  # --amp for mixed precision 

num_gpus=${1:-1}

mode=${2:-train}


if [ $num_gpus == 1 ]; then
    CMD='python'
else
    CMD='python -m multiproc'
fi

if [ $AMP == "--amp" ]; then
    BS=104
    AF=0.3
else
    BS=48
    AF=0.1
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
            -m Tacotron2 \
            -o $RESULTS \
            -lr 1e-3 \
            --epochs $EPOCHS \
            -bs $BS \
            --weight-decay 1e-6 \
            --grad-clip-thresh 1.0 \
            --cudnn-enabled \
            --log-file nvlog.json \
            --load-mel-from-disk \
            --training-files=filelists/ljs_mel_text_train_subset_2500_filelist.txt \
            --validation-files=filelists/ljs_mel_text_val_filelist.txt \
            --anneal-steps 500 1000 1500 \
            --anneal-factor $AF \
            $AMP \
            | tee $LOGFILE

#    python -m multiproc train.py \
#           -m Tacotron2 \
#           -o $RESULTS \
#           -lr 1e-3 \
#           --epochs $EPOCHS \
#           -bs 48 \
#           --weight-decay 1e-6 \
#           --grad-clip-thresh 1.0 \
#           --cudnn-enabled \
#           --log-file nvlog.json \
#           --anneal-steps 500 1000 1500 \
#           --anneal-factor 0.1 \
#           | tee $LOGFILE


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
result_name="segmentation"

echo "RESULT,$result_name,$seed,$result,$start_fmt"
