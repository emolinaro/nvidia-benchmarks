#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#8 GPUS x 4 batch/GPU

GPU=1
NAME='MRCNN_TRAIN'
CONFIG='configs/e2e_mask_rcnn_R_50_FPN_1x.yaml'
GLOBAL_BATCH=32
RESULTS='./results'
LOGFILE="$RESULTS/joblog.log"
DTYPE=${DTYPE:-'fp16'}  # it can be 'fp16' or 'fp32' 

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
    python -m torch.distributed.launch --nproc_per_node=$num_gpus tools/train_net.py \
            --config-file $CONFIG \
            --skip-test \
            DATASETS.TRAIN "(\"coco_2014_train\", \"coco_2014_valminusminival\")" \
            DATASETS.TEST "(\"coco_2014_val\",)" \
            SOLVER.BASE_LR 0.04 \
            SOLVER.MAX_ITER 3665 \
            SOLVER.STEPS "(30000, 40000)" \
            SOLVER.IMS_PER_BATCH $GLOBAL_BATCH \
            DTYPE "$DTYPE" \
            OUTPUT_DIR $RESULTS \
            | tee $LOGFILE
            
    time=`cat $LOGFILE | grep -F 'maskrcnn_benchmark.trainer INFO: Total training time' | tail -n 1 | awk -F'(' '{print $2}' | awk -F' s ' '{print $1}' | egrep -o [0-9.]+`
    statement=`cat $LOGFILE | grep -F 'maskrcnn_benchmark.trainer INFO: Total training time' | tail -n 1`
    calc=$(echo $time 1.0 $GLOBAL_BATCH | awk '{ printf "%f", $2 * $3 / $1 }')
    echo "Training perf is: "$calc" FPS"

else
    ## inference benchmark
    python3 -m torch.distributed.launch --nproc_per_node=1 tools/test_net.py \
        --config-file $CONFIG \
        --skip-eval \
        DATASETS.TEST "(\"coco_2014_minival\",)" \
        DTYPE "$DTYPE" \
        OUTPUT_DIR $FOLDER \
        TEST.IMS_PER_BATCH 1 \
        | tee $LOGFILE

    time=`cat $LOGFILE | grep -F 'maskrcnn_benchmark.inference INFO: Total inference time' | tail -n 1 | awk -F'(' '{print $2}' | awk -F' s ' '{print $1}' | egrep -o [0-9.]+`
    calc=$(echo $time 1.0 | awk '{ printf "%f", $2 / $1 }')
    echo "Inference perf is: "$calc" FPS"

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
