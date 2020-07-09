#!/bin/bash
# Copyright (c) 2018, NVIDIA CORPORATION. All rights reserved.
#8 GPUS x 4 batch/GPU

GPU=1
NAME='MRCNN_TRAIN'
GLOBAL_BATCH=32
RESULTS='./results'
LOGFILE="$RESULTS/joblog.log"
DTYPE=${DTYPE:-'--noamp'} 

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
    python mask_rcnn_main.py \
        --mode="train_and_eval" \
        --checkpoint="weights/resnet/resnet-nhwc-2018-10-14/model.ckpt-112602" \
        --eval_samples=5000 \
        --init_learning_rate=0.005 \
        --learning_rate_steps="240000,320000" \
        --model_dir="/results/" \
        --num_steps_per_eval=29568 \
        --total_steps=360000 \
        --train_batch_size=4 \
        --eval_batch_size=8 \
        --training_file_pattern="/data/train*.tfrecord" \
        --validation_file_pattern="/data/val*.tfrecord" \
        --val_json_file="/data/annotations/instances_val2017.json" \
        --noamp \
        --use_batched_nms \
        --xla \
        --nouse_custom_box_proposals_op
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
