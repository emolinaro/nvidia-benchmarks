#!/bin/bash

set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

BASEDIR=${BASEDIR:-'../../../Datasets'}
DATASET=${DATASET:-coco}
MODE=${MODE:-benchmark-training} # for training: 'benchmark-training'; for inference: 'benchmark-inference'
BS=${BS:-32} # batch size
AMP=${AMP:-'--amp'} # if '--amp' use Tensor Cores for benchmark training/inference
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}
EPOCHS=${EPOCHS:-65} # by default, training is running for 65 epochs


# Get command line seed
seed=${1:-1}

# Get command line nubber of GPUs
num_gpus=${2:-1}

# Get mode: training or inference
mode=${3:-train}


DATASET_DIR=${BASEDIR}/${DATASET}

if [ -d ${DATASET_DIR} ]
then

    ## start timing
    start=$(date +%s)
    start_fmt=$(date +%Y-%m-%d\ %r)
    echo "STARTING TIMING RUN AT $start_fmt"
    echo ""

    ## run program 
    if [ $mode = "train" ]; then
        ## training benchmark
        python -m torch.distributed.launch --nproc_per_node=$num_gpus \
                main.py --batch-size $BS \
                        --mode benchmark-training \
                        --epochs $EPOCHS \
                        --benchmark-warmup 100 \
                        --benchmark-iterations 200 \
                        $AMP \
                        --data $DATASET_DIR \
                        --seed $seed 
    else
        ## inference benchmark on 1 GPU
        python -m torch.distributed.launch \
                main.py --eval-batch-size $BS \
                        --mode benchmark-inference \
                        --benchmark-warmup 100 \
                        --benchmark-iterations 200 \
                        $AMP \
                        --data $DATASET_DIR \
                        --seed $seed 
    
    fi

    ## end timing
    end=$(date +%s)
    end_fmt=$(date +%Y-%m-%d\ %r)
    echo ""
    echo "ENDING TIMING RUN AT $end_fmt"

    ## report result
    result=$(( $end - $start ))
    result_name="recommendation"
                      
    echo "RESULT,$result_name,$seed,$result,$start_fmt"

else
    
    echo "Directory ${DATASET_DIR} does not exist"

fi
