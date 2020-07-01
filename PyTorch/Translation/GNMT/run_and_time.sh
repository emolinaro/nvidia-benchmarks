#!/bin/bash

set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

BASEDIR=${BASEDIR:-'../../../Datasets'}
DATASET=${DATASET:-wmt16_de_en}
MODE=${MODE:-benchmark-training} # for training: 'benchmark-training'; for inference: 'benchmark-inference'
BS=${BS:-1024} # batch size
AMP=${AMP:-'f16'} # if 'f16' use mixed precision and 'f32' use single precision
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}
EPOCHS=${EPOCHS:-6} # by default, training is running for 65 epochs


# Get command line seed
seed=${1:-2}

# Get command line number of GPUs
num_gpus=${2:-1}

# Get mode: training or inference
mode=${3:-train}
DATA_DIR=${BASEDIR}/${DATASET}

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
                train.py --train-global-batch-size $BS \
                         --dataset-dir $DATA_DIR
                         --math $AMP
                         --results-dir results \
                         --epochs $EPOCHS \
                         --seed $seed 
    else
        ## inference benchmark on 1 GPU
        python -m translate.py --input $DATA_DIR/newstest2014.en \
                               --reference $DATA_DIR/newstest2014.de \
                               --output /tmp/output \
                               --model results/gnmt/model_best.pth
    
    fi

    ## end timing
    end=$(date +%s)
    end_fmt=$(date +%Y-%m-%d\ %r)
    echo ""
    echo "ENDING TIMING RUN AT $end_fmt"

    ## report result
    result=$(( $end - $start ))
    result_name="translation"
                      
    echo "RESULT,$result_name,$seed,$result,$start_fmt"

else
    
    echo "Directory ${DATASET_DIR} does not exist"

fi
