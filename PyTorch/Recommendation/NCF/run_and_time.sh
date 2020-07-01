#!/bin/bash

set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

THRESHOLD=1.0
BASEDIR=${BASEDIR:-'../../../Datasets'}
DATASET=${DATASET:-ml-20m}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-checkpoints}
EPOCHS=${EPOCHS:-1000}
LOCAL_RANK=${LOCAL_RANK:-0}


# Get command line seed
seed=${1:-2}

# Get command line nubber of GPUs
num_gpus=${2:-1}

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
    python -m torch.distributed.launch --nproc_per_node=${num_gpus} --use_env ncf.py --seed ${seed} --data ${DATASET_DIR} --mode $mode --epochs ${EPOCHS} --checkpoint_dir ${CHECKPOINT_DIR}

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
