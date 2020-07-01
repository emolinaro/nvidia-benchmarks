#!/bin/bash

set -e

# runs benchmark and reports time to convergence
# to use the script:
#   run_and_time.sh <random seed 1-5>

BASEDIR=${BASEDIR:-'../../../Datasets'}
DATASET=${DATASET:-'LibriSpeech'}
MODEL_CONFIG=${MODEL_CONFIG:-'configs/jasper10x5dr_sp_offline_specaugment.toml'}
RESULT_DIR=${RESULT_DIR:-'results'}
CREATE_LOGFILE=${CREATE_LOGFILE:-'true'}
CUDNN_BENCHMARK=${CUDNN_BENCHMARK:-'true'}
NUM_STEPS=${NUM_STEPS:-'-1'}
MAX_DURATION=${MAX_DURATION:-16.7}
BATCH_SIZE=${BATCH_SIZE:-64}
LEARNING_RATE=${LEARNING_RATE:-'0.015'}
GRADIENT_ACCUMULATION_STEPS=${GRADIENT_ACCUMULATION_STEPS:-1}
PRINT_FREQUENCY=${PRINT_FREQUENCY:-1}
USE_PROFILER=${USE_PROFILER:-'false'}
PRECISION=${PRECISION:-'fp16'}

PREC=""
if [ "$PRECISION" = "fp16" ] ; then
   PREC=" --fp16"
elif [ "$PRECISION" = "fp32" ] ; then
   PREC=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

STEPS=""
if [ "$NUM_STEPS" -ne "-1" ] ; then
   STEPS=" --num_steps=$NUM_STEPS"
elif [ "$NUM_STEPS" = "-1" ] ; then
   STEPS=""
else
   echo "Unknown <precision> argument"
   exit -2
fi

CUDNN=""
if [ "$CUDNN_BENCHMARK" = "true" ] ; then
   CUDNN=" --cudnn"
else
   CUDNN=""
fi

if [ "${USE_PROFILER}" = "true" ] ; then
    PYTHON_ARGS+="-m cProfile  -s cumtime"
fi


# Get command line seed
seed=${1:-0}

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

        CMD="train.py"
        CMD+=" --batch_size=$BATCH_SIZE"
        CMD+=" --num_epochs=400"
        CMD+=" --output_dir=$RESULT_DIR"
        CMD+=" --model_toml=$MODEL_CONFIG"
        CMD+=" --lr=$LEARNING_RATE"
        CMD+=" --seed=$seed"
        CMD+=" --optimizer=novograd"
        CMD+=" --gradient_accumulation_steps=$GRADIENT_ACCUMULATION_STEPS"
        CMD+=" --dataset_dir=$DATA_DIR"
        CMD+=" --val_manifest=$DATA_DIR/librispeech-dev-clean-wav.json"
        CMD+=" --train_manifest=$DATA_DIR/librispeech-train-clean-100-wav.json,$DATA_DIR/librispeech-train-clean-360-wav.json,$DATA_DIR/librispeech-train-other-500-wav.json"
        CMD+=" --weight_decay=1e-3"
        CMD+=" --save_freq=100000"
        CMD+=" --eval_freq=100000"
        CMD+=" --max_duration=$MAX_DURATION"
        CMD+=" --pad_to_max"
        CMD+=" --train_freq=$PRINT_FREQUENCY"
        CMD+=" --lr_decay "
        CMD+=" $CUDNN"
        CMD+=" $PREC"
        CMD+=" $STEPS"

        if [ "$num_gpus" -gt 1  ] ; then
           CMD="python3 -m torch.distributed.launch --nproc_per_node=$num_gpus $CMD"
        else
           CMD="python3  $CMD"
        fi

        if [ "$CREATE_LOGFILE" = "true" ] ; then
          export GBS=$(expr $BATCH_SIZE \* $num_gpus)
          printf -v TAG "jasper_train_benchmark_%s_gbs%d" "$PRECISION" $GBS
          DATESTAMP=`date +'%y%m%d%H%M%S'`
          LOGFILE="${RESULT_DIR}/${TAG}.${DATESTAMP}.log"
          printf "Logs written to %s\n" "$LOGFILE"

        fi

        if [ -z "$LOGFILE" ] ; then

           set -x
           $CMD
           set +x
        else

           set -x
           (
             $CMD
           ) |& tee "$LOGFILE"

           set +x

           mean_latency=`cat "$LOGFILE" | grep 'Step time' | awk '{print $3}'  | tail -n +2 | egrep -o '[0-9.]+'| awk 'BEGIN {total=0} {total+=$1} END {printf("%.2f\n",total/NR)}'`
           mean_throughput=`python -c "print($BATCH_SIZE*$num_gpus/${mean_latency})"`
           training_wer_per_pgu=`cat "$LOGFILE" | grep 'training_batch_WER'| awk '{print $2}'  | tail -n 1 | egrep -o '[0-9.]+'`
           training_loss_per_pgu=`cat "$LOGFILE" | grep 'Loss@Step'| awk '{print $4}'  | tail -n 1 | egrep -o '[0-9.]+'`
           final_eval_wer=`cat "$LOGFILE" | grep 'Evaluation WER'| tail -n 1 | egrep -o '[0-9.]+'`
           final_eval_loss=`cat "$LOGFILE" | grep 'Evaluation Loss'| tail -n 1 | egrep -o '[0-9.]+'`

           echo "max duration: $MAX_DURATION s" | tee -a "$LOGFILE"
           echo "mean_latency: $mean_latency s" | tee -a "$LOGFILE"
           echo "mean_throughput: $mean_throughput sequences/s" | tee -a "$LOGFILE"
           echo "training_wer_per_pgu: $training_wer_per_pgu" | tee -a "$LOGFILE"
           echo "training_loss_per_pgu: $training_loss_per_pgu" | tee -a "$LOGFILE"
           echo "final_eval_loss: $final_eval_loss" | tee -a "$LOGFILE"
           echo "final_eval_wer: $final_eval_wer" | tee -a "$LOGFILE"
        fi


    else
        ## inference benchmark on 1 GPU

        CMD=" python inference_benchmark.py"
        CMD+=" --batch_size=$BATCH_SIZE"
        CMD+=" --model_toml=$MODEL_CONFIG"
        CMD+=" --seed=$SEED"
        CMD+=" --dataset_dir=$DATA_DIR"
        CMD+=" --val_manifest $DATA_DIR/librispeech-${DATASET}-wav.json "
        CMD+=" --ckpt=$CHECKPOINT"
        CMD+=" --max_duration=$MAX_DURATION"
        CMD+=" --pad_to=-1"
        CMD+=" $CUDNN_BENCHMARK"
        CMD+=" $PREC"
        CMD+=" $STEPS"

        if [ "$CREATE_LOGFILE" = "true" ] ; then
          export GBS=$(expr $BATCH_SIZE )
          printf -v TAG "jasper_inference_benchmark_%s_gbs%d" "$PRECISION" $GBS
          DATESTAMP=`date +'%y%m%d%H%M%S'`
          LOGFILE="${RESULT_DIR}/${TAG}.${DATESTAMP}.log"
          printf "Logs written to %s\n" "$LOGFILE"
        fi

        set -x
        if [ -z "$LOGFILE" ] ; then
           $CMD
        else
           (
             $CMD
           ) |& tee "$LOGFILE"
           grep 'latency' "$LOGFILE"
        fi
        set +x

    fi

    ## end timing
    end=$(date +%s)
    end_fmt=$(date +%Y-%m-%d\ %r)
    echo ""
    echo "ENDING TIMING RUN AT $end_fmt"

    ## report result
    result=$(( $end - $start ))
    result_name="speech recognition"
                      
    echo "RESULT,$result_name,$seed,$result,$start_fmt"

else
    
    echo "Directory ${DATASET_DIR} does not exist"

fi
