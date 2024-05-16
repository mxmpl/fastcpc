#!/bin/bash
#SBATCH --job-name=fastcpc
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1     # crucial - only 1 task per node
#SBATCH --cpus-per-task=40
#SBATCH --gres=gpu:4
#SBATCH --time=20:00:00
#SBATCH --output=%x-%j.out

echo "START TIME: $(date)"
set -e

module purge
eval "$(micromamba shell hook --shell bash)"
micromamba activate fastcpc

GPUS_PER_NODE=4
MASTER_PORT=6000
MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
NUM_PROCESSES=$(($SLURM_NNODES * $GPUS_PER_NODE))
THIS_SCRIPT_PATH=$(scontrol show job $SLURM_JOBID | awk -F= '/Command=/{print $2}' | awk '{print $1}')
ACCELERATE_CONFIG_FILE=$(dirname $THIS_SCRIPT_PATH)/accelerate.yaml

export TRITON_CACHE_DIR=$HOME/.cache/triton # Important - avoids OOM error when /tmp is full
#export CUDA_LAUNCH_BLOCKING=1 # Uncomment this for debugging
#export NCCL_DEBUG=INFO # Uncomment this for debugging

# note `\$SLURM_PROCID` we don't want it interpolated till `srun` since otherwise all nodes will get
# 0 and the launcher will hang
LAUNCHER="python -u -m accelerate.commands.launch \
    --rdzv_conf "rdzv_backend=c10d,rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT" \
    --config_file $ACCELERATE_CONFIG_FILE \
    --num_processes $NUM_PROCESSES \
    --num_machines $SLURM_NNODES \
    --main_process_ip $MASTER_ADDR \
    --main_process_port $MASTER_PORT \
    --machine_rank \$SLURM_PROCID \
    --role \$(hostname -s): --tee 3 \
    --no_python \
    "

SRUN_ARGS=" \
    --wait=60 \
    --kill-on-bad-exit=1 \
    --jobid $SLURM_JOB_ID \
    "

if [ -z "${5}" ]; then
    PROJECT="cpc"
else
    PROJECT="${5}"
fi

# bash -c is needed for the delayed interpolation of env vars to work
srun $SRUN_ARGS bash -c "$LAUNCHER fastcpc train $1 $2 $3 $4 --project $PROJECT"
echo "END TIME: $(date)"

