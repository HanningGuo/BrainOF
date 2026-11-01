#!/bin/bash
#SBATCH --account=brainof
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks-per-node=1
#SBATCH --partition=booster
#SBATCH --cpus-per-task=48
#SBATCH --output=/path/to/results/CamCAN_slurm-%j.out
#SBATCH --error=/path/to/results/CamCAN_slurm-%j.err
#SBATCH --time=03:00:00

# Propagate the specified number of CPUs per task to each `srun`.
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK
export OMP_NUM_THREADS=12

# so processes know who to talk to
export MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
# Allow communication over InfiniBand cells on JSC machines.
# MASTER_ADDR="$MASTER_ADDR"i
MASTER_ADDR="$(hostname -f)"

export MASTER_PORT=29500

# Prevent NCCL not figuring out how to initialize.
export NCCL_SOCKET_IFNAME=ib0
# Prevent Gloo not being able to communicate.
export GLOO_SOCKET_IFNAME=ib0

# activate the environment
source /path/to/miniforge3/bin/activate brainof

srun torchrun --nnodes=$SLURM_JOB_NUM_NODES --nproc_per_node=gpu \
    --rdzv_id=$SLURM_JOB_ID --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    /path/to/BrainOF/main_finetuning_multimodal.py \
    --cfg /path/to/BrainOF/configs/BrainOF_Base_CamCAN_multimodal.yaml \