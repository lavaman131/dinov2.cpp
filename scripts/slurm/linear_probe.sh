#!/bin/bash -l

#$ -P dl4ds
#$ -l h_rt=48:00:00
#$ -l gpus=2
#$ -l gpu_memory=40G
#$ -N pretokenization
#$ -pe omp 4
#$ -j y # Merge the error and output streams into a single file

module load cmake ffmpeg gcc/10.2.0 llvm/9.0.1 miniconda openmpi cuda/11.8

conda activate eff-cv

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID"
echo "Host : $host"
echo "=========================================================="

NCCL_SOCKET_IFNAME=ib # use all infiniband interfaces
MASTER_ADDR="localhost"
MASTER_PORT="9999"
RDZV_ID=$JOB_ID # job id
RDZV_BACKEND="c10d"
RDZV_ENDPOINT="$MASTER_ADDR:$MASTER_PORT"
NODE_RANK=0 # rank of the node

