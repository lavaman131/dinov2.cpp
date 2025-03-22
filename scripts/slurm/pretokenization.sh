#!/bin/bash -l

#$ -P dl4ds
#$ -l h_rt=12:00:00
#$ -l gpus=1
#$ -N pretokenization
#$ -pe omp 4
#$ -j y # Merge the error and output streams into a single file

module load cmake ffmpeg gcc/10.2.0 llvm/9.0.1 miniconda openmpi cuda/12.2

conda activate eff-cv

# Keep track of information related to the current job
echo "=========================================================="
echo "Start date : $(date)"
echo "Job name : $JOB_NAME"
echo "Job ID : $JOB_ID"
echo "Host : $host"
echo "=========================================================="

INPUT_DIR=/projectnb/dl4ds/materials/datasets/imagenet
OUTPUT_DIR=/projectnb/dl4ds/materials/datasets/imagenet-tokenized
 
python ../train/pretokenization.py --cached_path $OUTPUT_DIR --data_path $INPUT_DIR
