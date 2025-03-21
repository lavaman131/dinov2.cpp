#!/bin/bash -l

#$ -P dl4ds
#$ -l h_rt=72:00:00
#$ -N convert-imagenet-to-webdataset
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

OUTPUT_DIR=/projectnb/dl4ds/materials/datasets/imagenet
 
python ./scripts/data/convert_imagenet_to_wds.py --output_dir $OUTPUT_DIR