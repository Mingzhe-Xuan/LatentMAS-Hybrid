#!/bin/bash
#PBS -N x_aime25
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -j oe

export TASK="aime2025"
export FULL_EXP=true
export TASK_ONLY=true
export STATE_FILE="state_aime2025.txt"
exec bash run.sh
