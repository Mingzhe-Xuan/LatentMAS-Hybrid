#!/bin/bash
#PBS -N x_gsm8k
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -j oe

export TASK="gsm8k"
export FULL_EXP=true
export TASK_ONLY=true
export STATE_FILE="state_gsm8k.txt"
exec bash run.sh
