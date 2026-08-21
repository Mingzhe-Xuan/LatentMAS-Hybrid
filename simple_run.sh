#!/bin/bash
###############################################################################
# simple_run.sh - standard experiment suite with Planner -> Judger only
#
# All experiment parameters and supported environment overrides come from
# run.sh. Only the Python entry point and output locations differ.
#
# Submit current config:  qsub simple_run.sh
# Submit full matrix:     qsub -v FULL_EXP=true simple_run.sh
# Local full matrix:      bash simple_run.sh --full_exp
###############################################################################

## PBS directives must be present in the submitted file (directives in a sourced
## script are not interpreted by qsub).
#PBS -N xmz-simple
#PBS -P ds_ccds_wei.lu
#PBS -q gpu_ded
#PBS -l walltime=72:00:00
#PBS -l select=1:ncpus=12:ngpus=1
#PBS -j oe

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

export RUN_SCRIPT="simple_run.py"
export STATE_FILE="${STATE_FILE:-simple_state.txt}"
export LOG_ROOT="${LOG_ROOT:-simple_logging}"
export RESULT_ROOT="${RESULT_ROOT:-simple_result}"

exec bash "${SCRIPT_DIR}/run.sh" "$@"
