#!/bin/bash
# Usage:  bash zero_score_gens.sh -d $TASK_NAME

DATA_NAME="unk"

# Parse named arguments
while getopts ":d:" opt; do
  case $opt in
    d) DATA_NAME="$OPTARG"
    ;;
    \?) echo "Invalid option -$OPTARG" >&2
    ;;
  esac
done


if [ $DATA_NAME == "zebra_grid" ]; then
    # result_dirs/zebra-grid.summary.md
    python -m src.evaluation.zebra_grid_eval    
elif [ $DATA_NAME == "hendrycks-math" ]; then
    # result_dirs/zebra-grid.summary.md
    python -m src.evaluation.hendrycks_math_eval
elif [ $DATA_NAME == "mmlu-pro" ]; then
    # result_dirs/zebra-grid.summary.md
    python -m src.evaluation.mmlu_pro_eval mmlu-pro
elif [ $DATA_NAME == "mmlu-pro-short" ]; then
    # result_dirs/zebra-grid.summary.md
    python -m src.evaluation.mmlu_pro_eval mmlu-pro-short
else
    echo "Can't get accuracy scores for dataset. Invalid dataset_name: '$DATA_NAME'"
    exit 1
fi

echo "All evaluations completed!"