#!/bin/bash

# Define datasets and mods
datasets=("prime")
mods=("test" "val")

# Define scorer_name mapping using an associative array
declare -A dataset_scorer_map=(
  [mag]="ada"
  [amazon]="ada"
  [prime]="contriever"
)

# Loop through datasets and mods
for dataset in "${datasets[@]}"; do
  # Get the corresponding scorer_name for the dataset
  scorer_name="${dataset_scorer_map[$dataset]}"

  for mod in "${mods[@]}"; do
    echo "Processing dataset: $dataset with mod: $mod and scorer: $scorer_name"
    python eval.py --dataset_name "$dataset" --scorer_name "$scorer_name" --mod "$mod"
  done
done
