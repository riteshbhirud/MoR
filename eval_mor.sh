# !/bin/bash


datasets=("mag" "amazon" "prime")
# Define scorer_name mapping using an associative array
declare -A dataset_scorer_map=(
  [mag]="ada"
  [amazon]="ada"
  [prime]="contriever"
)

for dataset in "${datasets[@]}"; do
    # Get the corresponding scorer_name for the dataset
    scorer_name="${dataset_scorer_map[$dataset]}"
    echo "Processing dataset: $dataset with scorer: $scorer_name"
    python eval.py --dataset_name "$dataset" --scorer_name "$scorer_name" --mod "test" 

    cd Reranking
    python rerank.py --dataset_name "$dataset"
done
