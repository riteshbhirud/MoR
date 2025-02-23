# MoR

# Running the Evaluation and Reranking Script

## Installation
To set up the environment, you can install dependencies using Conda or pip:

### Using Conda
```bash
conda env create -f mor_env.yml
conda activate your_env_name  # Replace with actual environment name
```

### Using pip
```bash
pip install -r requirements.txt
```

## Inference
To run the inference script, execute the following command in the terminal:

```bash
bash eval_mor.sh
```

This script will automatically process three datasets using the pre-trained planning graph generator and the pre-trained reranker.

## Training
### Training the planning graph generator 

```bash
bash train_planner.sh
```

### Training the reranker

```bash
bash train_reranker.sh
```


