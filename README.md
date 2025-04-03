# MoR
This repository is an official PyTorch implementation of MoR in [Mixture of Structural-and-Textual Retrieval over Text-rich Graph Knowledge Bases](https://arxiv.org/pdf/2502.20317)

[![](https://img.shields.io/badge/Huggingface_Learderboard-online-yellow?style=plastic&logo=Hugging%20face)](https://huggingface.co/GagaLey/MoR)
[![](https://img.shields.io/badge/Arxiv-paper-red?style=plastic&logo=arxiv)](https://arxiv.org/pdf/2502.20317)

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

### Checkpoints and embeddings download
Before running the inference, please go to https://drive.google.com/drive/folders/1ldOYiyrIaZ3AVAKAmNeP0ZWfD3DLZu9D?usp=drive_link

(1) download the "checkpoints" and put it under the directory MoR/Planning/

(2) download the "data" and put it under the directory MoR/Reasoning/

(2) download the "model_checkpoint" and put it under the directory MoR/Reasoning/text_retrievers/


## Inference
To run the inference script, execute the following command in the terminal:

```bash
bash eval_mor.sh
```

This script will automatically process three datasets using the pre-trained planning graph generator and the pre-trained reranker.

## Training (Train MoR from Scratch)
### Step1: Training the planning graph generator 

```bash
bash train_planner.sh
```

### Step2: Train mixed traversal to collect candidates (note: there is no training process for reasoning)

```bash
bash run_reasoning.sh
```

### Step3: Training the reranker

```bash
bash train_reranker.sh
```

## Generating training data of Planner
### We provide codes to generate your own training data to finetune the Planner by using different LLMs.
#### If you are using Azure API

```bash
python get_llm_data.py --model "model_name" \
  --dataset_name "dataset_name" \
  --azure_api_key "your_azure_key" \
  --azure_endpoint "your_azure_endpoint" \
  --azure_api_version "your_azure_version"

```

#### If you are using OpenAI API

```bash
python get_llm_data.py --model "model_name" \
  --dataset_name "dataset_name" \
  --openai_api_key "your_openai_key" \
  --openai_endpoint "your_openai_endpoint"

```
