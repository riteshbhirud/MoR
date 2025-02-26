#!/bin/bash

# Navigate to the Reranking directory
cd Reranking

# Run the training script
# amazon    
python train_eval_path_amazon.py
# # mag
python train_eval_path_mag.py
# prime
python train_eval_path_prime.py

