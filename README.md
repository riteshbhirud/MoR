# MoR

# Project Title

## Installation
To install the required environment, use the `mor_env.yml` file:

```bash
conda env create -f mor_env.yml
conda activate your_env_name  # Replace with the actual environment name
```

Alternatively, you can install dependencies using `requirements.txt`. First, create a virtual environment (optional but recommended) and install the necessary packages:

```bash
python -m venv venv  # Create a virtual environment
source venv/bin/activate  # Activate on macOS/Linux
venv\Scripts\activate  # Activate on Windows

pip install -r requirements.txt  # Install dependencies
```

## How to Run the Program

### Step 1: Collect Traversed Candidates
Run `eval.py` to collect the traversed candidates:

```bash
python eval.py
```

This step processes the data and extracts candidate results for further evaluation.

### Step 2: Rerank and Evaluate Performance
Navigate to the `Router` directory and execute `rerank.py`:

```bash
cd Router
python rerank.py
```

This script reranks the collected candidates and evaluates the performance of the model.

## Configuring the Dataset Name
To change the dataset, modify the `dataset_name` parameter in `rerank.py`. Locate the following line in `rerank.py`:

```python
args.dataset_name = "your_dataset_name"
```
Replace `"your_dataset_name"` with the desired dataset name before running the script.

---
For further details, please check the documentation or raise an issue in the repository.

