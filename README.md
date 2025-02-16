# MoR

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

# Running the Evaluation and Reranking Script

## Usage
To run the script, execute the following command in the terminal:

```bash
bash eval_mor.sh
```

This script will automatically process predefined datasets and scorers.


---
For further details, please check the documentation or raise an issue in the repository.

