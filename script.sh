#!/bin/bash

echo "Scaffolding Retail MLOps Pipeline..."

# 1. Create Directory Structure
mkdir -p data/raw
mkdir -p data/processed
mkdir -p data/slices
mkdir -p src
mkdir -p notebooks
mkdir -p artifacts

# 2. Create the modular Python files specified in Chapter 3.11
touch src/__init__.py
touch src/data_ingest.py
touch src/preprocess.py
touch src/features.py
touch src/train.py
touch src/drift.py
touch src/retrain.py
touch src/mlflow_utils.py
touch src/evaluate.py

# 3. Create requirements and environment files
cat <<EOT > requirements.txt
pandas>=1.5.0
numpy>=1.23.0
scikit-learn>=1.2.0
mlflow>=2.0.0
scipy>=1.10.0
jupyterlab>=3.0.0
matplotlib>=3.6.0
EOT

touch README.md
touch .gitignore

echo "Project scaffolded successfully!"
echo "Directories created: data/, src/, notebooks/, artifacts/"
echo "Modules created in src/: data_ingest.py, preprocess.py, features.py, train.py, drift.py, retrain.py, mlflow_utils.py, evaluate.py"