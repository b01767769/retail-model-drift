# retail-model-drift
This repo contains the code or appendix required for my project dissertation
-----

# 🛒 Lightweight MLOps for Retail Concept Drift

[](https://python.org)
[](https://mlflow.org/)
[](https://scikit-learn.org/)
[](https://opensource.org/licenses/MIT)

## 📖 Project Overview

Retail predictive models frequently degrade over time due to shifts in customer behavior (seasonality, macroeconomic changes, promotions). Traditional "build-and-deploy" workflows assume static data, leading to hidden technical debt and silent model failures.

This project implements a **lightweight, auditable MLOps pipeline** designed to detect and manage concept drift in retail models. By combining interpretable Recency-Frequency-Monetary (RFM) features with distributional monitoring and a conservative automated retraining loop, this system ensures models remain accurate over time without unnecessary churn.

### To RUN A DEMO CLICK ON THE LINK BELOW TO DEMO THE PROJECT IN COLLAB 

https://colab.research.google.com/github/b01767769/retail-model-drift/blob/main/notebooks/orchestrator.ipynb

### 🎯 Key Features

  * **Chronological Evaluation:** Uses fixed time-slices to simulate real-world production and prevent temporal leakage.
  * **Distributional Monitoring (PSI):** Implements the Population Stability Index with fixed baseline quantile binning, epsilon adjustments, and strict minimum-count reliability checks to detect feature drift.
  * **Champion-Challenger Governance:** Automates model promotion using a trigger-based loop. Challengers are only promoted if they demonstrate statistically significant improvement (via Bootstrap Confidence Intervals for AUC).
  * **Complete Auditability:** Integrates **MLflow** as a system of record, logging all parameters, metrics, run roles, and model artifacts for full reproducibility.

-----

## 📂 Repository Structure

```text
├── artifacts/             # Stored outputs, MLflow JSON reports, and figures
├── data/
│   ├── raw/               # Downloaded UCI Online Retail II dataset
│   ├── processed/         # Cleaned and aggregated data
│   └── slices/            # Chronological data partitions
├── notebooks/
│   └── orchestrator.ipynb # Main Google Colab execution and reporting notebook
├── src/                   # Modular pipeline source code
│   ├── __init__.py
│   ├── data_ingest.py     # Secure data loading
│   ├── preprocess.py      # Chronological sorting and cleaning
│   ├── features.py        # RFM feature engineering & target assignment
│   ├── train.py           # Baseline Random Forest training
│   ├── drift.py           # PSI computation and drift trigger logic
│   ├── retrain.py         # Cumulative challenger training
│   ├── evaluate.py        # Statistical evaluation (AUC Bootstrap)
│   └── mlflow_utils.py    # Strict MLflow logging schema wrapper
├── requirements.txt       # Python dependencies
├── scaffold_project.sh    # Bash script to generate this structure
└── README.md
```

-----

## 🚀 How to Reproduce the Experiment

This project is optimized to run in **Google Colab** to bypass local compute limitations while maintaining strict artifact tracking.

### Step 1: Prepare the Environment

1.  Upload the repository to your GitHub account.
2.  Open Google Colab and create a new notebook, or open `notebooks/orchestrator.ipynb` directly from your GitHub repo via Colab.

### Step 2: Run the Orchestrator

Execute the cells in `orchestrator.ipynb` sequentially. The notebook will automatically:

1.  Clone this repository into the Colab environment.
2.  Download the **Online Retail II** dataset directly from the UCI Machine Learning Repository.
3.  Clean the data and generate 6 chronological slices.
4.  Train the **Baseline Champion** on Slice 1.
5.  Simulate production through Slices 2–6, calculating PSI and evaluating performance.
6.  Trigger the **Champion-Challenger loop** if corroborated drift is detected.

### Step 3: View the Automated Results

The final cell in the notebook queries the local MLflow database to automatically generate a Markdown report for Chapter 6 of the project. It outputs a table detailing:

  * The current Champion model for each slice.
  * The validation AUC.
  * Whether a Challenger was promoted or rejected.
  * The explicit `Run ID` for provenance tracking.

### Step 4: Export Artifacts

Because Colab runtimes are ephemeral, the orchestrator automatically zips the MLflow database (`mlflow.db`) and the `mlruns/` folder.

  * Download these files when prompted at the end of the execution.
  * These files contain your serialized `.pkl` models, scalers, and JSON PSI reports required for audit compliance.

-----

## 🔬 Running MLflow UI (Local Inspection)

If you wish to inspect the exact parameters, metrics, and models locally after downloading the `mlflow.db` and `mlruns/` folder from Colab:

1.  Clone this repository to your local machine.
2.  Install the requirements:
    ```bash
    pip install -r requirements.txt
    ```
3.  Place the downloaded `mlflow.db` and `mlruns/` folder in the root of the repository.
4.  Start the MLflow UI:
    ```bash
    mlflow ui --backend-store-uri sqlite:///mlflow.db
    ```
5.  Open your browser and navigate to `http://127.0.0.1:5000` to view the interactive dashboard, compare champion/challenger runs, and download model artifacts.

-----

## 🧠 Methodology Highlights

### Why PSI?

The Population Stability Index (PSI) is used because it does not require ground-truth labels, which are often delayed in retail environments (Label Lag). To ensure mathematical stability, the pipeline utilizes **epsilon adjustments ($10^{-6}$)** to prevent division-by-zero errors and fixes quantile bins to the baseline distribution.

### Why not retrain constantly?

Frequent retraining causes "model churn" and can overfit to temporary anomalies. This pipeline implements a **corroboration requirement**: a model is only retrained if the PSI exceeds 0.25 AND the validation AUC drops by a specified tolerance. Promotion requires a practical AUC improvement $\ge$ 0.02.

-----

## 📝 Acknowledgements & Data

  * **Dataset:** [Online Retail II Dataset](https://archive.ics.uci.edu/ml/datasets/Online+Retail+II) provided by the UCI Machine Learning Repository.
  * **Framework:** This pipeline was developed following MLOps assurance guidelines (Ashmore et al., 2021) and the ML Test Score rubric (Breck et al., 2017) to explicitly mitigate hidden technical debt (Sculley et al., 2015).
