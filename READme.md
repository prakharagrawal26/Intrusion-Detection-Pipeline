# End-to-End Network Anomaly Detection on CIC-IDS2017

## Overview

This project implements a comprehensive, end-to-end machine learning pipeline for detecting network intrusions using the CIC-IDS2017 dataset. It tackles the challenges of working with large-scale, real-world network traffic data, including data cleaning, feature engineering, dimensionality reduction, and model evaluation under significant class imbalance. The pipeline supports both binary anomaly detection (Benign vs. Attack) and multi-class classification (identifying specific attack types).

Built using Python and common data science libraries (Pandas, Scikit-learn, TensorFlow/Keras, XGBoost), this project demonstrates a practical workflow applicable to enhancing network infrastructure security.

## Features

*   **Data Loading & Merging:** Handles multiple large CSV files from the CIC-IDS2017 dataset, merging them efficiently using Pandas and saving intermediate results in Parquet format.
*   **Robust Data Cleaning:** Implements steps to handle missing values (NaNs), infinite values, and duplicate entries commonly found in network flow data.
*   **Exploratory Data Analysis (EDA):** Generates plots (optional via config) including correlation heatmaps, feature distributions, and feature-vs-label plots to understand the data characteristics.
*   **Feature Engineering:**
    *   Creates target labels for either binary or multi-class classification mode.
    *   Includes basic feature selection steps (low variance, high correlation removal).
    *   Provides a structure to easily add custom-engineered features.
*   **Preprocessing:**
    *   Scales numerical features using MinMaxScaler.
    *   Optionally applies Principal Component Analysis (PCA) for dimensionality reduction.
    *   Splits data into training and testing sets with stratification.
*   **Diverse Model Training:**
    *   Trains various **supervised** models (Logistic Regression, Random Forest, XGBoost).
    *   Trains **unsupervised** models (Isolation Forest, Autoencoder via TensorFlow/Keras) for anomaly detection.
    *   Handles class imbalance for supervised models using class weighting.
*   **Comprehensive Evaluation:**
    *   Evaluates models based on the selected mode (binary/multi-class).
    *   Calculates standard metrics (Accuracy, Precision, Recall, F1-Score, ROC AUC where applicable).
    *   Generates detailed classification reports and confusion matrices (with plots).
    *   Calculates thresholds for unsupervised models based on normal training data scores.
*   **Configuration Driven:** Pipeline behavior (sampling, modes, PCA, EDA, model selection, paths) is controlled via `src/config.py`.
*   **Modular Code:** Organized into distinct Python modules for better readability and maintenance.

## Dataset: CIC-IDS2017

This project uses the **Canadian Institute for Cybersecurity Intrusion Detection System 2017 (CIC-IDS2017)** dataset.

*   **Source:** [https://www.unb.ca/cic/datasets/ids-2017.html](https://www.unb.ca/cic/datasets/ids-2017.html)
*   **Content:** Realistic network traffic including benign flows and 14 common attack types (DoS, DDoS, Brute Force, Web Attacks, Botnet, etc.), captured over 5 days. Features (~80) are pre-extracted using CICFlowMeter.
*   **Requirement:** You need to **download the labeled flow CSV files** from the source website and place them in the `data/` directory *before* running the pipeline. The pipeline expects the `.csv` files, not the raw `.pcap` files.


## Installation

1.  **Clone the repository (Optional):**
    ```bash
    git clone [link-to-your-repo]
    cd AnomalyDetection
    ```
2.  **Create and activate a virtual environment (Recommended):**
    ```bash
    python -m venv venv
    # Windows:
    .\venv\Scripts\activate
    # Linux/macOS:
    source venv/bin/activate
    ```
3.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

1.  **Download Data:** Obtain the CIC-IDS2017 labeled flow **CSV files** from the official source and place them inside the `data/` directory.
2.  **Configure Pipeline:**
    *   Open `src/config.py`.
    *   Verify the `CSV_FILES` list matches your downloaded filenames exactly.
    *   Verify `CONFIGURED_LABEL_COLUMN` matches the label column header in the CSVs (likely `'Label'`).
    *   **Important:** Adjust `SAMPLE_FRACTION` (e.g., to `0.1` or `0.2`) if you have limited RAM or want faster initial runs. Use `1.0` for the full dataset (requires significant resources).
    *   Set `CLASSIFICATION_MODE` to `'binary'` or `'multiclass'`.
    *   Set `USE_PCA` to `True` or `False`.
    *   Set `GENERATE_EDA_PLOTS` to `True` or `False`.
    *   Review other parameters in `config.py` as needed.
3.  **Run the Pipeline:**
    *   Ensure your virtual environment is active.
    *   Navigate to the project's root directory (`AnomalyDetection/`) in your terminal.
    *   Execute the main script:
        ```bash
        python -m src.main
        ```
4.  **Monitor Output:** The script will print progress messages for each step. Loading, cleaning, and training can take a significant amount of time, especially on the full dataset.
5.  **Check Results:** Generated files (Parquet data, models, plots, summary CSV) will be saved in the `data/`, `models/`, and `results/` directories.

## Configuration (`src/config.py`)

Key parameters you can adjust:

*   `CSV_FILES`: List of input data files.
*   `CONFIGURED_LABEL_COLUMN`: Name of the label column in CSVs.
*   `CLASSIFICATION_MODE`: `'binary'` or `'multiclass'`.
*   `USE_PCA`: `True` / `False` to enable/disable PCA.
*   `PCA_N_COMPONENTS`: Variance ratio or number of components for PCA.
*   `SAMPLE_FRACTION`: Fraction of data to use (1.0 for all).
*   `GENERATE_EDA_PLOTS`: `True` / `False` to create EDA visualizations.
*   `MODELS_CONFIG`: Enable/disable specific models and set basic hyperparameters.
*   `FORCE_*` flags: Set to `True` to force re-running specific steps (e.g., `FORCE_CLEAN=True`).

## Evaluation

The pipeline evaluates models based on the `CLASSIFICATION_MODE`.
*   **Binary Mode:** Evaluates performance in distinguishing BENIGN vs. ATTACK.
*   **Multi-Class Mode:** Evaluates performance in classifying specific attack types.
*   **Unsupervised Models (IForest, AE):** Always evaluated in a binary context (predicting anomaly vs. normal) based on calculated thresholds.

Metrics calculated include Accuracy, Precision, Recall, F1-Score, and ROC AUC (where applicable). Results are saved in:
*   `results/evaluation_summary.csv`: Table of metrics per model.
*   `results/evaluation_plots/`: Confusion matrix plots.
*   `results/debug_plots/`: Score distribution plots for unsupervised models.

## Potential Improvements

*   **Hyperparameter Tuning:** Implement GridSearchCV/RandomizedSearchCV or Optuna for optimizing model parameters.
*   **Advanced Feature Selection:** Explore techniques like RFECV or SHAP-based selection.
*   **More Models:** Add LightGBM, SVM (with sampling), or more sophisticated Deep Learning architectures (CNNs, LSTMs for sequence data if features allow).
*   **Handling Imbalance:** Integrate techniques like SMOTE (from `imbalanced-learn`) applied *only* to the training set.
*   **Model Interpretability:** Use SHAP library to explain model predictions.
*   **Temporal Analysis:** Incorporate timestamp features and evaluate model performance over time to check for concept drift.
*   **Attack-Specific Analysis:** Deeper dive into performance against individual attack categories.