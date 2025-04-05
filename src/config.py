# src/config.py
import os

# --- Project Root ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# --- Data Paths ---
DATA_DIR = os.path.join(BASE_DIR, "data")
CSV_FILES = [ # List the EXACT names of your downloaded CIC-IDS2017 CSV files
    "Monday-WorkingHours.pcap_ISCX.csv",
    "Tuesday-WorkingHours.pcap_ISCX.csv",
    "Wednesday-workingHours.pcap_ISCX.csv", # Check exact case/name
    "Thursday-WorkingHours-Morning-WebAttacks.pcap_ISCX.csv",
    "Thursday-WorkingHours-Afternoon-Infilteration.pcap_ISCX.csv",
    "Friday-WorkingHours-Morning.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-PortScan.pcap_ISCX.csv",
    "Friday-WorkingHours-Afternoon-DDos.pcap_ISCX.csv",
]
MERGED_RAW_DATA_FILE = os.path.join(DATA_DIR, "cicids2017_merged_raw.parquet")
CLEANED_DATA_FILE = os.path.join(DATA_DIR, "cicids2017_cleaned.parquet")
FEATURE_ENGINEERED_DATA_FILE = os.path.join(DATA_DIR, "cicids2017_features.parquet")

# --- Model/Results Paths ---
MODEL_DIR = os.path.join(BASE_DIR, "models")
RESULT_DIR = os.path.join(BASE_DIR, "results")
SCALER_FILE = os.path.join(MODEL_DIR, "scaler.joblib")
PCA_FILE = os.path.join(MODEL_DIR, "pca_transformer.joblib")
LABEL_ENCODER_FILE = os.path.join(MODEL_DIR, "label_encoder.joblib")
AE_MODEL_FILE = os.path.join(MODEL_DIR, "autoencoder.keras")

# --- Pipeline Control ---
CLASSIFICATION_MODE = 'multiclass'
USE_PCA = True
GENERATE_EDA_PLOTS = False
SAMPLE_FRACTION = 1.0 # Use 1.0 for full data, < 1.0 for sampling
FORCE_RELOAD = False
FORCE_CLEAN = False
FORCE_ENGINEER = False
FORCE_TRAIN = False

# --- Feature Columns ---
CONFIGURED_LABEL_COLUMN = 'Label'
BINARY_LABEL_COLUMN = 'binary_label'
ENCODED_LABEL_COLUMN = 'encoded_label'
BENIGN_LABEL = 'BENIGN'
COLS_TO_DROP_INITIAL = []

# --- Modeling Parameters ---
TEST_SPLIT_RATIO = 0.2
RANDOM_STATE = 42
CLASS_WEIGHT_METHOD = 'balanced'
TRAIN_FRACTION_NORMAL = 0.15

# --- PCA Config ---
PCA_N_COMPONENTS = 0.98

# --- Autoencoder Config ---
AE_ENCODING_DIM = 16
AE_EPOCHS = 20
AE_BATCH_SIZE = 256
AE_THRESHOLD_PERCENTILE = 98
AE_EARLY_STOPPING_PATIENCE = 5
AE_VALIDATION_SPLIT = 0.1

# --- >>> ADD THIS SECTION <<< ---
# Isolation Forest Config
IFOREST_THRESHOLD_PERCENTILE = 95 # Percentile used in main.py for threshold calc
# --- >>> END ADDED SECTION <<< ---

# --- Models to train ---
MODELS_CONFIG = {
    "LogisticRegression": {"enabled": True, "multi_class": "auto"},
    "RandomForest": {"enabled": True, "n_estimators": 100, "max_depth": 20},
    "XGBoost": {"enabled": True, "n_estimators": 100, "max_depth": 10},
    "IsolationForest": {"enabled": True, "contamination": 'auto'},
    "Autoencoder": {"enabled": True},
}

# --- Feature Engineering Parameters ---
CORRELATION_THRESHOLD = 0.95
VARIANCE_THRESHOLD = 0.0

# --- Evaluation ---
EVALUATION_METRICS = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
AVERAGING_METHOD_BINARY = 'binary'
AVERAGING_METHOD_MULTICLASS = 'weighted'