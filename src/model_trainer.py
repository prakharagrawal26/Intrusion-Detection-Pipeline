import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from xgboost import XGBClassifier
from src import config, utils
import os
import joblib

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential, save_model, load_model
    from tensorflow.keras.layers import Input, Dense, Dropout
    from tensorflow.keras.callbacks import EarlyStopping
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

def train_autoencoder(X_train_benign, model_dir):
    """Trains an Autoencoder model ONLY on benign data."""
    if not TENSORFLOW_AVAILABLE:
        print("ERROR: TensorFlow not found. Cannot train Autoencoder.")
        return None

    print("--- Training Autoencoder (Unsupervised on Benign Data) ---")
    model_name = "Autoencoder"
    model_path = os.path.join(model_dir, f"{model_name}.keras")
    if X_train_benign.empty:
        print("ERROR: No benign training data provided for Autoencoder. Skipping.")
        return None

    X_train_numeric = X_train_benign.select_dtypes(include=np.number)
    if X_train_numeric.empty:
        print("ERROR: No numeric features in benign data for Autoencoder. Skipping.")
        return None

    input_dim = X_train_numeric.shape[1]
    encoding_dim = config.AE_ENCODING_DIM
    encoding_dim = min(encoding_dim, int(input_dim / 2)) if input_dim > 1 else 1
    if encoding_dim < 1:
        encoding_dim = 1
    print(f"Autoencoder Input Dim: {input_dim}, Bottleneck Dim: {encoding_dim}")

    autoencoder = Sequential([
        Input(shape=(input_dim,)),
        Dense(max(input_dim, 32), activation='relu'),
        Dense(max(int(input_dim / 2), encoding_dim * 2, 16), activation='relu'),
        Dropout(0.1),
        Dense(encoding_dim, activation='relu'),
        Dense(max(int(input_dim / 2), encoding_dim * 2, 16), activation='relu'),
        Dropout(0.1),
        Dense(input_dim, activation='sigmoid')
    ])
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.summary()

    early_stopping = EarlyStopping(monitor='val_loss', patience=config.AE_EARLY_STOPPING_PATIENCE, restore_best_weights=True, verbose=1)

    history = autoencoder.fit(X_train_numeric, X_train_numeric, epochs=config.AE_EPOCHS, batch_size=config.AE_BATCH_SIZE, shuffle=True, validation_split=config.AE_VALIDATION_SPLIT, callbacks=[early_stopping], verbose=1)

    utils.save_model(autoencoder, model_path)
    print(f"{model_name} trained and saved to {model_path}")
    return autoencoder

def train_model(X_train, y_train, model_name, model_config, model_dir):
    """Trains a specified model (handles supervised and unsupervised setup)."""
    print(f"\n--- Training {model_name} ---")
    model_path = os.path.join(model_dir, f"{model_name}.joblib")
    model = None
    is_unsupervised = model_name in ["IsolationForest", "Autoencoder"]

    if model_name == "IsolationForest":
        print("Training Isolation Forest (unsupervised)...")
        X_train_numeric = X_train.select_dtypes(include=np.number)
        if X_train_numeric.empty:
            print("ERROR: No numeric features for Isolation Forest. Skipping.")
            return None
        model = IsolationForest(contamination=model_config.get("contamination", 'auto'), random_state=config.RANDOM_STATE, n_jobs=-1)
        model.fit(X_train_numeric)
        utils.save_model(model, model_path)
        return model

    elif model_name == "Autoencoder":
        print("Autoencoder training is handled separately using only benign data.")
        return None

    print(f"Mode: {config.CLASSIFICATION_MODE}")
    target_label_col = config.BINARY_LABEL_COLUMN if config.CLASSIFICATION_MODE == 'binary' else config.ENCODED_LABEL_COLUMN

    num_classes = y_train.nunique()
    print(f"Number of classes detected in y_train: {num_classes}")

    class_weight = config.CLASS_WEIGHT_METHOD if num_classes > 1 else None

    X_train_numeric = X_train.select_dtypes(include=np.number)

    if model_name == "LogisticRegression":
        model = LogisticRegression(random_state=config.RANDOM_STATE, class_weight=class_weight, max_iter=1000, solver='liblinear', multi_class=model_config.get("multi_class", "auto"))
        X_fit = X_train_numeric

    elif model_name == "RandomForest":
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        model = RandomForestClassifier(n_estimators=model_config.get("n_estimators", 100), max_depth=model_config.get("max_depth", None), random_state=config.RANDOM_STATE, class_weight=class_weight, n_jobs=-1)
        X_fit = X_train

    elif model_name == "XGBoost":
        model_path = os.path.join(model_dir, f"{model_name}.joblib")
        objective = 'binary:logistic' if num_classes <= 2 else 'multi:softmax'
        eval_metric = 'logloss' if num_classes <= 2 else 'mlogloss'
        num_class_param = num_classes if num_classes > 2 else None

        scale_pos_weight = None
        if objective == 'binary:logistic' and class_weight == 'balanced':
            neg = (y_train == 0).sum()
            pos = (y_train == 1).sum()
            scale_pos_weight = neg / pos if pos > 0 else 1
            print(f"Using scale_pos_weight={scale_pos_weight:.2f} for XGBoost (binary)")

        model = XGBClassifier(n_estimators=model_config.get("n_estimators", 100), max_depth=model_config.get("max_depth", 6), random_state=config.RANDOM_STATE, objective=objective, scale_pos_weight=scale_pos_weight, num_class=num_class_param, use_label_encoder=False, eval_metric=eval_metric, n_jobs=-1)
        X_fit = X_train_numeric

    else:
        print(f"Model {model_name} not handled in supervised training section. Skipping.")
        return None

    if model is not None:
        try:
            print(f"Fitting {model_name} on X shape: {X_fit.shape}, y shape: {y_train.shape}")
            model.fit(X_fit, y_train)
            utils.save_model(model, model_path)
        except Exception as e:
            print(f"Error training {model_name}: {e}")
            import traceback
            traceback.print_exc()
            return None

    return model