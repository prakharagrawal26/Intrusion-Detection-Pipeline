# src/feature_engineer.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import VarianceThreshold
from src import config, utils
import joblib
import os

def create_labels(df, mode, benign_label, original_label_col, binary_col, encoded_col, encoder_file):
    """Creates binary or encoded multi-class labels."""
    print(f"Creating labels for mode: {mode}")
    if original_label_col not in df.columns:
        possible_labels = ['Label', 'label', 'LABEL']
        found_original = None
        for pl in possible_labels:
            if pl in df.columns:
                print(f"Warning: Using '{pl}' as original label.")
                original_label_col = pl
                found_original = True
                break
        if not found_original:
            raise ValueError(f"Original label column '{original_label_col}' not found.")

    encoder_instance = None
    if mode == 'binary':
        df[binary_col] = df[original_label_col].apply(lambda x: 0 if x == benign_label else 1)
        print(f"Binary label distribution:\n{df[binary_col].value_counts(normalize=True)}")
        if original_label_col in df.columns:
            df = df.drop(columns=[original_label_col])
        final_label_col = binary_col
    elif mode == 'multiclass':
        print("Encoding multi-class labels...")
        encoder_instance = LabelEncoder()
        df[encoded_col] = encoder_instance.fit_transform(df[original_label_col])
        utils.save_label_encoder(encoder_instance, encoder_file)
        print(f"Label encoder saved to {encoder_file}")
        print(f"Encoded label distribution:\n{df[encoded_col].value_counts(normalize=True)}")
        mapping = {cls: code for code, cls in enumerate(encoder_instance.classes_)}
        print(f"Label Encoding Map (Top 20): {dict(list(mapping.items())[:20])}...")
        if original_label_col in df.columns:
            df = df.drop(columns=[original_label_col])
        final_label_col = encoded_col
    else:
        raise ValueError(f"Invalid CLASSIFICATION_MODE: {mode}")

    print(f"Using '{final_label_col}' as target.")
    return df, final_label_col, encoder_instance

def remove_low_variance_features(df, threshold, label_col):
    """Removes features with variance below threshold."""
    print(f"Attempting variance threshold ({threshold})...")
    if threshold <= 0:
        print("Skipping: threshold <= 0.")
        return df
    if label_col not in df.columns:
        print(f"Skipping: label '{label_col}' not found.")
        return df

    features = df.drop(columns=[label_col])
    numeric_cols = features.select_dtypes(include=np.number).columns
    if numeric_cols.empty:
        print("Skipping: no numeric features.")
        return df

    X_numeric = df[numeric_cols]
    if X_numeric.shape[0] < 2:
        print("Skipping: Not enough samples for VarianceThreshold.")
        return df

    selector = VarianceThreshold(threshold=threshold)
    try:
        print("Fitting VarianceThreshold...")
        selector.fit(X_numeric)
        features_to_keep_mask = selector.get_support()
        features_to_drop = X_numeric.columns[~features_to_keep_mask]
        if len(features_to_drop) > 0:
            print(f"Dropping {len(features_to_drop)} low variance features: {list(features_to_drop)}")
            non_numeric_cols = features.select_dtypes(exclude=np.number).columns
            df = df[list(X_numeric.columns[features_to_keep_mask]) + list(non_numeric_cols) + [label_col]]
            print(f"Shape after variance threshold: {df.shape}")
        else:
            print("No features dropped by variance threshold.")
    except ValueError as ve:
        print(f"ValueError during VarianceThreshold: {ve}. Skipping.")
    except Exception as e:
        print(f"Error during VarianceThreshold: {e}. Skipping.")
    return df

def remove_highly_correlated_features(df, threshold, label_col):
    """Removes one feature from each pair with correlation > threshold."""
    print(f"Attempting correlation threshold ({threshold})...")
    if threshold > 1.0:
        print("Skipping: threshold > 1.")
        return df
    if label_col not in df.columns:
        print(f"Skipping: label '{label_col}' not found.")
        return df

    features = df.drop(columns=[label_col])
    numeric_cols = features.select_dtypes(include=np.number).columns
    if len(numeric_cols) < 2:
        print("Skipping: not enough numeric columns.")
        return df

    print("Calculating correlation matrix...")
    try:
        corr_matrix = df[numeric_cols].corr().abs()
    except TypeError as te:
        print(f"TypeError during correlation calculation: {te}. Check column dtypes.")
        for col in numeric_cols:
            if not pd.api.types.is_numeric_dtype(df[col]):
                print(f"  Problematic non-numeric column: {col}, dtype: {df[col].dtype}")
        return df

    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    cols_to_drop = set()
    for column in upper_tri.columns:
        correlated_features = upper_tri.index[upper_tri[column] > threshold].tolist()
        if correlated_features:
            cols_to_drop.update(correlated_features)

    if cols_to_drop:
        cols_to_drop = list(cols_to_drop.intersection(set(df.columns)))
        if cols_to_drop:
            print(f"Dropping {len(cols_to_drop)} correlated features: {cols_to_drop}")
            df = df.drop(columns=cols_to_drop)
            print(f"Shape after correlation threshold: {df.shape}")
        else:
            print("No features dropped (already removed or none found).")
    else:
        print("No features dropped by correlation threshold.")
    return df

def engineer_features(df_input, output_file, mode, force_engineer=False):
    """Applies feature engineering based on classification mode."""
    encoder = None
    if os.path.exists(output_file) and not force_engineer:
        print(f"Loading feature engineered data from {output_file}...")
        try:
            df_fe = pd.read_parquet(output_file)
            if config.BINARY_LABEL_COLUMN in df_fe.columns:
                final_label_col = config.BINARY_LABEL_COLUMN
            elif config.ENCODED_LABEL_COLUMN in df_fe.columns:
                final_label_col = config.ENCODED_LABEL_COLUMN
            else:
                guessed_label = config.BINARY_LABEL_COLUMN if mode == 'binary' else config.ENCODED_LABEL_COLUMN
                if guessed_label in df_fe.columns:
                    final_label_col = guessed_label
                    print(f"Warning: Using guessed label column '{final_label_col}'.")
                else:
                    raise ValueError(f"Could not find expected label column in {output_file}.")
            print(f"Loaded. Using label column: '{final_label_col}'")
            if mode == 'multiclass':
                print("Attempting to load label encoder...")
                encoder = utils.load_label_encoder(config.LABEL_ENCODER_FILE)
                if encoder is None:
                    print("Warning: Could not load label encoder.")
                else:
                    print("Label encoder loaded successfully.")
            return df_fe, final_label_col, encoder
        except Exception as e:
            print(f"Could not load {output_file}: {e}. Re-engineering.")

    print("--- Starting Feature Engineering ---")
    df = df_input.copy()
    original_cols = df.columns.tolist()

    df, final_label_col, label_encoder_instance = create_labels(
        df, mode, config.BENIGN_LABEL, config.CONFIGURED_LABEL_COLUMN,
        config.BINARY_LABEL_COLUMN, config.ENCODED_LABEL_COLUMN, config.LABEL_ENCODER_FILE
    )
    if mode == 'multiclass':
        encoder = label_encoder_instance

    cols_to_drop_init = [col for col in config.COLS_TO_DROP_INITIAL if col in df.columns]
    if cols_to_drop_init:
        print(f"Dropping initial columns: {cols_to_drop_init}")
        df = df.drop(columns=cols_to_drop_init)

    df = remove_low_variance_features(df, config.VARIANCE_THRESHOLD, final_label_col)
    df = remove_highly_correlated_features(df, config.CORRELATION_THRESHOLD, final_label_col)

    print("--- Feature Engineering Finished ---")
    print(f"Final shape after FE: {df.shape}")
    final_cols_list = df.columns.tolist()
    print(f"Final columns ({len(final_cols_list)}): {final_cols_list}")

    print(f"Saving to {output_file}...")
    try:
        df.to_parquet(output_file, index=False)
        print("Saved successfully.")
    except Exception as e:
        print(f"Error saving {output_file}: {e}")

    return df, final_label_col, encoder