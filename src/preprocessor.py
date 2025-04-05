import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from src import config, utils
import joblib
import os
import numpy as np

def scale_features(df, scaler_file, label_col):
    """Scales numerical features using MinMaxScaler."""
    print("Scaling features...")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame for scaling.")

    features = df.drop(columns=[label_col])
    labels = df[label_col]

    numeric_cols = features.select_dtypes(include=np.number).columns
    if numeric_cols.empty:
        print("No numeric columns found to scale.")
        return df.copy(), None

    scaler = MinMaxScaler()
    features_scaled = scaler.fit_transform(features[numeric_cols])
    utils.save_model(scaler, scaler_file)
    print(f"Scaler saved to {scaler_file}")

    features_scaled_df = pd.DataFrame(features_scaled, index=features.index, columns=numeric_cols)

    non_numeric_df = features.select_dtypes(exclude=np.number)
    labels = labels.loc[features_scaled_df.index]
    df_scaled = pd.concat([features_scaled_df, non_numeric_df, labels], axis=1)

    print("Scaling complete.")
    return df_scaled, scaler

def apply_pca(X_train, X_test, n_components, pca_file):
    """Applies PCA for dimensionality reduction."""
    print(f"Applying PCA with n_components={n_components}...")

    X_train_numeric = X_train.select_dtypes(include=np.number)
    X_test_numeric = X_test.select_dtypes(include=np.number)
    numeric_cols = X_train_numeric.columns

    if X_train_numeric.empty:
        print("No numeric features found for PCA. Skipping.")
        return X_train, X_test, None

    pca = PCA(n_components=n_components, random_state=config.RANDOM_STATE)
    X_train_pca = pca.fit_transform(X_train_numeric)
    X_test_pca = pca.transform(X_test_numeric)

    explained_variance = np.sum(pca.explained_variance_ratio_)
    actual_components = pca.n_components_
    print(f"PCA applied. Actual components: {actual_components}, Explained variance: {explained_variance:.4f}")
    utils.save_model(pca, pca_file)
    print(f"PCA transformer saved to {pca_file}")

    pca_cols = [f"PC_{i+1}" for i in range(actual_components)]
    X_train_pca_df = pd.DataFrame(X_train_pca, index=X_train.index, columns=pca_cols)
    X_test_pca_df = pd.DataFrame(X_test_pca, index=X_test.index, columns=pca_cols)

    X_train_non_numeric = X_train.select_dtypes(exclude=np.number)
    X_test_non_numeric = X_test.select_dtypes(exclude=np.number)

    X_train_final = pd.concat([X_train_pca_df, X_train_non_numeric], axis=1)
    X_test_final = pd.concat([X_test_pca_df, X_test_non_numeric], axis=1)

    return X_train_final, X_test_final, pca

def split_data(df, test_size, random_state, label_col):
    """Splits data into training and testing sets."""
    print(f"Splitting data using label '{label_col}'...")
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in DataFrame for splitting.")

    X = df.drop(columns=[label_col])
    y = df[label_col]

    if y.nunique() > 1 and all(y.value_counts() >= 2):
        stratify_option = y
        print("Using stratified splitting.")
    else:
        stratify_option = None
        print("Warning: Not enough samples in minority class for stratified splitting or only one class present. Using simple split.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=stratify_option)
    print(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    print(f"Test set shape: X={X_test.shape}, y={y_test.shape}")
    print(f"Train set label distribution:\n{y_train.value_counts(normalize=True).sort_index()}")
    print(f"Test set label distribution:\n{y_test.value_counts(normalize=True).sort_index()}")
    return X_train, X_test, y_train, y_test