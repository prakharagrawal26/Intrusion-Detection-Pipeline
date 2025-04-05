# src/data_cleaner.py
import pandas as pd
import numpy as np
from src import config
import os

def clean_data(df, output_file, force_clean=False):
    """
    Data cleaning: drop NaNs, Infs, duplicates.
    """
    if os.path.exists(output_file) and not force_clean:
        print(f"Loading cleaned data from {output_file}...")
        try:
            df_cleaned = pd.read_parquet(output_file)
            print("Loaded.")
            return df_cleaned
        except Exception as e:
            print(f"Error loading {output_file}: {e}. Re-cleaning.")

    print("--- Cleaning Data ---")
    print(f"Initial shape: {df.shape}")

    # Label column identification.
    actual_label_column = config.CONFIGURED_LABEL_COLUMN
    if actual_label_column not in df.columns:
        print(f"Warning: '{actual_label_column}' not found. Checking variations...")
        possible_labels = ['Label', 'label', 'LABEL']
        found_label = None
        for pl in possible_labels:
            if pl in df.columns:
                print(f"Found '{pl}'. Using '{pl}' as label.")
                actual_label_column = pl
                found_label = True
                break
        if not found_label:
            raise ValueError(f"Label '{config.CONFIGURED_LABEL_COLUMN}' or variations not found.")

    # Numeric conversion (excluding label).
    print(f"Converting columns to numeric (excluding '{actual_label_column}')...")
    cols_to_convert = [col for col in df.columns if col != actual_label_column]
    for col in cols_to_convert:
        df[col] = pd.to_numeric(df[col], errors='coerce')

    # NaN handling.
    nan_counts = df.isna().sum()
    print(f"NaNs before:\n{nan_counts[nan_counts > 0]}")

    percent_nan = df.drop(columns=[actual_label_column]).isnull().sum() / len(df) * 100
    cols_to_drop_nan = percent_nan[percent_nan > 50].index
    if not cols_to_drop_nan.empty:
        print(f"Dropping columns with >50% NaNs: {list(cols_to_drop_nan)}")
        df.drop(columns=cols_to_drop_nan, inplace=True)

    print("Dropping NaN rows...")
    initial_rows = df.shape[0]
    df.dropna(inplace=True)
    print(f"Dropped {initial_rows - df.shape[0]} rows.")
    print(f"Shape after NaN handling: {df.shape}")

    # Inf handling.
    cols_to_check_inf = [col for col in df.columns if col != actual_label_column]
    numeric_cols_for_inf = df[cols_to_check_inf].select_dtypes(include=np.number).columns

    if not numeric_cols_for_inf.empty:
        inf_mask = np.isinf(df[numeric_cols_for_inf])
        inf_counts = inf_mask.sum()
        print(f"Infs before:\n{inf_counts[inf_counts > 0]}")
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        print("Dropping new NaN rows from Inf replacement...")
        initial_rows = df.shape[0]
        df.dropna(inplace=True)
        print(f"Dropped {initial_rows - df.shape[0]} rows.")
        print(f"Shape after Inf handling: {df.shape}")
    else:
        print("No numeric columns to check for Infs.")

    # Duplicate removal.
    print("Checking for duplicates...")
    initial_rows = df.shape[0]
    df.drop_duplicates(inplace=True)
    print(f"Dropped {initial_rows - df.shape[0]} duplicates.")
    print(f"Shape after duplicates: {df.shape}")

    # Label check.
    if actual_label_column not in df.columns:
        raise ValueError(f"Label '{actual_label_column}' missing after cleaning.")
    print(f"Label distribution:\n{df[actual_label_column].value_counts(normalize=True)}")

    print("--- Cleaning Finished ---")
    print(f"Final shape: {df.shape}")

    print(f"Saving to {output_file}...")
    try:
        if actual_label_column != config.CONFIGURED_LABEL_COLUMN:
            print(f"Renaming '{actual_label_column}' to '{config.CONFIGURED_LABEL_COLUMN}'.")
            df.rename(columns={actual_label_column: config.CONFIGURED_LABEL_COLUMN}, inplace=True)

        df.to_parquet(output_file, index=False)
        print("Saved.")
    except Exception as e:
        print(f"Error saving: {e}")

    return df