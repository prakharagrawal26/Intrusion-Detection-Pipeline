# src/data_loader.py
import pandas as pd
import os
from glob import glob
from tqdm import tqdm
from src import config

def load_and_merge_csvs(data_dir, csv_files, output_file, sample_frac=1.0, force_reload=False):
    """Loads and merges CSVs, optionally samples, saves as Parquet."""
    if os.path.exists(output_file) and not force_reload:
        print(f"Loading merged data from {output_file}...")
        try:
            df = pd.read_parquet(output_file)
            print(f"Loaded: {df.shape}")
            if sample_frac < 1.0:
                current_len = len(df)
                target_len_from_current = int(current_len / sample_frac * sample_frac)
                if len(df) > target_len_from_current:
                    target_n_rows = int(len(df) * sample_frac)
                    print(f"Sampling {sample_frac*100:.1f}% ({target_n_rows} rows)...")
                    random_state = getattr(config, 'RANDOM_STATE', 42)
                    df = df.sample(n=target_n_rows, random_state=random_state).reset_index(drop=True)
                    print(f"Sampled: {df.shape}")
            print("Loaded.")
            return df
        except Exception as e:
            print(f"Error loading {output_file}: {e}. Reloading from CSVs.")

    all_files = [os.path.join(data_dir, f) for f in csv_files if os.path.exists(os.path.join(data_dir, f))]
    if not all_files:
        raise FileNotFoundError(f"No CSVs found in {data_dir}.")

    print(f"Merging {len(all_files)} CSVs.")
    df_list = []
    for filename in tqdm(all_files, desc="Loading"):
        try:
            df_temp = pd.read_csv(filename, encoding='latin-1', on_bad_lines='skip')
            df_temp.columns = df_temp.columns.str.strip()
            df_list.append(df_temp)
        except Exception as e:
            print(f"Error reading {filename}: {e}")

    if not df_list:
        raise ValueError("No data loaded.")

    print("Merging DataFrames...")
    merged_df = pd.concat(df_list, axis=0, ignore_index=True)
    print(f"Merged: {merged_df.shape}")
    del df_list

    if sample_frac < 1.0:
        print(f"Sampling {sample_frac*100:.1f}%...")
        random_state = getattr(config, 'RANDOM_STATE', 42)
        merged_df = merged_df.sample(frac=sample_frac, random_state=random_state).reset_index(drop=True)
        print(f"Sampled: {merged_df.shape}")

    print(f"Saving to {output_file}...")
    try:
        merged_df.to_parquet(output_file, index=False)
        print("Saved.")
    except Exception as e:
        print(f"Error saving: {e}")

    return merged_df