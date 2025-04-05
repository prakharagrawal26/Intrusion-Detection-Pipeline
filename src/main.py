import time
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from src import config
from src import data_loader, data_cleaner, feature_engineer, preprocessor, model_trainer, evaluator, utils

def run_pipeline(force_reload=False, force_clean=False, force_engineer=False, force_train=False):
    """Runs the entire end-to-end pipeline with configurable modes."""
    start_time_total = time.time()

    # Create necessary directories
    os.makedirs(config.DATA_DIR, exist_ok=True)
    os.makedirs(config.MODEL_DIR, exist_ok=True)
    os.makedirs(config.RESULT_DIR, exist_ok=True)

    # --- Step 1: Load/Merge Data ---
    print("\n===== Step 1: Load/Merge Data =====")
    start_time = time.time()
    df_raw = data_loader.load_and_merge_csvs(config.DATA_DIR, config.CSV_FILES, config.MERGED_RAW_DATA_FILE,
                                             sample_frac=config.SAMPLE_FRACTION,
                                             force_reload=force_reload)
    print(f"Load/Merge completed in {time.time() - start_time:.2f} seconds.")
    if df_raw is None or df_raw.empty:
        print("Failed to load data. Exiting.")
        return

    # --- Step 2: Clean Data ---
    print("\n===== Step 2: Clean Data =====")
    start_time = time.time()
    df_clean = data_cleaner.clean_data(df_raw, config.CLEANED_DATA_FILE, force_clean=force_clean)
    del df_raw
    print(f"Cleaning completed in {time.time() - start_time:.2f} seconds.")
    if df_clean is None or df_clean.empty:
        print("Failed to clean data. Exiting.")
        return

    # --- Step 2.5: Generate EDA Plots (Optional) ---
    if config.GENERATE_EDA_PLOTS:
        print("\n===== Step 2.5: Generate EDA Plots =====")
        start_time = time.time()
        temp_df_eda = df_clean.copy()
        eda_label_col = config.CONFIGURED_LABEL_COLUMN  # Name after cleaning
        if eda_label_col not in temp_df_eda.columns:
            print(f"ERROR: EDA - Label column '{eda_label_col}' not found. Skipping plots.")
        else:
            # Determine class names/map for plots
            class_names_map = {}
            plot_label_col_eda = eda_label_col  # Default to original text label
            if config.CLASSIFICATION_MODE == 'multiclass':
                encoder = utils.load_label_encoder(config.LABEL_ENCODER_FILE)  # Try loading
                if encoder:
                    class_names = encoder.classes_
                    class_names_map = {i: name for i, name in enumerate(class_names)}
                    # Temporarily add encoded col for plotting if encoder loaded
                    try:
                        temp_df_eda[config.ENCODED_LABEL_COLUMN] = encoder.transform(temp_df_eda[eda_label_col])
                        plot_label_col_eda = config.ENCODED_LABEL_COLUMN  # Use encoded for plot grouping
                    except Exception as e:
                        print(f"Warning: Could not apply loaded encoder for EDA plot: {e}. Using text labels.")
                        plot_label_col_eda = eda_label_col  # Fallback
                        class_names_map = {name: name for name in sorted(temp_df_eda[eda_label_col].unique())}
                else:
                    print("Warning: Label encoder not found for multiclass EDA names. Using unique text values.")
                    unique_labels = sorted(temp_df_eda[eda_label_col].unique())
                    class_names_map = {name: name for name in unique_labels}
                    plot_label_col_eda = eda_label_col

            elif config.CLASSIFICATION_MODE == 'binary':
                temp_df_eda[config.BINARY_LABEL_COLUMN] = temp_df_eda[eda_label_col].apply(lambda x: 0 if x == config.BENIGN_LABEL else 1)
                plot_label_col_eda = config.BINARY_LABEL_COLUMN
                class_names_map = {0: 'BENIGN', 1: 'ATTACK'}
            else:  # Fallback if mode invalid
                unique_labels = sorted(temp_df_eda[eda_label_col].unique())
                class_names_map = {name: name for name in unique_labels}
                plot_label_col_eda = eda_label_col

            # Generate Plots (pass the correct label column name for plotting)
            evaluator.plot_correlation_heatmap(temp_df_eda, plot_label_col_eda, config.RESULT_DIR)
            evaluator.plot_feature_distributions(temp_df_eda, plot_label_col_eda, config.RESULT_DIR)
            if plot_label_col_eda in temp_df_eda.columns:
                evaluator.plot_feature_vs_label(temp_df_eda, plot_label_col_eda, class_names_map, config.RESULT_DIR)
            else:
                print(f"Skipping feature_vs_label plot: Column '{plot_label_col_eda}' not found.")

        print(f"EDA Plot Generation completed in {time.time() - start_time:.2f} seconds.")

    # --- Step 3: Feature Engineering ---
    print("\n===== Step 3: Engineer Features =====")
    start_time = time.time()
    df_fe, final_label_col, label_encoder = feature_engineer.engineer_features(
        df_clean,
        config.FEATURE_ENGINEERED_DATA_FILE,
        config.CLASSIFICATION_MODE,
        force_engineer=force_engineer
    )
    del df_clean
    print(f"Feature Engineering completed in {time.time() - start_time:.2f} seconds.")
    if df_fe is None or df_fe.empty:
        print("Failed feature engineering. Exiting.")
        return

    # --- Step 4: Preprocessing (Scale & Split) ---
    print("\n===== Step 4: Preprocess Data (Scale, Split, PCA) =====")
    start_time = time.time()

    X_train, X_test, y_train, y_test = None, None, None, None
    scaler_fitted, pca_transformer = None, None

    df_scaled, scaler_fitted = preprocessor.scale_features(df_fe, config.SCALER_FILE, final_label_col)
    del df_fe
    if df_scaled is None:
        print("Scaling failed. Exiting.")
        return

    X_train, X_test, y_train, y_test = preprocessor.split_data(
        df_scaled, config.TEST_SPLIT_RATIO, config.RANDOM_STATE, final_label_col
    )
    del df_scaled

    # --- Step 4.5: Apply PCA (Optional) ---
    if config.USE_PCA and X_train is not None and not X_train.empty:
        X_train, X_test, pca_transformer = preprocessor.apply_pca(
            X_train, X_test, config.PCA_N_COMPONENTS, config.PCA_FILE
        )
    print(f"Preprocessing (Scale, Split, PCA) completed in {time.time() - start_time:.2f} seconds.")
    print(f"Final X_train shape: {X_train.shape if X_train is not None else 'N/A'}")
    print(f"Final X_test shape: {X_test.shape if X_test is not None else 'N/A'}")

    if X_train is None or X_test is None or y_train is None or y_test is None:
        print("ERROR: Data splitting failed. Train/Test sets not created. Exiting.")
        return

    # --- Step 5: Model Training ---
    print("\n===== Step 5: Train Models =====")
    start_time = time.time()
    trained_models = {}
    thresholds = {}
    scores_for_plotting = {}

    current_label_encoder = label_encoder

    X_train_benign_ae = None
    benign_mask = None
    if config.MODELS_CONFIG.get("Autoencoder", {}).get("enabled", False):
        print("Extracting benign data for Autoencoder training...")
        if config.CLASSIFICATION_MODE == 'binary':
            benign_mask = (y_train == 0)
        elif config.CLASSIFICATION_MODE == 'multiclass' and current_label_encoder is not None:
            try:
                benign_label_encoded = current_label_encoder.transform([config.BENIGN_LABEL])[0]
                benign_mask = (y_train == benign_label_encoded)
            except ValueError:
                print(f"Warning: BENIGN label '{config.BENIGN_LABEL}' not found by encoder. Cannot extract benign data for AE.")
                benign_mask = pd.Series([False] * len(y_train))
        else:
            print("Warning: Cannot determine benign data for AE training. AE will be skipped or config invalid.")
            benign_mask = pd.Series([False] * len(y_train))

        if benign_mask is not None and benign_mask.any():
            X_train_benign_ae = X_train[benign_mask]
            print(f"Benign training data shape for AE: {X_train_benign_ae.shape}")
        else:
            print("Warning: No benign training data found/extracted for Autoencoder.")

    for name, m_config in config.MODELS_CONFIG.items():
        if m_config.get("enabled", False):
            model = None
            is_ae = (name == "Autoencoder")
            model_load_path = config.AE_MODEL_FILE if is_ae else os.path.join(config.MODEL_DIR, f"{name}.joblib")

            if os.path.exists(model_load_path) and not force_train:
                print(f"\nLoading existing model for {name}...")
                model = utils.load_model(model_load_path)
                if model:
                    trained_models[name] = model
                else:
                    print(f"Failed to load {name}, will retrain if possible.")
                    model = None

            if model is None:
                if force_train:
                    print(f"\nForce retraining {name}...")
                if name == "Autoencoder":
                    model = model_trainer.train_autoencoder(X_train_benign_ae, config.MODEL_DIR)
                else:
                    model = model_trainer.train_model(X_train, y_train, name, m_config, config.MODEL_DIR)
                if model:
                    trained_models[name] = model
                else:
                    print(f"!!! Training failed for {name}. Model will be skipped. !!!")
                    continue

            if name in trained_models:
                model = trained_models[name]
                print(f"\nCalculating threshold for {name}...")
                if name == "IsolationForest":
                    X_train_numeric_if = X_train.select_dtypes(include=np.number)
                    if not X_train_numeric_if.empty:
                        scores_normal_if = []
                        if benign_mask is not None and benign_mask.any():
                            scores_normal_if = -1 * model.decision_function(X_train_numeric_if[benign_mask])
                            print(f"  Using {len(scores_normal_if)} benign samples for IForest threshold.")
                        elif config.CLASSIFICATION_MODE == 'binary':
                            scores_normal_if = -1 * model.decision_function(X_train_numeric_if[y_train == 0])
                            print("  Using binary fallback for IForest threshold.")
                        if len(scores_normal_if) > 0:
                            thresholds[name] = np.percentile(scores_normal_if, config.IFOREST_THRESHOLD_PERCENTILE)
                            print(f"  IForest Scores (Benign Train): Min={np.min(scores_normal_if):.4f}, Max={np.max(scores_normal_if):.4f}, Mean={np.mean(scores_normal_if):.4f}")
                            print(f"  Threshold for {name} ({config.IFOREST_THRESHOLD_PERCENTILE}th perc): {thresholds[name]:.6f}")
                            scores_for_plotting[name] = {'train_normal': scores_normal_if}
                        else:
                            print(f"  Warning: Could not get normal scores for {name} threshold.")
                            thresholds[name] = np.inf
                    else:
                        print(f"  Warning: No numeric data in X_train for {name} threshold.")
                        thresholds[name] = np.inf
                elif name == "Autoencoder":
                    if X_train_benign_ae is not None and not X_train_benign_ae.empty:
                        X_train_benign_numeric_ae = X_train_benign_ae.select_dtypes(include=np.number)
                        if not X_train_benign_numeric_ae.empty:
                            X_pred_benign_ae = model.predict(X_train_benign_numeric_ae)
                            mse_benign = np.mean(np.power(X_train_benign_numeric_ae.values - X_pred_benign_ae, 2), axis=1)
                            thresholds[name] = np.percentile(mse_benign, config.AE_THRESHOLD_PERCENTILE)
                            print(f"  AE MSE (Benign Train): Min={np.min(mse_benign):.6f}, Max={np.max(mse_benign):.6f}, Mean={np.mean(mse_benign):.6f}")
                            print(f"  Threshold for {name} ({config.AE_THRESHOLD_PERCENTILE}th perc): {thresholds[name]:.8f}")
                            scores_for_plotting[name] = {'train_normal': mse_benign}
                        else:
                            print(f"  Warning: No numeric benign data for {name} threshold.")
                            thresholds[name] = np.inf
                    else:
                        print(f"  Warning: Benign training data not available for {name} threshold.")
                        thresholds[name] = np.inf
        else:
            print(f"\nSkipping disabled model: {name}")

    print(f"\nModel Training & Threshold Calculation completed in {time.time() - start_time:.2f} seconds.")
    print(f"Models available for evaluation: {list(trained_models.keys())}")
    print(f"Calculated thresholds: {thresholds}")

    # --- Step 5.5: Plot Score Distributions (DEBUG) ---
    print("\n===== Step 5.5: Visualize Score Distributions =====")
    if X_test is not None:
        X_test_numeric = X_test.select_dtypes(include=np.number)
        if not X_test_numeric.empty:
            plot_path_dir = os.path.join(config.RESULT_DIR, "debug_plots")
            os.makedirs(plot_path_dir, exist_ok=True)
            for name, model in trained_models.items():
                if name in thresholds and name in scores_for_plotting:
                    plt.figure(figsize=(10, 6))
                    sns.histplot(scores_for_plotting[name]['train_normal'], color='blue', label='Normal Train Scores', kde=True, stat='density', bins=50)
                    test_scores = []
                    if name == "IsolationForest":
                        test_scores = -1 * model.decision_function(X_test_numeric)
                    elif name == "Autoencoder":
                        pred_test = model.predict(X_test_numeric)
                        test_scores = np.mean(np.power(X_test_numeric.values - pred_test, 2), axis=1)
                    if len(test_scores) > 0:
                        sns.histplot(test_scores, color='orange', label='Test Scores', kde=True, stat='density', bins=50, alpha=0.7)
                        print(f"  {name} Scores (Test): Min={np.min(test_scores):.6f}, Max={np.max(test_scores):.6f}, Mean={np.mean(test_scores):.6f}")
                    plt.axvline(thresholds[name], color='red', linestyle='--', label=f'Threshold ({thresholds[name]:.6f})')
                    plt.title(f"{name} Score Distribution (Train Normal vs Test)")
                    plt.legend()
                    plt.xlabel("Anomaly Score")
                    plt.ylabel("Density")
                    plt.savefig(os.path.join(plot_path_dir, f"{name}_score_distribution.png"))
                    plt.close()
                    print(f"  Saved score distribution plot for {name}.")
        else:
            print("Skipping score plots: X_test numeric is empty.")
    else:
        print("Skipping score plots: X_test is None.")

    # --- Step 6: Evaluation ---
    print("\n===== Step 6: Evaluate Models =====")
    start_time = time.time()
    all_results = []
    if not trained_models:
        print("No models were trained or loaded successfully. Skipping evaluation.")
    elif X_test is None or y_test is None:
        print("Test data not available. Skipping evaluation.")
    else:
        eval_label_encoder = current_label_encoder
        for name, model in trained_models.items():
            eval_threshold = thresholds.get(name, None)
            results = evaluator.evaluate_model(model, name, X_test, y_test, config.CLASSIFICATION_MODE, config.EVALUATION_METRICS, eval_label_encoder, eval_threshold)
            if results:
                all_results.append(results)

        if all_results:
            results_df = pd.DataFrame(all_results).set_index('model')
            print("\n--- Overall Results Summary ---")
            cols_to_show = [col for col in results_df.columns if 'matrix' not in col]
            print(results_df[cols_to_show])
            summary_path = os.path.join(config.RESULT_DIR, "evaluation_summary.csv")
            try:
                results_df.to_csv(summary_path)
                print(f"Evaluation summary saved to {summary_path}")
            except Exception as e:
                print(f"Error saving evaluation summary: {e}")
        else:
            print("No evaluation results generated.")

    print(f"\nEvaluation completed in {time.time() - start_time:.2f} seconds.")

    # --- Finish ---
    print(f"\n===== Pipeline Finished Successfully =====")
    print(f"Total execution time: {time.time() - start_time_total:.2f} seconds.")

if __name__ == "__main__":
    run_pipeline(
        force_reload=config.FORCE_RELOAD,
        force_clean=config.FORCE_CLEAN,
        force_engineer=config.FORCE_ENGINEER,
        force_train=config.FORCE_TRAIN
    )