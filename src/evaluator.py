# src/evaluator.py
import pandas as pd
import numpy as np
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                            roc_auc_score, confusion_matrix, classification_report)
from sklearn.preprocessing import LabelBinarizer
import matplotlib.pyplot as plt
import seaborn as sns
from src import config, utils
import os

try:
    import tensorflow as tf
    TENSORFLOW_AVAILABLE = True
except ImportError:
    TENSORFLOW_AVAILABLE = False

def plot_correlation_heatmap(df, label_col, output_dir):
    """Plots the correlation heatmap for numeric features."""
    print("Generating correlation heatmap...")
    plot_path_dir = os.path.join(output_dir, "eda_plots")
    os.makedirs(plot_path_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include=np.number).columns
    if label_col in numeric_cols:
        numeric_cols = numeric_cols.drop(label_col)
    if len(numeric_cols) < 2:
        print("Skipping: Not enough numeric features.")
        return
    plt.figure(figsize=(18, 14))
    correlation_matrix = df[numeric_cols].corr()
    sns.heatmap(correlation_matrix, cmap='coolwarm', annot=False, fmt=".2f")
    plt.title('Feature Correlation Matrix', fontsize=16)
    plt.xticks(rotation=90, fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plot_path = os.path.join(plot_path_dir, "correlation_heatmap.png")
    plt.savefig(plot_path, dpi=150, bbox_inches='tight')
    print(f"Correlation heatmap saved to {plot_path}")
    plt.close()

def plot_feature_distributions(df, label_col, output_dir, num_features_to_plot=20):
    """Plots distributions for a subset of numeric features."""
    print(f"Generating feature distribution plots (max {num_features_to_plot})...")
    plot_path_dir = os.path.join(output_dir, "eda_plots")
    os.makedirs(plot_path_dir, exist_ok=True)
    numeric_cols = df.select_dtypes(include=np.number).columns
    if label_col in numeric_cols:
        numeric_cols = numeric_cols.drop(label_col)
    if numeric_cols.empty:
        print("Skipping: No numeric features.")
        return
    cols_to_plot = numeric_cols[:min(len(numeric_cols), num_features_to_plot)]
    num_plots = len(cols_to_plot)
    num_cols_grid = 4
    num_rows_grid = (num_plots + num_cols_grid - 1) // num_cols_grid
    fig, axes = plt.subplots(num_rows_grid, num_cols_grid, figsize=(5 * num_cols_grid, 4 * num_rows_grid))
    if num_plots == 0:
        plt.close(fig)
        return
    axes = axes.flatten()
    plot_count = 0
    for i, col in enumerate(cols_to_plot):
        if i >= len(axes):
            break
        try:
            if df[col].nunique() > 1:
                sns.histplot(df[col], kde=True, ax=axes[i], bins=50, log_scale=False)
                axes[i].set_title(col, fontsize=10)
            else:
                axes[i].hist(df[col].dropna(), bins=1)
                axes[i].set_title(f"{col} (Constant)", fontsize=10)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('')
            plot_count += 1
        except Exception as e:
            print(f"Could not plot distribution for {col}: {e}")
    for j in range(plot_count, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle('Distribution of Key Features', fontsize=16, y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(plot_path_dir, "feature_distributions.png")
    plt.savefig(plot_path, dpi=120)
    print(f"Feature distribution plots saved to {plot_path}")
    plt.close(fig)

def plot_feature_vs_label(df_input, label_col, class_names_map, output_dir, num_features_to_plot=15):
    """Plots feature vs label."""
    print(f"Generating feature vs label plots (max {num_features_to_plot})...")
    plot_path_dir = os.path.join(output_dir, "eda_plots")
    os.makedirs(plot_path_dir, exist_ok=True)
    df = df_input.copy()
    plot_label_name_col = label_col
    if pd.api.types.is_numeric_dtype(df[label_col]):
        df['label_name'] = df[label_col].map(class_names_map)
        if df['label_name'].isnull().any():
            print("Warning: Label mapping resulted in NaNs. Using numeric labels.")
            plot_label_name_col = label_col
            df.drop(columns=['label_name'], inplace=True, errors='ignore')
        else:
            plot_label_name_col = 'label_name'
    elif not pd.api.types.is_string_dtype(df[label_col]) and not pd.api.types.is_categorical_dtype(df[label_col]):
        try:
            df[plot_label_name_col] = df[label_col].astype(str)
            print(f"Warning: Converted label column '{label_col}' to string for plotting.")
        except:
            print("ERROR: Could not convert label to string. Skipping plot.")
            return
    if plot_label_name_col not in df.columns:
        print(f"ERROR: Plot label column '{plot_label_name_col}' not found. Skipping plot.")
        return
    numeric_cols_plot = df.select_dtypes(include=np.number).columns
    if numeric_cols_plot.empty:
        print("Skipping: No numeric features.")
        return
    cols_to_plot = numeric_cols_plot[:min(len(numeric_cols_plot), num_features_to_plot)]
    num_plots = len(cols_to_plot)
    num_cols_grid = 3
    num_rows_grid = (num_plots + num_cols_grid - 1) // num_cols_grid
    fig, axes = plt.subplots(num_rows_grid, num_cols_grid, figsize=(7 * num_cols_grid, 6 * num_rows_grid))
    if num_plots == 0:
        plt.close(fig)
        return
    axes = axes.flatten()
    unique_labels_plot = sorted(df[plot_label_name_col].unique())
    plot_count = 0
    for i, col in enumerate(cols_to_plot):
        if i >= len(axes):
            break
        try:
            sns.violinplot(data=df, x=plot_label_name_col, y=col, ax=axes[i], cut=0, order=unique_labels_plot, scale='width', inner=None)
            axes[i].set_title(f'{col} vs Label', fontsize=10)
            axes[i].set_xlabel('Label', fontsize=9)
            axes[i].set_ylabel(col, fontsize=9)
            axes[i].tick_params(axis='x', rotation=60, labelsize=8, ha='right')
            plot_count += 1
        except Exception as e:
            print(f"Could not plot feature vs label for {col}: {e}")
    for j in range(plot_count, len(axes)):
        fig.delaxes(axes[j])
    fig.suptitle('Feature Distributions per Class Label', fontsize=16, y=1.02)
    plt.tight_layout()
    plot_path = os.path.join(plot_path_dir, "feature_vs_label_plots.png")
    plt.savefig(plot_path, dpi=120, bbox_inches='tight')
    print(f"Feature vs Label plots saved to {plot_path}")
    plt.close(fig)

def evaluate_model(model, model_name, X_test_input, y_test_input, mode, metrics, label_encoder=None, threshold=None):
    """Evaluates a trained model based on classification mode."""
    print(f"\n--- Evaluating {model_name} (Mode: {mode}) ---")
    results = {"model": model_name}
    X_test = X_test_input.copy()
    y_test = y_test_input.copy()

    X_eval = X_test
    models_needing_numeric = ["LogisticRegression", "XGBoost", "IsolationForest", "Autoencoder"]
    if model_name in models_needing_numeric:
        X_eval = X_test.select_dtypes(include=np.number)
        if X_eval.empty:
            print(f"ERROR: Numeric data empty for {model_name}. Skip eval.")
            return None
    if X_eval.empty:
        print(f"ERROR: Eval data empty for {model_name}. Skip eval.")
        return None

    y_pred, y_pred_proba = None, None
    target_names, eval_mode, avg_method = None, mode, None
    metrics_to_run = list(metrics)

    try:
        if model_name == "IsolationForest":
            print("Predicting with Isolation Forest...")
            scores_if = -1 * model.decision_function(X_eval)
            print(f"  IForest Test Scores: Min={np.min(scores_if):.6f}, Max={np.max(scores_if):.6f}, Mean={np.mean(scores_if):.6f}")
            if threshold is None or not np.isfinite(threshold):
                print("  WARNING: Invalid/No threshold provided for Isolation Forest. Predicting all as normal (0).")
                threshold = np.inf
                y_pred = np.zeros(len(X_eval), dtype=int)
            else:
                print(f"  Using IForest Threshold: {threshold:.6f}")
                y_pred = (scores_if > threshold).astype(int)
            target_names = ['BENIGN', 'ATTACK']
            eval_mode = 'binary'
            avg_method = config.AVERAGING_METHOD_BINARY
            if 'roc_auc' in metrics_to_run:
                metrics_to_run.remove('roc_auc')
        elif model_name == "Autoencoder":
            print("Predicting with Autoencoder...")
            if not TENSORFLOW_AVAILABLE or model is None:
                print("Skipping AE eval.")
                return None
            X_pred_ae = model.predict(X_eval)
            if X_eval.shape[1] != X_pred_ae.shape[1]:
                print(f"ERROR: AE input/pred dimensions mismatch. Skipping.")
                return None
            mse = np.mean(np.power(X_eval.values - X_pred_ae, 2), axis=1)
            print(f"  AE Test MSE: Min={np.min(mse):.8f}, Max={np.max(mse):.8f}, Mean={np.mean(mse):.8f}")
            if threshold is None or not np.isfinite(threshold):
                print("  WARNING: Invalid/No threshold provided for Autoencoder. Predicting all as normal (0).")
                threshold = np.inf
                y_pred = np.zeros(len(X_eval), dtype=int)
            else:
                print(f"  Using AE Threshold: {threshold:.8f}")
                y_pred = (mse > threshold).astype(int)
            target_names = ['BENIGN', 'ATTACK']
            eval_mode = 'binary'
            avg_method = config.AVERAGING_METHOD_BINARY
            if 'roc_auc' in metrics_to_run:
                metrics_to_run.remove('roc_auc')
        else:
            print(f"Predicting with {model_name}...")
            y_pred = model.predict(X_eval)
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_eval)
            if mode == 'binary':
                target_names = ['BENIGN', 'ATTACK']
                avg_method = config.AVERAGING_METHOD_BINARY
            elif mode == 'multiclass' and label_encoder is not None:
                target_names = list(label_encoder.classes_)
                avg_method = config.AVERAGING_METHOD_MULTICLASS
            else:
                unique_labels_test = sorted(np.unique(y_test))
                target_names = [str(i) for i in unique_labels_test]
                avg_method = config.AVERAGING_METHOD_MULTICLASS
    except Exception as pred_err:
        print(f"ERROR during prediction for {model_name}: {pred_err}")
        import traceback
        traceback.print_exc()
        return None
    if y_pred is None:
        print(f"No predictions for {model_name}. Skip metrics.")
        return None
    y_pred_unique, y_pred_counts = np.unique(y_pred, return_counts=True)
    print(f"\nUnique predicted values for {model_name}: {dict(zip(y_pred_unique, y_pred_counts))}")
    can_calc_binary_metrics = len(y_pred_unique) > 1
    print(f"Calculating metrics ({eval_mode}, avg: {avg_method})...")
    for metric in metrics_to_run:
        score = np.nan
        metric_key = f"{metric}_{avg_method}" if eval_mode != 'binary' or metric == 'accuracy' else metric
        try:
            if metric in ['precision', 'recall', 'f1'] and eval_mode == 'binary' and not can_calc_binary_metrics:
                print(f"  Skipping {metric} for {model_name}: Only one class predicted.")
                score = 0.0
            elif metric == 'accuracy':
                score = accuracy_score(y_test, y_pred)
            elif metric == 'precision':
                score = precision_score(y_test, y_pred, average=avg_method, zero_division=0)
            elif metric == 'recall':
                score = recall_score(y_test, y_pred, average=avg_method, zero_division=0)
            elif metric == 'f1':
                score = f1_score(y_test, y_pred, average=avg_method, zero_division=0)
            elif metric == 'roc_auc' and y_pred_proba is not None:
                if y_test.nunique() < 2:
                    print(f"  Skipping ROC AUC for {model_name}: Only one class in y_test.")
                elif eval_mode == 'binary':
                    score = roc_auc_score(y_test, y_pred_proba[:, 1])
                elif eval_mode == 'multiclass':
                    n_classes = len(target_names)
                    if y_pred_proba.shape[1] == n_classes:
                        lb = LabelBinarizer()
                        lb.fit(y_test)
                        y_test_bin = lb.transform(y_test)
                        if y_test_bin.shape[1] == 1 and n_classes == 2:
                            y_test_bin = np.hstack((1 - y_test_bin, y_test_bin))
                        if y_test_bin.shape[1] == y_pred_proba.shape[1]:
                            score = roc_auc_score(y_test_bin, y_pred_proba, average=avg_method, multi_class='ovr')
                        else:
                            print(f"  Shape mismatch ROC AUC: y_bin{y_test_bin.shape}, proba{y_pred_proba.shape}")
                    else:
                        print(f"  Proba shape {y_pred_proba.shape} != n_classes {n_classes} for ROC AUC.")
            results[metric_key] = score
            if not np.isnan(score):
                print(f"  {metric.capitalize()} ({avg_method if eval_mode != 'binary' else 'Overall'}): {score:.4f}")
        except ValueError as ve:
            print(f"  ValueError calculating {metric} ({metric_key}): {ve}")
            results[metric_key] = np.nan
        except Exception as e:
            print(f"  Could not calculate {metric} ({metric_key}): {e}")
            results[metric_key] = np.nan
    print("\nClassification Report:")
    try:
        present_labels = sorted(np.unique(np.concatenate((y_test, y_pred))))
        report_target_names = None
        if target_names and len(target_names) >= (max(present_labels) + 1):
            try:
                report_target_names = [target_names[i] for i in present_labels]
            except IndexError:
                print("  Warning: Label index mismatch for report names.")
                report_target_names = None
        print(classification_report(y_test, y_pred, labels=present_labels, target_names=report_target_names, zero_division=0))
    except Exception as e:
        print(f"  Could not generate report: {e}\n", classification_report(y_test, y_pred, zero_division=0))
    try:
        cm = confusion_matrix(y_test, y_pred, labels=present_labels)
        results['confusion_matrix_array'] = cm.tolist()
        print("\nConfusion Matrix:")
        print(cm)
        plot_cm_names = report_target_names if report_target_names else [str(l) for l in present_labels]
        if plot_cm_names:
            plot_confusion_matrix(cm, plot_cm_names, model_name, config.RESULT_DIR)
    except Exception as cm_e:
        print(f"Error generating confusion matrix: {cm_e}")
        results['confusion_matrix_array'] = None
    return results

def plot_confusion_matrix(cm, class_names, model_name, output_dir):
    """Plots and saves the confusion matrix."""
    plot_path_dir = os.path.join(output_dir, "evaluation_plots")
    os.makedirs(plot_path_dir, exist_ok=True)
    plt.figure(figsize=(max(6, len(class_names) * 0.7), max(5, len(class_names) * 0.5)))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, annot_kws={"size": 8})
    plt.title(f'Confusion Matrix: {model_name}', fontsize=12)
    plt.ylabel('Actual Label', fontsize=10)
    plt.xlabel('Predicted Label', fontsize=10)
    plt.xticks(rotation=60, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    plot_path = os.path.join(plot_path_dir, f"confusion_matrix_{model_name}.png")
    try:
        plt.savefig(plot_path, dpi=100, bbox_inches='tight')
        print(f"Confusion matrix plot saved to {plot_path}")
    except Exception as e:
        print(f"Error saving confusion matrix plot: {e}")
    plt.close()