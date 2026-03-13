# =============================================================================
# MODULE 4: Advanced Models — Random Forest & XGBoost
# Team Size: 1 member
# Deliverables: RF & XGBoost models, feature importance plots,
#               model comparison table
# Input: data/processed/ splits from Module 2
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, RocCurveDisplay
)
import xgboost as xgb
import pickle

# TODO: pip install xgboost if not installed

# -----------------------------------------------------------------------------
# SECTION 4.1: Load Processed Data
# -----------------------------------------------------------------------------

def load_processed_data(data_dir: str):
    """
    Load train/test splits from Module 2 output.

    TODO:
    - Load X_train, X_test, y_train, y_test from CSVs
    - Confirm shapes match Module 3 input (consistency check)
    - Return X_train, X_test, y_train, y_test
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 4.2: Train Random Forest
# -----------------------------------------------------------------------------

def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Fit a default Random Forest as starting point.

    TODO:
    - Instantiate RandomForestClassifier with:
        * n_estimators=100
        * random_state=42
        * class_weight='balanced' if dataset is imbalanced (check Module 1)
    - Fit on X_train, y_train
    - Print OOB score if oob_score=True
    - Return fitted model
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 4.3: Tune Random Forest
# -----------------------------------------------------------------------------

def tune_random_forest(X_train, y_train) -> RandomForestClassifier:
    """
    Hyperparameter tuning for Random Forest.

    TODO:
    - Define param_grid or param_distributions:
        * n_estimators: [100, 200, 300]
        * max_depth: [None, 10, 20, 30]
        * min_samples_split: [2, 5, 10]
        * min_samples_leaf: [1, 2, 4]
        * max_features: ['sqrt', 'log2']
    - Use RandomizedSearchCV (faster than GridSearch for large grids):
        * n_iter=20, cv=5, scoring='roc_auc', random_state=42
    - Print best params and best CV AUC
    - Return best estimator
    - NOTE FOR REPORT: Discuss how max_depth controls overfitting
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 4.4: Train XGBoost
# -----------------------------------------------------------------------------

def train_xgboost(X_train, y_train) -> xgb.XGBClassifier:
    """
    Fit a default XGBoost classifier.

    TODO:
    - Instantiate XGBClassifier with:
        * use_label_encoder=False
        * eval_metric='logloss'
        * random_state=42
        * scale_pos_weight = neg_count / pos_count  (handles class imbalance)
    - Fit on X_train, y_train
    - Return fitted model
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 4.5: Tune XGBoost
# -----------------------------------------------------------------------------

def tune_xgboost(X_train, y_train) -> xgb.XGBClassifier:
    """
    Hyperparameter tuning for XGBoost.

    TODO:
    - Define param_grid:
        * n_estimators: [100, 200, 300]
        * max_depth: [3, 5, 7]
        * learning_rate: [0.01, 0.05, 0.1, 0.2]
        * subsample: [0.7, 0.8, 1.0]
        * colsample_bytree: [0.7, 0.8, 1.0]
        * reg_alpha (L1): [0, 0.1, 1]
        * reg_lambda (L2): [1, 5, 10]
    - Use RandomizedSearchCV: n_iter=20, cv=5, scoring='roc_auc'
    - Print best params and best CV AUC
    - Return best estimator
    - NOTE FOR REPORT: Contrast XGBoost regularisation (alpha/lambda) with
      LR regularisation (C) from Module 3
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 4.6: Cross-Validation (Both Models)
# -----------------------------------------------------------------------------

def cross_validate_model(model, X_train, y_train, model_name: str, cv: int = 5) -> dict:
    """
    Evaluate model stability with k-fold cross-validation.

    TODO:
    - Run cross_val_score with scoring='roc_auc', cv=5
    - Print mean ± std for the model
    - Return dict {model_name: {'mean': ..., 'std': ...}}
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 4.7: Evaluate on Test Set
# -----------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Generate full evaluation metrics on the held-out test set.

    TODO:
    - Predict y_pred and y_prob (predict_proba[:,1])
    - Calculate: accuracy, precision, recall, F1, ROC-AUC
    - Return metrics dict for model comparison table (used in Module 5)
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 4.8: Feature Importance Plots
# -----------------------------------------------------------------------------

def plot_feature_importance(model, feature_names: list, model_name: str, top_n: int = 20) -> None:
    """
    Plot and save feature importance for tree-based models.

    TODO:
    - Extract feature importances:
        * RF: model.feature_importances_
        * XGB: model.feature_importances_ or plot_importance(model)
    - Sort top N features by importance
    - Plot horizontal bar chart
    - Save to /outputs/feature_importance_{model_name}.png
    - NOTE FOR REPORT: Compare top features across RF, XGB, and LR coefficients —
      do they agree? Disagreements are interesting to discuss
    - OPTIONAL: Use SHAP values for more interpretable importances
        * pip install shap
        * shap.TreeExplainer for RF/XGB
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 4.9: Model Comparison Table
# -----------------------------------------------------------------------------

def build_comparison_table(metrics_dict: dict) -> pd.DataFrame:
    """
    Compile all model metrics into a single comparison DataFrame.

    TODO:
    - metrics_dict structure: {model_name: {metric: value, ...}}
    - Include: Logistic Regression (from Module 3), RF, XGBoost
    - Columns: Model | Accuracy | Precision | Recall | F1 | ROC-AUC | CV AUC
    - Save as CSV to /outputs/model_comparison.csv
    - Print formatted table
    - Return DataFrame (will be used in Module 5 report)
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 4.10: ROC Curve Comparison
# -----------------------------------------------------------------------------

def plot_roc_comparison(models: dict, X_test, y_test) -> None:
    """
    Plot all model ROC curves on a single chart.

    TODO:
    - models: dict {model_name: fitted_model}
    - Plot each model's ROC curve with AUC in legend label
    - Include LR model loaded from models/logistic_regression.pkl
    - Add diagonal reference line (random classifier)
    - Save to /outputs/roc_comparison.png
    - This will be included in Module 5 final report
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 4.11: Save Models
# -----------------------------------------------------------------------------

def save_models(models: dict, output_dir: str) -> None:
    """
    Persist trained models for Module 5.

    TODO:
    - Save each model as a .pkl file in output_dir
    - Print paths for each saved model
    """
    pass


# =============================================================================
# MAIN PIPELINE
# =============================================================================

if __name__ == "__main__":
    DATA_DIR = "data/processed/"
    MODEL_DIR = "models/"

    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_processed_data(DATA_DIR)
    feature_names = list(X_train.columns)

    # Step 2 & 3: Random Forest
    rf_model = train_random_forest(X_train, y_train)
    rf_best = tune_random_forest(X_train, y_train)

    # Step 4 & 5: XGBoost
    xgb_model = train_xgboost(X_train, y_train)
    xgb_best = tune_xgboost(X_train, y_train)

    # Step 6: Cross-validation
    rf_cv = cross_validate_model(rf_best, X_train, y_train, "Random Forest")
    xgb_cv = cross_validate_model(xgb_best, X_train, y_train, "XGBoost")

    # Step 7: Test set evaluation
    rf_metrics = evaluate_model(rf_best, X_test, y_test, "Random Forest")
    xgb_metrics = evaluate_model(xgb_best, X_test, y_test, "XGBoost")

    # Step 8: Feature importance
    plot_feature_importance(rf_best, feature_names, "random_forest")
    plot_feature_importance(xgb_best, feature_names, "xgboost")

    # Step 9: Comparison table (include LR metrics loaded from Module 3 output)
    # TODO: Load LR metrics saved from Module 3
    lr_metrics = {}  # placeholder — load from Module 3 output
    all_metrics = {"Logistic Regression": lr_metrics, "Random Forest": rf_metrics, "XGBoost": xgb_metrics}
    comparison_df = build_comparison_table(all_metrics)
    print(comparison_df)

    # Step 10: ROC comparison
    models = {"Random Forest": rf_best, "XGBoost": xgb_best}
    plot_roc_comparison(models, X_test, y_test)

    # Step 11: Save models
    save_models({"random_forest": rf_best, "xgboost": xgb_best}, MODEL_DIR)

    print("✅ Module 4 complete. Models saved to:", MODEL_DIR)
