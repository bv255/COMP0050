# =============================================================================
# MODULE 3: Baseline Model — Logistic Regression
# Team Size: 1 member
# Deliverables: Logistic regression model results, coefficient analysis,
#               baseline performance metrics
# Input: data/processed/ splits from Module 2
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    RocCurveDisplay
)
import pickle

# -----------------------------------------------------------------------------
# SECTION 3.1: Load Processed Data
# -----------------------------------------------------------------------------

def load_processed_data(data_dir: str):
    """
    Load train/test splits from Module 2 output.

    TODO:
    - Load X_train, X_test, y_train, y_test from CSVs in data_dir
    - Confirm shapes and class balance
    - Return X_train, X_test, y_train, y_test
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 3.2: Train Logistic Regression (Default)
# -----------------------------------------------------------------------------

def train_baseline_lr(X_train, y_train) -> LogisticRegression:
    """
    Fit a default Logistic Regression as the initial baseline.

    TODO:
    - Instantiate LogisticRegression with:
        * solver='lbfgs' (good default)
        * max_iter=1000 (increase if convergence warning)
        * random_state=42
    - Fit on X_train, y_train
    - Print training accuracy as a sanity check
    - Return fitted model
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 3.3: Hyperparameter Tuning
# -----------------------------------------------------------------------------

def tune_logistic_regression(X_train, y_train) -> LogisticRegression:
    """
    Use GridSearchCV to find optimal hyperparameters.

    TODO:
    - Define param_grid:
        * C: [0.001, 0.01, 0.1, 1, 10, 100]  (inverse regularisation strength)
        * penalty: ['l1', 'l2']
        * solver: ['liblinear', 'saga']  (needed for l1)
    - Run GridSearchCV with cv=5, scoring='roc_auc'
    - Print best params and best CV score
    - Return best estimator
    - NOTE FOR REPORT: Discuss effect of C on bias-variance trade-off,
      and L1 vs L2 regularisation (links to lecture material)
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 3.4: Cross-Validation
# -----------------------------------------------------------------------------

def cross_validate_model(model, X_train, y_train, cv: int = 5) -> dict:
    """
    Evaluate model stability with k-fold cross-validation.

    TODO:
    - Run cross_val_score with scoring='roc_auc', cv=5
    - Print mean and std of CV scores
    - Return dict of scores for comparison table in Module 5
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 3.5: Evaluate on Test Set
# -----------------------------------------------------------------------------

def evaluate_model(model, X_test, y_test, model_name: str = "Logistic Regression") -> dict:
    """
    Generate full evaluation metrics on the held-out test set.

    TODO:
    - Predict y_pred and y_prob (predict_proba[:,1])
    - Calculate: accuracy, precision, recall, F1, ROC-AUC
    - Print classification_report
    - Return metrics dict (for comparison table in Module 5)
    - NOTE: If dataset is imbalanced, focus on F1 and AUC over accuracy
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 3.6: Confusion Matrix
# -----------------------------------------------------------------------------

def plot_confusion_matrix(model, X_test, y_test, model_name: str = "Logistic Regression") -> None:
    """
    Plot and save confusion matrix.

    TODO:
    - Generate confusion matrix
    - Plot as heatmap with labels (True Churn, False Churn etc.)
    - Annotate with TP, TN, FP, FN counts
    - Save to /outputs/confusion_matrix_lr.png
    - NOTE FOR REPORT: Discuss business cost of FP vs FN in churn context
      (missing a churner is usually more costly than a false alarm)
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 3.7: ROC Curve
# -----------------------------------------------------------------------------

def plot_roc_curve(model, X_test, y_test, model_name: str = "Logistic Regression") -> None:
    """
    Plot ROC curve and display AUC score.

    TODO:
    - Use RocCurveDisplay.from_estimator or manual roc_curve
    - Label curve with AUC value
    - Save to /outputs/roc_curve_lr.png
    - This will be overlaid with Module 4 models in Module 5
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 3.8: Coefficient Analysis
# -----------------------------------------------------------------------------

def analyse_coefficients(model, feature_names: list) -> pd.DataFrame:
    """
    Interpret Logistic Regression coefficients.

    TODO:
    - Extract model.coef_[0] and pair with feature_names
    - Sort by absolute coefficient value (descending)
    - Plot horizontal bar chart (positive = increases churn risk,
      negative = decreases churn risk)
    - Save to /outputs/lr_coefficients.png
    - Return DataFrame: feature | coefficient | direction
    - NOTE FOR REPORT: Tie top coefficients back to EDA findings from Module 1
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 3.9: Save Model
# -----------------------------------------------------------------------------

def save_model(model, output_path: str) -> None:
    """
    Persist the trained model for use in Module 5.

    TODO:
    - Save model using pickle to output_path
    - Print confirmation
    """
    pass


# =============================================================================
# MAIN PIPELINE
# =============================================================================

if __name__ == "__main__":
    DATA_DIR = "data/processed/"
    MODEL_OUTPUT_PATH = "models/logistic_regression.pkl"

    # Step 1: Load data
    X_train, X_test, y_train, y_test = load_processed_data(DATA_DIR)

    # Step 2: Baseline (untuned) model
    baseline_model = train_baseline_lr(X_train, y_train)

    # Step 3: Tune hyperparameters
    best_model = tune_logistic_regression(X_train, y_train)

    # Step 4: Cross-validation
    cv_scores = cross_validate_model(best_model, X_train, y_train)

    # Step 5: Evaluate on test set
    feature_names = list(X_train.columns)
    metrics = evaluate_model(best_model, X_test, y_test)
    print("Test Metrics:", metrics)

    # Step 6: Confusion matrix
    plot_confusion_matrix(best_model, X_test, y_test)

    # Step 7: ROC curve
    plot_roc_curve(best_model, X_test, y_test)

    # Step 8: Coefficient analysis
    coef_df = analyse_coefficients(best_model, feature_names)
    print(coef_df)

    # Step 9: Save model
    save_model(best_model, MODEL_OUTPUT_PATH)

    print("✅ Module 3 complete. Baseline LR model saved to:", MODEL_OUTPUT_PATH)
