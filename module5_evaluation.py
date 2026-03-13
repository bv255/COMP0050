# =============================================================================
# MODULE 5: Evaluation + Business Insights + Final Report
# Team Size: TBC (recommend 2 members)
# Deliverables: ROC curves, confusion matrices, model performance table,
#               final presentation and report
# Input: All outputs from Modules 1–4
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.metrics import (
    roc_curve, auc, confusion_matrix, RocCurveDisplay,
    precision_recall_curve, PrecisionRecallDisplay
)

# -----------------------------------------------------------------------------
# SECTION 5.1: Load All Models and Data
# -----------------------------------------------------------------------------

def load_all_models(model_dir: str) -> dict:
    """
    Load all trained models from Modules 3 & 4.

    TODO:
    - Load logistic_regression.pkl
    - Load random_forest.pkl
    - Load xgboost.pkl
    - Return dict: {model_name: model_object}
    """
    pass


def load_test_data(data_dir: str):
    """
    Load held-out test set from Module 2.

    TODO:
    - Load X_test, y_test
    - Confirm shapes
    - Return X_test, y_test
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 5.2: Final ROC Curves (All Models)
# -----------------------------------------------------------------------------

def plot_final_roc_curves(models: dict, X_test, y_test) -> None:
    """
    Publication-quality ROC curve plot comparing all three models.

    TODO:
    - Plot ROC curves for LR, RF, and XGBoost on one figure
    - Label each curve with model name and AUC score
    - Add random classifier baseline (dashed diagonal)
    - Use distinct colours and line styles for accessibility
    - Add title, axis labels, and legend
    - Save high-res to /outputs/final_roc_curves.png (dpi=300)
    - This figure will go directly into the report
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 5.3: Confusion Matrices (All Models)
# -----------------------------------------------------------------------------

def plot_all_confusion_matrices(models: dict, X_test, y_test) -> None:
    """
    Plot confusion matrices for all three models side by side.

    TODO:
    - Create a 1×3 subplot figure
    - For each model: compute and plot confusion matrix heatmap
    - Label axes: Predicted Churn / Predicted No Churn vs Actual
    - Annotate with counts AND percentages
    - Save to /outputs/final_confusion_matrices.png (dpi=300)
    - NOTE FOR REPORT: Discuss threshold sensitivity — at the default 0.5
      threshold, how many churners are missed?
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 5.4: Precision-Recall Curves (Optional but valuable for imbalanced data)
# -----------------------------------------------------------------------------

def plot_precision_recall_curves(models: dict, X_test, y_test) -> None:
    """
    Plot Precision-Recall curves — more informative than ROC for imbalanced classes.

    TODO:
    - Plot PR curves for all three models
    - Include average precision (AP) score in legend
    - Save to /outputs/precision_recall_curves.png
    - NOTE FOR REPORT: If class imbalance was noted in Module 1, this curve
      tells a more honest story than AUC-ROC
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 5.5: Final Model Performance Table
# -----------------------------------------------------------------------------

def compile_final_performance_table(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Create the definitive model comparison table for the report.

    TODO:
    - For each model compute: Accuracy, Precision, Recall, F1, AUC-ROC
    - Load CV results (mean ± std) from Module 3 & 4 outputs
    - Format as a clean DataFrame
    - Export to /outputs/final_model_performance.csv
    - Also export as LaTeX table (df.to_latex()) for direct use in report
    - Highlight best value per metric
    - Return DataFrame
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 5.6: Churn Risk Interpretation
# -----------------------------------------------------------------------------

def interpret_churn_risk(best_model, X_test, y_test, feature_names: list) -> None:
    """
    Translate model outputs into actionable churn risk insights.

    TODO:
    - Predict churn probabilities for test set customers
    - Segment customers into risk buckets:
        * High risk: prob > 0.7
        * Medium risk: 0.4 < prob ≤ 0.7
        * Low risk: prob ≤ 0.4
    - For high-risk segment: identify most common feature profiles
      (e.g. month-to-month contract + high monthly charges + low tenure)
    - OPTIONAL: Use SHAP to explain individual predictions
        * shap.TreeExplainer for RF/XGB
        * shap.LinearExplainer for LR
        * Generate summary plot and waterfall plot for a sample customer
    - Save risk distribution plot to /outputs/churn_risk_segments.png
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 5.7: Business Insights
# -----------------------------------------------------------------------------

def generate_business_insights(comparison_df: pd.DataFrame) -> None:
    """
    Synthesise findings into business-facing recommendations.

    TODO:
    - Summarise which model performed best and why it's recommended
    - Identify the top 3–5 churn drivers from feature importance analysis
    - Translate these into actionable retention strategies, e.g.:
        * Month-to-month contracts → offer annual contract incentives
        * High monthly charges → targeted discount or loyalty programme
        * Low tenure + tech support issues → proactive onboarding support
    - Discuss the cost trade-off: cost of false negatives (missed churners)
      vs false positives (unnecessary retention spend)
    - Suggest a decision threshold tuning approach (e.g. lower threshold
      to 0.3 if missing churners is very costly)
    - Print or export insights as a formatted text summary
    - NOTE FOR REPORT: This section directly addresses the "Critical Discussion"
      marking criterion
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 5.8: Limitations & Future Work
# -----------------------------------------------------------------------------

def document_limitations() -> None:
    """
    Critically reflect on the study's limitations for the report.

    TODO: Write notes on:
    - Dataset limitations (size, synthetic vs real, time period, geography)
    - Modelling limitations (no temporal modelling, static snapshot only)
    - Feature limitations (missing potentially important signals)
    - Evaluation limitations (single train/test split vs repeated CV)
    - Class imbalance handling — was SMOTE or other rebalancing considered?
    - What would you do differently with more time?
    - Possible extensions: survival analysis, deep learning, real-time scoring

    This function doesn't need to return anything — use it as a TODO checklist
    when writing the Critical Discussion section of the report.
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 5.9: Final Presentation Slides Structure
# -----------------------------------------------------------------------------

"""
PRESENTATION STRUCTURE — TODO for slide creation (PowerPoint / LaTeX Beamer)

Slide 1: Title
    - Project title, team names, module code, date

Slide 2: Problem & Motivation
    - What is customer churn? Why does it matter?
    - Dataset description (source, size, features)

Slide 3: EDA Highlights (Module 1)
    - Churn rate, key distributions, correlation heatmap
    - 2–3 most striking EDA findings

Slide 4: Feature Engineering (Module 2)
    - List of engineered features and rationale
    - Feature importance screening results

Slide 5: Baseline Model — Logistic Regression (Module 3)
    - Best hyperparameters, CV AUC
    - Top coefficients and interpretation

Slide 6: Advanced Models — RF & XGBoost (Module 4)
    - Best hyperparameters for each
    - Feature importance comparison

Slide 7: Model Comparison
    - Performance table (all metrics, all models)
    - ROC curve comparison figure

Slide 8: Business Insights
    - Top 3 churn drivers
    - Recommended retention actions
    - Suggested decision threshold

Slide 9: Limitations & Future Work
    - 3–4 honest limitations
    - 2–3 future directions

Slide 10: Conclusion
    - Best model recommendation + rationale
    - Key takeaway message

TODO: Assign slides to team members. Keep each slide to 3–5 bullet points max.
"""

# =============================================================================
# MAIN PIPELINE
# =============================================================================

if __name__ == "__main__":
    MODEL_DIR = "models/"
    DATA_DIR = "data/processed/"

    # Step 1: Load everything
    models = load_all_models(MODEL_DIR)
    X_test, y_test = load_test_data(DATA_DIR)
    feature_names = list(X_test.columns)

    # Step 2: ROC curves
    plot_final_roc_curves(models, X_test, y_test)

    # Step 3: Confusion matrices
    plot_all_confusion_matrices(models, X_test, y_test)

    # Step 4: Precision-Recall curves
    plot_precision_recall_curves(models, X_test, y_test)

    # Step 5: Performance table
    comparison_df = compile_final_performance_table(models, X_test, y_test)
    print(comparison_df)

    # Step 6: Churn risk interpretation
    # TODO: Select best model from comparison_df
    best_model_name = comparison_df["ROC-AUC"].idxmax()
    best_model = models[best_model_name]
    interpret_churn_risk(best_model, X_test, y_test, feature_names)

    # Step 7: Business insights
    generate_business_insights(comparison_df)

    # Step 8: Limitations (refer to document_limitations() as a checklist)
    document_limitations()

    print("✅ Module 5 complete. All evaluation outputs saved to /outputs/")
    print("📝 Next step: Write report using outputs. See presentation structure above.")
