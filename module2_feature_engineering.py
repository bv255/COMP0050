# =============================================================================
# MODULE 2: Feature Engineering
# Team Size: 1 member
# Deliverables: Processed modelling dataset, feature importance screening,
#               engineered feature table
# Input: data/processed/cleaned_churn.csv (from Module 1)
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2, f_classif
# TODO: Consider adding: from sklearn.inspection import permutation_importance

# -----------------------------------------------------------------------------
# SECTION 2.1: Load Cleaned Data
# -----------------------------------------------------------------------------

def load_cleaned_data(filepath: str) -> pd.DataFrame:
    """
    Load the cleaned dataset produced by Module 1.

    TODO:
    - Load CSV
    - Separate features (X) and target (y = 'Churn')
    - Confirm no missing values remain
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 2.2: Encode Categorical Variables
# -----------------------------------------------------------------------------

def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode non-numeric categorical columns.

    TODO:
    - Identify all categorical columns (dtype == object)
    - For BINARY categoricals (e.g. Yes/No, Male/Female):
        * Use Label Encoding (0/1)
    - For MULTI-CLASS categoricals with no ordinal relationship
      (e.g. InternetService, Contract, PaymentMethod):
        * Use One-Hot Encoding (drop_first=True to avoid dummy trap)
    - Keep track of which columns were encoded and how — log this for report
    - Return transformed DataFrame
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 2.3: Scale Numerical Features
# -----------------------------------------------------------------------------

def scale_numericals(df: pd.DataFrame, method: str = "standard") -> pd.DataFrame:
    """
    Scale numerical features for model readiness.

    TODO:
    - Identify numerical columns (e.g. tenure, MonthlyCharges, TotalCharges)
    - Apply chosen scaling method:
        * 'standard' → StandardScaler (zero mean, unit variance) — best for LR
        * 'minmax'   → MinMaxScaler (0 to 1 range)
    - Fit scaler on TRAINING set only — apply to test set (do the train/test
      split before calling this function, or return the fitted scaler)
    - Save the fitted scaler object (pickle) for use in Module 3 & 4 inference
    - Return scaled DataFrame + fitted scaler
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 2.4: Create Behavioural Indicator Features
# -----------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features that may capture churn-driving behaviour.

    TODO:
    - Tenure groups: bucket tenure into bins (e.g. 0-12, 13-24, 25-48, 48+)
      and create a categorical or ordinal column 'tenure_group'
    - Usage ratio: MonthlyCharges / TotalCharges (watch for div by zero)
    - Avg monthly spend: TotalCharges / (tenure + 1)
    - Service count: count how many add-on services each customer has
      (e.g. OnlineSecurity, OnlineBackup, StreamingTV...)
    - Contract risk flag: binary flag for month-to-month contracts
    - TODO: Brainstorm any other domain-specific features relevant to churn
    - Log all new features in a feature_notes.md for the report
    - Return DataFrame with new features appended
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 2.5: Train / Test Split
# -----------------------------------------------------------------------------

def split_data(df: pd.DataFrame, target_col: str = "Churn", test_size: float = 0.2,
               random_state: int = 42):
    """
    Split data into train and test sets.

    TODO:
    - Use stratified split to preserve churn ratio in both sets
    - Print class balance in train and test splits to confirm stratification
    - Return X_train, X_test, y_train, y_test
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 2.6: Feature Importance Screening
# -----------------------------------------------------------------------------

def screen_feature_importance(X_train, y_train) -> pd.DataFrame:
    """
    Initial feature importance screening before modelling.

    TODO:
    - Option A: Use SelectKBest with f_classif (ANOVA F-score) for numerical
    - Option B: Train a quick Random Forest and use .feature_importances_
    - Option C: Compute point-biserial correlation with the target
    - Rank features by importance score
    - Plot a horizontal bar chart of top N features
    - Save chart to /outputs/feature_importance_screen.png
    - Return a DataFrame: feature name | importance score | rank
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 2.7: Save Processed Dataset
# -----------------------------------------------------------------------------

def save_processed_data(X_train, X_test, y_train, y_test, output_dir: str) -> None:
    """
    Save all processed splits for use in Modules 3, 4, and 5.

    TODO:
    - Save X_train, X_test, y_train, y_test as separate CSVs in output_dir
    - Save the fitted scaler as a pickle file
    - Save feature names list (post-encoding) for interpretability later
    - Print saved file paths
    """
    pass


# =============================================================================
# MAIN PIPELINE
# =============================================================================

if __name__ == "__main__":
    CLEANED_DATA_PATH = "data/processed/cleaned_churn.csv"
    PROCESSED_OUTPUT_DIR = "data/processed/"

    # Step 1: Load
    df = load_cleaned_data(CLEANED_DATA_PATH)

    # Step 2: Encode categoricals
    df = encode_categoricals(df)

    # Step 3: Engineer new features
    df = engineer_features(df)

    # Step 4: Split (before scaling to avoid data leakage)
    X_train, X_test, y_train, y_test = split_data(df, target_col="Churn")

    # Step 5: Scale numericals (fit on train only)
    X_train, scaler = scale_numericals(X_train, method="standard")
    X_test, _ = scale_numericals(X_test, method="standard")  # TODO: apply fitted scaler

    # Step 6: Feature importance screening
    importance_df = screen_feature_importance(X_train, y_train)
    print(importance_df)

    # Step 7: Save
    save_processed_data(X_train, X_test, y_train, y_test, PROCESSED_OUTPUT_DIR)

    print("✅ Module 2 complete. Processed data saved to:", PROCESSED_OUTPUT_DIR)
