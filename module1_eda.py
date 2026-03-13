# =============================================================================
# MODULE 1: Exploratory Data Analysis (EDA)
# Team Size: 2 members
# Deliverables: Cleaned dataset CSVs, churn distribution plots,
#               correlation heatmap, interactive EDA dashboard
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# TODO: pip install plotly if using interactive dashboard
# import plotly.express as px
# import plotly.graph_objects as go

# -----------------------------------------------------------------------------
# SECTION 1.1: Data Ingestion
# -----------------------------------------------------------------------------

def load_data(filepath: str) -> pd.DataFrame:
    """
    Load raw dataset from CSV/Excel/etc.

    TODO:
    - Load the dataset from the given filepath
    - Print shape, dtypes, and first few rows for a sanity check
    - Return the raw DataFrame
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 1.2: Data Cleaning
# -----------------------------------------------------------------------------

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Handle data quality issues.

    TODO:
    - Drop or rename any irrelevant/duplicate columns (e.g. customer ID)
    - Standardise column names (lowercase, underscores)
    - Convert columns to correct dtypes (e.g. TotalCharges may be object → float)
    - Handle any obviously incorrect values (e.g. negative tenure)
    - Return cleaned DataFrame
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 1.3: Missing Value Handling
# -----------------------------------------------------------------------------

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Identify and resolve missing values.

    TODO:
    - Print a summary of missing values per column (count + %)
    - Decide strategy per column:
        * Drop rows if missingness is very low (<1%)
        * Impute with median/mode for numerical/categorical
        * Or flag as a separate category if missingness is informative
    - Document every decision made
    - Return DataFrame with no missing values
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 1.4: Churn Class Balance Analysis
# -----------------------------------------------------------------------------

def analyse_churn_balance(df: pd.DataFrame, target_col: str = "Churn") -> None:
    """
    Analyse and visualise class distribution of the target variable.

    TODO:
    - Print value counts and % split for churned vs non-churned
    - Plot a bar/pie chart of churn distribution
    - Note whether the dataset is imbalanced (important for Modules 3 & 4)
    - Save plot to /outputs/churn_distribution.png
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 1.5: Distribution Analysis
# -----------------------------------------------------------------------------

def plot_distributions(df: pd.DataFrame) -> None:
    """
    Visualise distributions of key features split by churn.

    TODO:
    - For numerical features (e.g. tenure, MonthlyCharges, TotalCharges):
        * Plot histograms / KDE plots, coloured by churn label
    - For categorical features (e.g. Contract, PaymentMethod):
        * Plot count plots grouped by churn label
    - Save plots to /outputs/distributions/
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 1.6: Correlation Heatmap
# -----------------------------------------------------------------------------

def plot_correlation_heatmap(df: pd.DataFrame) -> None:
    """
    Generate a correlation heatmap for numerical features.

    TODO:
    - Encode binary categoricals (e.g. Yes/No → 1/0) for correlation calculation
    - Compute correlation matrix (Pearson or Spearman)
    - Plot heatmap with annotations
    - Highlight any features strongly correlated with churn
    - Watch for multicollinearity between features (flag for Module 2)
    - Save plot to /outputs/correlation_heatmap.png
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 1.7: Interactive EDA Dashboard (Optional but listed as deliverable)
# -----------------------------------------------------------------------------

def build_eda_dashboard(df: pd.DataFrame) -> None:
    """
    Build an interactive EDA dashboard using Plotly/Dash or similar.

    TODO:
    - Option A (Plotly): Create a collection of interactive plotly figures
        * Dropdown to select feature
        * Churn split toggle
    - Option B (Dash): Build a simple Dash app with dropdowns and charts
    - Option C (Streamlit): Wrap analysis in a Streamlit app
    - Export or run locally — document how to launch it in README
    """
    pass


# -----------------------------------------------------------------------------
# SECTION 1.8: Save Cleaned Data
# -----------------------------------------------------------------------------

def save_cleaned_data(df: pd.DataFrame, output_path: str) -> None:
    """
    Export the cleaned dataset for downstream modules.

    TODO:
    - Save cleaned DataFrame to CSV at output_path
    - Print confirmation with shape and path
    - This file will be the input for Module 2
    """
    pass


# =============================================================================
# MAIN PIPELINE
# =============================================================================

if __name__ == "__main__":
    # TODO: Update filepath to your actual dataset
    RAW_DATA_PATH = "data/raw/customer_churn.csv"
    CLEANED_DATA_PATH = "data/processed/cleaned_churn.csv"

    # Step 1: Load
    df = load_data(RAW_DATA_PATH)

    # Step 2: Clean
    df = clean_data(df)

    # Step 3: Handle missing values
    df = handle_missing_values(df)

    # Step 4: Analyse churn balance
    analyse_churn_balance(df, target_col="Churn")

    # Step 5: Plot distributions
    plot_distributions(df)

    # Step 6: Correlation heatmap
    plot_correlation_heatmap(df)

    # Step 7: Interactive dashboard
    build_eda_dashboard(df)

    # Step 8: Save cleaned data
    save_cleaned_data(df, CLEANED_DATA_PATH)

    print("✅ Module 1 complete. Cleaned data saved to:", CLEANED_DATA_PATH)
