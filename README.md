# Customer Churn ML Project — COMP0050
# Project Skeleton & Workflow Overview

## Directory Structure

```
project/
├── data/
│   ├── raw/                    # Original dataset (DO NOT modify)
│   │   └── customer_churn.csv
│   └── processed/              # Outputs from Module 2
│       ├── cleaned_churn.csv   # From Module 1
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── models/                     # Trained model .pkl files
│   ├── logistic_regression.pkl
│   ├── random_forest.pkl
│   └── xgboost.pkl
│
├── outputs/                    # All plots and tables
│   ├── distributions/
│   ├── churn_distribution.png
│   ├── correlation_heatmap.png
│   ├── feature_importance_screen.png
│   ├── confusion_matrix_lr.png
│   ├── roc_curve_lr.png
│   ├── lr_coefficients.png
│   ├── feature_importance_random_forest.png
│   ├── feature_importance_xgboost.png
│   ├── roc_comparison.png
│   ├── final_roc_curves.png
│   ├── final_confusion_matrices.png
│   ├── precision_recall_curves.png
│   ├── churn_risk_segments.png
│   └── final_model_performance.csv
│
├── module1_eda.py
├── module2_feature_engineering.py
├── module3_baseline_model.py
├── module4_advanced_models.py
├── module5_evaluation.py
└── README.md
```

## Module Ownership & Dependencies

| Module | Owner(s) | Depends On | Produces |
|--------|----------|------------|----------|
| Module 1 (EDA) | Member A + B | Raw data | cleaned_churn.csv |
| Module 2 (Features) | Member C | Module 1 output | X/y train/test splits |
| Module 3 (LR) | Member D | Module 2 output | LR model + metrics |
| Module 4 (RF/XGB) | Member E | Module 2 output | RF/XGB models + metrics |
| Module 5 (Eval) | Member A + ? | Modules 3 & 4 | All final outputs + report |

## Running the Pipeline (in order)

```bash
python module1_eda.py
python module2_feature_engineering.py
python module3_baseline_model.py
python module4_advanced_models.py
python module5_evaluation.py
```

## Key TODOs Before Starting

- [ ] Agree on and confirm dataset with lecturer
- [ ] Set up shared repo (GitHub/GitLab) with this folder structure
- [ ] Confirm Python version and install dependencies (see below)
- [ ] Assign team members to modules (update the table above)
- [ ] Decide team policy on AI tool usage — document in AI Usage Statement

## Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost plotly shap
```

## Report Checklist (8 pages max)

- [ ] Problem statement & motivation
- [ ] Dataset description
- [ ] EDA summary (Module 1 highlights)
- [ ] Feature engineering decisions (Module 2)
- [ ] Methodology: LR, RF, XGBoost
- [ ] Results table + ROC curves + confusion matrices
- [ ] Critical discussion: interpret results, limitations, future work
- [ ] Teamwork & Contributions section (not in page limit)
- [ ] AI Usage Statement if applicable (not in page limit)

## Notes on the Marking Criteria

1. **Clarity** — every figure needs a caption, axes labels, and a title
2. **Results** — justify all modelling choices; report CV scores, not just test scores
3. **Critical Discussion** — don't just report numbers; interpret them in business context
   and honestly discuss what didn't work or could be improved
