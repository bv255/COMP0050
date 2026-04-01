import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report,
    ConfusionMatrixDisplay, RocCurveDisplay
)
import matplotlib.pyplot as plt
import os


# Continuous columns to scale
CONTINUOUS_COLS = ['tenure', 'MonthlyCharges', 'TotalCharges']


def load_data():
    df = pd.read_csv("WA_Fn-UseC_-Telco-Customer-Churn.csv")
    df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
    df.dropna(inplace=True)
    df.drop("customerID", axis=1, inplace=True)

    cleaned_df = df.copy()
    cleaned_df["Churn"] = cleaned_df["Churn"].map({"Yes": 1, "No": 0})
    cleaned_df = pd.get_dummies(cleaned_df, drop_first=True)
    bool_cols = cleaned_df.select_dtypes(include='bool').columns
    cleaned_df[bool_cols] = cleaned_df[bool_cols].astype(int)

    X = cleaned_df.drop(columns=['Churn'])
    y = cleaned_df['Churn']
    return X, y


def get_preprocessor(feature_names):
    """Scale only continuous columns, pass through the rest."""
    continuous_idx = [i for i, col in enumerate(feature_names) if col in CONTINUOUS_COLS]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('scale', StandardScaler(), continuous_idx)
        ],
        remainder='passthrough'
    )
    return preprocessor


def evaluate_model(model, X_test, y_test, model_name):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Create model subfolder
    folder_name = model_name.lower().replace(" ", "_")
    model_dir = f'results/{folder_name}'
    os.makedirs(model_dir, exist_ok=True)
    
    results = {
        'model': model_name,
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else None
    }
    
    # Print results
    print(f"\n{'='*50}")
    print(f"{model_name}")
    print('='*50)
    print(f"Accuracy:  {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall:    {results['recall']:.4f}")
    print(f"F1:        {results['f1']:.4f}")
    if results['roc_auc']:
        print(f"ROC-AUC:   {results['roc_auc']:.4f}")
    print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")
    print(f"\n{classification_report(y_test, y_pred)}")
    
    # Save confusion matrix
    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, ax=ax)
    ax.set_title(f'{model_name} - Confusion Matrix')
    plt.tight_layout()
    plt.savefig(f'{model_dir}/confusion_matrix.png', dpi=150)
    plt.close()
    
    # Save ROC curve
    if y_prob is not None:
        fig, ax = plt.subplots(figsize=(8, 6))
        RocCurveDisplay.from_predictions(y_test, y_prob, ax=ax)
        ax.set_title(f'{model_name} - ROC Curve')
        plt.tight_layout()
        plt.savefig(f'{model_dir}/roc_curve.png', dpi=150)
        plt.close()
    
    # Save predictions
    preds_df = pd.DataFrame({
        'y_true': y_test.values,
        'y_pred': y_pred,
        'y_prob': y_prob if y_prob is not None else np.nan
    })
    preds_df.to_csv(f'{model_dir}/predictions.csv', index=False)
    
    return results


if __name__ == "__main__":
    # Create results folders
    os.makedirs('results/all', exist_ok=True)
    
    # Load data
    X, y = load_data()
    
    # Consistent split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=50, stratify=y
    )
    
    # Consistent CV
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=50)
    
    # Get preprocessor that only scales continuous columns
    preprocessor = get_preprocessor(X.columns.tolist())
    
    # Define pipelines
    pipelines = {
        'Logistic Regression': Pipeline([
            ('preprocessor', preprocessor),
            ('model', LogisticRegression(max_iter=1000, random_state=50))
        ]),
        'SVM': Pipeline([
            ('preprocessor', preprocessor),
            ('model', SVC(probability=True, random_state=50))
        ]),
        'Random Forest': Pipeline([
            ('preprocessor', preprocessor),
            ('model', RandomForestClassifier(random_state=50))
        ])
    }
    
    # Param grids
    param_grids = {
        'Logistic Regression': {
            'model__C': [0.1, 1, 10],
            'model__penalty': ['l2']
        },
        'SVM': {
            'model__C': [0.1, 1, 10],
            'model__kernel': ['rbf', 'linear']
        },
        'Random Forest': {
            'model__n_estimators': [100, 200],
            'model__max_depth': [5, 10, None],
            'model__min_samples_split': [2, 5]
        }
    }
    
    # Train and evaluate all models
    all_results = []
    best_models = {}
    
    for name, pipe in pipelines.items():
        print(f"\nTraining {name}...")
        
        grid = GridSearchCV(
            pipe,
            param_grids[name],
            cv=cv,
            scoring='f1',
            n_jobs=-1
        )
        grid.fit(X_train, y_train)
        
        print(f"Best CV F1: {grid.best_score_:.4f}")
        print(f"Best params: {grid.best_params_}")
        
        # Evaluate on test set
        results = evaluate_model(grid.best_estimator_, X_test, y_test, name)
        results['best_cv_f1'] = grid.best_score_
        results['best_params'] = grid.best_params_
        
        all_results.append(results)
        best_models[name] = grid.best_estimator_
    
    # Summary comparison
    print("\n" + "="*60)
    print("FINAL COMPARISON")
    print("="*60)
    results_df = pd.DataFrame(all_results)
    print(results_df[['model', 'accuracy', 'precision', 'recall', 'f1', 'roc_auc']].to_string(index=False))
    
    # Save comparison CSV to results/all
    results_df.to_csv('results/all/model_comparison.csv', index=False)
    
    # Combined ROC curve to results/all
    fig, ax = plt.subplots(figsize=(10, 8))
    for name, model in best_models.items():
        if hasattr(model, 'predict_proba'):
            RocCurveDisplay.from_estimator(model, X_test, y_test, ax=ax, name=name)
    ax.set_title('ROC Curves - All Models')
    plt.tight_layout()
    plt.savefig('results/all/roc_curves.png', dpi=150)
    plt.close()
    
    print("\nResults saved to /results folder")