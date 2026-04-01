from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np
import pandas as pd


def split_data(X, y, test_size=0.2, random_state=50):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)


def train_svm(X_train, y_train, kernel='rbf', C=1.0, random_state=42):
    model = SVC(kernel=kernel, C=C, random_state=random_state)
    model.fit(X_train, y_train)
    return model


def compute_stats(model, X_test, y_test):
    y_pred = model.predict(X_test)
    
    return {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
        'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0),
        'confusion_matrix': confusion_matrix(y_test, y_pred),
        'support_vectors': model.n_support_,
        'predictions': y_pred
    }


def print_summary(stats, kernel, n_train, n_test):
    print("=" * 40)
    print("SVM Classification Results")
    print("=" * 40)
    print(f"Train size: {n_train}")
    print(f"Test size:  {n_test}")
    print(f"Kernel:     {kernel}")
    print("-" * 40)
    print(f"Accuracy:   {stats['accuracy']:.4f}")
    print(f"Precision:  {stats['precision']:.4f}")
    print(f"Recall:     {stats['recall']:.4f}")
    print(f"F1 Score:   {stats['f1']:.4f}")
    print("-" * 40)
    print(f"Support vectors per class: {stats['support_vectors']}")
    print(f"\nConfusion Matrix:\n{stats['confusion_matrix']}")


def run_svm(X, y, kernel='rbf', C=1.0, test_size=0.2, scale=True, random_state=42):
    """
    Main pipeline: takes a data function, runs SVM, returns stats.
    """
    # Split
    X_train, X_test, y_train, y_test = split_data(X, y, test_size, random_state)
    
    # Train
    model = train_svm(X_train, y_train, kernel, C, random_state)
    
    # Evaluate
    stats = compute_stats(model, X_test, y_test)
    
    # Print
    print_summary(stats, kernel, len(y_train), len(y_test))
    
    return stats, model

def get_data():
    df = pd.read_csv('telco_churn_cleaned.csv')
    X = df.drop(columns=['Churn'])
    y = df['Churn']
    return X, y


if __name__ == "__main__":
    X, y = get_data()
    stats, model = run_svm(X, y, kernel='rbf', C=1.0)