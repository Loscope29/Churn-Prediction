import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, precision_recall_curve
)
from loguru import logger

# Paths
BASE_DIR = Path(__file__).resolve().parents[2]
DATA_PATH = BASE_DIR / "data" / "processed" / "Bank_Churn_clean.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_PATH = MODEL_DIR / "best_churn_pipeline.pkl"


def train_and_save_pipeline():
    logger.info(f"Loading data from {DATA_PATH}...")
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Data file not found at {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)
    
    TARGET = "Exited"
    NUMERICAL_COLS = [
        "CreditScore", "Age", "Tenure", "Balance", 
        "NumOfProducts", "HasCrCard", "IsActiveMember", "EstimatedSalary"
    ]
    CATEGORICAL_COLS = ["Gender", "Geography"]
    
    X = df[NUMERICAL_COLS + CATEGORICAL_COLS]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    logger.info("Building Scikit-Learn Pipeline...")
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERICAL_COLS),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_COLS)
        ]
    )

    # Best hyperparameters from Optuna tuning in 7_Optimization.ipynb
    clf = RandomForestClassifier(
        n_estimators=350,
        max_depth=12,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        class_weight="balanced",
        random_state=42
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])

    logger.info("Fitting model pipeline on X_train...")
    pipeline.fit(X_train, y_train)

    # Evaluate probabilities on test set
    y_probs = pipeline.predict_proba(X_test)[:, 1]
    
    # Financial threshold optimization (from notebook 7)
    # C_FP = 50 (retention offer cost)
    # C_FN = 1200 * 0.30 = 360 (lost customer LTV * success rate)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_probs)
    P = (y_test == 1).sum()

    best_threshold = 0.35
    best_impact = -float("inf")

    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        tp = r * P
        predicted_pos = tp / p if p > 0 else 0
        net_impact = tp * 0.30 * 1200 - predicted_pos * 50
        if net_impact > best_impact:
            best_impact = net_impact
            best_threshold = float(t)

    # Predictions with optimal threshold
    y_pred_opt = (y_probs >= best_threshold).astype(int)

    acc = accuracy_score(y_test, y_pred_opt)
    prec = precision_score(y_test, y_pred_opt)
    rec = recall_score(y_test, y_pred_opt)
    f1 = f1_score(y_test, y_pred_opt)
    auc = roc_auc_score(y_test, y_probs)

    logger.info("=== Model Performance Metrics (Threshold: {:.3f}) ===".format(best_threshold))
    logger.info(f"Accuracy:  {acc:.4f}")
    logger.info(f"Precision: {prec:.4f}")
    logger.info(f"Recall:    {rec:.4f}")
    logger.info(f"F1-Score:  {f1:.4f}")
    logger.info(f"ROC-AUC:   {auc:.4f}")
    logger.info(f"Net Financial Impact: ${best_impact:,.2f}")

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    
    artifact = {
        "pipeline": pipeline,
        "optimal_threshold": best_threshold,
        "numerical_cols": NUMERICAL_COLS,
        "categorical_cols": CATEGORICAL_COLS,
        "metrics": {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1_score": f1,
            "roc_auc": auc,
            "net_impact": best_impact,
        }
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    logger.success(f"Successfully exported pipeline artifact to {MODEL_PATH}")
    return artifact


if __name__ == "__main__":
    train_and_save_pipeline()
