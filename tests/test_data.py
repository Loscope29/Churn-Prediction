from pathlib import Path
import pandas as pd
import pytest

DATA_PATH = Path(__file__).resolve().parents[1] / "data" / "processed" / "Bank_Churn_clean.csv"


def test_processed_dataset_exists():
    assert DATA_PATH.exists(), f"Processed data file missing at {DATA_PATH}"


def test_processed_dataset_columns_and_shape():
    df = pd.read_csv(DATA_PATH)
    expected_cols = [
        "CreditScore", "Geography", "Gender", "Age", "Tenure",
        "Balance", "NumOfProducts", "HasCrCard", "IsActiveMember",
        "EstimatedSalary", "Exited"
    ]
    assert list(df.columns) == expected_cols
    assert len(df) > 0
    assert df["Exited"].isin([0, 1]).all()
