import pickle
from pathlib import Path
import pandas as pd
import pytest

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_churn_pipeline.pkl"


def test_model_artifact_exists():
    assert MODEL_PATH.exists(), f"Model artifact does not exist at {MODEL_PATH}"


def test_model_pipeline_inference():
    with open(MODEL_PATH, "rb") as f:
        artifact = pickle.load(f)

    assert "pipeline" in artifact
    assert "optimal_threshold" in artifact
    assert "metrics" in artifact

    pipeline = artifact["pipeline"]
    threshold = artifact["optimal_threshold"]

    # Sample customer dataframe
    sample_customer = pd.DataFrame([{
        "CreditScore": 650,
        "Geography": "France",
        "Gender": "Female",
        "Age": 42,
        "Tenure": 5,
        "Balance": 75000.5,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0
    }])

    probabilities = pipeline.predict_proba(sample_customer)[:, 1]
    assert len(probabilities) == 1
    assert 0.0 <= probabilities[0] <= 1.0

    prediction = int(probabilities[0] >= threshold)
    assert prediction in (0, 1)
