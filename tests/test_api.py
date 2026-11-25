from fastapi.testclient import TestClient
import pytest
from api.main import app, load_model_artifact

# Load model artifact before running tests
load_model_artifact()
client = TestClient(app)


def test_root_endpoint():
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["service"] == "Bank Churn Prediction API"
    assert data["status"] == "online"


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["model_loaded"] is True
    assert "optimal_threshold" in data


def test_predict_endpoint_valid():
    payload = {
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
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "churn_prediction" in data
    assert data["churn_prediction"] in (0, 1)
    assert 0.0 <= data["churn_probability"] <= 1.0
    assert data["risk_level"] in ("Faible", "Moyen", "Élevé")
    assert isinstance(data["recommended_action"], str)


def test_predict_endpoint_invalid_data():
    # Invalid credit score (< 300) and invalid age (> 100)
    payload = {
        "CreditScore": 150,  # Invalid
        "Geography": "France",
        "Gender": "Female",
        "Age": 150,  # Invalid
        "Tenure": 5,
        "Balance": 75000.5,
        "NumOfProducts": 1,
        "HasCrCard": 1,
        "IsActiveMember": 1,
        "EstimatedSalary": 50000.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 422  # Unprocessable Entity (Pydantic validation error)


def test_predict_batch_endpoint():
    payload = {
        "customers": [
            {
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
            },
            {
                "CreditScore": 800,
                "Geography": "Germany",
                "Gender": "Male",
                "Age": 25,
                "Tenure": 1,
                "Balance": 120000.0,
                "NumOfProducts": 2,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 90000.0
            }
        ]
    }
    response = client.post("/predict_batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_customers"] == 2
    assert len(data["predictions"]) == 2
