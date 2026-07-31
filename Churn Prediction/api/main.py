import pickle
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List
import pandas as pd
from fastapi import FastAPI, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from loguru import logger

from api.schemas import (
    CustomerData,
    BatchCustomerData,
    PredictionResponse,
    BatchPredictionResponse,
    HealthResponse,
)

# Global variables for model pipeline
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_churn_pipeline.pkl"
artifact = None


def load_model_artifact():
    global artifact
    if not MODEL_PATH.exists():
        logger.error(f"Model artifact not found at {MODEL_PATH}")
        return None
    try:
        with open(MODEL_PATH, "rb") as f:
            artifact = pickle.load(f)
        logger.info("Successfully loaded ML model pipeline artifact.")
        return artifact
    except Exception as e:
        logger.error(f"Error loading model artifact: {e}")
        return None


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model_artifact()
    yield


# Initialize FastAPI App
app = FastAPI(
    title="Bank Churn Prediction API",
    description="""
    ## Service MLOps de Prédiction de Churn Client (Attrition Bancaire)
    Cette API expose un modèle **Random Forest / XGBoost** optimisé par Optuna pour prédire la probabilité de résiliation d'un client bancaire.
    
    ### Fonctionnalités principales :
    - **Validation stricte** des données d'entrée avec Pydantic v2.
    - **Seuil financier d'action** optimisé pour maximiser le ROI métier.
    - **Score de risque & recommandations** automatiques pour chaque client.
    - **Inférence à l'unité (`/predict`) et par lot (`/predict_batch`)**.
    """,
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def format_single_prediction(data_dict: dict) -> PredictionResponse:
    if artifact is None or "pipeline" not in artifact:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Le modèle ML n'est pas disponible ou chargé.",
        )

    pipeline = artifact["pipeline"]
    threshold = artifact.get("optimal_threshold", 0.371)

    df_single = pd.DataFrame([data_dict])
    prob = float(pipeline.predict_proba(df_single)[0, 1])
    pred = int(prob >= threshold)

    if prob < 0.25:
        risk_level = "Faible"
        recommended_action = "Client fidèle. Recommander des produits d'épargne ou carte premium."
    elif prob < threshold:
        risk_level = "Moyen"
        recommended_action = "Surveillance recommandée. Proposer un bilan bancaire personnalisé."
    else:
        risk_level = "Élevé"
        recommended_action = "Risque de churn imminence. Proposer une offre promotionnelle ou baisse des frais."

    return PredictionResponse(
        churn_prediction=pred,
        churn_probability=round(prob, 4),
        risk_level=risk_level,
        recommended_action=recommended_action,
    )


@app.get("/", tags=["Info"])
def root():
    return {
        "service": "Bank Churn Prediction API",
        "status": "online",
        "documentation": "/docs",
        "health_check": "/health",
    }


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health_check():
    is_loaded = artifact is not None and "pipeline" in artifact
    return HealthResponse(
        status="healthy" if is_loaded else "unhealthy",
        model_loaded=is_loaded,
        optimal_threshold=artifact.get("optimal_threshold", 0.371) if is_loaded else 0.0,
        roc_auc=artifact.get("metrics", {}).get("roc_auc") if is_loaded else None,
    )


@app.post("/predict", response_model=PredictionResponse, tags=["Prediction"])
def predict(customer: CustomerData):
    """
    Effectue une prédiction de risque de churn pour un client individuel.
    """
    try:
        data_dict = customer.model_dump()
        return format_single_prediction(data_dict)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction: {str(e)}",
        )


@app.post("/predict_batch", response_model=BatchPredictionResponse, tags=["Prediction"])
def predict_batch(batch: BatchCustomerData):
    """
    Effectue des prédictions de churn pour une liste de clients (traitement par lot).
    """
    try:
        predictions = []
        churn_count = 0
        for customer in batch.customers:
            res = format_single_prediction(customer.model_dump())
            predictions.append(res)
            if res.churn_prediction == 1:
                churn_count += 1

        return BatchPredictionResponse(
            total_customers=len(predictions),
            churn_count=churn_count,
            predictions=predictions,
        )
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Erreur lors de la prédiction par lot: {str(e)}",
        )
