from typing import List, Literal, Optional
from pydantic import BaseModel, Field, ConfigDict


class CustomerData(BaseModel):
    CreditScore: int = Field(
        ..., 
        ge=300, 
        le=850, 
        description="Score de crédit du client (entre 300 et 850)", 
        json_schema_extra={"example": 650}
    )
    Geography: Literal["France", "Spain", "Germany"] = Field(
        ..., 
        description="Pays de résidence du client", 
        json_schema_extra={"example": "France"}
    )
    Gender: Literal["Female", "Male"] = Field(
        ..., 
        description="Genre du client", 
        json_schema_extra={"example": "Female"}
    )
    Age: int = Field(
        ..., 
        ge=18, 
        le=100, 
        description="Âge du client (18-100 ans)", 
        json_schema_extra={"example": 42}
    )
    Tenure: int = Field(
        ..., 
        ge=0, 
        le=10, 
        description="Ancienneté en années (0-10 ans)", 
        json_schema_extra={"example": 5}
    )
    Balance: float = Field(
        ..., 
        ge=0.0, 
        description="Solde du compte bancaire (>= 0 €)", 
        json_schema_extra={"example": 75000.50}
    )
    NumOfProducts: int = Field(
        ..., 
        ge=1, 
        le=4, 
        description="Nombre de produits souscrits auprès de la banque (1-4)", 
        json_schema_extra={"example": 1}
    )
    HasCrCard: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Possession d'une carte de crédit (0: Non, 1: Oui)", 
        json_schema_extra={"example": 1}
    )
    IsActiveMember: int = Field(
        ..., 
        ge=0, 
        le=1, 
        description="Membre actif (0: Non, 1: Oui)", 
        json_schema_extra={"example": 1}
    )
    EstimatedSalary: float = Field(
        ..., 
        ge=0.0, 
        description="Salaire annuel estimé en euros (>= 0 €)", 
        json_schema_extra={"example": 50000.00}
    )

    model_config = ConfigDict(
        json_schema_extra={
            "example": {
                "CreditScore": 650,
                "Geography": "France",
                "Gender": "Female",
                "Age": 42,
                "Tenure": 5,
                "Balance": 75000.50,
                "NumOfProducts": 1,
                "HasCrCard": 1,
                "IsActiveMember": 1,
                "EstimatedSalary": 50000.00
            }
        }
    )


class BatchCustomerData(BaseModel):
    customers: List[CustomerData] = Field(..., description="Liste de dossiers clients à évaluer")


class PredictionResponse(BaseModel):
    churn_prediction: int = Field(..., description="Prédiction binaire du churn (0: Rétention, 1: Churn potentiel)")
    churn_probability: float = Field(..., description="Probabilité estimée de départ du client (0.000 à 1.000)")
    risk_level: str = Field(..., description="Niveau de risque (Faible, Moyen, Élevé)")
    recommended_action: str = Field(..., description="Action recommandée pour la fidélisation client")


class BatchPredictionResponse(BaseModel):
    total_customers: int
    churn_count: int
    predictions: List[PredictionResponse]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    optimal_threshold: float
    roc_auc: Optional[float] = None
