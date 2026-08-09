# 🏦 Bank Churn Prediction - Service MLOps & IA de Rétention Client

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)
[![Pytest](https://img.shields.io/badge/Pytest-Passed-success.svg)](https://docs.pytest.org/)

Projet MLOps complet de prédiction de l'attrition bancaire (*Churn Client*) et de recommandation automatique d'actions de rétention. Le système s'appuie sur un modèle de Machine Learning optimisé par Optuna, une API REST FastAPI sécurisée avec validation Pydantic, une interface graphique interactive Streamlit, une suite de tests Pytest, ainsi qu'une conteneurisation Docker multi-services.

---

## 📄 Rapport Décisionnel & Synthèse Métier

Pour consulter l'analyse détaillée destinée à la Direction Générale et à la Direction Marketing (insights SHAP, ROI financier, matrice de coûts et plan d'action) :
👉 **[Lire le Rapport Décisionnel (rapport_decisionnel_churn.md)](docs/rapport_decisionnel_churn.md)**

---

## 📊 Performances du Modèle IA

- **Algorithme :** Random Forest / XGBoost avec `ColumnTransformer` scikit-learn (`OneHotEncoder` + `StandardScaler`).
- **Score ROC-AUC :** `0.8602`
- **Taux de Détection (Recall) :** `82.31%`
- **Seuil Décisionnel Optimisé :** `0.371` (défini par minimisation de la matrice des coûts métier : $C_{FP}=50€, C_{FN}=360€$).
- **Impact Financier Net Préservé :** `+82 900,00 €` sur le jeu d'évaluation.

---

## 🏗️ Architecture du Projet

```
Churn Prediction/
├── api/                        <- Application microservice FastAPI & Schémas Pydantic
│   ├── main.py                 <- Endpoints REST (/health, /predict, /predict_batch)
│   └── schemas.py              <- Modèles de validation Pydantic v2
├── Churn Prediction/           <- Module Python & scripts MLOps
│   ├── config.py               <- Configuration des chemins du projet
│   └── modeling/
│       └── train_pipeline.py   <- Entraînement et sérialisation du pipeline Scikit-Learn
├── data/                       <- Données brutes et nettoyées
│   ├── raw/Bank_Churn.csv
│   └── processed/Bank_Churn_clean.csv
├── docs/                       <- Documentation projet et rapport décisionnel
│   └── rapport_decisionnel_churn.md <- Rapport stratégique pour les décideurs
├── models/                     <- Artefacts ML sérialisés (.pkl)
│   └── best_churn_pipeline.pkl
├── notebooks/                  <- Carnets Jupyter d'EDA, Prétraitement et Optimisation Optuna
├── reports/                    <- Figures d'explicabilité SHAP et présentations
│   └── figures/
├── streamlit_app/              <- Interface utilisateur web interactive
│   └── app.py                  <- Dashboard Streamlit (Formulaire & Import CSV par lot)
├── tests/                      <- Suite de tests unitaires et d'intégration Pytest
│   ├── test_api.py             <- Tests des endpoints FastAPI
│   ├── test_data.py            <- Tests d'intégrité des jeux de données
│   └── test_model.py           <- Tests d'inférence du pipeline ML
├── Dockerfile                  <- Conteneurisation Docker de l'API FastAPI
├── Dockerfile.streamlit        <- Conteneurisation Docker de Streamlit
├── docker-compose.yml          <- Orchestration multi-conteneurs (API + Streamlit)
├── pyproject.toml              <- Configuration Pytest, Ruff et métadonnées
└── requirements.txt            <- Liste des dépendances Python
```

---

## 🚀 Guide de Démarrage Rapide

### Option A : Exécution en Local avec Python Virtual Environment

1. **Cloner le projet et créer un environnement virtuel :**
   ```bash
   python -m venv .venv
   .\.venv\Scripts\activate   # Sur Windows
   ```

2. **Installer les dépendances requises :**
   ```bash
   pip install -r requirements.txt
   ```

3. **Entraîner et générer l'artefact du pipeline ML :**
   ```bash
   python "Churn Prediction/modeling/train_pipeline.py"
   ```

4. **Lancer le serveur API FastAPI (Port 8000) :**
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
   - Swagger OpenAPI UI : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - ReDoc UI : [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

5. **Lancer l'interface Streamlit (Port 8501) :**
   ```bash
   streamlit run streamlit_app/app.py
   ```
   - Interface Web : [http://127.0.0.1:8501](http://127.0.0.1:8501)

---

### Option B : Déploiement Conteneurisé avec Docker Compose

Lancer les deux services (`API` + `Streamlit`) en une seule commande :

```bash
docker compose up --build
```

- **Service API FastAPI :** `http://localhost:8000/docs`
- **Dashboard Streamlit :** `http://localhost:8501`

---

## CI/CD GitHub Actions

Le workflow `.github/workflows/ci-cd.yml` exécute les tests et les contrôles Ruff sur chaque pull request et chaque push. Après un push sur `main` ou `master`, il construit puis publie les deux images Docker sur Docker Hub :

- `loscope29/churnprediction-api`
- `loscope29/churnprediction-streamlit`

Dans les secrets du dépôt GitHub (`Settings > Secrets and variables > Actions`), ajouter :

- `DOCKERHUB_USERNAME` : nom d'utilisateur Docker Hub.
- `DOCKERHUB_TOKEN` : access token Docker Hub avec les droits de lecture et d'écriture sur les dépôts.

Les images sont publiées avec le tag `latest` et un tag correspondant au SHA du commit.

---

## 🧪 Exécution des Tests Automatisés

Lancer la suite de tests avec `pytest` :

```bash
pytest -v
```

Tous les 9 tests unitaires et d'intégration valident :
- L'intégrité des jeux de données nettoyés.
- L'inférence et la cohérence des probabilités du modèle ML.
- La réponse conforme des endpoints `/health`, `/predict` et `/predict_batch`.
- Le rejet et la validation stricte Pydantic des entrées invalides (HTTP 422).

---

## 🔌 API Endpoints Reference

### `GET /health`
Vérifie l'état du service et du modèle.
```json
{
  "status": "healthy",
  "model_loaded": true,
  "optimal_threshold": 0.3714,
  "roc_auc": 0.8602
}
```

### `POST /predict`
Évalue un client individuel.
```json
// Payload d'entrée
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
}
```

---

## 👨‍💻 Auteur & Licence

Projet développé dans le cadre de la mise en production MLOps pour la prédiction de churn bancaire.
