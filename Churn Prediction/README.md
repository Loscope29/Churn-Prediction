# Prédiction du churn bancaire — service MLOps de rétention client

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.25+-red.svg)](https://streamlit.io/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue.svg)](https://www.docker.com/)
[![Pytest](https://img.shields.io/badge/Tests-Pytest-blue.svg)](https://docs.pytest.org/)

Prédiction de l'attrition des clients d'une banque (*churn*) et recommandation d'actions de rétention. Le projet comprend un modèle de machine learning optimisé avec Optuna, une API REST FastAPI avec validation Pydantic, une interface Streamlit, une suite de tests Pytest et une conteneurisation Docker multi-services.

---

## Rapport décisionnel

L'analyse destinée à la direction générale et à la direction marketing (facteurs explicatifs SHAP, retour sur investissement, matrice de coûts et plan d'action) est disponible dans [docs/rapport_decisionnel_churn.md](docs/rapport_decisionnel_churn.md).

---

## Performances du modèle

- **Algorithme :** Random Forest / XGBoost avec `ColumnTransformer` scikit-learn (`OneHotEncoder` + `StandardScaler`).
- **ROC-AUC :** `0.8602`
- **Rappel (recall) :** `82.31%`
- **Seuil de décision :** `0.371` (défini par minimisation de la matrice des coûts métier : $C_{FP}=50€, C_{FN}=360€$).
- **Impact financier net :** `+82 900,00 €` sur le jeu d'évaluation.

---

## Architecture du projet

```
Churn Prediction/
├── api/                        <- API FastAPI et schémas Pydantic
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
├── notebooks/                  <- Notebooks : exploration, prétraitement, optimisation Optuna
├── reports/                    <- Figures d'explicabilité SHAP et présentations
│   └── figures/
├── streamlit_app/              <- Interface utilisateur web interactive
│   └── app.py                  <- Tableau de bord Streamlit (formulaire et import CSV par lot)
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

## Démarrage rapide

### Option A : exécution locale dans un environnement virtuel

1. **Créer et activer un environnement virtuel :**
   ```bash
   python -m venv .venv
   .\.venv\Scriptsctivate     # Windows
   source .venv/bin/activate    # Linux / macOS
   ```

2. **Installer les dépendances :**
   ```bash
   pip install -r requirements.txt
   ```

3. **Entraîner le pipeline et sérialiser le modèle :**
   ```bash
   python "Churn Prediction/modeling/train_pipeline.py"
   ```

4. **Lancer l'API FastAPI (port 8000) :**
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
   - Documentation Swagger : [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
   - Documentation ReDoc : [http://127.0.0.1:8000/redoc](http://127.0.0.1:8000/redoc)

5. **Lancer l'interface Streamlit (port 8501) :**
   ```bash
   streamlit run streamlit_app/app.py
   ```
   - Interface web : [http://127.0.0.1:8501](http://127.0.0.1:8501)

---

### Option B : Docker Compose

Lancer les deux services (`API` + `Streamlit`) en une seule commande :

```bash
docker compose up --build
```

- **API FastAPI :** `http://localhost:8000/docs`
- **Tableau de bord Streamlit :** `http://localhost:8501`

---

## CI/CD (GitHub Actions)

Les workflows `.github/workflows/ci.yaml`, `.github/workflows/cd.yaml` et `.github/workflows/cml.yaml` séparent respectivement la validation du code, la publication Docker et l'entraînement ML. La CI exécute les tests et les contrôles Ruff sur chaque pull request et chaque push. Après un push sur `main` ou `master`, la CD construit puis publie les deux images Docker sur Docker Hub :

- `loscope29/churnprediction-api`
- `loscope29/churnprediction-streamlit`

Dans les secrets du dépôt GitHub (`Settings > Secrets and variables > Actions`), ajouter :

- `DOCKERHUB_USERNAME` : nom d'utilisateur Docker Hub.
- `DOCKERHUB_TOKEN` : jeton d'accès Docker Hub avec les droits de lecture et d'écriture sur les dépôts.

Les images sont publiées avec le tag `latest` et un tag correspondant au SHA du commit.

---

## CML et suivi des modèles

Le job `cml` du workflow entraîne le pipeline avec `Churn Prediction/modeling/train_pipeline.py` et publie la sortie d'entraînement comme commentaire dans les pull requests. Le dataset utilisé par cet entraînement doit être disponible dans `data/processed/Bank_Churn_clean.csv` dans l'environnement CI.

Le répertoire `data/` étant ignoré par Git, configurez DVC ou un stockage externe avant d'utiliser le job CML si ce fichier n'est pas versionné. Dans ce cas, ajoutez une étape de téléchargement ou de `dvc pull` avant l'étape d'entraînement.

---

## Tests

Lancer la suite de tests avec `pytest` :

```bash
pytest -v
```

Les 9 tests unitaires et d'intégration couvrent :
- L'intégrité des jeux de données nettoyés.
- L'inférence et la cohérence des probabilités du modèle ML.
- Les réponses des endpoints `/health`, `/predict` et `/predict_batch`.
- Le rejet des entrées invalides par la validation Pydantic (HTTP 422).

---

## Référence de l'API

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
// Corps de la requête
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

## Auteur

Loscope29 — projet de portfolio data science, de l'analyse exploratoire à la mise en production d'un modèle de prédiction du churn bancaire.
