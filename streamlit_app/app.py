import os
import pickle
from pathlib import Path
import pandas as pd
import requests
import streamlit as st

# Page Configuration
st.set_page_config(
    page_title="Bank Churn Predictor",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS for rich aesthetics
st.markdown("""
<style>
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 24px;
        border-radius: 12px;
        color: white;
        text-align: center;
        margin-bottom: 24px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
    }
    .metric-card {
        background-color: #f8f9fa;
        border-left: 5px solid #2a5298;
        padding: 16px;
        border-radius: 8px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .risk-high {
        background-color: #ffebee;
        border-left: 6px solid #e53935;
        padding: 16px;
        border-radius: 8px;
    }
    .risk-medium {
        background-color: #fff8e1;
        border-left: 6px solid #fb8c00;
        padding: 16px;
        border-radius: 8px;
    }
    .risk-low {
        background-color: #e8f5e9;
        border-left: 6px solid #43a047;
        padding: 16px;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Header Banner
st.markdown("""
<div class="main-header">
    <h1>🏦 Plateforme MLOps de PRÉDICTION DE CHURN CLIENT</h1>
    <p>Évaluation du risque de résiliation bancaire & recommandations de rétention stratégique</p>
</div>
""", unsafe_allow_html=True)

# Configuration & API Endpoint
API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")
MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "best_churn_pipeline.pkl"


@st.cache_resource
def load_local_model():
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    return None


local_artifact = load_local_model()

# Navigation Tabs
tab1, tab2, tab3 = st.tabs(["👤 Prédiction Individuelle", "📊 Évaluation par Lot (CSV)", "ℹ️ Informations & API"])

# TAB 1: INDIVIDUAL PREDICTION
with tab1:
    st.subheader("Saisie des Caractéristiques Client")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        credit_score = st.number_input("Score de Crédit", min_value=300, max_value=850, value=650, step=10)
        geography = st.selectbox("Pays de résidence", ["France", "Germany", "Spain"])
        gender = st.selectbox("Genre", ["Female", "Male"])
        age = st.slider("Âge", min_value=18, max_value=100, value=40)
        
    with col2:
        tenure = st.slider("Ancienneté (Années)", min_value=0, max_value=10, value=5)
        balance = st.number_input("Solde du Compte (€)", min_value=0.0, value=75000.0, step=1000.0)
        num_products = st.selectbox("Nombre de Produits Souscrits", [1, 2, 3, 4], index=0)
        
    with col3:
        has_credit_card = st.radio("Carte de Crédit ?", [1, 0], format_func=lambda x: "Oui" if x == 1 else "Non", horizontal=True)
        is_active = st.radio("Membre Actif ?", [1, 0], format_func=lambda x: "Oui" if x == 1 else "Non", horizontal=True)
        estimated_salary = st.number_input("Salaire Estimé (€)", min_value=0.0, value=55000.0, step=1000.0)

    customer_payload = {
        "CreditScore": int(credit_score),
        "Geography": geography,
        "Gender": gender,
        "Age": int(age),
        "Tenure": int(tenure),
        "Balance": float(balance),
        "NumOfProducts": int(num_products),
        "HasCrCard": int(has_credit_card),
        "IsActiveMember": int(is_active),
        "EstimatedSalary": float(estimated_salary),
    }

    if st.button("🚀 Évaluer le Risque de Churn", type="primary", use_container_width=True):
        prediction_result = None
        
        # Try calling API first
        try:
            response = requests.post(f"{API_URL}/predict", json=customer_payload, timeout=3)
            if response.status_code == 200:
                prediction_result = response.json()
        except Exception:
            pass

        # Fallback to local model inference if API server is not running
        if prediction_result is None and local_artifact:
            pipeline = local_artifact["pipeline"]
            threshold = local_artifact.get("optimal_threshold", 0.371)
            df_in = pd.DataFrame([customer_payload])
            prob = float(pipeline.predict_proba(df_in)[0, 1])
            pred = int(prob >= threshold)
            
            if prob < 0.25:
                risk = "Faible"
                action = "Client fidèle. Recommander des produits d'épargne ou carte premium."
            elif prob < threshold:
                risk = "Moyen"
                action = "Surveillance recommandée. Proposer un bilan bancaire personnalisé."
            else:
                risk = "Élevé"
                action = "Risque de churn imminence. Proposer une offre promotionnelle ou baisse des frais."

            prediction_result = {
                "churn_prediction": pred,
                "churn_probability": round(prob, 4),
                "risk_level": risk,
                "recommended_action": action,
            }

        if prediction_result:
            prob = prediction_result["churn_probability"]
            pred = prediction_result["churn_prediction"]
            risk = prediction_result["risk_level"]
            action = prediction_result["recommended_action"]
            
            st.divider()
            res_col1, res_col2 = st.columns([1, 2])
            
            with res_col1:
                st.metric(label="Probabilité de Churn", value=f"{prob * 100:.1f}%")
                st.progress(prob)
                
            with res_col2:
                css_class = "risk-high" if risk == "Élevé" else ("risk-medium" if risk == "Moyen" else "risk-low")
                st.markdown(f"""
                <div class="{css_class}">
                    <h3>Résultat : <b>{'CHURN POTENTIEL ⚠️' if pred == 1 else 'CLIENT CONSERVÉ ✅'}</b></h3>
                    <p><b>Niveau de Risque :</b> {risk}</p>
                    <p><b>Action Recommandée :</b> {action}</p>
                </div>
                """, unsafe_allow_html=True)
        else:
            st.error("Impossible de calculer la prédiction. Vérifiez le service API ou la présence du modèle.")

# TAB 2: BATCH PREDICTION VIA CSV
with tab2:
    st.subheader("Importation de Fichier CSV pour Inférence par Lot")
    st.markdown("Téléchargez un fichier CSV contenant les colonnes clients requises.")

    sample_df = pd.DataFrame([{
        "CreditScore": 619, "Geography": "France", "Gender": "Female", "Age": 42,
        "Tenure": 2, "Balance": 0.0, "NumOfProducts": 1, "HasCrCard": 1,
        "IsActiveMember": 1, "EstimatedSalary": 101348.88
    }])
    st.download_button(
        "📥 Télécharger un modèle CSV d'exemple",
        data=sample_df.to_csv(index=False),
        file_name="sample_customers.csv",
        mime="text/csv",
    )

    uploaded_file = st.file_uploader("Choisissez un fichier CSV", type=["csv"])
    if uploaded_file:
        df_batch = pd.read_csv(uploaded_file)
        st.write("Aperçu des données chargées :", df_batch.head())
        
        if st.button("🔮 Lancer les prédictions sur le lot", type="primary"):
            customers_list = df_batch.to_dict(orient="records")
            results = []
            
            # Try API batch endpoint
            try:
                resp = requests.post(f"{API_URL}/predict_batch", json={"customers": customers_list}, timeout=5)
                if resp.status_code == 200:
                    results = resp.json()["predictions"]
            except Exception:
                pass

            # Local fallback batch prediction
            if not results and local_artifact:
                pipeline = local_artifact["pipeline"]
                threshold = local_artifact.get("optimal_threshold", 0.371)
                probs = pipeline.predict_proba(df_batch)[:, 1]
                for p in probs:
                    prob = float(p)
                    pred = int(prob >= threshold)
                    risk = "Élevé" if prob >= threshold else ("Moyen" if prob >= 0.25 else "Faible")
                    action = "Action urgente" if pred == 1 else "Rétention standard"
                    results.append({
                        "churn_prediction": pred,
                        "churn_probability": round(prob, 4),
                        "risk_level": risk,
                        "recommended_action": action,
                    })

            if results:
                df_res = pd.DataFrame(results)
                output_df = pd.concat([df_batch.reset_index(drop=True), df_res], axis=1)
                
                churn_rate = (output_df["churn_prediction"] == 1).mean() * 100
                st.success(f"Prédictions terminées ! Taux de churn détecté sur ce lot : **{churn_rate:.1f}%**")
                
                st.dataframe(output_df)
                
                st.download_button(
                    "💾 Télécharger les résultats (CSV)",
                    data=output_df.to_csv(index=False),
                    file_name="predictions_churn_results.csv",
                    mime="text/csv",
                )

# TAB 3: INFO & API
with tab3:
    st.subheader("Métadonnées du Service & Documentation")
    
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(f"""
        - **URL API FastAPI :** `{API_URL}`
        - **Documentation OpenAPI (Swagger) :** `{API_URL}/docs`
        - **Documentation ReDoc :** `{API_URL}/redoc`
        - **Endpoint Santé :** `{API_URL}/health`
        """)
        
    with col_b:
        if local_artifact:
            metrics = local_artifact.get("metrics", {})
            st.json({
                "seuil_optimal": local_artifact.get("optimal_threshold"),
                "roc_auc": metrics.get("roc_auc"),
                "recall": metrics.get("recall"),
                "f1_score": metrics.get("f1_score"),
                "impact_financier": f"${metrics.get('net_impact', 0):,.2f}",
            })
