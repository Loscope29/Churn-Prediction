# 📊 Rapport Décisionnel - Stratégie de Rétention & Prédiction du Churn Client

**À l'attention de :** Direction Générale, Direction Marketing & Responsables Relation Client  
**Auteur :** Équipe MLOps & Data Science  
**Date :** Juillet 2026  
**Objet :** Synthèse stratégique du modèle d'Intelligence Artificielle pour la prévention du Churn bancaire

---

## 1. Synthèse Exécutive & Enjeux Financiers

L'attrition bancaire (Churn) représente une perte directe de chiffre d'affaires et de valeur consommateur (*Lifetime Value - LTV*). Dans notre portefeuille analysé de 10 000 clients :
- **Taux de Churn Observé :** **20.37%** (soit plus d'un client sur cinq qui quitte la banque).
- **Coût d'Inaction :** Perte estimée à plusieurs millions d'euros en dépôts et frais de tenue de compte.
- **Objectif de l'IA :** Détecter de manière proactive les clients à fort risque d'attrition avant qu'ils ne résilient leurs comptes, afin de déployer des actions de rétention ciblées et rentables.

---

## 2. Facteurs Clés d'Attrition (Explicabilité SHAP & Data)

Grâce aux techniques d'explicabilité par valeurs SHAP (*SHapley Additive exPlanations*), nous avons identifié les 5 moteurs principaux du Churn :

```
                  Facteurs d'Attrition Majeurs
┌───────────────────────────┬──────────────────────────────────────────┐
│ Facteur                   │ Impact sur le Risque de Churn            │
├───────────────────────────┼──────────────────────────────────────────┤
│ 1. Âge du Client          │ Risque très élevé entre 40 et 60 ans     │
│ 2. Nombre de Produits     │ 1 seul produit ou >2 produits = Alerte   │
│ 3. Inactivité Membre      │ Les membres inactifs churnent 2x plus    │
│ 4. Localisation (Allemagne)│ Taux de churn (32%) vs France/Espagne(16%)│
│ 5. Solde Bancaire         │ Solde élevé + inactivité = Risque fort   │
└───────────────────────────┴──────────────────────────────────────────┘
```

### Insights Opérationnels :
1. **L'Âge (Facteur #1) :** Les clients seniors/d'âge mûr (40–60 ans) affichent le taux d'attrition le plus fort. Ils recherchent souvent de meilleurs rendements ou un accompagnement de gestion de patrimoine plus poussé.
2. **Le Nombre de Produits :** Les clients possédant **2 produits** sont les plus fidèles. En revanche, les clients avec 1 seul produit manquent d'ancrage, et ceux avec 3 ou 4 produits subissent une insatisfaction ou une sur-tarification.
3. **L'Activité du Compte :** Un client inactif sur les 12 derniers mois présente une probabilité de départ multipliée par 2.
4. **La Zone Géographique :** Les clients résidant en Allemagne ont un taux de churn de 32.4%, comparé à environ 16.1% pour la France et l'Espagne.

---

## 3. Analyse ROI & Optimisation du Seuil Décisionnel

Contrairement aux modèles de classification classiques qui visent uniquement la précision brute, nous avons calibré notre modèle sur la **matrice de coûts réels bancaires** :

- **Coût d'une Fausse Alerte ($C_{FP}$) = 50 €**  
  *(Coût d'une campagne de rétention proactive : appel conseiller, geste commercial, réduction temporaire de frais).*
- **Coût d'un Churner Non Détecté ($C_{FN}$) = 360 €**  
  *(Perte nette de LTV ajustée par le taux de succès d'une rétention = 1200 € x 30%).*

### Seuil d'Intervention Optimisé : `0.371`

Plutôt que d'attendre qu'un client ait 50% de chance de partir, le modèle déclenche l'alerte dès **37.1% de probabilité de risque**. 

```
┌──────────────────────────────────────────────────────────────────────┐
│                  RÉSULTATS DE L'OPTIMISATION MÉTIER                  │
├───────────────────────────────────┬──────────────────────────────────┤
│ Métrique                          │ Valeur                           │
├───────────────────────────────────┼──────────────────────────────────┤
│ Taux de Détection (Recall)        │ 82.31 % (8 clients sur 10 vus)   │
│ Score ROC-AUC (Qualité Globale)   │ 0.8602                           │
│ Impact Financier Net Généré       │ + 82 900.00 € (sur le jeu test)  │
└───────────────────────────────────┴──────────────────────────────────┘
```

---

## 4. Recommandations Stratégiques & Plan d'Action

Pour maximiser le retour sur investissement de cette solution IA, la Direction Marketing et le Réseau d'Agences doivent mettre en œuvre les actions suivantes :

### A. Automatisation des Campagnes de Rétention
- **Alerte Risque Élevé ($Prob \ge 37.1\%$) :**  
  Déclenchement automatique d'une tâche dans le CRM pour le conseiller référent. Proposition d'un bilan patrimonial gratuit et réduction des frais de tenue de compte.
- **Sensibilisation Risque Moyen ($25\% \le Prob < 37.1\%$) :**  
  Envoi d'une newsletter ciblée avec valorisation des nouveaux produits d'épargne ou d'assurance.

### B. Stratégie Produit & Cross-selling
- Encourager le passage de **1 produit à 2 produits** via des offres packagées avantageuses (ex: assurance-vie ou livret d'épargne associé au compte courant).

### C. Déploiement Métier (API & Dashboard)
- **Pour les Conseillers Bancaires :** Utilisation de l'application web [Streamlit](http://localhost:8501) pour évaluer en temps réel un client en rendez-vous ou analyser un portefeuille via fichier CSV.
- **Pour le Système d'Information (SI) :** Intégration de l'API REST [FastAPI](http://localhost:8000/docs) dans le CRM de la banque.

---

## 5. Déploiement & Gouvernance MLOps

Le modèle est intégralement sécurisé et automatisé :
- **Validation Pydantic v2 :** Empêche toute saisie erronée ou incohérente de données.
- **Architecture Docker Compose :** Déploiement en conteneurs légers et scalables (`churn_prediction_api` & `churn_prediction_streamlit`).
- **Tests d'Intégration Pytest :** Validation continue du bon fonctionnement des endpoints et du pipeline de prédiction.

---
*Ce document sert de support de décision pour le déploiement opérationnel de la solution de rétention client.*
