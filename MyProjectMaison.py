#!/usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("Tout est installé correctement ✅")

# 1. 📦 Import des bibliothèques
sns.set_style("whitegrid")
sns.set_palette("pastel")

# 2. 📥 Chargement du dataset
# Assurez-vous que train.csv est dans le même répertoire
df = pd.read_csv("train.csv")
df.shape, df.columns

# 3. 🧹 Exploration rapide et nettoyage de base
st.dataframe(df.head())  # Utilisation de Streamlit pour afficher le DataFrame

# Statistiques générales
st.write(df.describe())

# Données manquantes
missing = df.isnull().mean().sort_values(ascending=False)
st.write(missing[missing > 0])  # Seulement celles qui posent problème

# Suppression de colonnes très incomplètes
df = df.drop(columns=["Alley", "PoolQC", "Fence", "MiscFeature"], errors="ignore")

# 4. 📊 Analyse exploratoire
# Distribution du prix
sns.histplot(df['SalePrice'], kde=True)
plt.title("Distribution des prix de vente")
st.pyplot()  # Remplacé plt.show() par st.pyplot()

# Corrélation avec d’autres variables numériques
corr = df.corr(numeric_only=True)
top_corr = corr["SalePrice"].sort_values(ascending=False)[1:6]  # Top 5
st.write(top_corr)

# Scatterplot sur une variable très corrélée (ex: OverallQual)
sns.scatterplot(data=df, x="OverallQual", y="SalePrice")
plt.title("Qualité globale vs Prix de vente")
st.pyplot()  # Remplacé plt.show() par st.pyplot()

# 5. 🏗️ Préparation des données
features = ["GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF"]
X = df[features]
y = df["SalePrice"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. 🤖 Régression linéaire
model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

st.write(f"RMSE : {rmse:.2f}")
st.write(f"R² : {r2:.3f}")

# 7. 🧾 Conclusion rapide
st.write(f"Le modèle explique environ {r2 * 100:.1f}% de la variance du prix de vente.")
st.write(f"RMSE : erreur moyenne d’environ {rmse:.0f} $ sur l’échantillon de test.")

# À améliorer : traitement des variables catégorielles, plus de features, modèles non-linéaires…

# 🧹 Traitement des valeurs manquantes
for col in df.columns:
    if df[col].isnull().sum() > 0:
        if df[col].dtype == 'object':
            # C'est une colonne catégorielle → on remplace les NaN par le mode (valeur la plus fréquente)
            df[col].fillna(df[col].mode()[0], inplace=True)
        else:
            # C'est une colonne numérique → on remplace les NaN par la médiane
            df[col].fillna(df[col].median(), inplace=True)

# Vérification rapide que tout est propre
st.write("Valeurs manquantes restantes :")
st.write(df.isnull().sum().sum())  # doit afficher 0

# Visualisation du graphique : Relation entre surface habitable et prix de vente
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="GrLivArea", y="SalePrice", alpha=0.6, edgecolor=None)
plt.title("Relation entre surface habitable (GrLivArea) et prix de vente (SalePrice)")
plt.xlabel("Surface habitable (en pieds²)")
plt.ylabel("Prix de vente ($)")
plt.grid(True)
plt.tight_layout()
st.pyplot()  # Remplacé plt.show() par st.pyplot()

# Régression linéaire : Surface habitable vs Prix de vente
plt.figure(figsize=(10, 6))
sns.regplot(data=df, x="GrLivArea", y="SalePrice", scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})
plt.title("Régression linéaire : Surface habitable vs Prix de vente")
plt.xlabel("Surface habitable (en pieds²)")
plt.ylabel("Prix de vente ($)")
plt.grid(True)
plt.tight_layout()
st.pyplot()  # Remplacé plt.show() par st.pyplot()

# Encodage des variables catégorielles avec OneHotEncoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Sélection des variables d'entrée (X) et de sortie (y)
X = df_encoded.drop(columns=["SalePrice"])  # Variables d'entrée
y = df_encoded["SalePrice"]  # Variable cible (prix de vente)

# Séparation des données en entraînement (80%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialisation du modèle
model = LinearRegression()

# Entraînement du modèle sur les données d'entraînement
model.fit(X_train, y_train)

# Prédictions sur les données de test
y_pred = model.predict(X_test)

# Affichage des résultats
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

st.write(f"MSE (Mean Squared Error) : {mse}")
st.write(f"R² (R-squared) : {r2}")

# Affichage du graphique de prédictions vs valeurs réelles
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", lw=2)  # ligne parfaite
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title("Prédictions vs Valeurs réelles")
plt.grid(True)
plt.tight_layout()
st.pyplot()  # Remplacé plt.show() par st.pyplot()

# Random Forest - Prédictions et évaluations
from sklearn.ensemble import RandomForestRegressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# Évaluation
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

st.write(f"🌲 Random Forest - MSE : {mse_rf:.2f}")
st.write(f"🌲 Random Forest - R² : {r2_rf:.4f}")

# Visualisation des importances des variables
importances = rf_model.feature_importances_
features = X_train.columns
feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)

plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance[:10], y=feature_importance.index[:10])
plt.title("🔍 Importance des variables dans le modèle Random Forest")
plt.xlabel("Importance")
plt.ylabel("Variable")
plt.tight_layout()
st.pyplot()  # Remplacé plt.show() par st.pyplot()





