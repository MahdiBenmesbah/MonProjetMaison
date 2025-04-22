#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

print("Tout est installé correctement ✅")


# In[8]:


# 1. 📦 Import des bibliothèques
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Affichage plus lisible
sns.set_style("whitegrid")
sns.set_palette("pastel")


# In[19]:


# 2. 📥 Chargement du dataset
# Télécharger le fichier train.csv depuis Kaggle, puis le placer dans le même dossier que ce notebook

df = pd.read_csv("train.csv")
df.shape, df.columns


# In[20]:


# 3. 🧹 Exploration rapide et nettoyage de base

# Aperçu des données
df.head()

# Statistiques générales
df.describe()

# Données manquantes
missing = df.isnull().mean().sort_values(ascending=False)
missing[missing > 0]  # Seulement celles qui posent problème

# Suppression de colonnes très incomplètes
df = df.drop(columns=["Alley", "PoolQC", "Fence", "MiscFeature"], errors="ignore")


# In[21]:


# 4. 📊 Analyse exploratoire

# Distribution du prix
sns.histplot(df['SalePrice'], kde=True)
plt.title("Distribution des prix de vente")
plt.show()

# Corrélation avec d’autres variables numériques
corr = df.corr(numeric_only=True)
top_corr = corr["SalePrice"].sort_values(ascending=False)[1:6]  # Top 5

print(top_corr)

# Scatterplot sur une variable très corrélée (ex: OverallQual)
sns.scatterplot(data=df, x="OverallQual", y="SalePrice")
plt.title("Qualité globale vs Prix de vente")
plt.show()


# In[22]:


# 5. 🏗️ Préparation des données

# Variables explicatives simples
features = ["GrLivArea", "OverallQual", "GarageCars", "TotalBsmtSF"]
X = df[features]
y = df["SalePrice"]

# Train/Test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# In[23]:


# 6. 🤖 Régression linéaire

model = LinearRegression()
model.fit(X_train, y_train)

# Prédictions
y_pred = model.predict(X_test)

# Évaluation
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)

print(f"RMSE : {rmse:.2f}")
print(f"R² : {r2:.3f}")


# In[24]:


# 7. 🧾 Conclusion rapide

print("Le modèle explique environ {:.1f}% de la variance du prix de vente.".format(r2 * 100))
print("RMSE : erreur moyenne d’environ {:.0f} $ sur l’échantillon de test.".format(rmse))

# À améliorer : traitement des variables catégorielles, plus de features, modèles non-linéaires…


# In[17]:


import os

# Affiche le chemin actuel + les fichiers dans le dossier
print("Répertoire actuel :", os.getcwd())
print("Contenu du dossier :", os.listdir())


# In[18]:


import pandas as pd

df = pd.read_csv("train.csv")
df.head()


# In[25]:


# Aperçu des premières lignes
display(df.head())

# Dimensions du dataset
print(f"Nombre de lignes : {df.shape[0]}")
print(f"Nombre de colonnes : {df.shape[1]}")

# Types de données
print("\nTypes de données :")
print(df.dtypes)

# Valeurs manquantes
print("\nValeurs manquantes :")
print(df.isnull().sum())

# Statistiques générales
display(df.describe(include='all'))


# In[26]:


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
print("Valeurs manquantes restantes :")
print(df.isnull().sum().sum())  # doit afficher 0


# In[27]:


import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x="GrLivArea", y="SalePrice", alpha=0.6, edgecolor=None)

plt.title("Relation entre surface habitable (GrLivArea) et prix de vente (SalePrice)")
plt.xlabel("Surface habitable (en pieds²)")
plt.ylabel("Prix de vente ($)")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[28]:


plt.figure(figsize=(10, 6))
sns.regplot(data=df, x="GrLivArea", y="SalePrice", scatter_kws={"alpha": 0.6}, line_kws={"color": "red"})

plt.title("Régression linéaire : Surface habitable vs Prix de vente")
plt.xlabel("Surface habitable (en pieds²)")
plt.ylabel("Prix de vente ($)")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[31]:


plt.figure(figsize=(12, 8))
sns.lmplot(
    data=df,
    x="GrLivArea",
    y="SalePrice",
    hue="Neighborhood",       # Couleur par quartier
    height=12,
    aspect=1.6,
    scatter_kws={"alpha": 0.5, "s": 40},
    line_kws={"linewidth": 2}
)

plt.title("Régressions par quartier : Surface habitable vs Prix de vente", fontsize=14)
plt.xlabel("Surface habitable (en pieds²)")
plt.ylabel("Prix de vente ($)")
plt.tight_layout()
plt.show()


# In[32]:


# 🧹 Encodage des variables catégorielles avec OneHotEncoding
df_encoded = pd.get_dummies(df, drop_first=True)

# Vérification : quelques colonnes après encodage
df_encoded.head()


# In[33]:


# Sélection des variables d'entrée (X) et de sortie (y)
X = df_encoded.drop(columns=["SalePrice"])  # Variables d'entrée
y = df_encoded["SalePrice"]  # Variable cible (prix de vente)

# Affichage de la forme des données X et y
print(f"Shape of X: {X.shape}")
print(f"Shape of y: {y.shape}")


# In[34]:


from sklearn.model_selection import train_test_split

# Séparation des données en entraînement (80%) et test (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print(f"Shape of X_train: {X_train.shape}")
print(f"Shape of X_test: {X_test.shape}")


# In[35]:


from sklearn.linear_model import LinearRegression

# Initialisation du modèle
model = LinearRegression()

# Entraînement du modèle sur les données d'entraînement
model.fit(X_train, y_train)

# Prédictions sur les données de test
y_pred = model.predict(X_test)

# Affichage des résultats
from sklearn.metrics import mean_squared_error, r2_score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"MSE (Mean Squared Error) : {mse}")
print(f"R² (R-squared) : {r2}")


# In[36]:


plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.plot([min(y_test), max(y_test)], [min(y_test), max(y_test)], color="red", lw=2)  # ligne parfaite
plt.xlabel("Valeurs réelles")
plt.ylabel("Prédictions")
plt.title("Prédictions vs Valeurs réelles")
plt.grid(True)
plt.tight_layout()
plt.show()


# In[37]:


from sklearn.ensemble import RandomForestRegressor

rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

y_pred_rf = rf_model.predict(X_test)

# Évaluation
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"🌲 Random Forest - MSE : {mse_rf:.2f}")
print(f"🌲 Random Forest - R² : {r2_rf:.4f}")


# In[38]:


import matplotlib.pyplot as plt
import seaborn as sns

# Calcul des importances des features
importances = rf_model.feature_importances_
features = X_train.columns
feature_importance = pd.Series(importances, index=features).sort_values(ascending=False)

# Visualisation des 10 premières variables les plus importantes
plt.figure(figsize=(10,6))
sns.barplot(x=feature_importance[:10], y=feature_importance.index[:10])
plt.title("🔍 Importance des variables dans le modèle Random Forest")
plt.xlabel("Importance")
plt.ylabel("Variable")
plt.tight_layout()
plt.show()


# In[ ]:




