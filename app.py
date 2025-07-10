import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.cluster import KMeans, DBSCAN
from sklearn.decomposition import PCA
from imblearn.over_sampling import SMOTE
import shap
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(layout="wide")
st.title("📊 Dashboard de Análise de Risco de Crédito")

# Carrega a base de dados embutida
df = pd.read_csv("credit_customers.csv")
df_original = df.copy()

st.subheader("Visualização da Base de Dados")
st.dataframe(df.head())

# Codificar variável alvo
df['class'] = df['class'].map({'good': 0, 'bad': 1})
X = df.drop("class", axis=1)
y = df["class"]

cat_features = X.select_dtypes(include='object').columns.tolist()
num_features = X.select_dtypes(include=['int64', 'float64']).columns.tolist()

# Pré-processamento
preprocessor = ColumnTransformer([
    ("num", StandardScaler(), num_features),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features)
])

pipeline = Pipeline([
    ("preprocessor", preprocessor)
])

X_processed = pipeline.fit_transform(X)

# Balanceamento com SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_processed, y)

# Modelo
rf = RandomForestClassifier(random_state=42)
rf.fit(X_res, y_res)

st.subheader("Classificação de Risco")
prob = rf.predict_proba(X_processed)[:, 1]
df_original["Risco de Inadimplência (%)"] = (prob * 100).round(2)
st.dataframe(df_original[["Risco de Inadimplência (%)"]].join(df[["class"]]))

# SHAP
st.subheader("Explicabilidade com SHAP")
explainer = shap.Explainer(rf, X_res)
shap_values = explainer(X_res[:100])  # limitar para desempenho

fig, ax = plt.subplots(figsize=(10, 6))
shap.plots.beeswarm(shap_values, max_display=10, show=False)
st.pyplot(fig)

# Escolha individual
idx = st.number_input("Selecione o índice do cliente para explicação individual:", min_value=0, max_value=len(X)-1, value=0)
shap_ind = explainer(X_processed[idx:idx+1])
st.write(f"Cliente {idx}: Classe real = {'bad' if y.iloc[idx] == 1 else 'good'}")
st_shap = shap.plots.waterfall(shap_ind[0], show=False)
st.pyplot(bbox_inches='tight', dpi=300)

# Clustering com KMeans
st.subheader("Clusterização com KMeans")
kmeans = KMeans(n_clusters=3, random_state=42)
cluster_labels = kmeans.fit_predict(X_processed)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_processed)
cluster_df = pd.DataFrame(X_pca, columns=["PC1", "PC2"])
cluster_df["Cluster"] = cluster_labels
cluster_df["Classe"] = y.values

fig, ax = plt.subplots()
sns.scatterplot(data=cluster_df, x="PC1", y="PC2", hue="Cluster", style="Classe", palette="Set2", ax=ax)
st.pyplot(fig)

# DBSCAN para outliers
st.subheader("Detecção de Outliers com DBSCAN")
dbscan = DBSCAN(eps=3.7, min_samples=30)
outlier_labels = dbscan.fit_predict(X_processed)
cluster_df["DBSCAN"] = outlier_labels

fig, ax = plt.subplots()
sns.scatterplot(data=cluster_df, x="PC1", y="PC2", hue="DBSCAN", palette="Spectral", ax=ax)
st.pyplot(fig)

st.success("✅ Dashboard completo com classificação, explicabilidade, clusters e outliers.")
