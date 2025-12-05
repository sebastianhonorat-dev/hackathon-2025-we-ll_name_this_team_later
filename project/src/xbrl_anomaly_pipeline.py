import pandas as pd
import numpy as np
import plotly.express as px

import plotly.io as pio
pio.renderers.default = "browser"

df = pd.read_csv("../data/organized_sec_xbrl_clean.csv")

adj_df = df.set_index(["cik", "period"])

cik_group = {}

for cik in adj_df.index.get_level_values(0).unique():
    cik_group[cik] = adj_df.loc[cik]

epsilon = 10**-9

log_cik_group = {cik: df.copy() for cik, df in cik_group.items()}

for cik, df  in log_cik_group.items():

    for col in df.columns:
        smallest = df[col].loc[df[col] != 0].min()
        largest = df[col].loc[df[col] != 0].max()
        if pd.isna(largest):
            continue
        ratio = max(abs(largest),abs(smallest))/min(abs(largest),abs(smallest))

        if ratio > 10:
            pos_mask = df[col] > 0
            neg_mask = df[col] < 0

            df.loc[pos_mask, col] = np.log(df[col].loc[pos_mask] + epsilon)
            df.loc[neg_mask, col] = -1 * np.log(abs(df[col].loc[neg_mask]) + epsilon)

###################
#PCA 2D and 3D
###################

cik = 320193

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

def clust_up(cik):
    X_in = log_cik_group[cik]
    X_scaled_in = StandardScaler().fit_transform(X_in)

    pca_2d = PCA(n_components = 2)
    pca_3d = PCA(n_components = 3)

    coords_2d_in = pca_2d.fit_transform(X_scaled_in)
    coords_3d_in = pca_3d.fit_transform(X_scaled_in)

    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters = 10, random_state = 42)
    labels = kmeans.fit_predict(coords_2d_in)

    cluster_results = pd.DataFrame({
        "Period": log_cik_group[cik].index,
        "Cluster": labels
    })

    fig2_in = px.scatter(
    log_cik_group[cik],
    x = coords_2d_in[:,0],
    y = coords_2d_in[:,1],
    color = cluster_results["Cluster"]
    )

    fig3_in = px.scatter_3d(
    log_cik_group[cik],
    x = coords_3d_in[:,0],
    y = coords_3d_in[:,1],
    z = coords_3d_in[:,2],
    color = cluster_results["Cluster"]
    )
    return cluster_results,fig2_in,fig3_in, coords_2d_in, coords_3d_in, X_scaled_in
_,fig2,fig3,coords_2d, coords_3d, X_scaled  = clust_up(cik)

##############
# Isolation Forest
##############
def iso_for(coords_3d_in,cik_in):
    from sklearn.ensemble import IsolationForest

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(log_cik_group[cik_in])

    iso_in = IsolationForest(
        n_estimators=100,
        contamination = 0.05,
        random_state = 42
    )

    iso_in.fit(X_scaled)
    anomaly_score = iso_in.decision_function(X_scaled)
    anomaly_label = iso_in.predict(X_scaled)

    anomaly_results_in = pd.DataFrame({
        "CIK": cik_in,
        "Period": [i for i in log_cik_group[cik_in].index],
        "AnomalyScore": anomaly_score,
        "AnomalyLabel": anomaly_label
    }).set_index("Period")

    fig4 = px.scatter_3d(
        x = coords_3d_in[:,0],
        y = coords_3d_in[:,1],
        z = coords_3d_in[:,2],
        color = anomaly_label.astype(str),
        color_discrete_map = {"-1": "red", "1": "grey"},
        hover_name = log_cik_group[cik_in].index,
        labels = {"x": "PC1", "y": "PC2", "color": "AnomalyLabel"},
        title = f"{cik_in} - Isolation Forest Anomalies"
    )

    return fig4, anomaly_results_in, iso_in, X_scaled

fig_if, anomaly_results, iso, X_iso  = iso_for(coords_3d, cik)

###############
# SHAP
###############
import shap

def shap_explain_if(iso_model_in, X_in, cik_in, n_samples = 200):

    #picking a background sample
    if X_in.shape[0] > n_samples:
        background = X_in[:n_samples]
    else:
        background = X_in

    #Create SHAP explainer
    explainer = shap.TreeExplainer(iso_model_in)

    #Compute contributions for each point
    shap_values = explainer.shap_values(X_in)

    shap_df = pd.DataFrame(
        shap_values,
        index = log_cik_group[cik_in].index,
        columns = log_cik_group[cik_in].columns
    )

    preds = iso_model_in.predict(X_in)
    shap_df["AnomalyLabel"] = preds
    shap_df["CIK"] = cik_in

    return shap_df

shap_results = shap_explain_if(iso, X_iso, cik)