import pandas as pd
import numpy as np
import plotly.express as px

import plotly.io as pio
pio.renderers.default = "browser"

df = pd.read_csv("../data/organized_sec_xbrl_clean.csv")

adj_df = df.set_index(["cik", "period"])

epsilon = 10**-9
log_df = adj_df.copy()

for col in log_df.columns:
    smallest = log_df[col].loc[log_df[col] != 0].min()
    largest = log_df[col].loc[log_df[col] != 0].max()
    ratio = max(abs(largest),abs(smallest))/min(abs(largest),abs(smallest))
    if ratio > 10:
        log_df[col].loc[log_df[col] > 0] = np.log(log_df[col].loc[log_df[col] > 0] + epsilon)
        log_df[col].loc[log_df[col] < 0] = -1 * np.log(abs(log_df[col].loc[log_df[col] < 0]) + epsilon)

###################
#PCA 2D and 3D
###################
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

X = log_df
X_scaled = StandardScaler().fit_transform(X)

pca_2d = PCA(n_components = 2)
pca_3d = PCA(n_components = 3)

coords_2d = pca_2d.fit_transform(X_scaled)
coords_3d = pca_3d.fit_transform(X_scaled)

################
#KMeans
###############
from sklearn.cluster import KMeans

kmeans = KMeans(n_clusters = 10, random_state = 42)
labels = kmeans.fit_predict(coords_2d)

cluster_results = pd.DataFrame({
    "CIK, Period": log_df.index,
    "Cluster": labels
}).set_index("CIK, Period")

##################
#Scatter plots
##################
text = [f"{i[0]} {i[1]}" for i in log_df.index]

fig2 = px.scatter(
    log_df,
    x = coords_2d[:,0],
    y = coords_2d[:,1],
    color = cluster_results["Cluster"],
    color_continuous_scale = ["pink","red","black"]
)

fig2.show()

fig3 = px.scatter_3d(
    log_df,
    x = coords_3d[:,0],
    y = coords_3d[:,1],
    z = coords_3d[:,2],
    color = cluster_results["Cluster"],
    color_continuous_scale = ["pink","red","black"]
)

fig3.show()

##############
# Cik Cluster Movement
##############
for item in cluster_results:
    pass