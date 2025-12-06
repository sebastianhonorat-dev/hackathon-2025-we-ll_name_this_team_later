import finnhub
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler

finnhub_client = finnhub.Client(api_key="d4nj2vhr01qk2nuc6b20d4nj2vhr01qk2nuc6b2g")

#get 10 years worth of form 4 per company ticker
def form4_pull(tckr_in):
    data = finnhub_client.stock_insider_transactions(f"{tckr_in}", '2014-10-01', '2025-12-01')
    return data

#
tckr = "AIR"
form_4s = form4_pull(tckr)

df = pd.DataFrame(form_4s["data"]).set_index("name")
fig1 = px.line(df, x = "transactionDate", y = "share", color = df.index)
#fig1.show()

adj_df = df[["transactionDate","change","share"]] #reduce df to valuable quantitative columns

#create diction with sub-dataframes per insider name
ins_df = {name: adj_df.loc[name].set_index("transactionDate") for name in adj_df.index.unique()}

#drop names with less than 7 rows. NOT ENOUGH INFO
pop_names = []
for name, df_in in ins_df.items():

        if df_in.shape[0] < 7:
            pop_names.append(name)

for name in pop_names:
    del ins_df[name]

#LOG values, incase spread is too large
def log_df_func(df_dict: dict):
    epsilon = 10**-9
    column = ["change","share"]
    log_ins_group = {name_in: df_in.copy() for name_in, df_in in df_dict.items()}

    for name, df_in in log_ins_group.items():

        for col in column:
            df_in[col] = df_in[col].astype("float64")

            smallest = df_in[col].loc[df_in[col] != 0].min()
            largest = df_in[col].loc[df_in[col] != 0].max()
            if pd.isna(largest):
                continue
            ratio = max(abs(largest),abs(smallest))/min(abs(largest),abs(smallest))

            if ratio > 10:
                pos_mask = df_in[col] > 0
                neg_mask = df_in[col] < 0

                df_in.loc[pos_mask, col] = np.log(df_in[col].loc[pos_mask] + epsilon)
                df_in.loc[neg_mask, col] = -1 * np.log(abs(df_in[col].loc[neg_mask]) + epsilon)
    return log_ins_group
log_dict = log_df_func(ins_df)

#name list
name_list = [name for name in log_dict.keys()]



#DBSCAN instead of Iso Forest, because there's only two quantitative columns
def db_scan_func(name_in):
    df_in = log_dict[name_in]

    X_in = df_in
    X_scaled_in = StandardScaler().fit_transform(X_in)

    db = DBSCAN(eps=0.5, min_samples=5)
    labels = db.fit_predict(X_scaled_in)

    df_in["cluster"] = labels.astype(str)
    df_in["is_anomaly"] = (labels == -1).astype(int)

    #######
    # 2) Get all non-anomaly clusters as strings
    unique_clusters = sorted(c for c in df_in["cluster"].unique() if c != "-1")

    # 3) Take a list of blues (slice, not single item!)
    base_blues = px.colors.sequential.Blues  # this is a list like ['#f7fbff', ..., '#08306b']

    # pick as many as we need
    blue_palette = base_blues[:len(unique_clusters)]  # or base_blues[-len(unique_clusters):]

    # 4) Build the color map
    color_map = {"-1": "red"}  # anomaly color
    for c, color in zip(unique_clusters, blue_palette):
        color_map[c] = color
    #######

    fig2_in = px.scatter(
        df_in,
        x="change",
        y="share",
        color="cluster",
        color_discrete_map=color_map,
        hover_name=df_in.index,
        title=f"{name_in} - DBSCAN"
    )

    return df_in, fig2_in, X_scaled_in