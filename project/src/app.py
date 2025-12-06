# app.py
# Streamlit dashboard wiring together:
#  - XBRL anomaly pipeline (fundamentals)
#  - Form 4 insider trading anomaly pipeline

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from sklearn.cluster import DBSCAN, KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
import shap
import finnhub
import requests

SEC_COMPANY_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

# SEC wants a real User-Agent header
SEC_HEADERS = {
    "User-Agent": "your-name your-email@example.com",
    "Accept-Encoding": "gzip, deflate",
    "Host": "www.sec.gov",
}


@st.cache_data(show_spinner=False, ttl=24 * 3600)
def load_ticker_cik_mapping():
    """
    Load SEC company_tickers.json and return:
      - df: DataFrame with columns [cik, ticker, name]
      - mapping: dict[ticker_upper] -> {"cik": int, "name": str}
    """
    resp = requests.get(SEC_COMPANY_TICKERS_URL, headers=SEC_HEADERS)
    resp.raise_for_status()
    raw = resp.json()  # keys are 0,1,2,... values have cik_str, ticker, title

    df = pd.DataFrame.from_dict(raw, orient="index")
    # columns: cik_str, ticker, title
    df["cik"] = df["cik_str"].astype(int)
    df.rename(columns={"title": "name"}, inplace=True)
    df["ticker"] = df["ticker"].str.upper()

    mapping = {
        row["ticker"]: {"cik": row["cik"], "name": row["name"]}
        for _, row in df.iterrows()
    }
    return df, mapping

def highlight_anomalies(df: pd.DataFrame, label_col: str = "AnomalyLabel"):
    """Return a styled DataFrame with anomaly rows highlighted."""
    def style_row(row):
        val = row.get(label_col, None)
        if val == -1:  # anomaly
            return ["background-color: #5a1a1a; color: #ffffff"] * len(row)
        return [""] * len(row)

    return df.style.apply(style_row, axis=1)

# -------------------------------------------------------------------
# CONFIG
# -------------------------------------------------------------------
st.set_page_config(
    page_title="SECure AI",
    page_icon="📈",
    layout="wide",
)


# Global style overrides
st.markdown(
    """
    <style>
    html, body, [data-testid="stApp"], [data-testid="stSidebar"] * {
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI",
                     Roboto, Oxygen, Ubuntu, Cantarell,
                     "Open Sans", "Helvetica Neue", sans-serif;
    }

    h1 {
        font-weight: 650;
        letter-spacing: 0.08em;
        font-size: 3.4rem !important;   /* <<< make title bigger */
        margin-top: 0.25rem !important;
    }

    h2, h3, h4 {
        font-weight: 600;
        letter-spacing: 0.02em;
    }

    .block-container {
        padding-top: 1rem !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


st.title("SECure AI")
st.write("Issuer anomaly surveillance – XBRL & Form 4")

# ⚠️ Put your Finnhub key here or use st.secrets["FINNHUB_API_KEY"]
FINNHUB_API_KEY = "d4nj2vhr01qk2nuc6b20d4nj2vhr01qk2nuc6b2g"
finnhub_client = finnhub.Client(api_key=FINNHUB_API_KEY)

# Path to your XBRL CSV
XBRL_CSV_PATH = "project/data/organized_sec_xbrl_clean.csv"


epsilon = 10 ** -9


# -------------------------------------------------------------------
# PIPELINE HELPERS – INSIDER FORM 4
# (same logic as your script, just wrapped so we can call it per ticker)
# -------------------------------------------------------------------
def form4_pull(tckr_in: str):
    """Get ~10 years of Form 4 insider transactions for a ticker."""
    data = finnhub_client.stock_insider_transactions(
        tckr_in, "2014-10-01", "2025-12-01"
    )
    return data


def log_df_func_insider(df_dict: dict):
    """
    Your log_df_func for insiders; unchanged logic.
    Applies log transform to 'change' and 'share' when spread is large.
    """
    column = ["change", "share"]
    log_ins_group = {name_in: df_in.copy() for name_in, df_in in df_dict.items()}

    for name, df_in in log_ins_group.items():
        for col in column:
            df_in[col] = df_in[col].astype("float64")

            smallest = df_in[col].loc[df_in[col] != 0].min()
            largest = df_in[col].loc[df_in[col] != 0].max()
            if pd.isna(largest):
                continue
            ratio = max(abs(largest), abs(smallest)) / min(
                abs(largest), abs(smallest)
            )

            if ratio > 10:
                pos_mask = df_in[col] > 0
                neg_mask = df_in[col] < 0

                df_in.loc[pos_mask, col] = np.log(
                    df_in[col].loc[pos_mask] + epsilon
                )
                df_in.loc[neg_mask, col] = -1 * np.log(
                    abs(df_in[col].loc[neg_mask]) + epsilon
                )
    return log_ins_group


@st.cache_data(show_spinner=False)
def prepare_insider_data(ticker: str):
    """
    Run your Form 4 pipeline for a given ticker:
      - pull data from Finnhub
      - build df + adj_df
      - build dict of insider dataframes (>= 7 trades)
      - log transform
    Returns:
      df_raw, fig_timeseries, log_dict, name_list
    """
    raw = form4_pull(ticker)
    rows = raw.get("data", [])
    if not rows:
        return None, None, None, []

    df = pd.DataFrame(rows)
    if df.empty:
        return None, None, None, []

    # Align with your script: index by insider name
    df["transactionDate"] = pd.to_datetime(df["transactionDate"])
    df = df.set_index("name")

    fig_timeseries = px.line(
        df.reset_index(),
        x="transactionDate",
        y="share",
        color="name",
        hover_name=df.index,
        title=f"{ticker} – Insider share holdings over time",
    )

    # Reduce to relevant quantitative columns
    adj_df = df[["transactionDate", "change", "share"]]

    # Build sub-dataframes per insider name, index = transactionDate
    ins_df = {
        name: df_in.set_index("transactionDate").copy()
        for name, df_in in adj_df.groupby(adj_df.index)
    }

    ins_df = {
        name: df_in for name, df_in in ins_df.items() if df_in.shape[0] >= 7
    }

    if not ins_df:
        return df, fig_timeseries, None, []

    # Log transform (your logic)
    log_dict = log_df_func_insider(ins_df)
    name_list = list(log_dict.keys())

    return df, fig_timeseries, log_dict, name_list


def run_dbscan_for_insider(name_in: str, log_dict: dict):
    """
    Your DBSCAN pipeline for a single insider (name).
    Same logic, but takes log_dict as argument instead of global.
    """
    df_in = log_dict[name_in].copy()

    X_in = df_in[["change", "share"]]
    X_scaled_in = StandardScaler().fit_transform(X_in)

    db = DBSCAN(eps=0.5, min_samples=5)
    labels = db.fit_predict(X_scaled_in)

    df_in["cluster"] = labels.astype(str)
    df_in["is_anomaly"] = (labels == -1).astype(int)

    # Build color map: red for anomalies, BlUE scale for others
    unique_clusters = sorted(c for c in df_in["cluster"].unique() if c != "-1")
    base_blues = px.colors.sequential.Blues
    blue_palette = base_blues[: len(unique_clusters)]

    color_map = {"-1": "red"}
    for c, color in zip(unique_clusters, blue_palette):
        color_map[c] = color

    fig = px.scatter(
        df_in.reset_index(),
        x="change",
        y="share",
        color="cluster",
        color_discrete_map=color_map,
        hover_name= df_in.index,
        title=f"{name_in} – DBSCAN clusters (Form 4 trades)",
    )

    return df_in, fig


# -------------------------------------------------------------------
# PIPELINE HELPERS – XBRL FUNDAMENTALS
# (same logic as your second script, just wrapped)
# -------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def load_xbrl_log_data():
    """
    Load your organized_sec_xbrl_clean.csv and apply the log transform.
    CIKs are normalized to int so they match the SEC ticker mapping.
    Returns:
      - log_cik_group: dict[int cik] -> log-transformed DataFrame
      - cik_list: sorted list of CIK ints
    """
    df = pd.read_csv(XBRL_CSV_PATH)

    # 🔴 Normalize CIK to int (handles '000001750', '1750', etc.)
    df["cik"] = (
        df["cik"]
        .astype(str)
        .str.extract(r"(\d+)")[0]  # grab digits
        .astype(int)
    )

    # MultiIndex on (cik, period)
    adj_df = df.set_index(["cik", "period"])

    # Group per CIK (keys will be ints)
    cik_group = {}
    for cik in adj_df.index.get_level_values(0).unique():
        cik_int = int(cik)
        cik_group[cik_int] = adj_df.loc[cik_int]

    # Your log-transform logic
    epsilon = 10 ** -9
    log_cik_group = {cik: df_cik.copy() for cik, df_cik in cik_group.items()}

    for cik, df_cik in log_cik_group.items():
        for col in df_cik.columns:
            smallest = df_cik[col].loc[df_cik[col] != 0].min()
            largest = df_cik[col].loc[df_cik[col] != 0].max()
            if pd.isna(largest):
                continue
            ratio = max(abs(largest), abs(smallest)) / min(
                abs(largest), abs(smallest)
            )

            if ratio > 10:
                pos_mask = df_cik[col] > 0
                neg_mask = df_cik[col] < 0

                df_cik.loc[pos_mask, col] = np.log(
                    df_cik[col].loc[pos_mask] + epsilon
                )
                df_cik.loc[neg_mask, col] = -1 * np.log(
                    abs(df_cik[col].loc[neg_mask]) + epsilon
                )

    cik_list = sorted(log_cik_group.keys())  # list of ints
    return log_cik_group, cik_list


def clust_up(cik, log_cik_group):
    """
    Your clust_up pipeline: PCA(2D & 3D) + KMeans(10).
    Returns:
      - cluster_results (Period, Cluster)
      - fig2_in (2D PCA scatter)
      - fig3_in (3D PCA scatter)
    """
    X_in = log_cik_group[cik]
    X_scaled_in = StandardScaler().fit_transform(X_in)

    pca_2d = PCA(n_components=2)
    pca_3d = PCA(n_components=3)

    coords_2d_in = pca_2d.fit_transform(X_scaled_in)
    coords_3d_in = pca_3d.fit_transform(X_scaled_in)

    kmeans = KMeans(n_clusters=10, random_state=42)
    labels = kmeans.fit_predict(coords_2d_in)

    cluster_results = pd.DataFrame(
        {
            "Period": log_cik_group[cik].index,
            "Cluster": labels,
        }
    )

    fig2_in = px.scatter(
        x=coords_2d_in[:, 0],
        y=coords_2d_in[:, 1],
        color=cluster_results["Cluster"].astype(str),
        hover_name=log_cik_group[cik].index,
        title=f"{cik} – PCA(2D) + KMeans clusters",
        labels={"x": "PC1", "y": "PC2"},
    )

    fig3_in = px.scatter_3d(
        x=coords_3d_in[:, 0],
        y=coords_3d_in[:, 1],
        z=coords_3d_in[:, 2],
        color=cluster_results["Cluster"].astype(str),
        hover_name=log_cik_group[cik].index,
        title=f"{cik} – PCA(3D) + KMeans clusters",
        labels={"x": "PC1", "y": "PC2", "z": "PC3"},
    )

    return cluster_results, fig2_in, fig3_in


def iso_for(cik_in, log_cik_group):
    """
    Your Isolation Forest pipeline for fundamentals.
    Returns:
      - fig4 (3D PCA with anomalies)
      - anomaly_results_in (DataFrame)
      - iso_model (IsolationForest)
      - iso_X_scaled_in (scaled feature matrix)
    """
    scaler = StandardScaler()
    iso_X_scaled_in = scaler.fit_transform(log_cik_group[cik_in])

    pca_3d = PCA(n_components=3)
    coords_3d_in = pca_3d.fit_transform(iso_X_scaled_in)

    iso_in = IsolationForest(
        n_estimators=100,
        contamination=0.05,
        random_state=42,
    )

    iso_in.fit(iso_X_scaled_in)
    anomaly_score = iso_in.decision_function(iso_X_scaled_in)
    anomaly_label = iso_in.predict(iso_X_scaled_in)

    anomaly_results_in = pd.DataFrame(
        {
            "CIK": cik_in,
            "Period": [i for i in log_cik_group[cik_in].index],
            "AnomalyScore": anomaly_score,
            "AnomalyLabel": anomaly_label,
        }
    ).set_index("Period")

    fig4 = px.scatter_3d(
        x=coords_3d_in[:, 0],
        y=coords_3d_in[:, 1],
        z=coords_3d_in[:, 2],
        color=anomaly_label.astype(str),
        color_discrete_map={"-1": "red", "1": "grey"},
        hover_name=log_cik_group[cik_in].index,
        labels={"x": "PC1", "y": "PC2", "z": "PC3"},
        title=f"{cik_in} – Isolation Forest anomalies",
    )

    return fig4, anomaly_results_in, iso_in, iso_X_scaled_in


def shap_explain_if(iso_model_in, iso_X_scaled_in, cik_in, log_cik_group, n_samples=200):
    """
    Your SHAP explanation function, modified only to receive log_cik_group.
    """
    # background sample
    if iso_X_scaled_in.shape[0] > n_samples:
        background = iso_X_scaled_in[:n_samples]
    else:
        background = iso_X_scaled_in

    explainer = shap.TreeExplainer(iso_model_in)
    shap_values = explainer.shap_values(iso_X_scaled_in)

    shap_df = pd.DataFrame(
        shap_values,
        index=log_cik_group[cik_in].index,
        columns=log_cik_group[cik_in].columns,
    )

    preds = iso_model_in.predict(iso_X_scaled_in)
    shap_df["AnomalyLabel"] = preds
    shap_df["CIK"] = cik_in

    return shap_df


# -------------------------------------------------------------------
# SIDEBAR – USER INPUTS
# -------------------------------------------------------------------
st.sidebar.header("Company Selection")
# ---- SEC mapping (you already added load_ticker_cik_mapping) ----
company_df, ticker_map = load_ticker_cik_mapping()

# ---- XBRL data ----
try:
    log_cik_group, cik_list = load_xbrl_log_data()
except Exception as e:
    log_cik_group, cik_list = {}, []
    st.sidebar.error(f"Error loading XBRL data: {e}")

# ---- Ticker input (single source of truth) ----
user_ticker = st.sidebar.text_input("Ticker", value="AIR").upper()

cik_from_ticker = None
company_info = ticker_map.get(user_ticker)
if company_info is None:
    st.sidebar.warning("Ticker not found in SEC mapping.")
else:
    cik_from_ticker = int(company_info["cik"])
    name = company_info["name"]
    st.sidebar.success(f"{user_ticker} → CIK {cik_from_ticker} ({name})")

st.sidebar.caption(
    "Tip: Use the **same company** (its CIK and ticker) to compare "
    "fundamental anomalies vs insider trading behavior."
)

# -------------------------------------------------------------------
# MAIN LAYOUT – TWO TABS
# -------------------------------------------------------------------
tab_fundamentals, tab_insiders = st.tabs(
    ["🏛 Fundamentals anomalies (XBRL)", "Insider Trading Anomalies (Form 4)"]
)

# ------------------------------ FUNDAMENTALS TAB --------------------
with tab_fundamentals:
    st.subheader("Fundamentals Anomalies (XBRL)")

    # Conditions where we can't run the fundamentals pipeline
    if not log_cik_group:
        st.info("XBRL data not loaded.")
        st.stop()

    if cik_from_ticker is None:
        st.info("Enter a valid ticker in the sidebar to look up its CIK.")
        st.stop()

    if cik_from_ticker not in log_cik_group:
        st.warning(
            f"CIK {cik_from_ticker} (from ticker {user_ticker}) "
            "is not present in the XBRL dataset."
        )
        st.stop()

    cik_key = cik_from_ticker  # this is the key for log_cik_group

    # PCA + KMeans
    with st.spinner("Running PCA + KMeans clustering..."):
        cluster_results, fig2, fig3 = clust_up(cik_key, log_cik_group)

    # Isolation Forest
    with st.spinner("Running Isolation Forest anomaly detection..."):
        fig4, anomaly_results, iso_model, iso_X_scaled = iso_for(
            cik_key, log_cik_group
        )

    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(fig2, use_container_width=True)
    with col2:
        st.plotly_chart(fig3, use_container_width=True)

    st.plotly_chart(fig4, use_container_width=True)

    st.markdown("### Anomaly table")
    st.dataframe(highlight_anomalies(anomaly_results))

    with st.expander("Explain Anomalies with SHAP"):
        st.write(
            "This computes SHAP values to see which features drive the Isolation Forest anomaly scores."
        )
        if st.button("Compute SHAP for this CIK"):
            with st.spinner("Computing SHAP values..."):
                shap_df = shap_explain_if(
                    iso_model, iso_X_scaled, cik_key, log_cik_group
                )

            st.markdown("#### SHAP values (how anomalous each column was)")
            st.write("Showing first 100 rows:")
            st.dataframe(highlight_anomalies(shap_df.head(100)))

            # ---- NEW: join in the underlying fundamentals ("raw" data) ----
            st.markdown("#### Underlying fundamentals for these periods")

            # log_cik_group[cik_key] has the feature values per Period
            fundamentals_df = log_cik_group[cik_key].copy()

            # anomaly_results has AnomalyLabel per Period (index = Period)
            # make sure index aligns by Period
            fundamentals_with_labels = fundamentals_df.join(
                anomaly_results[["AnomalyLabel"]],
                how="left"
            )

            st.dataframe(highlight_anomalies(fundamentals_with_labels))

# ------------------------------ INSIDERS TAB ------------------------
with tab_insiders:
    st.subheader("Insider Trading Anomalies (Form 4)")

    # No ticker? Show prompt.
    if not user_ticker:
        st.info("Enter a ticker in the sidebar to analyze insider trades.")
        st.stop()

    st.write(f" Form 4 Data Analysis for **{user_ticker}**")

    # Pull and prep data
    try:
        with st.spinner("Pulling Form 4 insider transactions..."):
            df_raw, fig_timeseries, log_dict, name_list = prepare_insider_data(
                user_ticker
            )
    except Exception as e:
        st.error(f"Error while fetching / processing Form 4 data: {e}")
        st.stop()

    # No data case
    if df_raw is None:
        st.warning(f"No Form 4 data returned for ticker '{user_ticker}'.")
        st.stop()

    # Always show the time series chart + some raw data
    if fig_timeseries is not None:
        st.plotly_chart(fig_timeseries, use_container_width=True)

    with st.expander("Raw Form 4 data (first 100 rows)"):
        st.dataframe(df_raw.head(100))

    # Not enough history per insider for DBSCAN
    if not log_dict or not name_list:
        st.info(
            "Not enough history per insider (need at least 7 trades per insider) "
            "to run the DBSCAN anomaly analysis."
        )
        st.stop()

    # Full insider-level anomaly analysis
    st.markdown("### Insider-level anomaly analysis")

    selected_insider = st.selectbox(
        "Choose an insider (name)", name_list
    )

    df_insider, fig_dbscan = run_dbscan_for_insider(
        selected_insider, log_dict
    )

    st.plotly_chart(fig_dbscan, use_container_width=True)

    st.markdown("#### Trades for this insider (with anomaly flag)")
    st.dataframe(df_insider)

    num_anom = int(df_insider["is_anomaly"].sum())
    st.info(
        f"Detected **{num_anom}** anomalous trades for insider **{selected_insider}** "
        f"(cluster label = -1 in the DBSCAN output)."
    )
