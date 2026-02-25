import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import plotly.express as px
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(
    page_title="Tourism Analytics",
    page_icon="🌍",
    layout="wide"
)

# ---------------------------------------------------
# CSS Styling
# ---------------------------------------------------
st.markdown("""
<style>
.main {background-color: #f8fafc;}
h1 {color: #1e293b; font-weight: 700; padding-bottom: 1rem; border-bottom: 4px solid #3b82f6;}
.stTabs [data-baseweb="tab-list"] {gap: 4px;}
.stTabs [data-baseweb="tab"] {background: #e2e8f0; border-radius: 8px; padding: 8px 20px; font-weight: 600;}
.stTabs [aria-selected="true"] {background: #3b82f6; color: white;}
div[data-testid="stMetricValue"] {font-size: 28px; font-weight: 700; color: #1e293b;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------
# Load Data
# ---------------------------------------------------
@st.cache_data
def load_data():
    try:
        df = pd.read_csv("master_dataset.csv")
        return df
    except:
        return pd.DataFrame()

# ---------------------------------------------------
# Load Models
# ---------------------------------------------------
@st.cache_resource
def load_models():

    try:
        with open("models/regression_model.pkl", "rb") as f:
            reg_package = pickle.load(f)

        with open("models/classification_model.pkl", "rb") as f:
            clf_package = pickle.load(f)

        return reg_package, clf_package

    except:
        return None, None

# ---------------------------------------------------
# Header
# ---------------------------------------------------
st.title("🌍 Tourism Experience Analytics Platform")
st.markdown("**Advanced Machine Learning for Tourism Insights**")
st.markdown("---")

df = load_data()

if df.empty:
    st.error("❌ master_dataset.csv not found")
    st.stop()

reg_pkg, clf_pkg = load_models()

if reg_pkg is None or clf_pkg is None:
    st.error("❌ Models not found in /models folder")
    st.stop()

# Extract models
reg_model = reg_pkg["model"]
reg_scaler = reg_pkg["scaler"]
reg_features = reg_pkg["feature_columns"]

clf_model = clf_pkg["model"]
clf_features = clf_pkg["feature_columns"]

# ---------------------------------------------------
# Sidebar Inputs
# ---------------------------------------------------
with st.sidebar:

    st.markdown("### ⚙️ Prediction Inputs")
    st.markdown("---")

    year = st.selectbox("Visit Year", sorted(df["VisitYear"].unique()))
    month = st.selectbox("Visit Month", sorted(df["VisitMonth"].unique()))

    user_visits = st.slider("User Visit Count", 1, 50, 5)
    attraction_visits = st.slider("Attraction Visit Count", 1, 500, 50)
    user_avg = st.slider("User Avg Rating", 1.0, 5.0, 4.0)
    attraction_avg = st.slider("Attraction Avg Rating", 1.0, 5.0, 4.0)
    type_pop = st.slider("Attraction Type Popularity", 1, 500, 50)

# ---------------------------------------------------
# Tabs
# ---------------------------------------------------
tab1, tab2, tab3 = st.tabs(
    ["📊 Dashboard", "🎯 Predictions", "📈 Analytics"]
)

# ===================================================
# TAB 1 — DASHBOARD
# ===================================================
with tab1:

    st.subheader("📊 Executive Dashboard")

    c1, c2, c3, c4 = st.columns(4)

    c1.metric("Total Records", f"{len(df):,}")
    c2.metric("Users", f"{df['UserId'].nunique():,}")
    c3.metric("Attractions", f"{df['AttractionId'].nunique():,}")
    c4.metric("Avg Rating", f"{df['Rating'].mean():.2f}")

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        yearly = df.groupby("VisitYear").size().reset_index(name="Visits")
        fig = px.line(yearly, x="VisitYear", y="Visits", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        top_types = df["AttractionType"].value_counts().head(6).reset_index()
        top_types.columns = ["Type", "Count"]
        fig = px.pie(top_types, values="Count", names="Type")
        st.plotly_chart(fig, use_container_width=True)

# ===================================================
# TAB 2 — PREDICTIONS
# ===================================================
with tab2:

    st.subheader("🎯 ML Predictions")

    # ---------------- Regression Input ----------------
    reg_input_dict = {
        "VisitYear": year,
        "VisitMonth": month,
        "UserVisitCount": user_visits,
        "AttractionVisitCount": attraction_visits,
        "UserAvgRating": user_avg,
        "AttractionAvgRating": attraction_avg,
        "AttractionTypePopularity": type_pop
    }

    # Fill missing encoded features with 0
    for col in reg_features:
        if col not in reg_input_dict:
            reg_input_dict[col] = 0

    reg_input = pd.DataFrame([reg_input_dict])[reg_features]

    if reg_scaler is not None:
        reg_input = reg_scaler.transform(reg_input)

    pred_rating = reg_model.predict(reg_input)[0]

    st.metric("⭐ Predicted Rating", f"{pred_rating:.2f} / 5")

    # ---------------- Classification ----------------
    clf_input_dict = reg_input_dict.copy()

    for col in clf_features:
        if col not in clf_input_dict:
            clf_input_dict[col] = 0

    clf_input = pd.DataFrame([clf_input_dict])[clf_features]

    pred_mode = clf_model.predict(clf_input)[0]

    st.metric("🧭 Predicted Visit Mode", pred_mode)

# ===================================================
# TAB 3 — ANALYTICS
# ===================================================
with tab3:

    st.subheader("📈 Advanced Analytics")

    top_attr = df["Attraction"].value_counts().head(10).reset_index()
    top_attr.columns = ["Attraction", "Visits"]

    fig = px.bar(
        top_attr,
        x="Visits",
        y="Attraction",
        orientation="h"
    )

    st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------
# Footer
# ---------------------------------------------------
st.markdown("---")
st.markdown(
    "<p style='text-align:center'>Tourism Analytics Platform © 2026</p>",
    unsafe_allow_html=True
)
