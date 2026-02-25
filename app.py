"""
Tourism Experience Analytics — Streamlit Application
======================================================
Features:
  • 🗺️  Home / Dashboard  — KPIs + charts
  • ⭐  Rating Predictor   — Regression model
  • 🧳  Visit Mode Predictor — Classification model
  • 🎯  Attraction Recommender — Collaborative filtering
  • 📊  EDA Explorer        — Interactive charts

Run:
    streamlit run app.py

Required pickle files (output/ folder):
    regression_model.pkl
    classification_model.pkl
    recommendation_system.pkl
    label_encoders.pkl
    feature_metadata.pkl
"""

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import os
import warnings
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Tourism Analytics AI",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Root palette ── */
:root {
    --bg:        #0d1117;
    --bg2:       #161b22;
    --bg3:       #1c2333;
    --border:    #30363d;
    --accent:    #f97316;
    --accent2:   #fb923c;
    --gold:      #f59e0b;
    --teal:      #2dd4bf;
    --text:      #e6edf3;
    --muted:     #8b949e;
    --success:   #3fb950;
    --danger:    #f85149;
    --radius:    14px;
}

html, body, .stApp { background: var(--bg) !important; color: var(--text) !important; font-family: 'DM Sans', sans-serif !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: var(--bg2) !important;
    border-right: 1px solid var(--border) !important;
}
[data-testid="stSidebar"] .stRadio label { color: var(--text) !important; font-weight: 500; }
[data-testid="stSidebar"] .stRadio [data-baseweb="radio"] { gap: 4px; }

/* ── Metric cards ── */
[data-testid="metric-container"] {
    background: var(--bg2) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius) !important;
    padding: 18px 20px !important;
}
[data-testid="metric-container"] label { color: var(--muted) !important; font-size: 0.78rem !important; text-transform: uppercase; letter-spacing: 0.06em; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: var(--text) !important; font-size: 2rem !important; font-weight: 700 !important; }

/* ── Inputs ── */
.stSelectbox > div > div, .stNumberInput > div > div, .stSlider {
    background: var(--bg3) !important;
    border-color: var(--border) !important;
    color: var(--text) !important;
    border-radius: 8px !important;
}
.stSelectbox label, .stNumberInput label, .stSlider label, .stMultiSelect label { color: var(--muted) !important; font-size: 0.82rem !important; }
div[data-baseweb="select"] { background: var(--bg3) !important; }
div[data-baseweb="select"] * { color: var(--text) !important; background: var(--bg3) !important; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, var(--accent), var(--gold)) !important;
    color: white !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    padding: 10px 28px !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 15px rgba(249,115,22,0.35) !important;
}
.stButton > button:hover { transform: translateY(-2px) !important; box-shadow: 0 8px 25px rgba(249,115,22,0.45) !important; }

/* ── Section headings ── */
.section-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.1rem;
    font-weight: 900;
    color: var(--text);
    margin-bottom: 6px;
}
.section-sub {
    color: var(--muted);
    font-size: 0.92rem;
    margin-bottom: 28px;
}

/* ── Prediction result box ── */
.pred-box {
    background: linear-gradient(135deg, #1a2744, #0d1117);
    border: 1px solid var(--accent);
    border-radius: var(--radius);
    padding: 28px 32px;
    text-align: center;
    margin-top: 18px;
    box-shadow: 0 0 40px rgba(249,115,22,0.18);
}
.pred-label { color: var(--muted); font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.08em; margin-bottom: 8px; }
.pred-value { font-family: 'Playfair Display', serif; font-size: 3.2rem; font-weight: 900; color: var(--accent); line-height: 1.1; }
.pred-sub { color: var(--muted); font-size: 0.83rem; margin-top: 6px; }

/* ── Rec card ── */
.rec-card {
    background: var(--bg2);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 16px 20px;
    margin-bottom: 10px;
    display: flex;
    align-items: center;
    gap: 14px;
    transition: border-color 0.2s;
}
.rec-card:hover { border-color: var(--accent); }
.rec-rank { font-size: 1.5rem; font-weight: 900; color: var(--accent); min-width: 36px; }
.rec-name { font-weight: 600; color: var(--text); font-size: 1rem; }
.rec-type { color: var(--muted); font-size: 0.78rem; margin-top: 2px; }
.rec-score { margin-left: auto; text-align: right; }
.rec-score-val { font-weight: 700; color: var(--teal); font-size: 0.95rem; }
.rec-score-lbl { color: var(--muted); font-size: 0.72rem; }

/* ── Info pills ── */
.pill {
    display: inline-block;
    background: rgba(249,115,22,0.15);
    color: var(--accent2);
    border: 1px solid rgba(249,115,22,0.3);
    border-radius: 100px;
    padding: 3px 12px;
    font-size: 0.78rem;
    font-weight: 600;
    margin: 2px;
}

/* ── Divider ── */
.fancy-divider {
    height: 1px;
    background: linear-gradient(90deg, transparent, var(--border), transparent);
    margin: 28px 0;
}

/* ── Table ── */
.stDataFrame { background: var(--bg2) !important; }
thead tr th { background: var(--bg3) !important; color: var(--muted) !important; font-size: 0.78rem !important; text-transform: uppercase !important; letter-spacing: 0.06em !important; }
tbody tr td { color: var(--text) !important; }

/* ── Expander ── */
.streamlit-expanderHeader { background: var(--bg2) !important; color: var(--text) !important; border: 1px solid var(--border) !important; border-radius: 10px !important; }

/* ── Alerts ── */
.stAlert { border-radius: 10px !important; }

/* ── Plots ── */
.stPlotlyChart, .stImage { border-radius: var(--radius) !important; }
</style>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────
# CONSTANTS & MODEL LOADING
# ──────────────────────────────────────────────
OUTPUT_DIR = r'C:\Users\rohit\Desktop\labmentix\models'

@st.cache_resource(show_spinner="Loading models…")
def load_models():
    """Load all pickle files from the output directory."""
    models = {}
    files = {
        "regression":    "regression_model.pkl",
        "classification":"classification_model.pkl",
        "recommendation":"recommendation_system.pkl",
        "encoders":      "label_encoders.pkl",
        "metadata":      "feature_metadata.pkl",
    }
    missing = []
    for key, fname in files.items():
        path = os.path.join(OUTPUT_DIR, fname)
        if os.path.exists(path):
            with open(path, "rb") as f:
                models[key] = pickle.load(f)
        else:
            missing.append(fname)
    if missing:
        st.warning(f"⚠️ Missing pickle files: {', '.join(missing)}\n\nRun the notebook first to generate them in `{OUTPUT_DIR}/`.")
    return models

models = load_models()

# ── Helper: safe encoder transform ──
def safe_encode(encoder, value):
    """Transform with fallback to 0 for unseen labels."""
    try:
        return int(encoder.transform([str(value)])[0])
    except Exception:
        try:
            return int(encoder.transform([encoder.classes_[0]])[0])
        except Exception:
            return 0

# ──────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='text-align:center; padding: 10px 0 24px;'>
        <div style='font-size:2.6rem;'>🌍</div>
        <div style='font-family: "Playfair Display", serif; font-size:1.25rem; font-weight:900; color:#e6edf3;'>Tourism AI</div>
        <div style='color:#8b949e; font-size:0.75rem; margin-top:4px;'>Analytics & Recommendations</div>
    </div>
    """, unsafe_allow_html=True)

    page = st.radio(
        "Navigate",
        ["🏠 Dashboard", "⭐ Rating Predictor", "🧳 Visit Mode Predictor",
         "🎯 Recommender", "📊 EDA Explorer"],
        label_visibility="collapsed",
    )

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # Model status
    st.markdown("<div style='color:#8b949e; font-size:0.72rem; text-transform:uppercase; letter-spacing:0.08em; margin-bottom:10px;'>Model Status</div>", unsafe_allow_html=True)

    status_items = [
        ("Regression",     "regression"    in models),
        ("Classification", "classification" in models),
        ("Recommender",    "recommendation" in models),
        ("Encoders",       "encoders"      in models),
    ]
    for name, ok in status_items:
        icon  = "🟢" if ok else "🔴"
        label = "Ready" if ok else "Missing"
        st.markdown(f"<div style='display:flex; justify-content:space-between; font-size:0.8rem; padding:3px 0;'><span style='color:#8b949e;'>{icon} {name}</span><span style='color:{'#3fb950' if ok else '#f85149'};'>{label}</span></div>", unsafe_allow_html=True)

    # Metadata summary
    if "metadata" in models:
        meta = models["metadata"]
        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.75rem; color:#8b949e;'>Regression features: <b style='color:#e6edf3;'>{len(meta.get('regression_features',[]))}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.75rem; color:#8b949e;'>Classification features: <b style='color:#e6edf3;'>{len(meta.get('classification_features',[]))}</b></div>", unsafe_allow_html=True)
        st.markdown(f"<div style='font-size:0.75rem; color:#8b949e;'>Visit modes: <b style='color:#e6edf3;'>{len(meta.get('visit_mode_classes',[]))}</b></div>", unsafe_allow_html=True)

# ══════════════════════════════════════════════
# PAGE 1 — DASHBOARD
# ══════════════════════════════════════════════
if page == "🏠 Dashboard":
    st.markdown("<div class='section-title'>Tourism Analytics Dashboard</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>AI-powered travel intelligence — ratings, visit modes, and personalized recommendations</div>", unsafe_allow_html=True)

    # KPI metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("🤖 Models Loaded", f"{sum(1 for k in ['regression','classification','recommendation'] if k in models)} / 3")
    with col2:
        if "recommendation" in models:
            mat = models["recommendation"]["user_item_matrix"]
            st.metric("👤 Users Indexed", f"{mat.shape[0]:,}")
        else:
            st.metric("👤 Users Indexed", "—")
    with col3:
        if "recommendation" in models:
            mat = models["recommendation"]["user_item_matrix"]
            st.metric("🏛️ Attractions", f"{mat.shape[1]:,}")
        else:
            st.metric("🏛️ Attractions", "—")
    with col4:
        if "classification" in models:
            clf_meta = models["classification"]
            acc = clf_meta.get("metrics", {}).get("accuracy", 0)
            st.metric("🎯 Classifier Accuracy", f"{acc*100:.1f}%")
        else:
            st.metric("🎯 Classifier Accuracy", "—")

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # Model performance summary
    st.markdown("### 📈 Trained Model Performance")
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Regression — Rating Prediction**")
        if "regression" in models:
            rm = models["regression"]
            metrics = rm.get("metrics", {})
            st.markdown(f"""
            <div style='background:#161b22; border:1px solid #30363d; border-radius:12px; padding:18px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:10px;'>
                    <span style='color:#8b949e; font-size:0.8rem;'>Best Model</span>
                    <span class='pill'>{rm.get('model_name','—')}</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                    <span style='color:#8b949e; font-size:0.85rem;'>R² Score</span>
                    <span style='color:#3fb950; font-weight:700;'>{metrics.get('r2', 0):.4f}</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                    <span style='color:#8b949e; font-size:0.85rem;'>RMSE</span>
                    <span style='color:#e6edf3; font-weight:600;'>{metrics.get('rmse', 0):.4f}</span>
                </div>
                <div style='display:flex; justify-content:space-between;'>
                    <span style='color:#8b949e; font-size:0.85rem;'>MAE</span>
                    <span style='color:#e6edf3; font-weight:600;'>{metrics.get('mae', 0):.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Regression model not loaded.")

    with col_b:
        st.markdown("**Classification — Visit Mode Prediction**")
        if "classification" in models:
            cm = models["classification"]
            cmet = cm.get("metrics", {})
            st.markdown(f"""
            <div style='background:#161b22; border:1px solid #30363d; border-radius:12px; padding:18px;'>
                <div style='display:flex; justify-content:space-between; margin-bottom:10px;'>
                    <span style='color:#8b949e; font-size:0.8rem;'>Best Model</span>
                    <span class='pill'>{cm.get('model_name','—')}</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                    <span style='color:#8b949e; font-size:0.85rem;'>Accuracy</span>
                    <span style='color:#3fb950; font-weight:700;'>{cmet.get('accuracy', 0)*100:.2f}%</span>
                </div>
                <div style='display:flex; justify-content:space-between; margin-bottom:6px;'>
                    <span style='color:#8b949e; font-size:0.85rem;'>F1-Score</span>
                    <span style='color:#e6edf3; font-weight:600;'>{cmet.get('f1', 0):.4f}</span>
                </div>
                <div style='display:flex; justify-content:space-between;'>
                    <span style='color:#8b949e; font-size:0.85rem;'>Precision</span>
                    <span style='color:#e6edf3; font-weight:600;'>{cmet.get('precision', 0):.4f}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info("Classification model not loaded.")

    st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

    # Feature columns preview
    if "metadata" in models:
        meta = models["metadata"]
        with st.expander("🔍 View Feature Columns Used in Models"):
            col_r, col_c = st.columns(2)
            with col_r:
                st.markdown("**Regression Features**")
                for f in meta.get("regression_features", []):
                    st.markdown(f"<span class='pill'>{f}</span>", unsafe_allow_html=True)
            with col_c:
                st.markdown("**Classification Features**")
                for f in meta.get("classification_features", []):
                    st.markdown(f"<span class='pill'>{f}</span>", unsafe_allow_html=True)

    # How to use
    st.markdown("### 🚀 How to Use This App")
    cols = st.columns(3)
    cards = [
        ("⭐", "Rating Predictor", "Input user demographics & attraction details to predict the rating a visitor will give."),
        ("🧳", "Visit Mode Predictor", "Predict whether the visitor is travelling as Business, Family, Couples, Friends, etc."),
        ("🎯", "Recommender", "Enter a User ID or select an attraction to get personalized recommendations."),
    ]
    for col, (icon, title, desc) in zip(cols, cards):
        with col:
            st.markdown(f"""
            <div style='background:#161b22; border:1px solid #30363d; border-radius:14px; padding:22px; height:150px;'>
                <div style='font-size:1.8rem; margin-bottom:8px;'>{icon}</div>
                <div style='font-weight:700; color:#e6edf3; font-size:1rem; margin-bottom:6px;'>{title}</div>
                <div style='color:#8b949e; font-size:0.82rem; line-height:1.45;'>{desc}</div>
            </div>
            """, unsafe_allow_html=True)


# ══════════════════════════════════════════════
# PAGE 2 — RATING PREDICTOR
# ══════════════════════════════════════════════
elif page == "⭐ Rating Predictor":
    st.markdown("<div class='section-title'>⭐ Rating Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Predict what rating a user will give to a tourist attraction based on their profile and visit context.</div>", unsafe_allow_html=True)

    if "regression" not in models or "encoders" not in models:
        st.error("❌ Regression model or encoders not found. Run the notebook first.")
        st.stop()

    reg_model  = models["regression"]["model"]
    reg_feats  = models["regression"]["feature_columns"]
    reg_scaler = models["regression"].get("scaler")
    encoders   = models["encoders"]
    meta       = models.get("metadata", {})

    # ── Get valid encoder classes
    def get_classes(col):
        if col in encoders:
            return sorted(encoders[col].classes_.tolist())
        return ["Unknown"]

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### 👤 User Profile")
        continent_name  = st.selectbox("Continent",       get_classes("ContinentName"))
        region_name     = st.selectbox("Region",          get_classes("RegionName"))
        country_name    = st.selectbox("Country",         get_classes("CountryName"))
        user_city_name  = st.selectbox("User City",       get_classes("UserCityName"))

        st.markdown("#### 📅 Visit Details")
        visit_year  = st.slider("Visit Year",  2000, 2024, 2022)
        visit_month = st.slider("Visit Month", 1, 12, 6,
                                 format="%d", help="1=Jan … 12=Dec")
        season_map  = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",
                       5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Autumn",
                       10:"Autumn",11:"Autumn"}
        season = season_map[visit_month]
        st.info(f"Season: **{season}**")

    with col2:
        st.markdown("#### 🏛️ Attraction Details")
        attraction_type  = st.selectbox("Attraction Type",    get_classes("AttractionType"))
        attr_city_name   = st.selectbox("Attraction City",    get_classes("AttractionCityName"))

        st.markdown("#### 📊 Historical Stats")
        user_visit_count   = st.number_input("User's Previous Visit Count",        min_value=1, max_value=500, value=5)
        user_avg_rating    = st.slider("User's Average Rating",         1.0, 5.0, 3.8, 0.1)
        user_rating_std    = st.slider("User Rating Std Dev",           0.0, 2.0, 0.5, 0.1)
        attr_visit_count   = st.number_input("Attraction's Total Visits",           min_value=1, max_value=10000, value=200)
        attr_avg_rating    = st.slider("Attraction's Average Rating",   1.0, 5.0, 4.0, 0.1)
        attr_rating_std    = st.slider("Attraction Rating Std Dev",     0.0, 2.0, 0.4, 0.1)

    is_peak_month         = 1 if visit_month in [6,7,8,12] else 0
    years_since           = 2024 - visit_year
    visit_mode_popularity = 500
    attr_type_popularity  = 300

    predict_rating = st.button("🔮 Predict Rating", use_container_width=True)

    if predict_rating:
        # Build feature dict
        feature_vals = {
            "VisitYear":                 visit_year,
            "VisitMonth":                visit_month,
            "UserVisitCount":            user_visit_count,
            "UserAvgRating":             user_avg_rating,
            "UserRatingStd":             user_rating_std,
            "AttractionVisitCount":      attr_visit_count,
            "AttractionAvgRating":       attr_avg_rating,
            "AttractionRatingStd":       attr_rating_std,
            "VisitModePopularity":       visit_mode_popularity,
            "AttractionTypePopularity":  attr_type_popularity,
            "YearsSinceVisit":           years_since,
            "IsPeakMonth":               is_peak_month,
            "ContinentName_Encoded":     safe_encode(encoders.get("ContinentName"), continent_name),
            "RegionName_Encoded":        safe_encode(encoders.get("RegionName"), region_name),
            "CountryName_Encoded":       safe_encode(encoders.get("CountryName"), country_name),
            "AttractionType_Encoded":    safe_encode(encoders.get("AttractionType"), attraction_type),
            "Season_Encoded":            safe_encode(encoders.get("Season"), season),
            "UserCityName_Encoded":      safe_encode(encoders.get("UserCityName"), user_city_name),
            "AttractionCityName_Encoded":safe_encode(encoders.get("AttractionCityName"), attr_city_name),
        }

        # Build input row aligned to feature list
        row = pd.DataFrame([{f: feature_vals.get(f, 0) for f in reg_feats}])
        row = row.fillna(0)

        # Predict
        model_name = models["regression"].get("model_name", "")
        needs_scale = "Linear" in model_name or "Ridge" in model_name or "Lasso" in model_name
        if needs_scale and reg_scaler:
            row_in = reg_scaler.transform(row)
        else:
            row_in = row.values

        pred = float(reg_model.predict(row_in)[0])
        pred = max(1.0, min(5.0, pred))

        # Stars
        full_stars = int(pred)
        half_star  = "½" if (pred - full_stars) >= 0.5 else ""
        stars_str  = "★" * full_stars + half_star + "☆" * (5 - full_stars - (1 if half_star else 0))

        st.markdown(f"""
        <div class='pred-box'>
            <div class='pred-label'>Predicted Rating</div>
            <div class='pred-value'>{pred:.2f} / 5</div>
            <div style='font-size:1.8rem; color:#f59e0b; margin:8px 0;'>{stars_str}</div>
            <div class='pred-sub'>Model: {models["regression"].get("model_name","—")} &nbsp;|&nbsp; Season: {season} &nbsp;|&nbsp; {attraction_type}</div>
        </div>
        """, unsafe_allow_html=True)

        # Rating gauge
        fig, ax = plt.subplots(figsize=(7, 2.2), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        ax.barh([0], [5], color="#1c2333", height=0.4, edgecolor="#30363d")
        color = "#3fb950" if pred >= 4 else "#f59e0b" if pred >= 3 else "#f85149"
        ax.barh([0], [pred], color=color, height=0.4)
        ax.set_xlim(0, 5)
        ax.set_yticks([])
        for spine in ax.spines.values(): spine.set_visible(False)
        ax.set_xlabel("Rating Scale (1–5)", color="#8b949e", fontsize=9)
        ax.xaxis.label.set_color("#8b949e")
        ax.tick_params(colors="#8b949e")
        ax.text(pred, 0, f"  {pred:.2f}", va="center", color=color, fontweight="bold", fontsize=12)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()


# ══════════════════════════════════════════════
# PAGE 3 — VISIT MODE PREDICTOR
# ══════════════════════════════════════════════
elif page == "🧳 Visit Mode Predictor":
    st.markdown("<div class='section-title'>🧳 Visit Mode Predictor</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Predict how a user is likely to travel — Business, Family, Couples, Friends, or Solo.</div>", unsafe_allow_html=True)

    if "classification" not in models or "encoders" not in models:
        st.error("❌ Classification model or encoders not found. Run the notebook first.")
        st.stop()

    clf_model  = models["classification"]["model"]
    clf_feats  = models["classification"]["feature_columns"]
    clf_scaler = models["classification"].get("scaler")
    clf_classes= models["classification"].get("classes", [])
    encoders   = models["encoders"]

    def get_classes(col):
        if col in encoders:
            return sorted(encoders[col].classes_.tolist())
        return ["Unknown"]

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 👤 User Demographics")
        continent_name   = st.selectbox("Continent",    get_classes("ContinentName"))
        region_name      = st.selectbox("Region",       get_classes("RegionName"))
        country_name     = st.selectbox("Country",      get_classes("CountryName"))
        user_city_name   = st.selectbox("User City",    get_classes("UserCityName"))
        visit_year       = st.slider("Visit Year",  2000, 2024, 2022)
        visit_month      = st.slider("Visit Month", 1, 12, 7)

    with col2:
        st.markdown("#### 🏛️ Attraction & Stats")
        attraction_type  = st.selectbox("Attraction Type",   get_classes("AttractionType"))
        attr_city_name   = st.selectbox("Attraction City",   get_classes("AttractionCityName"))
        rating           = st.slider("Rating Given", 1.0, 5.0, 4.0, 0.1)
        user_visit_count = st.number_input("User's Total Visits",        1, 500, 10)
        user_avg_rating  = st.slider("User Avg Rating",    1.0, 5.0, 3.8, 0.1)
        user_rating_std  = st.slider("User Rating Std",    0.0, 2.0, 0.5, 0.1)
        attr_visit_count = st.number_input("Attraction Total Visits",    1, 10000, 300)
        attr_avg_rating  = st.slider("Attraction Avg Rating", 1.0, 5.0, 4.0, 0.1)
        attr_rating_std  = st.slider("Attraction Rating Std", 0.0, 2.0, 0.4, 0.1)

    season_map = {12:"Winter",1:"Winter",2:"Winter",3:"Spring",4:"Spring",
                  5:"Spring",6:"Summer",7:"Summer",8:"Summer",9:"Autumn",
                  10:"Autumn",11:"Autumn"}
    season = season_map[visit_month]

    predict_mode = st.button("🔮 Predict Visit Mode", use_container_width=True)

    if predict_mode:
        feature_vals = {
            "VisitYear":                 visit_year,
            "VisitMonth":                visit_month,
            "Rating":                    rating,
            "UserVisitCount":            user_visit_count,
            "UserAvgRating":             user_avg_rating,
            "UserRatingStd":             user_rating_std,
            "AttractionVisitCount":      attr_visit_count,
            "AttractionAvgRating":       attr_avg_rating,
            "AttractionRatingStd":       attr_rating_std,
            "AttractionTypePopularity":  300,
            "YearsSinceVisit":           2024 - visit_year,
            "IsPeakMonth":               1 if visit_month in [6,7,8,12] else 0,
            "ContinentName_Encoded":     safe_encode(encoders.get("ContinentName"), continent_name),
            "RegionName_Encoded":        safe_encode(encoders.get("RegionName"), region_name),
            "CountryName_Encoded":       safe_encode(encoders.get("CountryName"), country_name),
            "AttractionType_Encoded":    safe_encode(encoders.get("AttractionType"), attraction_type),
            "Season_Encoded":            safe_encode(encoders.get("Season"), season),
            "UserCityName_Encoded":      safe_encode(encoders.get("UserCityName"), user_city_name),
            "AttractionCityName_Encoded":safe_encode(encoders.get("AttractionCityName"), attr_city_name),
        }

        row = pd.DataFrame([{f: feature_vals.get(f, 0) for f in clf_feats}]).fillna(0)

        pred_class = clf_model.predict(row)[0]
        proba = None
        if hasattr(clf_model, "predict_proba"):
            proba = clf_model.predict_proba(row)[0]

        mode_icons = {
            "Business": "💼", "Family": "👨‍👩‍👧", "Couples": "💑",
            "Friends": "👥", "Solo": "🧍", "Other": "🌐"
        }
        icon = mode_icons.get(str(pred_class), "🧳")

        st.markdown(f"""
        <div class='pred-box'>
            <div class='pred-label'>Predicted Visit Mode</div>
            <div style='font-size:3rem; margin-bottom:4px;'>{icon}</div>
            <div class='pred-value'>{pred_class}</div>
            <div class='pred-sub'>Season: {season} &nbsp;|&nbsp; {attraction_type} &nbsp;|&nbsp; Rating: {rating}</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability chart
        if proba is not None and len(clf_classes) > 0:
            st.markdown("#### Prediction Confidence")
            proba_df = pd.DataFrame({
                "Visit Mode": clf_classes,
                "Probability": proba
            }).sort_values("Probability", ascending=True)

            fig, ax = plt.subplots(figsize=(8, max(3, len(clf_classes) * 0.55)), facecolor="#0d1117")
            ax.set_facecolor("#0d1117")
            colors = ["#f97316" if cls == pred_class else "#30363d" for cls in proba_df["Visit Mode"]]
            bars = ax.barh(proba_df["Visit Mode"], proba_df["Probability"],
                           color=colors, edgecolor="none", height=0.6)
            for bar, val in zip(bars, proba_df["Probability"]):
                ax.text(bar.get_width() + 0.005, bar.get_y() + bar.get_height() / 2,
                        f"{val*100:.1f}%", va="center", color="#e6edf3", fontsize=9)
            ax.set_xlim(0, 1.08)
            ax.set_xlabel("Probability", color="#8b949e", fontsize=9)
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.tick_params(colors="#8b949e", labelsize=9)
            ax.yaxis.tick_left()
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ══════════════════════════════════════════════
# PAGE 4 — RECOMMENDER
# ══════════════════════════════════════════════
elif page == "🎯 Recommender":
    st.markdown("<div class='section-title'>🎯 Personalized Recommender</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Collaborative filtering — find the best attractions for a user, or discover places similar to one you love.</div>", unsafe_allow_html=True)

    if "recommendation" not in models:
        st.error("❌ Recommendation system not found. Run the notebook first.")
        st.stop()

    rec_pkg    = models["recommendation"]
    uim        = rec_pkg["user_item_matrix"]
    item_sim   = rec_pkg["item_similarity"]
    user_sim   = rec_pkg.get("user_similarity")
    item_info  = rec_pkg.get("item_info", pd.DataFrame())

    tab1, tab2 = st.tabs(["👤 For a User", "🏛️ Similar Attractions"])

    # ── Tab 1: User recommendations
    with tab1:
        st.markdown("Enter a User ID to get personalized attraction recommendations.")

        available_users = uim.index.tolist()

        col_a, col_b = st.columns([2, 1])
        with col_a:
            user_input = st.text_input("User ID", value=str(available_users[0]) if available_users else "")
        with col_b:
            n_recs = st.slider("Number of Recommendations", 5, 20, 10)

        sample_btn = st.button("🎲 Use Random User")
        if sample_btn:
            user_input = str(np.random.choice(available_users))
            st.info(f"Random user selected: **{user_input}**")

        get_recs = st.button("🔮 Get Recommendations", key="get_user_rec")

        if get_recs and user_input:
            uid = user_input.strip()

            if uid not in [str(u) for u in uim.index]:
                st.warning(f"User `{uid}` not found in the training data. Try a different ID.")
            else:
                # User-user collaborative filtering
                if user_sim is not None and uid in user_sim.index:
                    sim_scores = user_sim[uid].drop(uid, errors="ignore").sort_values(ascending=False)
                    top_similar = sim_scores.head(20).index

                    user_rated_mask = uim.loc[uid] > 0
                    user_rated = uim.loc[uid][user_rated_mask].index

                    similar_ratings = uim.loc[top_similar]
                    unrated_cols = [c for c in similar_ratings.columns if c not in user_rated]

                    if unrated_cols:
                        weights = sim_scores[top_similar].values
                        scores  = similar_ratings[unrated_cols].T.dot(weights) / (weights.sum() + 1e-9)
                        top_recs = scores.sort_values(ascending=False).head(n_recs)

                        st.markdown(f"#### Top {len(top_recs)} Recommendations for User `{uid}`")
                        for rank, (attr_id, score) in enumerate(top_recs.items(), 1):
                            name = attr_id
                            atype = "—"
                            if not item_info.empty:
                                row_info = item_info[item_info["AttractionId"].astype(str) == str(attr_id)]
                                if not row_info.empty:
                                    name  = row_info.iloc[0].get("Attraction", attr_id)
                                    atype = row_info.iloc[0].get("AttractionTypeId", "—")

                            medal = ["🥇","🥈","🥉"][rank-1] if rank <= 3 else f"#{rank}"
                            st.markdown(f"""
                            <div class='rec-card'>
                                <div class='rec-rank'>{medal}</div>
                                <div>
                                    <div class='rec-name'>{name}</div>
                                    <div class='rec-type'>Type: {atype}</div>
                                </div>
                                <div class='rec-score'>
                                    <div class='rec-score-val'>{score:.3f}</div>
                                    <div class='rec-score-lbl'>score</div>
                                </div>
                            </div>
                            """, unsafe_allow_html=True)

                        # Show what the user has rated
                        with st.expander(f"📋 User `{uid}`'s rated attractions ({user_rated_mask.sum()} total)"):
                            rated_df = uim.loc[uid][user_rated_mask].reset_index()
                            rated_df.columns = ["AttractionId", "Rating"]
                            rated_df = rated_df.sort_values("Rating", ascending=False).head(20)
                            st.dataframe(rated_df, use_container_width=True)
                    else:
                        st.warning("No unrated attractions to recommend (user has rated everything!).")
                else:
                    st.warning("User similarity data unavailable.")

    # ── Tab 2: Similar attractions
    with tab2:
        st.markdown("Select an attraction to find the most similar ones.")

        all_attractions = item_sim.index.tolist()
        sel_attraction  = st.selectbox("Select Attraction ID", all_attractions[:500])
        n_sim = st.slider("Number of Similar Attractions", 5, 20, 10, key="n_sim")
        find_sim = st.button("🔍 Find Similar Attractions")

        if find_sim:
            sim_row = item_sim[sel_attraction].drop(sel_attraction, errors="ignore").sort_values(ascending=False).head(n_sim)

            st.markdown(f"#### Top {n_sim} Attractions Similar to `{sel_attraction}`")
            for rank, (aid, score) in enumerate(sim_row.items(), 1):
                name  = str(aid)
                atype = "—"
                if not item_info.empty:
                    row_info = item_info[item_info["AttractionId"].astype(str) == str(aid)]
                    if not row_info.empty:
                        name  = row_info.iloc[0].get("Attraction", aid)
                        atype = row_info.iloc[0].get("AttractionTypeId", "—")

                medal = ["🥇","🥈","🥉"][rank-1] if rank <= 3 else f"#{rank}"
                st.markdown(f"""
                <div class='rec-card'>
                    <div class='rec-rank'>{medal}</div>
                    <div>
                        <div class='rec-name'>{name}</div>
                        <div class='rec-type'>Type ID: {atype}</div>
                    </div>
                    <div class='rec-score'>
                        <div class='rec-score-val'>{score:.3f}</div>
                        <div class='rec-score-lbl'>similarity</div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # Bar chart
            fig, ax = plt.subplots(figsize=(8, max(3, n_sim * 0.45)), facecolor="#0d1117")
            ax.set_facecolor("#0d1117")
            sim_df = pd.DataFrame({"Attraction": sim_row.index.astype(str), "Similarity": sim_row.values})
            ax.barh(sim_df["Attraction"], sim_df["Similarity"],
                    color=sns.color_palette("YlOrRd", len(sim_df))[::-1], edgecolor="none")
            ax.set_xlabel("Cosine Similarity", color="#8b949e", fontsize=9)
            for spine in ax.spines.values(): spine.set_visible(False)
            ax.tick_params(colors="#8b949e", labelsize=8)
            plt.tight_layout(pad=0.5)
            st.pyplot(fig, use_container_width=True)
            plt.close()


# ══════════════════════════════════════════════
# PAGE 5 — EDA EXPLORER
# ══════════════════════════════════════════════
elif page == "📊 EDA Explorer":
    st.markdown("<div class='section-title'>📊 EDA Explorer</div>", unsafe_allow_html=True)
    st.markdown("<div class='section-sub'>Upload the merged dataset CSV to explore interactive visualizations.</div>", unsafe_allow_html=True)

    uploaded = st.file_uploader("Upload `merged_tourism_data.csv` (or any tourism CSV)", type=["csv", "xlsx"])

    @st.cache_data(show_spinner="Loading data…")
    def load_csv(file):
        if file.name.endswith(".xlsx"):
            return pd.read_excel(file)
        return pd.read_csv(file)

    if uploaded:
        df = load_csv(uploaded)
        st.success(f"✅ Loaded `{uploaded.name}` — {len(df):,} rows × {len(df.columns)} columns")

        st.markdown("#### Data Preview")
        st.dataframe(df.head(10), use_container_width=True)

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

        # Choose columns to explore
        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(include="object").columns.tolist()

        c1, c2 = st.columns(2)

        with c1:
            st.markdown("#### Numerical Distribution")
            if num_cols:
                sel_num = st.selectbox("Select numerical column", num_cols,
                                       index=num_cols.index("Rating") if "Rating" in num_cols else 0)
                fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0d1117")
                ax.set_facecolor("#0d1117")
                ax.hist(df[sel_num].dropna(), bins=30, color="#f97316", edgecolor="#0d1117", alpha=0.85)
                ax.axvline(df[sel_num].mean(), color="#f59e0b", lw=2, label=f"Mean: {df[sel_num].mean():.2f}")
                ax.axvline(df[sel_num].median(), color="#2dd4bf", lw=2, linestyle="--", label=f"Median: {df[sel_num].median():.2f}")
                ax.set_xlabel(sel_num, color="#8b949e")
                ax.set_ylabel("Count", color="#8b949e")
                ax.tick_params(colors="#8b949e")
                for spine in ax.spines.values(): spine.set_visible(False)
                ax.legend(labelcolor="#e6edf3", facecolor="#161b22", edgecolor="#30363d", fontsize=8)
                plt.tight_layout(pad=0.5)
                st.pyplot(fig, use_container_width=True)
                plt.close()

                # Stats
                stat = df[sel_num].describe()
                st.dataframe(stat.to_frame().T.style.format("{:.3f}"), use_container_width=True)

        with c2:
            st.markdown("#### Categorical Frequency")
            if cat_cols:
                sel_cat = st.selectbox("Select categorical column", cat_cols,
                                       index=cat_cols.index("VisitMode") if "VisitMode" in cat_cols else 0)
                top_n = st.slider("Top N values", 5, 30, 10, key="top_n_cat")
                vc = df[sel_cat].value_counts().head(top_n)
                fig, ax = plt.subplots(figsize=(7, 4), facecolor="#0d1117")
                ax.set_facecolor("#0d1117")
                colors = sns.color_palette("muted", len(vc))
                ax.barh(vc.index[::-1], vc.values[::-1], color=colors, edgecolor="none")
                ax.set_xlabel("Count", color="#8b949e")
                ax.tick_params(colors="#8b949e", labelsize=8)
                for spine in ax.spines.values(): spine.set_visible(False)
                plt.tight_layout(pad=0.5)
                st.pyplot(fig, use_container_width=True)
                plt.close()

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

        # ── Correlation heatmap
        if len(num_cols) >= 2:
            st.markdown("#### Correlation Heatmap")
            sel_corr_cols = st.multiselect("Select columns for correlation",
                                            num_cols,
                                            default=num_cols[:min(8, len(num_cols))])
            if len(sel_corr_cols) >= 2:
                corr = df[sel_corr_cols].corr()
                fig, ax = plt.subplots(figsize=(max(7, len(sel_corr_cols)), max(5, len(sel_corr_cols) * 0.7)), facecolor="#0d1117")
                ax.set_facecolor("#0d1117")
                mask = np.triu(np.ones_like(corr, dtype=bool))
                sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                            square=True, linewidths=0.3, ax=ax,
                            cbar_kws={"shrink": 0.8})
                ax.tick_params(colors="#8b949e", labelsize=8)
                plt.tight_layout(pad=0.3)
                st.pyplot(fig, use_container_width=True)
                plt.close()

        st.markdown("<div class='fancy-divider'></div>", unsafe_allow_html=True)

        # ── Groupby analysis
        st.markdown("#### Group Analysis")
        col_g1, col_g2, col_g3 = st.columns(3)
        with col_g1:
            group_col  = st.selectbox("Group by", cat_cols)
        with col_g2:
            agg_col    = st.selectbox("Aggregate", num_cols,
                                      index=num_cols.index("Rating") if "Rating" in num_cols else 0,
                                      key="agg_col")
        with col_g3:
            agg_func   = st.selectbox("Function", ["mean", "median", "sum", "count"])

        grp = df.groupby(group_col)[agg_col].agg(agg_func).sort_values(ascending=False).head(15)
        fig, ax = plt.subplots(figsize=(10, 4), facecolor="#0d1117")
        ax.set_facecolor("#0d1117")
        ax.bar(grp.index, grp.values, color="#f97316", edgecolor="none", alpha=0.85)
        ax.set_xlabel(group_col, color="#8b949e")
        ax.set_ylabel(f"{agg_func}({agg_col})", color="#8b949e")
        ax.tick_params(colors="#8b949e", labelsize=8, axis="x", rotation=30)
        ax.tick_params(colors="#8b949e", labelsize=8, axis="y")
        for spine in ax.spines.values(): spine.set_visible(False)
        plt.tight_layout(pad=0.5)
        st.pyplot(fig, use_container_width=True)
        plt.close()

    else:
        st.markdown("""
        <div style='background:#161b22; border:1px dashed #30363d; border-radius:14px;
                    padding:48px; text-align:center; color:#8b949e;'>
            <div style='font-size:2.5rem; margin-bottom:12px;'>📂</div>
            <div style='font-size:1.1rem; font-weight:600; color:#e6edf3; margin-bottom:8px;'>Upload a dataset to explore</div>
            <div style='font-size:0.85rem;'>Drag & drop your <code>merged_tourism_data.csv</code> or any tourism CSV/XLSX above.<br>
            You can export the merged dataframe from the notebook by uncommenting the save cell.</div>
        </div>
        """, unsafe_allow_html=True)

# ── Footer
st.markdown("""
<div style='text-align:center; color:#30363d; font-size:0.75rem; margin-top:48px; padding-top:20px; border-top:1px solid #21262d;'>
    Tourism Analytics AI &nbsp;•&nbsp; Powered by Streamlit &nbsp;•&nbsp; Models: Random Forest / XGBoost / LightGBM
</div>
""", unsafe_allow_html=True)
