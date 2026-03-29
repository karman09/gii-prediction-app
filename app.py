# ============================================================
# D-LOGII STREAMLIT DASHBOARD (FULL VERSION)
# Mantık birebir korunmuştur
# ============================================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import re
import math
import shap
import os

# ============================================================
# PAGE CONFIG
# ============================================================
st.set_page_config(page_title="D-LOGII Dashboard", layout="wide")

TARGET_YEAR = 2025
LAG_PERIOD = 2
INPUT_YEAR = TARGET_YEAR - LAG_PERIOD

# ============================================================
# PATH SETUP (CRITICAL FIX)
# ============================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def path(*args):
    return os.path.join(BASE_DIR, *args)

# ============================================================
# LOAD SYSTEM
# ============================================================
@st.cache_resource
def load_system():
    df_raw = pd.read_excel(path("data", "FINAL_DATA.xlsx"))
    df_proc = pd.read_excel(path("data", "FINAL_PREPROCESSED_DATA.xlsx"))

    scaler = joblib.load(path("models", "SCALER.pkl"))
    scaler_cols = joblib.load(path("models", "SCALER_COLUMNS.pkl"))
    cost_cols = joblib.load(path("models", "COST_COLS.pkl"))

    model = joblib.load(path("models", "BEST_MODEL.pkl"))
    model_features = joblib.load(path("models", "BEST_MODEL_FEATURES.pkl"))

    return df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features

df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features = load_system()

# ============================================================
# PREP
# ============================================================
def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_]", "", name)

sanitized_to_original = {sanitize_name(c): c for c in scaler_cols}

country_col = [c for c in df_raw.columns if "country" in c.lower() or "economy" in c.lower()][0]

latest_raw = df_raw[df_raw["year"] == INPUT_YEAR].set_index(country_col)
latest_proc = df_proc[df_proc["year"] == INPUT_YEAR].set_index(country_col)

countries = sorted(latest_raw.index.tolist())

feature_map = {}
ui_inputs = []

for feat in model_features:
    base = re.sub(r"_lag\d+$", "", feat)
    if base in sanitized_to_original:
        orig = sanitized_to_original[base]
        feature_map[orig] = feat
        ui_inputs.append(orig)

reverse_feature_map = {v: k for k, v in feature_map.items()}

# ============================================================
# CORE FUNCTION (UNCHANGED)
# ============================================================
def calculate_score(country, inputs):
    row_raw = latest_raw.loc[country]
    row_proc = latest_proc.loc[country]

    model_input = pd.DataFrame(0.0, index=[0], columns=model_features)

    for i, feat in enumerate(ui_inputs):
        user_val = float(inputs[i])

        base_val = row_raw.get(feat, 0.0)
        base_val = 0.0 if pd.isna(base_val) else float(base_val)

        if math.isclose(user_val, base_val, rel_tol=1e-5):
            final_val = row_proc[feat]
        else:
            idx = scaler_cols.index(feat)
            mean = scaler.mean_[idx]
            scale = scaler.scale_[idx]

            new_scaled = (user_val - mean) / scale

            if feat.lower().strip() in cost_cols:
                new_scaled = -new_scaled

            final_val = new_scaled

        model_input.at[0, feature_map[feat]] = final_val

    pred = model.predict(model_input)[0]
    return max(0, min(100, pred))

# ============================================================
# SHAP FUNCTION
# ============================================================
def get_shap(country):
    row_proc = latest_proc.loc[country]

    model_input = pd.DataFrame(0.0, index=[0], columns=model_features)

    for feat in ui_inputs:
        model_input.at[0, feature_map[feat]] = row_proc[feat]

    explainer = shap.Explainer(model)
    shap_values = explainer(model_input)

    fig = plt.figure(figsize=(8, 5))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.tight_layout()

    return fig

# ============================================================
# HEADER
# ============================================================
st.title("D-LOGII")
st.caption("Dynamic Lasso-Optimized Global Innovation Index")

# ============================================================
# TABS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs([
    "Simülatör",
    "Duyarlılık",
    "Karşılaştırma",
    "SHAP Analizi"
])

# ============================================================
# TAB 1 - SIMULATOR
# ============================================================
with tab1:
    st.subheader("Senaryo Simülatörü")

    country = st.selectbox("Ülke", countries)

    row = latest_raw.loc[country]

    inputs = []
    cols = st.columns(2)

    for i, feat in enumerate(ui_inputs):
        val = row.get(feat, 0.0)
        val = 0.0 if pd.isna(val) else float(val)

        with cols[i % 2]:
            inputs.append(st.number_input(feat, value=val))

    if st.button("Tahmin Hesapla"):
        score = calculate_score(country, inputs)
        st.success(f"{TARGET_YEAR} GII Tahmini: {score:.2f}")

# ============================================================
# TAB 2 - SENSITIVITY
# ============================================================
with tab2:
    st.subheader("Duyarlılık Analizi")

    country = st.selectbox("Ülke", countries, key="sens")

    inputs = latest_raw.loc[country].fillna(0).tolist()[:len(ui_inputs)]
    base_score = calculate_score(country, inputs)

    results = []

    for i, feat in enumerate(ui_inputs):
        val = inputs[i]

        if feat.lower().strip() in cost_cols:
            new_val = val * 0.9
        else:
            new_val = val * 1.1

        temp = inputs.copy()
        temp[i] = new_val

        new_score = calculate_score(country, temp)
        gain = new_score - base_score

        if gain > 0.01:
            results.append((feat, gain))

    results.sort(key=lambda x: x[1], reverse=True)

    for r in results[:5]:
        st.write(f"{r[0]} → +{r[1]:.3f}")

# ============================================================
# TAB 3 - COMPARISON
# ============================================================
with tab3:
    st.subheader("Karşılaştırma")

    c1 = st.selectbox("Ülke A", countries)
    c2 = st.selectbox("Ülke B", countries, index=1)

    if st.button("Karşılaştır"):
        row1 = latest_proc.loc[c1]
        row2 = latest_proc.loc[c2]

        vals1, vals2 = [], []

        for f in ui_inputs:
            vals1.append(row1[f])
            vals2.append(row2[f])

        y = np.arange(len(ui_inputs))

        fig, ax = plt.subplots(figsize=(10, 6))
        ax.barh(y, vals1, alpha=0.7, label=c1)
        ax.barh(y, vals2, alpha=0.7, label=c2)

        ax.set_yticks(y)
        ax.set_yticklabels(ui_inputs)
        ax.legend()

        st.pyplot(fig)

# ============================================================
# TAB 4 - SHAP
# ============================================================
with tab4:
    st.subheader("SHAP Analizi")

    country = st.selectbox("Ülke", countries, key="shap")

    if st.button("SHAP Oluştur"):
        fig = get_shap(country)
        st.pyplot(fig)
