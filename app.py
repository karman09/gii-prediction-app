import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import re
import math
import shap
import os

# Sayfa Yapılandırması
st.set_page_config(page_title="D-LOGII Dashboard", layout="wide")

# ============================================================
# 1. DOSYA YOLLARI (GÖRECELİ YOLLAR)
# ============================================================
# GitHub'daki dosya isimleriyle birebir aynı olmalı (Büyük/Küçük harf duyarlı)
FILE_PATHS = {
    "raw_data": "FINAL_DATA.xlsx",
    "proc_data": "FINAL_PREPROCESSED_DATA.xlsx",
    "scaler": "SCALER.pkl",
    "scaler_cols": "SCALER_COLUMNS.pkl",
    "cost_cols": "COST_COLS.pkl",
    "model": "BEST_MODEL.pkl",
    "features": "BEST_MODEL_FEATURES.pkl"
}

@st.cache_resource
def load_system():
    # Dosya varlık kontrolü
    for key, path in FILE_PATHS.items():
        if not os.path.exists(path):
            st.error(f"❌ Dosya Bulunamadı: {path}")
            st.info("Lütfen bu dosyanın GitHub deponuzun ana dizininde olduğundan emin olun.")
            st.stop()
    
    try:
        df_raw = pd.read_excel(FILE_PATHS["raw_data"])
        df_proc = pd.read_excel(FILE_PATHS["proc_data"])
        scaler = joblib.load(FILE_PATHS["scaler"])
        scaler_cols = joblib.load(FILE_PATHS["scaler_cols"])
        cost_cols = joblib.load(FILE_PATHS["cost_cols"])
        model = joblib.load(FILE_PATHS["model"])
        model_features = joblib.load(FILE_PATHS["features"])
        return df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features
    except Exception as e:
        st.error(f"Sistem yüklenirken hata oluştu: {e}")
        st.stop()

# Verileri Yükle
assets = load_system()
df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features = assets

# ============================================================
# 2. YARDIMCI FONKSİYONLAR VE MANTIK
# ============================================================
def sanitize_name(name): return re.sub(r"[^A-Za-z0-9_]", "", name)

sanitized_to_original = {sanitize_name(col): col for col in scaler_cols}
country_col = [c for c in df_raw.columns if "country" in c.lower() or "economy" in c.lower()][0]
year_col = "year"
TARGET_YEAR, INPUT_YEAR = 2025, 2023

latest_data_raw = df_raw[df_raw[year_col] == INPUT_YEAR].set_index(country_col)
latest_data_proc = df_proc[df_proc[year_col] == INPUT_YEAR].set_index(country_col)
country_list = sorted(latest_data_raw.index.tolist())

feature_map = {}
ui_input_names = []
for feat in model_features:
    base_clean = re.sub(r"_lag\d+$", "", feat)
    if base_clean in sanitized_to_original:
        original_name = sanitized_to_original[base_clean]
        feature_map[original_name] = feat
        ui_input_names.append(original_name)

reverse_feature_map = {v: k for k, v in feature_map.items()}

def calculate_score(country_name, input_vals_dict):
    row_raw = latest_data_raw.loc[country_name]
    row_proc = latest_data_proc.loc[country_name]
    model_input = pd.DataFrame(0.0, index=[0], columns=model_features)

    for feat_ui in ui_input_names:
        user_val = float(input_vals_dict[feat_ui])
        base_raw_val = float(row_raw.get(feat_ui, 0.0))
        
        if math.isclose(user_val, base_raw_val, rel_tol=1e-5):
            final_val = row_proc[feat_ui]
        else:
            idx = scaler_cols.index(feat_ui)
            new_scaled = (user_val - scaler.mean_[idx]) / scaler.scale_[idx]
            if feat_ui.lower().strip() in cost_cols: new_scaled = -new_scaled
            final_val = new_scaled
        
        model_input.at[0, feature_map[feat_ui]] = final_val
    
    pred = model.predict(model_input)[0]
    return max(0, min(100, pred))

# ============================================================
# 3. ARAYÜZ (STREAMLIT UI)
# ============================================================
st.title("🚀 D-LOGII Dashboard")
st.markdown("Dynamic Lasso-Optimized Global Innovation Index")

tab1, tab2, tab3, tab4 = st.tabs(["Simülatör", "Hassasiyet", "Kıyasla", "SHAP"])

with tab1:
    st.header("Senaryo Simülatörü")
    selected_country = st.selectbox("Ülke Seç", country_list)
    
    with st.expander("Verileri Düzenle (Ham Veri)"):
        user_inputs = {}
        cols = st.columns(2)
        current_vals = latest_data_raw.loc[selected_country]
        for i, name in enumerate(ui_input_names):
            user_inputs[name] = cols[i%2].number_input(name, value=float(current_vals.get(name, 0.0)))

    if st.button("Hesapla"):
        res = calculate_score(selected_country, user_inputs)
        st.metric(f"{TARGET_YEAR} GII Tahmini", f"{res:.2f}")

with tab4:
    st.header("SHAP Açıklanabilirlik")
    if st.button("Analiz Et"):
        row_proc = latest_data_proc.loc[selected_country]
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)
        for f in ui_input_names: model_input.at[0, feature_map[f]] = row_proc[f]
        
        explainer = shap.Explainer(model)
        shap_values = explainer(model_input)
        
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        st.pyplot(fig)

st.divider()
st.caption("2026 Karar Destek Sistemi")
