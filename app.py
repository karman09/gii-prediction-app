import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import re
import math
import shap
import os
from huggingface_hub import hf_hub_download

# ============================================================
# 1. PAGE CONFIG & STYLING
# ============================================================
st.set_page_config(page_title="GII 2025 Strategy Dashboard", layout="wide")

# Custom CSS for look and feel
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stButton>button { background-color: #0f766e; color: white; border-radius: 8px; }
    h3 { color: #0f766e; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 2. DATA LOADING (CACHED)
# ============================================================
@st.cache_resource
def load_all_data():
    HF_TOKEN = os.environ.get("HF_TOKEN")
    REPO_ID = "karman09/D-LOGII-Data"
    
    def fetch_file(file_name):
        return hf_hub_download(repo_id=REPO_ID, filename=file_name, repo_type="dataset", token=HF_TOKEN)

    try:
        raw_data_path = fetch_file("FINAL_DATA.xlsx")
        proc_data_path = fetch_file("FINAL_PREPROCESSED_DATA.xlsx")
        scaler = joblib.load(fetch_file("SCALER.pkl"))
        scaler_cols = joblib.load(fetch_file("SCALER_COLUMNS.pkl"))
        cost_cols = joblib.load(fetch_file("COST_COLS.pkl"))
        model = joblib.load(fetch_file("BEST_MODEL.pkl"))
        model_features = joblib.load(fetch_file("BEST_MODEL_FEATURES.pkl"))
        
        df_raw = pd.read_excel(raw_data_path)
        df_proc = pd.read_excel(proc_data_path)
        return df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features
    except Exception as e:
        st.error(f"Veri yükleme hatası: {e}")
        return None

data_bundle = load_all_data()
if data_bundle:
    df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features = data_bundle
else:
    st.stop()

# ============================================================
# 3. UTILITY & CORE LOGIC (UNCHANGED)
# ============================================================
def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_]", "", name)

sanitized_to_original = {sanitize_name(c): c for c in scaler_cols}

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

# Core Calculation Function
def calculate_score(country_name, raw_inputs_list):
    try:
        row_proc = latest_data_proc.loc[country_name]
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)
        
        for i, feat_ui in enumerate(ui_input_names):
            user_val = float(raw_inputs_list[i])
            # Check if user modified the default value
            base_raw_val = latest_data_raw.loc[country_name].get(feat_ui, 0.0)
            
            if math.isclose(user_val, base_raw_val, rel_tol=1e-5):
                final_scaled_feat = row_proc[feat_ui]
            else:
                idx = scaler_cols.index(feat_ui)
                new_scaled = (user_val - scaler.mean_[idx]) / scaler.scale_[idx]
                if feat_ui.lower().strip() in cost_cols:
                    new_scaled = -new_scaled
                final_scaled_feat = new_scaled
            
            model_input.at[0, feature_map[feat_ui]] = final_scaled_feat
            
        pred = model.predict(model_input)[0]
        return max(0, min(100, pred))
    except: return 0.0

# ============================================================
# 4. HEADER & LOGO
# ============================================================
col_logo1, col_logo2, col_logo3 = st.columns([1, 2, 1])
with col_logo2:
    try:
        st.image("logo.png", use_container_width=True)
    except:
        pass
    st.markdown("<h1 style='text-align: center; color: #0f766e;'>Dynamic Lasso-Optimized GII</h1>", unsafe_allow_html=True)

# ============================================================
# 5. UI TABS (TR / EN)
# ============================================================
tab_tr, tab_en = st.tabs(["🇹🇷 Türkçe", "🇬🇧 English"])

# Helper function to render a language UI
def render_ui(lang):
    is_tr = lang == "tr"
    
    # Tool Selection inside Language Tab
    tool = st.selectbox(
        "Araç Seçin / Select Tool" if is_tr else "Select Tool",
        ["Senaryo Simülatörü", "Duyarlılık Analizi", "Karşılaştırmalı Analiz", "Hedef ve SHAP Analizi", "Geçmiş Trendler"] if is_tr else
        ["Scenario Simulator", "Sensitivity Analysis", "Comparative Analysis", "Target & SHAP Analysis", "Historical Trends"]
    )
    st.divider()

    if tool in ["Senaryo Simülatörü", "Scenario Simulator"]:
        st.subheader("What-If Analysis")
        selected_country = st.selectbox("Ülke Seç" if is_tr else "Select Country", country_list, key=f"d1_{lang}")
        
        # Get defaults
        defaults = []
        row = latest_data_raw.loc[selected_country]
        for col in ui_input_names:
            val = row.get(col, 0.0)
            defaults.append(float(val) if not pd.isna(val) else 0.0)

        # Inputs Grid
        with st.expander("Değişkenleri Düzenle" if is_tr else "Edit Variables"):
            cols = st.columns(2)
            user_inputs = []
            for i, name in enumerate(ui_input_names):
                with cols[i % 2]:
                    val = st.number_input(name, value=defaults[i], key=f"in_{lang}_{name}")
                    user_inputs.append(val)
        
        if st.button("Hesapla" if is_tr else "Calculate", key=f"btn1_{lang}"):
            res = calculate_score(selected_country, user_inputs)
            st.success(f"Tahmini Skor: {res:.2f}")

    elif tool in ["Duyarlılık Analizi", "Sensitivity Analysis"]:
        selected_country = st.selectbox("Ülke Seç" if is_tr else "Select Country", country_list, key=f"d2_{lang}")
        if st.button("Analizi Başlat" if is_tr else "Start Analysis", key=f"btn2_{lang}"):
            # Reusing the scenario_advisor logic (simplified version for demo)
            current_vals = [latest_data_raw.loc[selected_country].get(n, 0.0) for n in ui_input_names]
            report = scenario_advisor(lang, selected_country, current_vals) # Dışarıdaki fonksiyonu çağırır
            st.text_area("Rapor", report, height=400)

    elif tool in ["Karşılaştırmalı Analiz", "Comparative Analysis"]:
        c1 = st.selectbox("Ülke A", country_list, key=f"c1_{lang}")
        c2 = st.selectbox("Ülke B", country_list, key=f"c2_{lang}")
        if st.button("Grafiği Oluştur" if is_tr else "Generate Chart"):
            fig, msg = scenario_benchmark(lang, c1, c2)
            if fig: st.pyplot(fig)
            st.info(msg)

    elif tool in ["Hedef ve SHAP Analizi", "Target & SHAP Analysis"]:
        selected_country = st.selectbox("Ülke Seç", country_list, key=f"d4_{lang}")
        target = st.number_input("Hedef Skor", value=0.0)
        if st.button("SHAP Analizi Çalıştır"):
            fig, txt1, txt2 = get_shap_analysis(lang, selected_country, target)
            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.write(txt1)
                st.write(txt2)
            with col_b:
                if fig: st.pyplot(fig)

    elif tool in ["Geçmiş Trendler", "Historical Trends"]:
        selected_country = st.selectbox("Ülke Seç", country_list, key=f"d5_{lang}")
        trend_candidates = [c for c in df_raw.columns if c not in [country_col, year_col]]
        feat = st.selectbox("Değişken", trend_candidates)
        if st.button("Trendi Göster"):
            fig, err = plot_historical_trend(lang, selected_country, feat)
            if fig: st.pyplot(fig)
            else: st.error(err)

# Logic for scenario_advisor, benchmark, etc. (Directly copy your functions here)
# scenario_advisor(...) 
# scenario_benchmark(...)
# get_shap_analysis(...)
# plot_historical_trend(...)

with tab_tr: render_ui("tr")
with tab_en: render_ui("en")

st.markdown("---")
st.caption("2025 Strategic Forecast & Decision Support System")
