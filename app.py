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
# 1. SAYFA AYARLARI VE TASARIM
# ============================================================
st.set_page_config(page_title="D-LOGII Dashboard", layout="wide")

# Kurumsal Renkler ve Stil
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stButton>button { 
        background-color: #0f766e; 
        color: white; 
        border-radius: 8px; 
        width: 100%;
        font-weight: bold;
    }
    .stTextInput>div>div>input, .stNumberInput>div>div>input {
        background-color: #ffffff;
    }
    h1, h2, h3 { color: #0f766e; text-align: center; }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 2. VERİ YÜKLEME (GITHUB DOSYALARINDAN)
# ============================================================
@st.cache_resource
def load_system_files():
    try:
        # Dosyaların GitHub ana dizininde olduğu varsayılmaktadır
        df_raw = pd.read_excel("FINAL_DATA.xlsx")
        df_proc = pd.read_excel("FINAL_PREPROCESSED_DATA.xlsx")
        scaler = joblib.load("SCALER.pkl")
        scaler_cols = joblib.load("SCALER_COLUMNS.pkl")
        cost_cols = joblib.load("COST_COLS.pkl")
        model = joblib.load("BEST_MODEL.pkl")
        model_features = joblib.load("BEST_MODEL_FEATURES.pkl")
        return df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features
    except Exception as e:
        st.error(f"⚠️ Kritik Hata: Dosyalar yüklenemedi. Hata: {e}")
        return None

data = load_system_files()
if data is None:
    st.stop()

df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features = data

# ============================================================
# 3. YARDIMCI FONKSİYONLAR VE AYARLAR
# ============================================================
def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_]", "", name)

sanitized_to_original = {sanitize_name(c): c for c in scaler_cols}
country_col = [c for c in df_raw.columns if "country" in c.lower() or "economy" in c.lower()][0]
year_col = "year"
TARGET_YEAR, INPUT_YEAR = 2025, 2023

# Veri filtreleme
latest_data_raw = df_raw[df_raw[year_col] == INPUT_YEAR].set_index(country_col)
latest_data_proc = df_proc[df_proc[year_col] == INPUT_YEAR].set_index(country_col)
country_list = sorted(latest_data_raw.index.tolist())

# Özellik eşleştirme
feature_map = {}
ui_input_names = []
for feat in model_features:
    base_clean = re.sub(r"_lag\d+$", "", feat)
    if base_clean in sanitized_to_original:
        original_name = sanitized_to_original[base_clean]
        feature_map[original_name] = feat
        ui_input_names.append(original_name)

reverse_feature_map = {v: k for k, v in feature_map.items()}

# ============================================================
# 4. HESAPLAMA MOTORU (CORE LOGIC)
# ============================================================
def calculate_score(country_name, raw_inputs_list):
    try:
        row_proc = latest_data_proc.loc[country_name]
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)
        
        for i, feat_ui in enumerate(ui_input_names):
            user_val = float(raw_inputs_list[i])
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

# --- SHAP Analizi Fonksiyonu ---
def get_shap_analysis(lang, country, target_score):
    row_proc = latest_data_proc.loc[country]
    model_input = pd.DataFrame(0.0, index=[0], columns=model_features)
    for feat_ui in ui_input_names:
        model_input.at[0, feature_map[feat_ui]] = row_proc[feat_ui]
    
    pred = model.predict(model_input)[0]
    pred_clamped = max(0, min(100, pred))
    
    explainer = shap.Explainer(model)
    shap_values = explainer(model_input)
    
    fig = plt.figure(figsize=(10, 6))
    shap.plots.waterfall(shap_values[0], max_display=10, show=False)
    plt.tight_layout()
    
    msg = f"Tahmini Skor: {pred_clamped:.2f}"
    if target_score > 0:
        gap = target_score - pred_clamped
        msg += f" | Hedef Farkı: {gap:.2f}"
        
    return fig, msg

# ============================================================
# 5. ARAYÜZ (STREAMLIT UI)
# ============================================================
# Logo ve Başlık
col_l, col_m, col_r = st.columns([1, 2, 1])
with col_m:
    if os.path.exists("logo.png"):
        st.image("logo.png", use_container_width=True)
    st.markdown(f"### Dynamic Lasso-Optimized Global Innovation Index")

# Dil Seçimi
tab_tr, tab_en = st.tabs(["🇹🇷 Türkçe", "🇬🇧 English"])

def render_content(lang):
    is_tr = lang == "tr"
    
    # Sidebar benzeri araç seçimi
    tool = st.radio(
        "Analiz Türü Seçin" if is_tr else "Select Analysis Type",
        ["Senaryo Simülatörü", "Karşılaştırmalı Analiz", "Hedef & SHAP Analizi"] if is_tr else 
        ["Scenario Simulator", "Comparative Analysis", "Target & SHAP Analysis"],
        horizontal=True
    )

    if "Senaryo Simülatörü" in tool or "Scenario Simulator" in tool:
        st.subheader("What-If Analysis")
        selected_country = st.selectbox("Ülke Seç" if is_tr else "Select Country", country_list, key=f"sel_{lang}")
        
        # Varsayılan değerleri çek
        defaults = [float(latest_data_raw.loc[selected_country].get(name, 0.0)) for name in ui_input_names]
        
        with st.expander("Değişkenleri Düzenle" if is_tr else "Edit Variables"):
            cols = st.columns(2)
            user_inputs = []
            for i, name in enumerate(ui_input_names):
                with cols[i % 2]:
                    val = st.number_input(name, value=defaults[i], key=f"inp_{lang}_{name}")
                    user_inputs.append(val)
        
        if st.button("Tahmin Et" if is_tr else "Predict", key=f"btn_{lang}"):
            score = calculate_score(selected_country, user_inputs)
            st.metric(label="2025 GII Tahmini", value=f"{score:.2f}")

    elif "Hedef" in tool or "Target" in tool:
        col1, col2 = st.columns([1, 2])
        with col1:
            country = st.selectbox("Ülke Seç", country_list, key=f"shap_c_{lang}")
            target = st.number_input("Hedef Skor", 0.0, 100.0, 45.0)
            run = st.button("Analiz Et" if is_tr else "Analyze")
        
        if run:
            fig, msg = get_shap_analysis(lang, country, target)
            st.info(msg)
            st.pyplot(fig)

    elif "Karşılaştırmalı" in tool or "Comparative" in tool:
        c1 = st.selectbox("Ülke 1", country_list, key=f"c1_{lang}")
        c2 = st.selectbox("Ülke 2", country_list, key=f"c2_{lang}")
        if st.button("Karşılaştır" if is_tr else "Compare"):
            # Özet karşılaştırma mantığı
            s1 = calculate_score(c1, [latest_data_raw.loc[c1].get(n, 0.0) for n in ui_input_names])
            s2 = calculate_score(c2, [latest_data_raw.loc[c2].get(n, 0.0) for n in ui_input_names])
            st.write(f"**{c1}:** {s1:.2f}  |  **{c2}:** {s2:.2f}")
            st.bar_chart(pd.DataFrame({"Skor": [s1, s2]}, index=[c1, c2]))

with tab_tr: render_content("tr")
with tab_en: render_content("en")

st.markdown("---")
st.caption("2025 Strategic Forecast & Decision Support System | D-LOGII")
