# ==============================================================================
# GII 2025 STRATEGY DASHBOARD (FINAL STABLE & BILINGUAL VERSION)
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import re
import math
import shap

# --- SAYFA YAPILANDIRMASI ---
st.set_page_config(page_title="D-LOGII Dashboard", page_icon="📊", layout="wide")

# --- TEMA VE STİL AYARLARI ---
st.markdown("""
    <style>
    .stApp { background-color: #f0f4f8; }
    [data-testid="stSidebar"], .stTabs, div[data-testid="stExpander"], .stAlert {
        background-color: #ffffff !important;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #64748b;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6fa8dc !important;
        color: white !important;
    }
    .stButton>button {
        background-color: #6fa8dc !important;
        color: white !important;
        border-radius: 8px !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================
# 1. DOSYA YOLLARI VE YÜKLEME 
# ============================================================
@st.cache_resource(show_spinner="Dosyalar yükleniyor... / Loading files...")
def load_system_files():
    try:
        df_raw  = pd.read_excel("FINAL_DATA.xlsx")
        df_proc = pd.read_excel("FINAL_PREPROCESSED_DATA.xlsx")
        scaler = joblib.load("SCALER.pkl")
        scaler_cols = joblib.load("SCALER_COLUMNS.pkl")
        cost_cols = joblib.load("COST_COLS.pkl")
        model = joblib.load("BEST_MODEL.pkl")
        model_features = joblib.load("BEST_MODEL_FEATURES.pkl")
        return df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features
    except Exception as e:
        st.error(f"Dosya Yükleme Hatası: {e}")
        st.stop()

df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features = load_system_files()

# ============================================================
# 2. İSİM EŞLEŞTİRMELERİ & VERİ HAZIRLIĞI
# ============================================================
def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_]", "", name)

sanitized_to_original = {sanitize_name(orig): orig for orig in scaler_cols}

country_col = [c for c in df_raw.columns if "country" in c.lower() or "economy" in c.lower()][0]
year_col = "year"

TARGET_YEAR = 2025
INPUT_YEAR  = 2023

latest_data_raw  = df_raw[df_raw[year_col] == INPUT_YEAR].set_index(country_col)
latest_data_proc = df_proc[df_proc[year_col] == INPUT_YEAR].set_index(country_col)
country_list = sorted(latest_data_raw.index.tolist())

feature_map = {}
ui_input_names = []
for feat in model_features:
    base_clean = re.sub(r"_lag\d+$", "", feat)
    if base_clean in sanitized_to_original:
        orig_name = sanitized_to_original[base_clean]
        feature_map[orig_name] = feat
        ui_input_names.append(orig_name)

trend_candidates = [c for c in df_raw.columns if c not in [country_col, year_col]]
gii_col_exact = [c for c in df_raw.columns if "global innovation index" in c.lower()]

# ============================================================
# 3. CORE HESAPLAMA MANTIĞI (DÜZELTİLMİŞ)
# ============================================================
def calculate_score(country_name, raw_inputs_list):
    try:
        if country_name not in latest_data_proc.index: return 0.0
        
        row_proc = latest_data_proc.loc[country_name]
        row_raw = latest_data_raw.loc[country_name]
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)
        
        for i, feat_ui in enumerate(ui_input_names):
            user_val = float(raw_inputs_list[i])
            orig_raw_val = float(row_raw.get(feat_ui, 0.0))
            
            # Değer değişmediyse pre-processed veriyi kullan (Tutarlılık için)
            if math.isclose(user_val, orig_raw_val, rel_tol=1e-3):
                final_scaled_feat = row_proc[feat_ui]
            else:
                # Değer değiştiyse manuel scaler kullan
                idx = scaler_cols.index(feat_ui)
                new_scaled = (user_val - scaler.mean_[idx]) / scaler.scale_[idx]
                # Maliyet kontrolü
                if any(c.strip().lower() == feat_ui.strip().lower() for c in cost_cols):
                    new_scaled = -new_scaled
                final_scaled_feat = new_scaled
                
            model_input.at[0, feature_map[feat_ui]] = final_scaled_feat
            
        return max(0, min(100, model.predict(model_input)[0]))
    except Exception:
        return 0.0

def get_actual_gii(country, lang):
    try:
        mask = (df_raw[country_col] == country) & (df_raw[year_col] == TARGET_YEAR)
        if mask.any() and gii_col_exact:
            val = df_raw.loc[mask, gii_col_exact[0]].values[0]
            return f"{val:.2f}" if not pd.isna(val) else "---"
    except: pass
    return "---"

# ============================================================
# 4. STREAMLIT ARAYÜZÜ
# ============================================================
lang_choice = st.sidebar.radio("Language / Dil", ["🇹🇷 Türkçe", "🇬🇧 English"])
lang = "tr" if "Türkçe" in lang_choice else "en"

if lang == "tr":
    st.markdown("<h2 style='text-align: center; color: #6fa8dc;'>GII 2025 Tahmin ve Karar Destek Sistemi</h2>", unsafe_allow_html=True)
else:
    st.markdown("<h2 style='text-align: center; color: #6fa8dc;'>GII 2025 Forecast & Decision Support System</h2>", unsafe_allow_html=True)

with st.expander("Metodoloji Hakkında / About Methodology"):
    if lang == "tr":
        st.write("Bu model, GII skorunu en çok etkileyen **22 kritik değişken** üzerinden tahmin üretir.")
    else:
        st.write("This model predicts based on the **22 critical variables** that most impact the GII score.")

tab_names = ["Senaryo Simülatörü", "Duyarlılık Analizi", "Karşılaştırmalı Analiz", "Hedef ve SHAP", "Trend Analizi"] if lang=="tr" else ["Scenario Simulator", "Sensitivity Analysis", "Comparative Analysis", "Target & SHAP", "Trend Analysis"]
t1, t2, t3, t4, t5 = st.tabs(tab_names)

# --- SEKME 1: SİMÜLATÖR (DÜZELTİLMİŞ) ---
with t1:
    st.markdown("### " + ("Senaryo Bazlı Tahmin" if lang=="tr" else "Scenario-Based Prediction"))
    country_sim = st.selectbox("Ülke Seç / Select Country", country_list, key="sim_country_sel")
    
    current_vals = [float(latest_data_raw.loc[country_sim].get(col, 0.0)) for col in ui_input_names]
    user_inputs = []
    
    with st.expander("Değişkenleri Düzenle / Edit Variables"):
        cols = st.columns(2)
        for i, name in enumerate(ui_input_names):
            with cols[i % 2]:
                # KEY ÖNEMLİ: Ülke değiştikçe değerlerin güncellenmesi için key'e ülke eklendi
                val = st.number_input(name, value=current_vals[i], format="%.5f", key=f"sim_{country_sim}_{name}")
                user_inputs.append(val)
                
    if st.button("Tahmini Hesapla / Calculate Forecast", type="primary"):
        score = calculate_score(country_sim, user_inputs)
        actual = get_actual_gii(country_sim, lang)
        col_a, col_b = st.columns(2)
        col_a.metric("GII 2025 Tahmini", f"{score:.2f}")
        col_b.metric("GII 2025 Gerçekleşen", actual)

# --- SEKME 2: DUYARLILIK ---
with t2:
    adv_country = st.selectbox("Ülke Seç / Select Country", country_list, key="sens_c")
    if st.button("Analizi Başlat / Start Analysis"):
        adv_inputs = [float(latest_data_raw.loc[adv_country].get(col, 0.0)) for col in ui_input_names]
        base_score = calculate_score(adv_country, adv_inputs)
        impacts = []
        for i, name in enumerate(ui_input_names):
            temp = adv_inputs.copy()
            is_cost = any(c.strip().lower() == name.strip().lower() for c in cost_cols)
            temp[i] = temp[i] * 0.9 if is_cost else temp[i] * 1.1
            gain = calculate_score(adv_country, temp) - base_score
            if gain > 0.001: impacts.append((name, gain))
        
        impacts.sort(key=lambda x: x[1], reverse=True)
        st.write(f"**Baz Skor:** {base_score:.2f}")
        for n, g in impacts[:5]:
            st.success(f"**{n}**: +{g:.3f} puan katkı potansiyeli.")

# --- SEKME 3: KARŞILAŞTIRMA ---
with t3:
    c1_col, c2_col = st.columns(2)
    c1 = c1_col.selectbox("Ülke A", country_list, key="c1")
    c2 = c2_col.selectbox("Ülke B", country_list, index=1, key="c2")
    if st.button("Kıyasla"):
        z1 = [latest_data_proc.loc[c1][f] for f in ui_input_names]
        z2 = [latest_data_proc.loc[c2][f] for f in ui_input_names]
        fig, ax = plt.subplots(figsize=(10, 8))
        y = np.arange(len(ui_input_names))
        ax.barh(y-0.2, z1, 0.4, label=c1, color="#6fa8dc")
        ax.barh(y+0.2, z2, 0.4, label=c2, color="#cbd5e1")
        ax.set_yticks(y); ax.set_yticklabels([n[:30] for n in ui_input_names])
        ax.legend(); st.pyplot(fig)

# --- SEKME 4: SHAP ---
with t4:
    d4 = st.selectbox("Ülke Seç", country_list, key="shap_country")
    if st.button("SHAP Analizi"):
        row_proc = latest_data_proc.loc[d4]
        m_in = pd.DataFrame(0.0, index=[0], columns=model_features)
        for f_ui in ui_input_names: m_in.at[0, feature_map[f_ui]] = row_proc[f_ui]
        
        pred = model.predict(m_in)[0]
        st.metric("Model Tahmini", f"{pred:.2f}")
        explainer = shap.Explainer(model)
        shap_vals = explainer(m_in)
        fig = plt.figure(); shap.plots.waterfall(shap_vals[0], show=False)
        st.pyplot(fig)

# --- SEKME 5: TREND ---
with t5:
    d5 = st.selectbox("Ülke Seç", country_list, key="trend_country")
    feat_tr = st.selectbox("Değişken", ["GII Skoru"] + trend_candidates)
    if st.button("Trend Çiz"):
        c_data = df_raw[df_raw[country_col] == d5].sort_values(year_col)
        y_col = gii_col_exact[0] if "GII" in feat_tr else feat_tr
        fig, ax = plt.subplots(); ax.plot(c_data[year_col], c_data[y_col], marker='o')
        st.pyplot(fig)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>2025 Strategic Forecast System</p>", unsafe_allow_html=True)
