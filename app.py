# GII 2025 STRATEGY DASHBOARD (LOCAL GITHUB BILINGUAL FULL VERSION)
# ==============================================================================
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import re
import math
import shap

st.set_page_config(page_title="D-LOGII Dashboard", page_icon="📊", layout="wide")

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
        st.error(f"Dosya Yükleme Hatası / File Load Error: {e}")
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
LAG_PERIOD  = 2
INPUT_YEAR  = TARGET_YEAR - LAG_PERIOD  # 2023

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

reverse_feature_map = {v: k for k, v in feature_map.items()}

trend_candidates = [c for c in df_raw.columns if c != country_col and c != year_col]
gii_col_exact = [c for c in df_raw.columns if "global innovation index" in c.lower()]
if gii_col_exact and gii_col_exact[0] in trend_candidates:
    trend_candidates.remove(gii_col_exact[0])
trend_features_tr = ["GII Skoru (Gerçekleşen)"] + trend_candidates
trend_features_en = ["GII Score (Actual)"] + trend_candidates

# ============================================================
# 3. CORE HESAPLAMA MANTIĞI VE YARDIMCI FONKSİYONLAR
# ============================================================
def get_raw_values(country_name):
    if country_name not in latest_data_raw.index: return [0.0] * len(ui_input_names)
    row = latest_data_raw.loc[country_name]
    return [float(row.get(col, 0.0) if not pd.isna(row.get(col, 0.0)) else 0.0) for col in ui_input_names]

def calculate_score(country_name, raw_inputs_list):
    try:
        if country_name not in latest_data_raw.index: return 0.0
        row_raw = latest_data_raw.loc[country_name]
        row_proc = latest_data_proc.loc[country_name]
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)
        
        for i, feat_ui in enumerate(ui_input_names):
            user_val = float(raw_inputs_list[i]) if raw_inputs_list[i] is not None else 0.0
            base_raw_val = row_raw.get(feat_ui, np.nan)
            displayed_base_val = 0.0 if pd.isna(base_raw_val) else float(base_raw_val)
            
            # KRİTİK DÜZELTME: row_proc içinden doğru anahtarla (feature_map) veriyi çekiyoruz
            if math.isclose(user_val, displayed_base_val, rel_tol=1e-5, abs_tol=1e-5):
                final_scaled_feat = row_proc[feature_map[feat_ui]]
            else:
                idx = scaler_cols.index(feat_ui)
                new_scaled = (user_val - scaler.mean_[idx]) / scaler.scale_[idx]
                if feat_ui.lower().strip() in cost_cols: new_scaled = -new_scaled
                final_scaled_feat = new_scaled
                
            model_input.at[0, feature_map[feat_ui]] = final_scaled_feat
            
        return max(0, min(100, model.predict(model_input)[0]))
    except Exception:
        return 0.0

def get_actual_gii(country, lang):
    actual_val_str = "Veri Bulunamadı" if lang == "tr" else "Data Not Found"
    try:
        mask = (df_raw[country_col] == country) & (df_raw[year_col] == TARGET_YEAR)
        if mask.any():
            gii_col = [c for c in df_raw.columns if "global innovation index" in c.lower()]
            if gii_col:
                val = df_raw.loc[mask, gii_col[0]].values[0]
                if not pd.isna(val): actual_val_str = f"{val:.2f}"
    except Exception: pass
    return actual_val_str

# ============================================================
# 4. STREAMLIT ARAYÜZÜ
# ============================================================

lang_choice = st.sidebar.radio("Language / Dil", ["🇹🇷 Türkçe", "🇬🇧 English"])
lang = "tr" if "Türkçe" in lang_choice else "en"

# Logo ve Başlık
col1, col2, col3 = st.columns([1,2,1])
with col2:
    try:
        st.image("logo.png", use_container_width=True)
    except:
        pass 

st.markdown("<h2 style='text-align: center; color: #6fa8dc; font-weight: bold;'>GII 2025 Tahmin ve Karar Destek Sistemi <br> <span style='font-size: 0.8em;'>Forecast & Decision Support System</span></h2>", unsafe_allow_html=True)

with st.expander("Metodoloji Hakkında / About Methodology"):
    if lang == "tr":
        st.markdown("""
        **1. Klasik GII Hesaplaması (WIPO Metodolojisi):** Küresel İnovasyon Endeksi (GII), 2025 yılı itibarıyla **7 ana sütun** altında toplanan tam **78 farklı göstergenin** ağırlıklı ortalaması alınarak hesaplanır. 
        **2. Geliştirilen Yapay Zeka Modeli:** Model, GII skorunu en çok etkileyen **"22 Kritik Belirleyici"** tespit etmiştir.
        """)
    else:
        st.markdown("""
        **1. Classical GII Calculation (WIPO Methodology):** Calculated by taking the weighted average of **78 indicators**.
        **2. Developed AI Model:** The model has identified **"22 Critical Determinants"** that most affect the GII score.
        """)

# --- SEKMELER ---
if lang == "tr":
    t1, t2, t3, t4, t5 = st.tabs(["Senaryo Simülatörü", "Duyarlılık Analizi", "Karşılaştırmalı Analiz", "Hedef ve SHAP", "Trend Analizi"])
else:
    t1, t2, t3, t4, t5 = st.tabs(["Scenario Simulator", "Sensitivity Analysis", "Comparative Analysis", "Target & SHAP", "Trend Analysis"])

# SEKME 1: SİMÜLATÖR
with t1:
    st.markdown("### " + ("Senaryo Bazlı Tahmin Simülasyonu" if lang=="tr" else "Scenario-Based Prediction Simulation"))
    country_sim = st.selectbox("Ülke Seç / Select Country" , country_list, key="c_sim")
    raw_vals = get_raw_values(country_sim)
    user_inputs = []
    
    with st.expander("Değişkenleri Düzenle / Edit Variables" if lang=="tr" else "View / Edit Variables"):
        cols = st.columns(2)
        for i, name in enumerate(ui_input_names):
            with cols[i % 2]:
                val = st.number_input(name, value=float(raw_vals[i]), format="%.5f", key=f"inp_{name}")
                user_inputs.append(val)
                
    if st.button("Tahmini Hesapla / Calculate Forecast", type="primary"):
        score = calculate_score(country_sim, user_inputs)
        actual = get_actual_gii(country_sim, lang)
        if lang == "tr":
            st.success(f"**{country_sim} İçin {TARGET_YEAR} GII Tahmini:** {score:.2f}\n\n**{TARGET_YEAR} GII Gerçekleşen Değeri:** {actual}")
        else:
            st.success(f"**{TARGET_YEAR} GII Forecast for {country_sim}:** {score:.2f}\n\n**{TARGET_YEAR} GII Actual Value:** {actual}")

# SEKME 2: DUYARLILIK ANALİZİ
with t2:
    st.markdown("### " + (f"{INPUT_YEAR} Verileri Üzerinden Etki Analizi" if lang=="tr" else f"Impact Analysis based on {INPUT_YEAR} Data"))
    adv_country = st.selectbox("Ülke Seç / Select Country", country_list, key="adv_country")
    
    if st.button("Analizi Başlat / Start Analysis", type="primary", key="adv_btn"):
        adv_inputs = get_raw_values(adv_country)
        current_score = calculate_score(adv_country, adv_inputs)
        impacts = []
        for i, orig_name in enumerate(ui_input_names):
            val = adv_inputs[i]
            is_cost = orig_name.lower().strip() in cost_cols
            new_val = val * 0.90 if is_cost else val * 1.10
            act = "AZALTILIRSA" if lang == "tr" else "DECREASED" if is_cost else "ARTIRILIRSA" if lang == "tr" else "INCREASED"
            temp = adv_inputs.copy()
            temp[i] = new_val
            new_score = calculate_score(adv_country, temp)
            gain = new_score - current_score
            if not np.isnan(gain) and gain > 0.01:
                impacts.append({"Feat": orig_name, "Gain": gain, "Act": act, "Val": new_val})
                
        impacts.sort(key=lambda x: x["Gain"], reverse=True)
        if lang == "tr":
            st.write(f"**Baz Tahmin:** {current_score:.2f}")
            for item in impacts[:5]:
                st.write(f"- **{item['Feat']}** %10 {item['Act']} -> +{item['Gain']:.3f} puan artış")
        else:
            st.write(f"**Base Forecast:** {current_score:.2f}")
            for item in impacts[:5]:
                st.write(f"- **{item['Feat']}** 10% {item['Act']} -> +{item['Gain']:.3f} point gain")

# SEKME 3: KARŞILAŞTIRMALI ANALİZ
with t3:
    st.markdown("### " + ("Performans Matrisi" if lang=="tr" else "Performance Matrix"))
    c1_col, c2_col = st.columns(2)
    with c1_col: c1 = st.selectbox("Ülke A / Country A", country_list, key="bench_c1")
    with c2_col: c2 = st.selectbox("Ülke B / Country B", country_list, key="bench_c2", index=1)
    
    if st.button("Grafiği Oluştur / Generate Chart", type="primary", key="bench_btn"):
        v1, v2 = get_raw_values(c1), get_raw_values(c2)
        s1, s2 = calculate_score(c1, v1), calculate_score(c2, v2)
        row_proc_1, row_proc_2 = latest_data_proc.loc[c1], latest_data_proc.loc[c2]
        
        lbls, z1, z2 = [], [], []
        for f in ui_input_names:
            lbls.append(f)
            z1.append(row_proc_1[feature_map[f]])
            z2.append(row_proc_2[feature_map[f]])
            
        fig, ax = plt.subplots(figsize=(10, 8))
        y = np.arange(len(lbls))
        ax.barh(y - 0.2, z1, 0.4, label=c1, color="#0f766e")
        ax.barh(y + 0.2, z2, 0.4, label=c2, color="#64748b")
        ax.set_yticks(y); ax.set_yticklabels(lbls)
        ax.legend(); plt.tight_layout()
        st.pyplot(fig)

# SEKME 4: SHAP ANALİZİ
with t4:
    st.markdown("### " + ("Model Açıklanabilirliği (SHAP)" if lang=="tr" else "Model Explainability (SHAP)"))
    shap_c, target_c = st.columns([2,1])
    with shap_c: d4 = st.selectbox("Ülke Seç / Select Country", country_list, key="shap_c")
    with target_c: target_score = st.number_input("Hedef Skor / Target", value=0.0)
    
    if st.button("Analizi Başlat / Start Analysis", type="primary", key="shap_btn"):
        row_proc = latest_data_proc.loc[d4]
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)
        for f in ui_input_names: model_input.at[0, feature_map[f]] = row_proc[feature_map[f]]
        
        pred = model.predict(model_input)[0]
        explainer = shap.Explainer(model)
        shap_values = explainer(model_input)
        
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], show=False)
        st.pyplot(plt.gcf())

# SEKME 5: TRENDLER
with t5:
    st.markdown("### " + ("Trend Analizi" if lang=="tr" else "Trend Analysis"))
    d5_c, f5_c = st.columns(2)
    with d5_c: d5 = st.selectbox("Ülke Seç", country_list, key="trend_c")
    with f5_c: feat_dropdown = st.selectbox("Değişken", trend_features_tr if lang=="tr" else trend_features_en)
    
    if st.button("Trendi Çiz", type="primary"):
        country_data = df_raw[df_raw[country_col] == d5].sort_values(by=year_col)
        actual_col = [c for c in df_raw.columns if "global innovation index" in c.lower()][0] if "GII" in feat_dropdown else feat_dropdown
        
        if actual_col in country_data.columns:
            fig, ax = plt.subplots()
            ax.plot(country_data[year_col], country_data[actual_col], marker='o')
            st.pyplot(fig)

# --- FOOTER ---
st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>2025 Stratejik Tahmin Sistemi</p>", unsafe_allow_html=True)
