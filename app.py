# ==============================================================================
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
# 3. CORE HESAPLAMA MOTORU (TÜM SEKMELER İÇİN TEK KAYNAK)
# ============================================================
def calculate_score_engine(country_name, raw_inputs_list):
    """
    Simülatör, SHAP ve Duyarlılık analizinin TAMAMEN aynı sonucu vermesini sağlar.
    """
    try:
        if country_name not in latest_data_raw.index: return 0.0, None
        row_raw = latest_data_raw.loc[country_name]
        row_proc = latest_data_proc.loc[country_name]
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)
        
        for i, feat_ui in enumerate(ui_input_names):
            user_val = float(raw_inputs_list[i])
            base_raw_val = float(row_raw.get(feat_ui, 0.0))
            
            # SHAP ile aynı mantık: Değer değişmediyse orijinal proc verisini kullan
            if math.isclose(user_val, base_raw_val, rel_tol=1e-5):
                final_scaled_feat = row_proc[feat_ui]
            else:
                # Kullanıcı yeni değer girdiyse scaler ile hesapla
                idx = scaler_cols.index(feat_ui)
                new_scaled = (user_val - scaler.mean_[idx]) / scaler.scale_[idx]
                if feat_ui.lower().strip() in cost_cols:
                    new_scaled = -new_scaled
                final_scaled_feat = new_scaled
            
            model_input.at[0, feature_map[feat_ui]] = final_scaled_feat
            
        pred = model.predict(model_input)[0]
        return max(0, min(100, pred)), model_input
    except Exception:
        return 0.0, None

def get_raw_values(country_name):
    if country_name not in latest_data_raw.index: return [0.0] * len(ui_input_names)
    row = latest_data_raw.loc[country_name]
    return [float(row.get(col, 0.0) if not pd.isna(row.get(col, 0.0)) else 0.0) for col in ui_input_names]

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

if lang == "tr":
    st.markdown("<h2 style='text-align: center; color: #6fa8dc; font-weight: bold;'>GII 2025 Tahmin ve Karar Destek Sistemi</h2>", unsafe_allow_html=True)
else:
    st.markdown("<h2 style='text-align: center; color: #6fa8dc; font-weight: bold;'>GII 2025 Forecast & Decision Support System</h2>", unsafe_allow_html=True)

with st.expander("Metodoloji Hakkında / About Methodology" if lang=="tr" else "About Methodology"):
    if lang == "tr":
        st.markdown("""
        **1. Klasik GII Hesaplaması:** 78 farklı göstergenin ağırlıklı ortalamasıdır.
        **2. Yapay Zeka Modeli:** 22 kritik belirleyici üzerinden tahminsel sonuçlar üretir.
        **3. SHAP Entegrasyonu:** Tahmin sonuçlarının simülatör ile %100 uyumlu olmasını sağlar.
        """)
    else:
        st.markdown("""
        **1. Classical GII:** Weighted average of 78 indicators.
        **2. AI Model:** Predictive results via 22 critical determinants.
        **3. SHAP Integration:** Ensures 100% consistency between simulator and analytics.
        """)

# --- SEKMELER ---
t1, t2, t3, t4, t5 = st.tabs(["Senaryo Simülatörü", "Duyarlılık Analizi", "Karşılaştırmalı Analiz", "Hedef ve SHAP", "Trend Analizi"] if lang=="tr" else ["Scenario Simulator", "Sensitivity Analysis", "Comparative Analysis", "Target & SHAP", "Trend Analysis"])

# SEKME 1: SENARYO SİMÜLATÖRÜ (YENİLENMİŞ MOTOR)
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
        # SHAP ile aynı motor kullanılıyor:
        score, _ = calculate_score_engine(country_sim, user_inputs)
        actual = get_actual_gii(country_sim, lang)
        
        if lang == "tr":
            st.success(f"**{country_sim} İçin {TARGET_YEAR} GII Tahmini:** {score:.2f}\n\n**{TARGET_YEAR} GII Gerçekleşen Değeri:** {actual}")
        else:
            st.success(f"**{TARGET_YEAR} GII Forecast for {country_sim}:** {score:.2f}\n\n**{TARGET_YEAR} GII Actual Value:** {actual}")

# SEKME 4: HEDEF VE SHAP (YENİLENMİŞ MOTOR)
with t4:
    st.markdown("### " + ("Model Açıklanabilirliği (XAI) ve Stratejik Hedef Takibi" if lang=="tr" else "Model Explainability (XAI) and Strategic Target Tracking"))
    
    shap_c, target_c = st.columns([2,1])
    with shap_c: d4 = st.selectbox("Ülke Seç / Select Country", country_list, key="shap_c")
    with target_c: target_score = st.number_input("Hedeflenen GII Skoru" if lang=="tr" else "Targeted Score", value=0.0)
    
    if st.button("Analizi Başlat / Start Analysis", type="primary", key="shap_btn"):
        # Mevcut ham verileri al ve motoru çalıştır
        raw_vals_current = get_raw_values(d4)
        pred, model_input = calculate_score_engine(d4, raw_vals_current)
        actual_val_str = get_actual_gii(d4, lang)
        
        target_text = f"**Ülke:** {d4}\n\n**{TARGET_YEAR} GII Tahmini:** {pred:.2f}\n\n**{TARGET_YEAR} Gerçekleşen:** {actual_val_str}\n\n---\n"
        if target_score > 0:
            gap = target_score - pred
            target_text += f"🎯 **Hedef:** {target_score:.2f} | " + (f"📉 **Fark:** {gap:.2f}" if gap > 0 else "🎉 **Durum:** Hedefin üzerindesiniz!")

        # SHAP İçgörüleri
        explainer = shap.Explainer(model)
        shap_values = explainer(model_input)
        impacts = list(zip(shap_values[0].feature_names, shap_values[0].values))
        pos_impacts = sorted([x for x in impacts if x[1] > 0], key=lambda x: x[1], reverse=True)
        neg_impacts = sorted([x for x in impacts if x[1] < 0], key=lambda x: x[1])
        
        shap_text = "**Spesifik İçgörüler:**\n\n" if lang=="tr" else "**Specific Insights:**\n\n"
        if pos_impacts:
            shap_text += "**Güçlü Yönler:**\n"
            for f, v in pos_impacts[:3]: shap_text += f"- {reverse_feature_map.get(f, f)}: +{v:.2f}\n"
        if neg_impacts:
            shap_text += "\n**Gelişim Alanları:**\n"
            for f, v in neg_impacts[:3]: shap_text += f"- {reverse_feature_map.get(f, f)}: {v:.2f}\n"

        col_txt, col_plot = st.columns([1,2])
        with col_txt:
            st.info(target_text)
            st.warning(shap_text)
        with col_plot:
            fig = plt.figure(figsize=(9, 6))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig)

# DİĞER SEKMELER (Duyarlılık, Karşılaştırma, Trend)
with t2:
    st.markdown("### " + ("Duyarlılık Analizi" if lang=="tr" else "Sensitivity Analysis"))
    adv_country = st.selectbox("Ülke Seç / Select Country", country_list, key="adv_country")
    if st.button("Analizi Başlat / Start Analysis", key="adv_btn"):
        adv_inputs = get_raw_values(adv_country)
        current_score, _ = calculate_score_engine(adv_country, adv_inputs)
        st.write(f"**Baz Tahmin:** {current_score:.2f}")

with t3:
    st.markdown("### Karşılaştırmalı Analiz")
    c1_col, c2_col = st.columns(2)
    with c1_col: c1 = st.selectbox("Ülke A", country_list, key="bench_c1")
    with c2_col: c2 = st.selectbox("Ülke B", country_list, key="bench_c2", index=1)
    if st.button("Grafiği Oluştur"):
        v1, v2 = get_raw_values(c1), get_raw_values(c2)
        s1, _ = calculate_score_engine(c1, v1)
        s2, _ = calculate_score_engine(c2, v2)
        st.write(f"{c1}: {s1:.2f} vs {c2}: {s2:.2f}")

with t5:
    st.markdown("### Trend Analizi")
    d5_c = st.selectbox("Ülke Seç / Select Country", country_list, key="trend_c")
    feat_dropdown = st.selectbox("Değişken", trend_features_tr if lang=="tr" else trend_features_en)
    if st.button("Trendi Çiz"):
        country_data = df_raw[df_raw[country_col] == d5_c].sort_values(by=year_col)
        fig, ax = plt.subplots()
        ax.plot(country_data[year_col], country_data.get(feat_dropdown, country_data[gii_col_exact[0]]))
        st.pyplot(fig)

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>{TARGET_YEAR} Strategic Decision Support System</p>", unsafe_allow_html=True)
