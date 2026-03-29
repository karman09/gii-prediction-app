# ==============================================================================
# GII 2025 STRATEGY DASHBOARD (OPTIMIZED CALCULATION VERSION)
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
@st.cache_resource(show_spinner="Dosyalar yükleniyor...")
def load_system_files():
    try:
        df_raw  = pd.read_excel("FINAL_DATA.xlsx")
        df_proc = pd.read_excel("FINAL_PREPROCESSED_DATA.xlsx")
        scaler = joblib.load("SCALER.pkl")
        scaler_cols = list(joblib.load("SCALER_COLUMNS.pkl"))
        cost_cols = joblib.load("COST_COLS.pkl")
        model = joblib.load("BEST_MODEL.pkl")
        model_features = list(joblib.load("BEST_MODEL_FEATURES.pkl"))
        
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

# Trend özellikleri için hazırlık
trend_candidates = [c for c in df_raw.columns if c not in [country_col, year_col]]
gii_col_exact = [c for c in df_raw.columns if "global innovation index" in c.lower()]
trend_features_tr = ["GII Skoru (Gerçekleşen)"] + [c for c in trend_candidates if c not in gii_col_exact]
trend_features_en = ["GII Score (Actual)"] + [c for c in trend_candidates if c not in gii_col_exact]

# ============================================================
# 3. DÜZELTİLMİŞ HESAPLAMA MANTIĞI
# ============================================================
def calculate_score(country_name, raw_inputs_list):
    try:
        if country_name not in latest_data_raw.index: return 0.0
        
        # A. Scaler'ın beklediği tüm sütunları içeren boş bir DataFrame oluştur
        # Sıralama: scaler_cols ile tam uyumlu olmalı
        input_raw_df = pd.DataFrame(0.0, index=[0], columns=scaler_cols)
        
        # B. Ülkenin 2023 ham verilerini doldur
        orig_row_raw = latest_data_raw.loc[country_name]
        for col in scaler_cols:
            input_raw_df.at[0, col] = float(orig_row_raw.get(col, 0.0))
            
        # C. UI'dan gelen manuel değişiklikleri üzerine yaz
        for i, feat_ui in enumerate(ui_input_names):
            input_raw_df.at[0, feat_ui] = float(raw_inputs_list[i])
            
        # D. Scaler ile dönüşüm (Manuel hesaplama yerine scaler.transform kullanıyoruz)
        scaled_array = scaler.transform(input_raw_df)
        scaled_df = pd.DataFrame(scaled_array, columns=scaler_cols)
        
        # E. Model Girişini Hazırla (Özellik sıralaması model_features ile birebir aynı olmalı)
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)
        
        for feat_ui in ui_input_names:
            target_feat_name = feature_map[feat_ui]
            # Scaled değerden al
            val = scaled_df.at[0, feat_ui]
            # Cost kontrolü (Eğer eğitim verisi terslenmişse)
            if feat_ui.lower().strip() in [c.lower().strip() for c in cost_cols]:
                val = -val
            model_input.at[0, target_feat_name] = val
            
        # F. Tahmin
        pred = model.predict(model_input)[0]
        return max(0, min(100, float(pred)))
        
    except Exception as e:
        st.error(f"Hesaplama Hatası: {e}")
        return 0.0

def get_actual_gii(country):
    try:
        mask = (df_raw[country_col] == country) & (df_raw[year_col] == TARGET_YEAR)
        if mask.any():
            gii_col = [c for c in df_raw.columns if "global innovation index" in c.lower()]
            val = df_raw.loc[mask, gii_col[0]].values[0]
            return f"{val:.2f}" if not pd.isna(val) else "---"
    except: pass
    return "---"

# ============================================================
# 4. STREAMLIT ARAYÜZÜ
# ============================================================

lang_choice = st.sidebar.radio("Language / Dil", ["🇹🇷 Türkçe", "🇬🇧 English"])
lang = "tr" if "Türkçe" in lang_choice else "en"

# Başlık
st.markdown(f"<h2 style='text-align: center; color: #6fa8dc; font-weight: bold;'>{'GII 2025 Tahmin Sistemi' if lang=='tr' else 'GII 2025 Forecast System'}</h2>", unsafe_allow_html=True)

tab_names = ["Simülatör", "Duyarlılık", "Karşılaştırma", "Hedef & SHAP", "Trendler"] if lang=="tr" else ["Simulator", "Sensitivity", "Comparison", "Target & SHAP", "Trends"]
t1, t2, t3, t4, t5 = st.tabs(tab_names)

# SEKME 1: SİMÜLATÖR
with t1:
    st.markdown("### " + ("Senaryo Tahmini" if lang=="tr" else "Scenario Forecast"))
    country_sim = st.selectbox("Ülke / Country" , country_list, key="c_sim")
    
    # Mevcut verileri çek
    current_raw_vals = [float(latest_data_raw.loc[country_sim].get(col, 0.0)) for col in ui_input_names]
    
    user_inputs = []
    with st.expander("Göstergeleri Düzenle / Edit Indicators"):
        cols = st.columns(2)
        for i, name in enumerate(ui_input_names):
            with cols[i % 2]:
                val = st.number_input(name, value=current_raw_vals[i], format="%.5f", key=f"inp_{name}")
                user_inputs.append(val)
                
    if st.button("Hesapla / Calculate", type="primary", key="btn_sim"):
        score = calculate_score(country_sim, user_inputs)
        actual = get_actual_gii(country_sim)
        
        st.markdown("---")
        c_a, c_b = st.columns(2)
        with c_a: st.metric(label="2025 GII Tahmini", value=f"{score:.2f}")
        with c_b: st.metric(label="2025 GII Gerçekleşen", value=actual)

# SEKME 2: DUYARLILIK
with t2:
    st.markdown("### " + ("Duyarlılık Analizi" if lang=="tr" else "Sensitivity Analysis"))
    adv_country = st.selectbox("Ülke / Country", country_list, key="adv_country")
    
    if st.button("Analiz Et / Analyze", type="primary"):
        adv_inputs = [float(latest_data_raw.loc[adv_country].get(col, 0.0)) for col in ui_input_names]
        base_score = calculate_score(adv_country, adv_inputs)
        
        results = []
        for i, name in enumerate(ui_input_names):
            temp_inputs = adv_inputs.copy()
            # %10 İyileştirme (Maliyetse azalt, faydaysa artır)
            is_cost = name.lower().strip() in [c.lower().strip() for c in cost_cols]
            temp_inputs[i] = adv_inputs[i] * 0.9 if is_cost else adv_inputs[i] * 1.1
            
            new_score = calculate_score(adv_country, temp_inputs)
            results.append({"Gösterge": name, "Artış": new_score - base_score})
            
        results_df = pd.DataFrame(results).sort_values(by="Artış", ascending=False)
        st.write("🎯 **En etkili iyileştirme alanları:**")
        st.table(results_df.head(5))

# SEKME 3: KARŞILAŞTIRMA
with t3:
    c1_col, c2_col = st.columns(2)
    with c1_col: c1 = st.selectbox("Ülke A", country_list, key="b1")
    with c2_col: c2 = st.selectbox("Ülke B", country_list, key="b2", index=1)
    
    if st.button("Kıyasla"):
        # Z-skor karşılaştırması için proc verisinden çek
        z1 = [latest_data_proc.loc[c1].get(f, 0) for f in ui_input_names]
        z2 = [latest_data_proc.loc[c2].get(f, 0) for f in ui_input_names]
        
        fig, ax = plt.subplots(figsize=(10, 8))
        y = np.arange(len(ui_input_names))
        ax.barh(y - 0.2, z1, 0.4, label=c1, color="#6fa8dc")
        ax.barh(y + 0.2, z2, 0.4, label=c2, color="#cbd5e1")
        ax.set_yticks(y); ax.set_yticklabels([n[:30] for n in ui_input_names])
        ax.legend(); plt.tight_layout()
        st.pyplot(fig)

# SEKME 4: SHAP
with t4:
    shap_c = st.selectbox("Ülke Seç", country_list, key="sh")
    if st.button("SHAP Analizi"):
        # Model girdisini hazırla
        input_raw = pd.DataFrame([latest_data_raw.loc[shap_c][scaler_cols]], columns=scaler_cols)
        scaled_input = scaler.transform(input_raw)
        
        # Modelin beklediği sıraya diz
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)
        for f_ui in ui_input_names:
            val = scaled_input[0][scaler_cols.index(f_ui)]
            if f_ui.lower().strip() in [c.lower().strip() for c in cost_cols]: val = -val
            model_input.at[0, feature_map[f_ui]] = val
            
        explainer = shap.Explainer(model)
        shap_vals = explainer(model_input)
        fig = plt.figure()
        shap.plots.waterfall(shap_vals[0], max_display=10, show=False)
        st.pyplot(plt.gcf())

# SEKME 5: TRENDLER
with t5:
    d5 = st.selectbox("Ülke", country_list, key="tr_c")
    feat_dropdown = st.selectbox("Gösterge", trend_features_tr if lang=="tr" else trend_features_en)
    
    if st.button("Trendi Göster"):
        c_data = df_raw[df_raw[country_col] == d5].sort_values(year_col)
        target_col = [c for c in df_raw.columns if "global innovation index" in c.lower()][0] if "GII" in feat_dropdown else feat_dropdown
        
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(c_data[year_col], c_data[target_col], marker='o', color='#6fa8dc')
        st.pyplot(fig)

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray;'>GII 2025 Strategic Forecast System</p>", unsafe_allow_html=True)
