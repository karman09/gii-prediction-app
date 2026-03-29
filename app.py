import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import re
import math
import shap

# Sayfa Konfigürasyonu (En üstte olmalı)
st.set_page_config(page_title="D-LOGII Dashboard", layout="wide")

# ============================================================
# 1. DOSYA YOLLARI VE YÜKLEME (Cache kullanarak hızı artırıyoruz)
# ============================================================
@st.cache_resource
def load_assets():
    # Dosya yollarını GitHub deponuzdaki isimlere göre güncelleyin
    raw_data_path      = "FINAL_DATA.xlsx"
    proc_data_path     = "FINAL_PREPROCESSED_DATA.xlsx"
    scaler_path        = "SCALER.pkl"
    scaler_cols_path   = "SCALER_COLUMNS.pkl"
    cost_cols_path     = "COST_COLS.pkl"
    model_path         = "BEST_MODEL.pkl"
    features_path      = "BEST_MODEL_FEATURES.pkl"

    try:
        df_raw  = pd.read_excel(raw_data_path)
        df_proc = pd.read_excel(proc_data_path)
        scaler = joblib.load(scaler_path)
        scaler_cols = joblib.load(scaler_cols_path)
        cost_cols = joblib.load(cost_cols_path)
        model = joblib.load(model_path)
        model_features = joblib.load(features_path)
        return df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features
    except Exception as e:
        st.error(f"Dosya yükleme hatası: {e}")
        return None

assets = load_assets()
if assets:
    df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features = assets

# ============================================================
# 2. YARDIMCI FONKSİYONLAR (Mantık Değişmedi)
# ============================================================
def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_]", "", name)

sanitized_to_original = {sanitize_name(col): col for col in scaler_cols}

country_col = [c for c in df_raw.columns if "country" in c.lower() or "economy" in c.lower()][0]
year_col, TARGET_YEAR, LAG_PERIOD = "year", 2025, 2023

latest_data_raw  = df_raw[df_raw[year_col] == LAG_PERIOD].set_index(country_col)
latest_data_proc = df_proc[df_proc[year_col] == LAG_PERIOD].set_index(country_col)
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

# ============================================================
# 3. HESAPLAMA MOTORU (Core Logic)
# ============================================================
def calculate_score_logic(country_name, user_inputs_dict):
    row_raw = latest_data_raw.loc[country_name]
    row_proc = latest_data_proc.loc[country_name]
    model_input = pd.DataFrame(0.0, index=[0], columns=model_features)

    for feat_ui in ui_input_names:
        user_val = float(user_inputs_dict[feat_ui])
        base_raw_val = row_raw.get(feat_ui, np.nan)
        displayed_base_val = 0.0 if pd.isna(base_raw_val) else float(base_raw_val)

        if math.isclose(user_val, displayed_base_val, rel_tol=1e-5, abs_tol=1e-5):
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

# ============================================================
# 4. STREAMLIT UI (Profesyonel Tema)
# ============================================================
st.markdown(f"""
    <div style='text-align: center;'>
        <h1 style='color: #0f766e;'>D-LOGII</h1>
        <h3 style='color: #64748b;'>Dynamic Lasso-Optimized Global Innovation Index</h3>
    </div>
    <hr>
    """, unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "Senaryo Simülatörü", 
    "Duyarlılık Analizi", 
    "Karşılaştırmalı Analiz", 
    "SHAP Analizi"
])

# --- TAB 1: SİMÜLATÖR ---
with tab1:
    st.subheader("Senaryo Bazlı Tahmin Simülasyonu")
    selected_country = st.selectbox("Ülke Seçiniz", country_list, key="sb1")
    
    with st.expander("Değişkenleri Düzenle (Ham Veriler)", expanded=False):
        col1, col2 = st.columns(2)
        user_inputs = {}
        raw_vals = latest_data_raw.loc[selected_country]
        
        for i, name in enumerate(ui_input_names):
            current_val = float(raw_vals.get(name, 0.0)) if not pd.isna(raw_vals.get(name)) else 0.0
            target_col = col1 if i % 2 == 0 else col2
            user_inputs[name] = target_col.number_input(f"{name}", value=current_val, key=f"in_{name}")

    if st.button("Tahmini Hesapla", type="primary"):
        score = calculate_score_logic(selected_country, user_inputs)
        st.success(f"**{selected_country}** İçin {TARGET_YEAR} GII Tahmini: **{score:.2f}**")

# --- TAB 3: KIYASLA (Örnek olarak Benchmark eklenmiştir) ---
with tab3:
    st.subheader("Performans Karşılaştırma (Z-Skor)")
    c_col1, c_col2 = st.columns(2)
    country_a = c_col1.selectbox("Ülke A", country_list, index=0)
    country_b = c_col2.selectbox("Ülke B", country_list, index=1)
    
    if st.button("Analizi Görselleştir"):
        # Grafik mantığını buraya taşıyoruz (Gradio fonksiyonundan kopyala-yapıştır yapabilirsiniz)
        fig, ax = plt.subplots(figsize=(10, 8))
        # ... (Gradio'daki ax.barh çizim kodları buraya gelecek)
        st.pyplot(fig)

# --- ALT BİLGİ ---
st.markdown("---")
st.caption("2025 Stratejik Tahmin ve Karar Destek Sistemi - D-LOGII")
