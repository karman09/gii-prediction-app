import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import re
import math
import shap

# Sayfa Konfigürasyonu
st.set_page_config(page_title="D-LOGII Dashboard", layout="wide")

# ============================================================
# 1. VERİ VE MODEL YÜKLEME (Streamlit Cache ile Optimize)
# ============================================================
@st.cache_resource
def load_assets():
    # Dosya yolları GitHub/Streamlit için göreceli (relative) yapıldı
    try:
        df_raw = pd.read_excel("FINAL_DATA.xlsx")
        df_proc = pd.read_excel("FINAL_PREPROCESSED_DATA.xlsx")
        scaler = joblib.load("SCALER.pkl")
        scaler_cols = joblib.load("SCALER_COLUMNS.pkl")
        cost_cols = joblib.load("COST_COLS.pkl")
        model = joblib.load("BEST_MODEL.pkl")
        model_features = joblib.load("BEST_MODEL_FEATURES.pkl")
        return df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features
    except Exception as e:
        st.error(f"Dosya yükleme hatası: {e}. Lütfen dosyaların ana dizinde olduğundan emin olun.")
        return None

assets = load_assets()
if assets:
    df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features = assets

# --- Sabitler ve Eşleştirmeler ---
def sanitize_name(name): return re.sub(r"[^A-Za-z0-9_]", "", name)

sanitized_to_original = {sanitize_name(col): col for col in scaler_cols}
country_col = [c for c in df_raw.columns if "country" in c.lower() or "economy" in c.lower()][0]
year_col = "year"
TARGET_YEAR, LAG_PERIOD = 2025, 2
INPUT_YEAR = TARGET_YEAR - LAG_PERIOD

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

# ============================================================
# 2. HESAPLAMA MOTORU (MANTIK DEĞİŞMEDİ)
# ============================================================
def calculate_score(country_name, input_values_dict):
    try:
        row_raw = latest_data_raw.loc[country_name]
        row_proc = latest_data_proc.loc[country_name]
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)

        for feat_ui in ui_input_names:
            user_val = float(input_values_dict[feat_ui])
            base_raw_val = float(row_raw.get(feat_ui, 0.0))

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
# 3. STREAMLIT UI TASARIMI
# ============================================================

# Başlık Bölümü
st.markdown(f"""
    <div style='text-align: center;'>
        <h1 style='color: #0f766e;'>D-LOGII</h1>
        <p style='font-size: 1.2em; color: #64748b;'>Dynamic Lasso-Optimized Global Innovation Index</p>
    </div>
    """, unsafe_allow_html=True)

with st.expander("Metodoloji Hakkında: WIPO Standartları vs. Yapay Zeka Modeli"):
    st.info("Bu sistem, GII skorunu en çok etkileyen kritik belirleyicileri kullanarak 2025 yılı için tahminsel bir yaklaşım sunar.")

tabs = st.tabs(["Senaryo Simülatörü", "Duyarlılık Analizi", "Kıyaslama", "SHAP Analizi"])

# --- TAB 1: SIMULATOR ---
with tabs[0]:
    st.subheader("Senaryo Bazlı Tahmin Simülasyonu")
    col_c, col_b = st.columns([3, 1])
    selected_country = col_c.selectbox("Ülke Seçin", country_list, key="sim_country")
    
    st.markdown("#### Değişkenleri Düzenle (Ham Veriler)")
    current_raw_vals = latest_data_raw.loc[selected_country]
    
    user_inputs = {}
    exp = st.expander("Girdi Değişkenlerini Görüntüle / Değiştir", expanded=False)
    with exp:
        cols = st.columns(2)
        for i, name in enumerate(ui_input_names):
            user_inputs[name] = cols[i % 2].number_input(name, value=float(current_raw_vals.get(name, 0.0)), key=f"in_{name}")

    if st.button("Tahmini Hesapla", type="primary"):
        score = calculate_score(selected_country, user_inputs)
        
        # Gerçekleşen değer çekme
        actual_val = "N/A"
        mask = (df_raw[country_col] == selected_country) & (df_raw[year_col] == TARGET_YEAR)
        if mask.any():
            gii_col = [c for c in df_raw.columns if "global innovation index" in c.lower()]
            actual_val = f"{df_raw.loc[mask, gii_col[0]].values[0]:.2f}"
        
        st.success(f"**{selected_country}** İçin {TARGET_YEAR} GII Tahmini: **{score:.2f}**")
        st.info(f"{TARGET_YEAR} GII Gerçekleşen Değeri: {actual_val}")

# --- TAB 2: SENSITIVITY ---
with tabs[1]:
    st.subheader("Değişken Bazlı Duyarlılık Analizi")
    s_country = st.selectbox("Analiz Edilecek Ülke", country_list, key="sens_country")
    
    if st.button("Analizi Başlat"):
        with st.spinner("Hesaplanıyor..."):
            base_inputs = {n: latest_data_raw.loc[s_country].get(n, 0.0) for n in ui_input_names}
            current_score = calculate_score(s_country, base_inputs)
            
            impacts = []
            for name in ui_input_names:
                is_cost = name.lower().strip() in cost_cols
                temp_inputs = base_inputs.copy()
                temp_inputs[name] *= 0.90 if is_cost else 1.10
                
                gain = calculate_score(s_country, temp_inputs) - current_score
                if gain > 0.01:
                    impacts.append((name, gain, temp_inputs[name], "AZALTILIRSA" if is_cost else "ARTIRILIRSA"))
            
            impacts.sort(key=lambda x: x[1], reverse=True)
            
            st.write(f"**Baz Skor ({INPUT_YEAR}):** {current_score:.2f}")
            for name, gain, val, act in impacts[:5]:
                st.write(f"**{name}** %10 {act} → **+{gain:.3f}** puan (Yeni Değer: {val:.2f})")

# --- TAB 3: BENCHMARK ---
with tabs[2]:
    st.subheader("Performans Karşılaştırma (Z-Skor)")
    c1_col, c2_col = st.columns(2)
    country_a = c1_col.selectbox("Ülke A", country_list, index=0)
    country_b = c2_col.selectbox("Ülke B", country_list, index=1)
    
    if st.button("Kıyasla"):
        row_a = latest_data_proc.loc[country_a]
        row_b = latest_data_proc.loc[country_b]
        
        z1, z2, lbls = [], [], []
        for f in ui_input_names:
            z1.append(row_a[f])
            z2.append(row_b[f])
            lbls.append(f + (" (-)" if f.lower().strip() in cost_cols else ""))

        fig, ax = plt.subplots(figsize=(10, len(lbls)*0.4))
        y = np.arange(len(lbls))
        ax.barh(y - 0.2, z1, 0.4, label=country_a, color="#0f766e")
        ax.barh(y + 0.2, z2, 0.4, label=country_b, color="#64748b")
        ax.set_yticks(y)
        ax.set_yticklabels(lbls)
        ax.legend()
        ax.axvline(0, color='black', linestyle='--', alpha=0.3)
        st.pyplot(fig)

# --- TAB 4: SHAP ---
with tabs[3]:
    st.subheader("Yapay Zeka Karar Açıklanabilirliği")
    shap_country = st.selectbox("Ülke Seç", country_list, key="shap_country")
    
    if st.button("SHAP Analizi Oluştur"):
        row_proc = latest_data_proc.loc[shap_country]
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)
        for f in ui_input_names: model_input.at[0, feature_map[f]] = row_proc[f]
        
        explainer = shap.Explainer(model)
        shap_values = explainer(model_input)
        
        col_t, col_p = st.columns([1, 2])
        
        with col_p:
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            st.pyplot(plt.gcf())
            
        with col_t:
            st.write(f"**{shap_country}** için Önemli Faktörler:")
            # Pozitif/Negatif etkileri listeleme mantığı buraya eklenebilir.
            st.write("Grafikte kırmızı barlar skoru yükselten, mavi barlar düşüren faktörleri temsil eder.")

st.markdown("---")
st.caption("2026 Stratejik Tahmin ve Karar Destek Sistemi | D-LOGII")
