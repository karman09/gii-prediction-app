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
    .stTabs [data-baseweb="tab-list"] { gap: 10px; background-color: transparent; }
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
        border: 1px solid #6fa8dc !important;
    }
    [data-baseweb="tab-border-highlight"] { background-color: #6fa8dc !important; }
    .stButton>button {
        background-color: #6fa8dc !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover { opacity: 0.9; transform: translateY(-1px); }
    .stNumberInput input { border-radius: 8px !important; }
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
        cost_cols = [c.lower().strip() for c in joblib.load("COST_COLS.pkl")]
        model = joblib.load("BEST_MODEL.pkl")
        model_features = joblib.load("BEST_MODEL_FEATURES.pkl")
        return df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features
    except Exception as e:
        st.error(f"Dosya Yükleme Hatası: {e}")
        st.stop()

df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features = load_system_files()

# ============================================================
# 2. VERİ HAZIRLIĞI VE EŞLEŞTİRME (DÜZELTİLMİŞ)
# ============================================================
country_col = [c for c in df_raw.columns if "country" in c.lower() or "economy" in c.lower()][0]
year_col = "year"

TARGET_YEAR = 2025
LAG_PERIOD  = 2
INPUT_YEAR  = TARGET_YEAR - LAG_PERIOD 

latest_data_raw  = df_raw[df_raw[year_col] == INPUT_YEAR].set_index(country_col)
latest_data_proc = df_proc[df_proc[year_col] == INPUT_YEAR].set_index(country_col)
country_list = sorted(latest_data_raw.index.tolist())

# Model özellikleri ile UI arasındaki köprüyü kur
ui_input_names = []
feature_to_ui_map = {}
for feat in model_features:
    base_name = re.sub(r"_lag\d+$", "", feat)
    if base_name in scaler_cols:
        ui_input_names.append(base_name)
        feature_to_ui_map[feat] = base_name

ui_input_names = list(dict.fromkeys(ui_input_names))

# Trend ve Diğer Yardımcı Listeler
trend_candidates = [c for c in df_raw.columns if c not in [country_col, year_col]]
gii_col_exact = [c for c in df_raw.columns if "global innovation index" in c.lower()]
if gii_col_exact and gii_col_exact[0] in trend_candidates:
    trend_candidates.remove(gii_col_exact[0])

# ============================================================
# 3. CORE HESAPLAMA MANTIĞI (SÖZLÜK TABANLI VE GÜVENLİ)
# ============================================================
def calculate_score(country_name, user_inputs_dict):
    try:
        if country_name not in latest_data_proc.index: return 0.0
        row_raw = latest_data_raw.loc[country_name]
        row_proc = latest_data_proc.loc[country_name]
        
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)
        
        for feat_model in model_features:
            feat_ui = feature_to_ui_map.get(feat_model)
            if not feat_ui: continue
            
            user_val = float(user_inputs_dict.get(feat_ui, 0.0))
            orig_raw_val = float(row_raw.get(feat_ui, 0.0))
            
            # Değer değişmediyse orijinal işlenmiş veriyi kullan (Hata payını sıfırlar)
            if math.isclose(user_val, orig_raw_val, rel_tol=1e-5):
                final_z = row_proc.get(feat_model, 0.0)
            else:
                idx = scaler_cols.index(feat_ui)
                final_z = (user_val - scaler.mean_[idx]) / scaler.scale_[idx]
                if feat_ui.lower().strip() in cost_cols:
                    final_z = -final_z
            
            model_input.at[0, feat_model] = final_z
            
        return max(0, min(100, model.predict(model_input)[0]))
    except: return 0.0

def get_actual_gii(country):
    mask = (df_raw[country_col] == country) & (df_raw[year_col] == TARGET_YEAR)
    if mask.any():
        gii_cols = [c for c in df_raw.columns if "global innovation index" in c.lower()]
        if gii_cols:
            val = df_raw.loc[mask, gii_cols[0]].values[0]
            return f"{val:.2f}" if not pd.isna(val) else "---"
    return "---"

# ============================================================
# 4. STREAMLIT ARAYÜZÜ
# ============================================================
lang_choice = st.sidebar.radio("Language / Dil", ["🇹🇷 Türkçe", "🇬🇧 English"])
lang = "tr" if "Türkçe" in lang_choice else "en"

# Logo ve Başlık
col1, col2, col3 = st.columns([1,2,1])
with col2:
    try: st.image("logo.png", use_container_width=True)
    except: pass

title_text = "GII 2025 Tahmin ve Karar Destek Sistemi" if lang=="tr" else "GII 2025 Forecast & Decision Support System"
st.markdown(f"<h2 style='text-align: center; color: #6fa8dc; font-weight: bold;'>{title_text}</h2>", unsafe_allow_html=True)

# Sekme İsimleri
tab_labels = ["Senaryo Simülatörü", "Duyarlılık Analizi", "Karşılaştırmalı Analiz", "Hedef ve SHAP", "Trend Analizi"] if lang=="tr" else ["Scenario Simulator", "Sensitivity Analysis", "Comparative Analysis", "Target & SHAP", "Trend Analysis"]
t1, t2, t3, t4, t5 = st.tabs(tab_labels)

# --- SEKME 1: SİMÜLATÖR ---
with t1:
    st.markdown("### " + ("Senaryo Bazlı Tahmin" if lang=="tr" else "Scenario-Based Prediction"))
    country_sim = st.selectbox("Ülke Seç / Select Country", country_list, key="c_sim")
    
    current_vals = {col: float(latest_data_raw.loc[country_sim].get(col, 0.0)) for col in ui_input_names}
    sim_inputs = {}
    
    with st.expander("Değişkenleri Düzenle / Edit Variables"):
        cols = st.columns(2)
        for i, name in enumerate(ui_input_names):
            with cols[i % 2]:
                sim_inputs[name] = st.number_input(name, value=current_vals[name], format="%.5f", key=f"inp_{name}")
                
    if st.button("Tahmini Hesapla / Calculate Forecast", type="primary"):
        score = calculate_score(country_sim, sim_inputs)
        actual = get_actual_gii(country_sim)
        st.markdown("---")
        c_a, c_b = st.columns(2)
        c_a.metric(f"{TARGET_YEAR} Tahmini / Forecast", f"{score:.2f}")
        c_b.metric(f"{TARGET_YEAR} Gerçekleşen / Actual", actual)

# --- SEKME 2: DUYARLILIK ---
with t2:
    st.markdown("### " + ("Etki Analizi" if lang=="tr" else "Impact Analysis"))
    adv_country = st.selectbox("Ülke Seç / Select Country", country_list, key="adv_c")
    if st.button("Analizi Başlat / Start Analysis", key="adv_btn"):
        base_vals = {col: float(latest_data_raw.loc[adv_country].get(col, 0.0)) for col in ui_input_names}
        current_score = calculate_score(adv_country, base_vals)
        impacts = []
        for name in ui_input_names:
            val = base_vals[name]
            is_cost = name.lower().strip() in cost_cols
            temp_vals = base_vals.copy()
            temp_vals[name] = val * 0.90 if is_cost else val * 1.10
            new_score = calculate_score(adv_country, temp_vals)
            gain = new_score - current_score
            if gain > 0.001: impacts.append({"Feat": name, "Gain": gain})
        
        impacts.sort(key=lambda x: x["Gain"], reverse=True)
        st.write(f"**Baz Skor:** {current_score:.2f}")
        for item in impacts[:5]:
            st.write(f"- **{item['Feat']}**: +{item['Gain']:.3f} puan artış.")

# --- SEKME 3: KARŞILAŞTIRMA ---
with t3:
    st.markdown("### " + ("Z-Skor Karşılaştırma" if lang=="tr" else "Z-Score Comparison"))
    c1_col, c2_col = st.columns(2)
    with c1_col: c1 = st.selectbox("Ülke A", country_list, key="b1")
    with c2_col: c2 = st.selectbox("Ülke B", country_list, key="b2", index=1)
    if st.button("Grafiği Oluştur", key="b_btn"):
        z1 = [latest_data_proc.loc[c1].get(ui_to_f, 0) for ui_to_f in [ui_to_feature_map.get(f, f) for f in ui_input_names]] # Basitleştirilmiş mantık
        # Not: Karşılaştırma grafiği proc verisinden doğrudan alınır.
        fig, ax = plt.subplots(figsize=(10, 8))
        labels = [f[:30] for f in ui_input_names]
        y = np.arange(len(labels))
        ax.barh(y-0.2, [latest_data_proc.loc[c1].get(f + "_lag2", 0) for f in ui_input_names], 0.4, label=c1, color="#6fa8dc")
        ax.barh(y+0.2, [latest_data_proc.loc[c2].get(f + "_lag2", 0) for f in ui_input_names], 0.4, label=c2, color="#cbd5e1")
        ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=9); ax.legend()
        st.pyplot(fig)

# --- SEKME 4: SHAP ---
with t4:
    st.markdown("### " + ("Model Açıklanabilirliği" if lang=="tr" else "Model Explainability"))
    d4 = st.selectbox("Ülke Seç", country_list, key="sh_c")
    if st.button("SHAP Analizi", key="sh_btn"):
        row_proc = latest_data_proc.loc[d4]
        model_input = pd.DataFrame([row_proc[model_features].values], columns=model_features)
        explainer = shap.Explainer(model)
        shap_values = explainer(model_input)
        fig = plt.figure(); shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.tight_layout(); st.pyplot(fig)

# --- SEKME 5: TRENDLER ---
with t5:
    st.markdown("### " + ("Tarihsel Trend" if lang=="tr" else "Historical Trend"))
    d5 = st.selectbox("Ülke Seç", country_list, key="tr_c")
    feat_tr = st.selectbox("Değişken Seç", ["GII Skoru"] + trend_candidates)
    if st.button("Trendi Çiz"):
        c_data = df_raw[df_raw[country_col] == d5].sort_values(year_col)
        target_col = gii_col_exact[0] if "GII" in feat_tr else feat_tr
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.plot(c_data[year_col], c_data[target_col], marker='o', color='#6fa8dc')
        st.pyplot(fig)

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>2025 Strategic Forecast System</p>", unsafe_allow_html=True)
