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

# --- TEMA VE STİL AYARLARI (Açık, Ferah ve Pastel Mavi) ---
st.markdown("""
    <style>
    /* Ana arka plan: Çok hafif buz mavisi */
    .stApp {
        background-color: #f0f4f8;
    }
    
    /* Yan Menü ve Kartlar: Saf Beyaz */
    [data-testid="stSidebar"], .stTabs, div[data-testid="stExpander"], .stAlert {
        background-color: #ffffff !important;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.02);
    }

    /* Sekme (Tab) Tasarımı - Pastel Mavi Odaklı */
    .stTabs [data-baseweb="tab-list"] {
        gap: 10px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px 8px 0 0;
        padding: 10px 20px;
        color: #64748b;
        font-weight: 500;
    }
    /* Aktif Sekme Stili */
    .stTabs [aria-selected="true"] {
        background-color: #6fa8dc !important;
        color: white !important;
        border: 1px solid #6fa8dc !important;
    }
    /* Sekme altındaki vurgu çizgisi */
    [data-baseweb="tab-border-highlight"] {
        background-color: #6fa8dc !important;
    }

    /* Buton Tasarımı */
    .stButton>button {
        background-color: #6fa8dc !important;
        color: white !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 0.5rem 1rem !important;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        opacity: 0.9;
        transform: translateY(-1px);
    }

    /* Input Alanları */
    .stNumberInput input {
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
# 3. CORE HESAPLAMA MANTIĞI (DÜZELTİLMİŞ / TUTARLI)
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
            
            # Hassasiyeti artırdık (1e-5 -> 1e-3) ve doğrudan pre-processed veriye güveniyoruz
            if math.isclose(user_val, orig_raw_val, rel_tol=1e-3):
                final_scaled_feat = row_proc[feat_ui]
            else:
                # Değer değiştiyse manuel scaler kullan
                idx = scaler_cols.index(feat_ui)
                new_scaled = (user_val - scaler.mean_[idx]) / scaler.scale_[idx]
                
                # Maliyet sütunu kontrolü (küçük/büyük harf duyarlılığı giderildi)
                is_cost = any(c.strip().lower() == feat_ui.strip().lower() for c in cost_cols)
                if is_cost: 
                    new_scaled = -new_scaled
                final_scaled_feat = new_scaled
                
            model_input.at[0, feature_map[feat_ui]] = final_scaled_feat
            
        return max(0, min(100, model.predict(model_input)[0]))
    except Exception as e:
        print(f"Hata: {e}") # Konsolda hata takibi için
        return 0.0

def get_actual_gii(country, lang):
    actual_val_str = "---"
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

# --- MENÜ VE DİL SEÇİMİ ---
lang_choice = st.sidebar.radio("Language / Dil", ["🇹🇷 Türkçe", "🇬🇧 English"])
lang = "tr" if "Türkçe" in lang_choice else "en"

# Logo ve Dinamik Başlık
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

# METODOLOJİ (22 KRİTİK DEĞİŞKEN VURGUSU)
with st.expander("Metodoloji Hakkında / About Methodology"):
    if lang == "tr":
        st.markdown("""
        **1. Klasik GII Hesaplaması (WIPO Metodolojisi):** Küresel İnovasyon Endeksi (GII), 2025 yılı itibarıyla **7 ana sütun** altında toplanan tam **78 farklı göstergenin** ağırlıklı ortalaması alınarak hesaplanır.
        
        **2. Geliştirilen Yapay Zeka Modeli (Bu Çalışma):** Bu sistem, 78 değişkenin tamamını manuel hesaplamak yerine, **tahminsel (predictive)** bir yaklaşım sunar:
        * **Kritik Değişken Seçimi:** Model, 78 gösterge arasından GII skorunu en çok etkileyen **"22 Kritik Belirleyici"** tespit etmiştir.
        * **Doğrusal Olmayan Öğrenme:** Model seçilen **22 değişken** arasındaki karmaşık ilişkileri öğrenerek sonuç üretir.
        * **Avantajı:** Sadece stratejik öneme sahip değişkenlere odaklanarak hızlı senaryo analizi sağlar.
        """)
    else:
        st.markdown("""
        **1. Classical GII Calculation (WIPO Methodology):** Calculated by weighted average of 78 indicators under 7 pillars.
        
        **2. Developed AI Model (This Study):** Instead of calculating all 78 variables, this system offers a **predictive** approach:
        * **Critical Variable Selection:** The model identified the **"22 Critical Determinants"** that most affect the GII score.
        * **Non-linear Learning:** The model learns complex relationships between these **22 variables**.
        * **Advantage:** Rapid scenario analysis by focusing on high-impact variables.
        """)

# --- SEKMELER ---
tab_names = ["Senaryo Simülatörü", "Duyarlılık Analizi", "Karşılaştırmalı Analiz", "Hedef ve SHAP", "Trend Analizi"] if lang=="tr" else ["Scenario Simulator", "Sensitivity Analysis", "Comparative Analysis", "Target & SHAP", "Trend Analysis"]
t1, t2, t3, t4, t5 = st.tabs(tab_names)

# SEKME 1: SİMÜLATÖR (Güncellenmiş Kısım)
with t1:
    st.markdown("### " + ("Senaryo Bazlı Tahmin Simülasyonu" if lang=="tr" else "Scenario-Based Prediction Simulation"))
    # ... (Bilgi mesajları aynı kalabilir)

    # Key eklendi: Ülke değişince inputlar sıfırlanacak
    country_sim = st.selectbox("Ülke Seç / Select Country", country_list, key="c_sim_selector")
    
    # Seçili ülkenin ham verilerini çek
    current_country_row = latest_data_raw.loc[country_sim]
    
    user_inputs = []
    with st.expander("Değişkenleri Düzenle / Edit Variables"):
        cols = st.columns(2)
        for i, name in enumerate(ui_input_names):
            # Ham veriyi güvenli bir şekilde al
            default_val = float(current_country_row.get(name, 0.0))
            
            with cols[i % 2]:
                # KRİTİK DÜZELTME: Key içine {country_sim} eklendi
                val = st.number_input(
                    name, 
                    value=default_val, 
                    format="%.5f", 
                    key=f"inp_{country_sim}_{name}" 
                )
                user_inputs.append(val)
                
    if st.button("Tahmini Hesapla / Calculate Forecast", type="primary", key="btn_sim_exec"):
        score = calculate_score(country_sim, user_inputs)
        actual = get_actual_gii(country_sim, lang)
        
        st.markdown("---")
        c_a, c_b = st.columns(2)
        with c_a: st.metric(label=f"{TARGET_YEAR} GII Tahmini / Forecast", value=f"{score:.2f}")
        with c_b: st.metric(label=f"{TARGET_YEAR} GII Gerçekleşen / Actual", value=actual)

# SEKME 2: DUYARLILIK ANALİZİ
with t2:
    st.markdown("### " + (f"{INPUT_YEAR} Verileri Üzerinden Etki Analizi" if lang=="tr" else f"Impact Analysis based on {INPUT_YEAR} Data"))
    if lang == "tr":
        st.info("💡 **Bu modül**, mevcut değişkenlerdeki %10'luk varsayımsal bir iyileşmenin genel skora etkisini ölçerek öncelikli alanları belirler.")
    else:
        st.info("💡 **This module** identifies priority areas by measuring the impact of a hypothetical 10% improvement on the overall score.")

    adv_country = st.selectbox("Ülke Seç / Select Country", country_list, key="adv_country")
    
    if st.button("Analizi Başlat / Start Analysis", type="primary", key="adv_btn"):
        adv_inputs = [float(latest_data_raw.loc[adv_country].get(col, 0.0)) for col in ui_input_names]
        current_score = calculate_score(adv_country, adv_inputs)
        impacts = []
        for i, orig_name in enumerate(ui_input_names):
            val = adv_inputs[i]
            is_cost = orig_name.lower().strip() in cost_cols
            new_val = val * 0.90 if is_cost else val * 1.10
            temp = adv_inputs.copy()
            temp[i] = new_val
            new_score = calculate_score(adv_country, temp)
            gain = new_score - current_score
            if not np.isnan(gain) and gain > 0.001:
                impacts.append({"Feat": orig_name, "Gain": gain})
                
        impacts.sort(key=lambda x: x["Gain"], reverse=True)
        
        st.write(f"**Baz Skor ({INPUT_YEAR}):** {current_score:.2f}")
        if impacts:
            st.success("🎯 **En Yüksek Artış Potansiyeli Olan İlk 5 Alan:**")
            for item in impacts[:5]:
                st.write(f"- **{item['Feat']}**: +{item['Gain']:.3f} puan artış potansiyeli.")
        else:
            st.warning("Belirgin bir artış tespit edilemedi.")

# SEKME 3: KARŞILAŞTIRMALI ANALİZ
with t3:
    st.markdown("### " + ("Performans Karşılaştırma Matrisi (Z-Skor)" if lang=="tr" else "Performance Comparison (Z-Score)"))
    if lang == "tr":
        st.info("💡 **Bu modül**, iki ülkenin performansını standartlaştırılmış Z-skorları üzerinden kıyaslamanızı sağlar.")
    else:
        st.info("💡 **This module** allows benchmarking two countries using standardized Z-scores.")

    c1_col, c2_col = st.columns(2)
    with c1_col: c1 = st.selectbox("Ülke A / Country A", country_list, key="bench_c1")
    with c2_col: c2 = st.selectbox("Ülke B / Country B", country_list, key="bench_c2", index=1)
    
    if st.button("Grafiği Oluştur / Generate Chart", type="primary", key="bench_btn"):
        row_proc_1, row_proc_2 = latest_data_proc.loc[c1], latest_data_proc.loc[c2]
        z1, z2, lbls = [], [], []
        for f in ui_input_names:
            lbls.append(f[:30] + ".." if len(f)>30 else f)
            z1.append(row_proc_1[f])
            z2.append(row_proc_2[f])
            
        fig, ax = plt.subplots(figsize=(10, 8))
        y = np.arange(len(lbls))
        ax.barh(y - 0.2, z1, 0.4, label=f"{c1}", color="#6fa8dc")
        ax.barh(y + 0.2, z2, 0.4, label=f"{c2}", color="#cbd5e1")
        ax.set_yticks(y); ax.set_yticklabels(lbls, fontsize=9)
        ax.legend(); ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)

# SEKME 4: SHAP ANALİZİ
with t4:
    st.markdown("### " + ("Model Açıklanabilirliği (XAI)" if lang=="tr" else "Model Explainability (XAI)"))
    if lang == "tr":
        st.info("💡 **Bu modül**, tahminin hangi değişkenlerden ne kadar etkilendiğini (SHAP) gösterir.")
    else:
        st.info("💡 **This module** shows how each variable contributes to the forecast (SHAP).")

    shap_c, target_c = st.columns([2,1])
    with shap_c: d4 = st.selectbox("Ülke Seç / Select Country", country_list, key="shap_c")
    with target_c: target_score = st.number_input("Hedeflenen GII Skoru / Targeted Score", value=0.0)
    
    if st.button("Analizi Başlat / Start Analysis", type="primary", key="shap_btn"):
        row_proc = latest_data_proc.loc[d4]
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)
        for feat_ui in ui_input_names: model_input.at[0, feature_map[feat_ui]] = row_proc[feat_ui]
        
        pred = model.predict(model_input)[0]
        actual_val = get_actual_gii(d4, lang)
        
        col_res, col_plt = st.columns([1,2])
        with col_res:
            st.metric("Model Tahmini / Forecast", f"{pred:.2f}")
            st.metric("Gerçekleşen / Actual", actual_val)
            if target_score > 0:
                gap = target_score - pred
                st.write(f"🎯 **Hedef Farkı:** {gap:.2f}")

        explainer = shap.Explainer(model)
        shap_values = explainer(model_input)
        fig = plt.figure()
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.tight_layout()
        with col_plt: st.pyplot(fig)

# SEKME 5: TRENDLER
with t5:
    st.markdown("### " + ("Tarihsel Trend Analizi" if lang=="tr" else "Historical Trend Analysis"))
    d5_c, f5_c = st.columns(2)
    with d5_c: d5 = st.selectbox("Ülke Seç / Select Country", country_list, key="trend_c")
    with f5_c: feat_dropdown = st.selectbox("Değişken Seç / Select Variable", trend_features_tr if lang=="tr" else trend_features_en)
    
    if st.button("Trendi Çiz / Plot Trend", type="primary", key="trend_btn"):
        country_data = df_raw[df_raw[country_col] == d5].copy().sort_values(by=year_col)
        actual_col = feat_dropdown if feat_dropdown not in ["GII Skoru (Gerçekleşen)", "GII Score (Actual)"] else [c for c in df_raw.columns if "global innovation index" in c.lower()][0]
        
        if actual_col in country_data.columns:
            fig, ax = plt.subplots(figsize=(8, 4))
            ax.plot(country_data[year_col], country_data[actual_col], marker='o', color='#6fa8dc', linewidth=2)
            ax.set_title(f"{d5} - {feat_dropdown}")
            st.pyplot(fig)

# --- FOOTER ---
st.markdown("---")
footer_text = "2025 Stratejik Tahmin ve Karar Destek Sistemi" if lang=="tr" else "2025 Strategic Forecast & Decision Support System"
st.markdown(f"<p style='text-align: center; color: gray;'>{footer_text}</p>", unsafe_allow_html=True)
