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
        st.info("💡 **Bu modül**, mevcut değişkenlerdeki %10'luk varsayımsal bir iyileşmenin veya kötüleşmenin genel skora etkisini otomatik ölçerek, politika yapıcılar için öncelikli müdahale alanlarını belirler.")
    else:
        st.info("💡 **This module** automatically identifies priority intervention areas for policymakers by measuring the impact of a hypothetical 10% improvement or deterioration in current variables on the overall score.")

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
            report = f"**Analiz Edilen Ülke:** {adv_country}\n\n**Baz Yıl ({INPUT_YEAR}) Tahmini:** {current_score:.2f}\n\n---\n\n"
            if not impacts: report += "Değişkenlerde %10'luk değişim belirgin fark yaratmadı."
            else:
                report += f"**{TARGET_YEAR} SKORU İÇİN ÖNCELİKLİ ALANLAR:**\n\n"
                for item in impacts[:5]: report += f"- **[{item['Feat']}]** -> %10 {item['Act']}\n  - Yeni Değer: {item['Val']:.2f} | Beklenen Artış: **+{item['Gain']:.3f} puan**\n"
        else:
            report = f"**Analyzed Country:** {adv_country}\n\n**Base Year ({INPUT_YEAR}) Forecast:** {current_score:.2f}\n\n---\n\n"
            if not impacts: report += "A 10% change in variables did not make a significant difference."
            else:
                report += f"**PRIORITY AREAS FOR {TARGET_YEAR} SCORE:**\n\n"
                for item in impacts[:5]: report += f"- **[{item['Feat']}]** -> 10% {item['Act']}\n  - New Value: {item['Val']:.2f} | Expected Gain: **+{item['Gain']:.3f} points**\n"
        st.markdown(report)

# SEKME 3: KARŞILAŞTIRMALI ANALİZ
with t3:
    st.markdown("### " + ("Standardize Edilmiş Performans Matrisi (Z-Skor)" if lang=="tr" else "Standardized Performance Matrix (Z-Score)"))
    
    if lang == "tr":
        st.info("💡 **Bu modül**, iki farklı ülkenin kritik göstergelerdeki performansını standartlaştırılmış Z-skorları üzerinden görselleştirerek doğrudan kıyaslamanızı sağlar.")
    else:
        st.info("💡 **This module** allows you to benchmark the performance of two different countries across critical indicators by visualizing their standardized Z-scores.")

    c1_col, c2_col = st.columns(2)
    with c1_col: c1 = st.selectbox("Ülke A / Country A", country_list, key="bench_c1")
    with c2_col: c2 = st.selectbox("Ülke B / Country B", country_list, key="bench_c2", index=1)
    
    if st.button("Grafiği Oluştur / Generate Chart", type="primary", key="bench_btn"):
        v1, v2 = get_raw_values(c1), get_raw_values(c2)
        s1, s2 = calculate_score(c1, v1), calculate_score(c2, v2)
        row_proc_1, row_proc_2 = latest_data_proc.loc[c1], latest_data_proc.loc[c2]
        
        z1, z2, lbls = [], [], []
        for f in ui_input_names:
            lbls.append(f + " (-)" if f.lower().strip() in cost_cols else f)
            z1.append(row_proc_1[f])
            z2.append(row_proc_2[f])
            
        fig_height = max(6, len(lbls) * 0.4)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        y = np.arange(len(lbls))
        ax.barh(y - 0.175, z1, 0.35, label=f"{c1}", color="#0f766e", alpha=0.9, edgecolor='none')
        ax.barh(y + 0.175, z2, 0.35, label=f"{c2}", color="#64748b", alpha=0.9, edgecolor='none')
        ax.set_yticks(y); ax.set_yticklabels(lbls, fontsize=10, color='#334155')
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#cbd5e1'); ax.tick_params(axis='both', which='both', length=0)
        ax.legend(fontsize=10, frameon=False, loc='lower center', bbox_to_anchor=(0.5, 1.02), ncol=2)
        ax.grid(axis='x', color='#f1f5f9', linestyle='-', linewidth=1, alpha=0.8); ax.axvline(0, color='#94a3b8', linewidth=1.5, linestyle='--')
        
        if lang == "tr":
            ax.set_xlabel("Standartlaştırılmış Performans (Z-Skor)\n(0 = Küresel Ortalama | Sağ Taraf = Daha İyi Performans)", fontsize=10, color='#64748b')
            ax.set_title(f"Performans Karşılaştırma Matrisi: {c1} vs {c2}", fontsize=13, fontweight='bold', color='#0f172a', pad=30)
            st.success(f"{c1}: {s1:.2f} | {c2}: {s2:.2f}\nFark: {abs(s1-s2):.2f}")
        else:
            ax.set_xlabel("Standardized Performance (Z-Score)\n(0 = Global Average | Right Side = Better Performance)", fontsize=10, color='#64748b')
            ax.set_title(f"Performance Comparison Matrix: {c1} vs {c2}", fontsize=13, fontweight='bold', color='#0f172a', pad=30)
            st.success(f"{c1}: {s1:.2f} | {c2}: {s2:.2f}\nDifference: {abs(s1-s2):.2f}")
            
        plt.tight_layout()
        st.pyplot(fig)

# SEKME 4: SHAP ANALİZİ
with t4:
    st.markdown("### " + ("Model Açıklanabilirliği (XAI) ve Stratejik Hedef Takibi" if lang=="tr" else "Model Explainability (XAI) and Strategic Target Tracking"))
    
    if lang == "tr":
        st.info("💡 **Bu modül**, belirlediğiniz hedef skor ile makine öğrenmesi tahmini arasındaki farkı hesaplar ve Açıklanabilir Yapay Zeka (SHAP) grafikleriyle ülkenin en güçlü yönlerini ve acil gelişim alanlarını listeler.")
    else:
        st.info("💡 **This module** calculates the gap between your target score and the AI forecast, using Explainable AI (SHAP) charts to list the country's main strengths and urgent areas for improvement.")

    shap_c, target_c = st.columns([2,1])
    with shap_c: d4 = st.selectbox("Ülke Seç / Select Country", country_list, key="shap_c")
    with target_c: target_score = st.number_input("Hedeflenen GII Skoru / Targeted Score" if lang=="tr" else "Targeted GII Score", value=0.0)
    
    if st.button("Analizi Başlat / Start Analysis", type="primary", key="shap_btn"):
        row_proc = latest_data_proc.loc[d4]
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)
        for feat_ui in ui_input_names: model_input.at[0, feature_map[feat_ui]] = row_proc[feat_ui]
        
        pred = model.predict(model_input)[0]
        pred_clamped = max(0, min(100, pred))
        actual_val_str = get_actual_gii(d4, lang)
        
        if lang == "tr":
            target_text = f"**Analiz Edilen Ülke:** {d4}\n\n**{TARGET_YEAR} GII Tahmini:** {pred_clamped:.2f}\n\n**{TARGET_YEAR} Gerçekleşen:** {actual_val_str}\n\n---\n"
            if target_score > 0:
                gap = target_score - pred_clamped
                target_text += f"🎯 **Belirlenen Hedef:** {target_score:.2f}\n\n"
                if gap > 0: target_text += f"📉 **Fark:** {gap:.2f} Puan\n\n💡 Öneri: Aşağıdaki 'Gelişim Alanları'na odaklanın."
                else: target_text += "🎉 Durum: Mevcut tahmin, hedeflenen değerin üzerinde!"
            else: target_text += "\n💡 Fark analizi için yukarıdan bir hedef skor giriniz."
        else:
            target_text = f"**Analyzed Country:** {d4}\n\n**{TARGET_YEAR} GII Forecast:** {pred_clamped:.2f}\n\n**{TARGET_YEAR} Actual:** {actual_val_str}\n\n---\n"
            if target_score > 0:
                gap = target_score - pred_clamped
                target_text += f"🎯 **Set Target:** {target_score:.2f}\n\n"
                if gap > 0: target_text += f"📉 **Gap:** {gap:.2f} Points\n\n💡 Suggestion: Focus on 'Areas for Improvement' below."
                else: target_text += "🎉 Status: Current forecast exceeds the targeted value!"
            else: target_text += "\n💡 Enter a target score above to see gap analysis."
            
        plt.clf()
        explainer = shap.Explainer(model)
        shap_values = explainer(model_input)
        impacts = list(zip(shap_values[0].feature_names, shap_values[0].values))
        pos_impacts = sorted([x for x in impacts if x[1] > 0], key=lambda x: x[1], reverse=True)
        neg_impacts = sorted([x for x in impacts if x[1] < 0], key=lambda x: x[1])
        
        shap_text = "**Spesifik İçgörüler:**\n\n" if lang=="tr" else "**Specific Insights:**\n\n"
        if pos_impacts:
            shap_text += "**Güçlü Yönler (Artıran):**\n" if lang=="tr" else "**Strengths (Increasing):**\n"
            for f, v in pos_impacts[:3]: shap_text += f"- {reverse_feature_map.get(f, f)}: +{v:.2f}\n"
        if neg_impacts:
            shap_text += "\n**Gelişim Alanları (Düşüren):**\n" if lang=="tr" else "\n**Improvement Areas (Decreasing):**\n"
            for f, v in neg_impacts[:3]: shap_text += f"- {reverse_feature_map.get(f, f)}: {v:.2f}\n"

        fig = plt.figure(figsize=(9, 5.5))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        ax = plt.gca()
        ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
        ax.spines['bottom'].set_color('#cbd5e1'); ax.tick_params(axis='y', length=0)
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_color('#334155'); item.set_fontsize(10)
        fig = plt.gcf(); plt.tight_layout()

        col_txt, col_plot = st.columns([1,2])
        with col_txt:
            st.info(target_text)
            st.warning(shap_text)
        with col_plot:
            st.pyplot(fig)

# SEKME 5: TRENDLER
with t5:
    st.markdown("### " + ("5 Yıllık Trend Analizi" if lang=="tr" else "5-Year Trend Analysis"))
    
    if lang == "tr":
        st.info("💡 **Bu modül**, seçilen ülkenin belirli bir göstergedeki (veya genel GII skorundaki) geçmiş 5 yıllık tarihsel değişimini ve trendini görselleştirir.")
    else:
        st.info("💡 **This module** visualizes the historical 5-year change and trend of a specific indicator (or overall GII score) for the selected country.")

    d5_c, f5_c = st.columns(2)
    with d5_c: d5 = st.selectbox("Ülke Seç / Select Country", country_list, key="trend_c")
    with f5_c: feat_dropdown = st.selectbox("İncelenecek Değişken / Variable to Examine", trend_features_tr if lang=="tr" else trend_features_en)
    
    if st.button("Trendi Çiz / Plot Trend", type="primary", key="trend_btn"):
        country_data = df_raw[df_raw[country_col] == d5].copy().sort_values(by=year_col)
        actual_col = None
        if feat_dropdown in ["GII Skoru (Gerçekleşen)", "GII Score (Actual)"]:
            gii_cols = [c for c in df_raw.columns if "global innovation index" in c.lower()]
            if gii_cols: actual_col = gii_cols[0]
        else: actual_col = feat_dropdown 
        
        if actual_col and actual_col in country_data.columns:
            x, y = country_data[year_col].astype(int).tolist(), country_data[actual_col].tolist()
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.plot(x, y, marker='o', color='#0f766e', linewidth=2.5, markersize=8, markerfacecolor='white', markeredgewidth=2)
            ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False); ax.spines['left'].set_visible(False)
            ax.spines['bottom'].set_color('#cbd5e1'); ax.tick_params(axis='both', which='both', length=0, labelsize=10, colors='#64748b')
            ax.grid(axis='y', color='#f1f5f9', linestyle='-', linewidth=1.5, alpha=0.8); ax.set_xticks(x)
            title = f"{d5} - {feat_dropdown} Trendi" if lang == "tr" else f"{d5} - {feat_dropdown} Trend"
            ax.set_title(title, fontsize=14, fontweight='bold', color='#0f172a', pad=20)
            y_range = max(y) - min(y) if max(y) != min(y) else max(y)
            offset = y_range * 0.05 if y_range > 0 else 0.5
            for i, v in enumerate(y):
                if not pd.isna(v): ax.text(x[i], v + offset, f"{v:.2f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color='#0f766e')
            plt.tight_layout()
            st.pyplot(fig)
        else:
            st.error("Veri bulunamadı. / Data not found.")

# --- FOOTER ---
st.markdown("---")
if lang == "tr":
    st.markdown("<p style='text-align: center; color: gray;'>2025 Stratejik Tahmin ve Karar Destek Sistemi</p>", unsafe_allow_html=True)
else:
    st.markdown("<p style='text-align: center; color: gray;'>2025 Strategic Forecast and Decision Support System</p>", unsafe_allow_html=True)















