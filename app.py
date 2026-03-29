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

# Trend özellikleri hazırlığı
trend_candidates = [c for c in df_raw.columns if c != country_col and c != year_col]
gii_col_exact = [c for c in df_raw.columns if "global innovation index" in c.lower()]
if gii_col_exact and gii_col_exact[0] in trend_candidates:
    trend_candidates.remove(gii_col_exact[0])
trend_features_tr = ["GII Skoru (Gerçekleşen)"] + trend_candidates
trend_features_en = ["GII Score (Actual)"] + trend_candidates

# ============================================================
# 3. CORE HESAPLAMA MANTIĞI (SIMULATOR VE SHAP TUTARLILIĞI)
# ============================================================
def get_raw_values(country_name):
    if country_name not in latest_data_raw.index: return [0.0] * len(ui_input_names)
    row = latest_data_raw.loc[country_name]
    return [float(row.get(col, 0.0) if not pd.isna(row.get(col, 0.0)) else 0.0) for col in ui_input_names]

# KOD 2'DEN ALINAN HESAPLAMA FONKSİYONU (SEKME 1 İÇİN)
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
            
            if math.isclose(user_val, displayed_base_val, rel_tol=1e-5, abs_tol=1e-5):
                final_scaled_feat = row_proc[feat_ui]
            else:
                idx = scaler_cols.index(feat_ui)
                new_scaled = (user_val - scaler.mean_[idx]) / scaler.scale_[idx]
                if feat_ui.lower().strip() in cost_cols: new_scaled = -new_scaled
                final_scaled_feat = new_scaled
                
            model_input.at[0, feature_map[feat_ui]] = final_scaled_feat
            
        return max(0, min(100, model.predict(model_input)[0]))
    except Exception:
        return 0.0

def calculate_score_engine(country_name, raw_inputs_list):
    """Hem simülatör hem SHAP için ortak, güvenilir hesaplama motoru"""
    try:
        row_raw = latest_data_raw.loc[country_name]
        row_proc = latest_data_proc.loc[country_name]
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)
        
        for i, feat_ui in enumerate(ui_input_names):
            user_val = float(raw_inputs_list[i])
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
        return max(0, min(100, pred)), model_input
    except Exception as e:
        return 0.0, None

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
        **1. Klasik GII Hesaplaması (WIPO Metodolojisi):** Küresel İnovasyon Endeksi (GII), 2025 yılı itibarıyla **7 ana sütun** altında toplanan tam **78 farklı göstergenin** ağırlıklı ortalaması alınarak hesaplanır.
        
        **2. Geliştirilen Yapay Zeka Modeli:** Bu sistem, 78 değişkenin tamamını manuel hesaplamak yerine, **tahminsel (predictive)** bir yaklaşım sunar:
        * **Kritik Değişken Seçimi:** Model, 78 gösterge arasından GII skorunu en çok etkileyen **"22 Kritik Belirleyici"** tespit etmiştir.
        * **Doğrusal Olmayan Öğrenme:** Model seçilen **22 değişken** arasındaki karmaşık ilişkileri öğrenerek sonuç üretir.
        * **SHAP (XAI):** Tahminlerin nedenini açıklayan "Açıklanabilir Yapay Zeka" teknolojisi entegre edilmiştir.
        """)
    else:
        st.markdown("""
        **1. Classical GII Calculation (WIPO Methodology):** The GII is calculated by taking the weighted average of exactly **78 indicators** grouped under **7 main pillars**.
        
        **2. Developed AI Model:** Instead of manual calculation, this system offers a **predictive** approach:
        * **Critical Variable Selection:** The model identified **"22 Critical Determinants"** that most impact the score.
        * **Non-linear Learning:** The model produces results by learning complex relationships between the selected **22 variables**.
        * **SHAP (XAI):** Integrated "Explainable AI" to show the reasoning behind each forecast.
        """)

# --- SEKMELER ---
t1, t2, t3, t4, t5 = st.tabs(["Senaryo Simülatörü", "Duyarlılık Analizi", "Karşılaştırmalı Analiz", "Hedef ve SHAP", "Trend Analizi"] if lang=="tr" else ["Scenario Simulator", "Sensitivity Analysis", "Comparative Analysis", "Target & SHAP", "Trend Analysis"])

# SEKME 1: SİMÜLATÖR (SHAP İLE %100 SENKRONİZE EDİLMİŞ VERSİYON)
with t1:
    st.markdown("### " + ("Senaryo Bazlı Tahmin Simülasyonu" if lang=="tr" else "Scenario-Based Prediction Simulation"))
    
    if lang == "tr":
        st.info("💡 **Bu modül**, seçtiğiniz bir ülkenin mevcut gösterge değerlerini değiştirerek, yeni senaryoların 2025 GII skoru üzerindeki etkisini anında tahmin etmenizi sağlar.")
    else:
        st.info("💡 **This module** allows you to instantly forecast the impact of new scenarios on the 2025 GII score by modifying the current indicator values of a selected country.")

    country_sim = st.selectbox("Ülke Seç / Select Country" , country_list, key="c_sim")
    
    # 1. Mevcut ham değerleri al
    raw_vals = get_raw_values(country_sim)
    user_inputs = []
    
    with st.expander("Değişkenleri Düzenle / Edit Variables" if lang=="tr" else "View / Edit Variables"):
        cols = st.columns(2)
        for i, name in enumerate(ui_input_names):
            with cols[i % 2]:
                # Kullanıcı girişini tanımla
                val = st.number_input(name, value=float(raw_vals[i]), format="%.5f", key=f"inp_{name}")
                user_inputs.append(val)
                
    if st.button("Tahmini Hesapla / Calculate Forecast", type="primary"):
        try:
            # 2. Boş bir girdi DataFrame'i oluştur (Modelin tam beklediği sütun yapısında)
            model_input_final = pd.DataFrame(0.0, index=[0], columns=model_features)
            
            # 3. Manuel Scaling ve Mapping (SHAP ile aynı matematiksel yol)
            for i, feat_ui in enumerate(ui_input_names):
                u_val = float(user_inputs[i])
                
                # Scaler içindeki doğru indeksi bul
                if feat_ui in scaler_cols:
                    idx = scaler_cols.index(feat_ui)
                    
                    # Standartlaştırma: (x - mean) / std
                    scaled_val = (u_val - scaler.mean_[idx]) / scaler.scale_[idx]
                    
                    # Maliyet sütunu ise ters çevir (Modelin eğitildiği yapı)
                    if feat_ui.lower().strip() in cost_cols:
                        scaled_val = -scaled_val
                    
                    # Teknik ismi bul (Örn: 'Gösterge' -> 'Gösterge_lag2')
                    tech_feature_name = feature_map.get(feat_ui)
                    if tech_feature_name:
                        model_input_final.at[0, tech_feature_name] = scaled_val

            # 4. Tahmin Üret
            prediction_raw = model.predict(model_input_final)[0]
            final_score = max(0, min(100, prediction_raw))
            
            # 5. Sonuçları Bas
            actual_val = get_actual_gii(country_sim, lang)
            if lang == "tr":
                st.success(f"**{country_sim} İçin {TARGET_YEAR} GII Tahmini:** {final_score:.2f}\n\n**{TARGET_YEAR} GII Gerçekleşen Değeri:** {actual_val}")
            else:
                st.success(f"**{TARGET_YEAR} GII Forecast for {country_sim}:** {final_score:.2f}\n\n**{TARGET_YEAR} GII Actual Value:** {actual_val}")
            
            # (Opsiyonel) Debug: Eğer hala fark varsa konsola değerleri basar
            # print(model_input_final.values) 

        except Exception as e:
            st.error(f"Simülasyon Hatası / Simulation Error: {str(e)}")

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
        current_score, _ = calculate_score_engine(adv_country, adv_inputs)
        impacts = []
        for i, orig_name in enumerate(ui_input_names):
            val = adv_inputs[i]
            is_cost = orig_name.lower().strip() in cost_cols
            new_val = val * 0.90 if is_cost else val * 1.10
            act = "AZALTILIRSA" if lang == "tr" else "DECREASED" if is_cost else "ARTIRILIRSA" if lang == "tr" else "INCREASED"
            temp = adv_inputs.copy()
            temp[i] = new_val
            new_score, _ = calculate_score_engine(adv_country, temp)
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
        s1, _ = calculate_score_engine(c1, v1)
        s2, _ = calculate_score_engine(c2, v2)
        row_proc_1, row_proc_2 = latest_data_proc.loc[c1], latest_data_proc.loc[c2]
        
        z1, z2, lbls = [], [], []
        for f in ui_input_names:
            lbls.append(f + " (-)" if f.lower().strip() in cost_cols else f)
            z1.append(row_proc_1[f])
            z2.append(row_proc_2[f])
            
        fig_height = max(6, len(lbls) * 0.4)
        fig, ax = plt.subplots(figsize=(10, fig_height))
        y = np.arange(len(lbls))
        ax.barh(y - 0.175, z1, 0.35, label=f"{c1}", color="#0f766e", alpha=0.9)
        ax.barh(y + 0.175, z2, 0.35, label=f"{c2}", color="#64748b", alpha=0.9)
        ax.set_yticks(y); ax.set_yticklabels(lbls, fontsize=10)
        ax.legend()
        ax.axvline(0, color='black', linewidth=1, linestyle='--')
        plt.tight_layout()
        st.pyplot(fig)

# SEKME 4: HEDEF VE SHAP
with t4:
    st.markdown("### " + ("Model Açıklanabilirliği (XAI) ve Stratejik Hedef Takibi" if lang=="tr" else "Model Explainability (XAI) and Strategic Target Tracking"))
    
    st.info("💡 " + ("Bu modül, hedef skor ile tahmin arasındaki farkı hesaplar ve SHAP grafikleriyle en güçlü yönleri ve gelişim alanlarını listeler." if lang=="tr" else "This module calculates the gap between target and forecast, listing strengths and improvement areas via SHAP."))

    shap_c, target_c = st.columns([2,1])
    with shap_c: d4 = st.selectbox("Ülke Seç / Select Country", country_list, key="shap_c")
    with target_c: target_score = st.number_input("Hedeflenen GII Skoru" if lang=="tr" else "Targeted Score", value=0.0)
    
    if st.button("Analizi Başlat / Start Analysis", type="primary", key="shap_btn"):
        raw_vals_current = get_raw_values(d4)
        pred, model_input = calculate_score_engine(d4, raw_vals_current)
        actual_val_str = get_actual_gii(d4, lang)
        
        target_text = f"**Analiz Edilen Ülke:** {d4}\n\n**{TARGET_YEAR} GII Tahmini:** {pred:.2f}\n\n**{TARGET_YEAR} Gerçekleşen:** {actual_val_str}\n\n---\n"
        if target_score > 0:
            gap = target_score - pred
            target_text += f"🎯 **Hedef:** {target_score:.2f} | " + (f"📉 **Fark:** {gap:.2f} Puan" if gap > 0 else "🎉 **Durum:** Hedefin üzerindesiniz!")

        explainer = shap.Explainer(model)
        shap_values = explainer(model_input)
        impacts = list(zip(shap_values[0].feature_names, shap_values[0].values))
        pos_impacts = sorted([x for x in impacts if x[1] > 0], key=lambda x: x[1], reverse=True)
        neg_impacts = sorted([x for x in impacts if x[1] < 0], key=lambda x: x[1])
        
        shap_text = "**Spesifik İçgörüler:**\n\n" if lang=="tr" else "**Specific Insights:**\n\n"
        if pos_impacts:
            shap_text += "**Güçlü Yönler (Skoru Artıranlar):**\n" if lang=="tr" else "**Strengths (Drivers):**\n"
            for f, v in pos_impacts[:3]:
                fname = reverse_feature_map.get(f, f)
                shap_text += f"- {fname}: +{v:.2f}\n"
        if neg_impacts:
            shap_text += "\n**Gelişim Alanları (Skoru Düşürenler):**\n" if lang=="tr" else "\n**Improvement Areas:**\n"
            for f, v in neg_impacts[:3]:
                fname = reverse_feature_map.get(f, f)
                shap_text += f"- {fname}: {v:.2f}\n"

        col_txt, col_plot = st.columns([1,2])
        with col_txt:
            st.info(target_text)
            st.warning(shap_text)
        with col_plot:
            fig = plt.figure(figsize=(9, 6))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(plt.gcf())

# SEKME 5: TREND ANALİZİ
with t5:
    st.markdown("### " + ("5 Yıllık Trend Analizi" if lang=="tr" else "5-Year Trend Analysis"))
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
            ax.plot(x, y, marker='o', color='#0f766e', linewidth=2.5)
            ax.set_title(f"{d5} - {feat_dropdown}")
            st.pyplot(fig)
        else:
            st.error("Veri bulunamadı.")

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>{TARGET_YEAR} Strategic Decision Support System</p>", unsafe_allow_html=True)
