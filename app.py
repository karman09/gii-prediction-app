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
import io
import tempfile 
from fpdf import FPDF # terminalde: pip install fpdf2

st.set_page_config(page_title="D-LOGII Dashboard", page_icon="📊", layout="wide")

# ============================================================
# PROFESYONEL TEMA VE ARAYÜZ CSS ENJEKSİYONU
# ============================================================
custom_css = """
<style>
/* Ana Başlık (Canlı ve Gradient Mavi Ton) */
.main-title {
    text-align: center;
    background: -webkit-linear-gradient(45deg, #1A5276, #5DADE2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
    font-size: 2.8em;
    margin-bottom: 20px;
    letter-spacing: 1px;
}

/* Metrik Kutuları (Skorlar ve Farklar İçin Belirgin Yapı) */
div[data-testid="stMetric"] {
    background-color: #f0f7ff;
    border: 1px solid #cce3ff;
    padding: 15px 20px;
    border-radius: 10px;
    box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
    transition: transform 0.2s ease-in-out;
}
div[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 6px 10px rgba(0, 0, 0, 0.1);
}
div[data-testid="stMetric"] label {
    color: #00509e !important;
    font-weight: 700;
    font-size: 1.1em;
}

/* Sekme (Tab) Başlıklarını Daha Belirgin Mavi Yapma */
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 16px;
    font-weight: 600;
    color: #2874A6;
}

/* Genişletilebilir Paneller (Expander) */
div[data-testid="stExpander"] {
    border: 1px solid #b3d4ff;
    border-radius: 8px;
    background-color: #fafcff;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================
# ============================================================
# PDF OLUŞTURMA YARDIMCI FONKSİYONU (KESİN ÇÖZÜM)
# ============================================================
def generate_pdf_report(title, text_content="", fig=None):
    """Metin ve grafikleri alıp PDF byte verisine çeviren yardımcı fonksiyon."""
    # Türkçe karakterleri standart İngilizce karakterlere dönüştüren harita
    tr_map = str.maketrans("ğüşiöçĞÜŞİÖÇıI", "gusiocGUSIOCiI")
    
    pdf = FPDF()
    pdf.add_page()
    
    # Başlık
    pdf.set_font("Helvetica", 'B', 16)
    safe_title = str(title).translate(tr_map).encode('latin-1', 'ignore').decode('latin-1')
    pdf.cell(0, 10, txt=safe_title, ln=True, align='C')
    pdf.ln(8)
    
    # Metin
    if text_content:
        pdf.set_font("Helvetica", size=11)
        safe_text = str(text_content).translate(tr_map).encode('latin-1', 'ignore').decode('latin-1')
        
        # multi_cell yerine doğrudan write kullanıyoruz.
        # write fonksiyonu metni akıtır, uzun kelimelerde veya boşluklarda çökmez.
        pdf.write(h=8, txt=safe_text)
        pdf.ln(10) # Metin bitince grafikle arasına boşluk bırak
        
    # Grafik
    if fig:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, format='png', bbox_inches='tight')
            pdf.image(tmpfile.name, w=170)
            
    return bytes(pdf.output())

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
# 3. CORE HESAPLAMA MANTIĞI
# ============================================================
def get_raw_values(country_name):
    if country_name not in latest_data_raw.index: return [0.0] * len(ui_input_names)
    row = latest_data_raw.loc[country_name]
    return [float(row.get(col, 0.0) if not pd.isna(row.get(col, 0.0)) else 0.0) for col in ui_input_names]

def calculate_score_engine(country_name, raw_inputs_list):
    """Analizler için ortak hesaplama motoru"""
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
    st.markdown("<h1 class='main-title'>GII 2025 Tahmin ve Karar Destek Sistemi</h1>", unsafe_allow_html=True)
else:
    st.markdown("<h1 class='main-title'>GII 2025 Forecast & Decision Support System</h1>", unsafe_allow_html=True)

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
        
# --- SEKMELER (SIRA GÜNCELLENDİ: 1 ve 5 YER DEĞİŞTİRDİ) ---
t1, t2, t3, t4, t5 = st.tabs([
    "Senaryo Simülatörü", "Karşılaştırmalı Analiz", "Hedef ve SHAP", "Trend Analizi", "Duyarlılık Analizi"
] if lang=="tr" else [
    "Scenario Simulator", "Comparative Analysis", "Target & SHAP", "Trend Analysis", "Sensitivity Analysis"
])

# ============================================================
# SEKME 1: SENARYO SİMÜLATÖRÜ (ESKİ SEKME 5)
# ============================================================
with t1:
    st.markdown("### " + ("GII 2025 Senaryo Simülatörü" if lang=="tr" else "GII 2025 Scenario Simulator"))
    
    st.info("💡 " + (
        "Bu bölümde, seçilen ülkenin 2023 ham verilerini (baz değerler) görebilir ve bu değerleri değiştirerek "
        "2025 tahmin skorunun nasıl etkilendiğini anlık olarak simüle edebilirsiniz."
        if lang=="tr" else 
        "In this section, you can view the 2023 raw data (baseline) for the selected country and simulate "
        "how the 2025 forecast score is affected by changing these values in real-time."
    ))

    # Ülke Seçimi
    sim_country = st.selectbox("Simülasyon İçin Ülke Seç / Select Country", country_list, key="sim_country_box")
    
    # Seçilen ülkenin 2023 ham verilerini çek
    base_raw_values = get_raw_values(sim_country)
    
    # Başlangıç (Orijinal) 2025 Tahmini
    base_pred, _ = calculate_score_engine(sim_country, base_raw_values)
    
    st.divider()
    
    # Sol kolon: Değişken girişleri | Sağ kolon: Sonuç metrikleri
    col_input, col_res = st.columns([2, 1])
    
    new_sim_values = []
    
    with col_input:
        st.subheader("📝 " + ("Değişken Ham Değerleri" if lang=="tr" else "Raw Variable Values"))
        # Değişkenleri 2 alt kolona bölerek daha derli toplu gösterelim
        sub_c1, sub_c2 = st.columns(2)
        
        for i, feat_name in enumerate(ui_input_names):
            current_val = float(base_raw_values[i])
            target_sub_col = sub_c1 if i % 2 == 0 else sub_c2
            
            # Kullanıcıya ham değerleri değiştirme imkanı tanıyan input
            u_val = target_sub_col.number_input(
                label=f"{feat_name}",
                value=current_val,
                format="%.3f",
                key=f"input_{sim_country}_{i}" # Ülke değiştikçe key değişmeli
            )
            new_sim_values.append(u_val)

    with col_res:
        st.subheader("🎯 " + ("Simülasyon Çıktısı" if lang=="tr" else "Simulation Output"))
        
        # Yeni değerlerle anlık hesaplama
        sim_pred, _ = calculate_score_engine(sim_country, new_sim_values)
        diff = sim_pred - base_pred
        
        # Skor Kartı
        st.metric(
            label=f"{TARGET_YEAR} " + ("Simüle Edilen Skor" if lang=="tr" else "Simulated Score"),
            value=f"{sim_pred:.2f}",
            delta=f"{diff:.2f} " + ("Puan Değişimi" if lang=="tr" else "Point Change")
        )
        
        st.write("---")
        st.write(f"**{sim_country} ({INPUT_YEAR})** " + ("Orijinal Tahmin:" if lang=="tr" else "Original Forecast:"))
        st.success(f"**{base_pred:.2f}**")
        
        # Özet Analiz
        if abs(diff) > 0.01:
            direction = "artış" if diff > 0 else "düşüş"
            if lang == "en": direction = "increase" if diff > 0 else "decrease"
            st.warning(f"⚠️ " + (
                f"Yapılan değişiklikler skorda {abs(diff):.2f} puanlık bir {direction} yaratıyor."
                if lang=="tr" else
                f"The changes made result in a {abs(diff):.2f} point {direction} in the score."
            ))
        
        if st.button("Değerleri Sıfırla / Reset Values" if lang=="tr" else "Reset Values"):
            # Mevcut simülasyon ülkesine ait tüm input key'lerini temizle
            for key in list(st.session_state.keys()):
                if key.startswith(f"input_{sim_country}_"):
                    del st.session_state[key]
            
            # Sayfayı yenile
            st.rerun()

    # Alt kısma küçük bir karşılaştırma tablosu
    with st.expander("Değişim Detaylarını Gör / See Change Details"):
        comparison_df = pd.DataFrame({
            "Değişken / Variable": ui_input_names,
            "2023 Baz (Raw)": base_raw_values,
            "Simülasyon (Raw)": new_sim_values
        })
        comparison_df["Fark / Diff"] = comparison_df["Simülasyon (Raw)"] - comparison_df["2023 Baz (Raw)"]
        changed_df = comparison_df[comparison_df["Fark / Diff"] != 0]
        st.table(changed_df)
        
        # PDF İNDİRME BUTONU
        pdf_title = "Senaryo Simulatoru Raporu" if lang=="tr" else "Scenario Simulator Report"
        pdf_text = f"Ulke: {sim_country}\nOrijinal Tahmin: {base_pred:.2f}\nSimule Edilen Skor: {sim_pred:.2f}\nFark: {diff:.2f} Puan\n\n--- Degistirilen Degiskenler ---\n"
        
        if not changed_df.empty:
            for index, row in changed_df.iterrows():
                pdf_text += f"{row['Değişken / Variable']}: {row['2023 Baz (Raw)']:.2f} -> {row['Simülasyon (Raw)']:.2f} (Fark: {row['Fark / Diff']:.2f})\n"
        else:
            pdf_text += "Herhangi bir degisiklik yapilmamistir."

        pdf_bytes = generate_pdf_report(pdf_title, pdf_text)
        st.download_button(
            label="📥 Senaryo Raporunu PDF Olarak İndir / Download Scenario PDF",
            data=pdf_bytes,
            file_name=f"Senaryo_{sim_country}.pdf",
            mime="application/pdf",
            key="dl_t1_new"
        )


# SEKME 2: KARŞILAŞTIRMALI ANALİZ
with t2:
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
        
        # PDF İNDİRME BUTONU
        pdf_title = "Karsilastirmali Analiz" if lang=="tr" else "Comparative Analysis"
        pdf_text = f"{c1} (Tahmin/Forecast: {s1:.2f}) VS {c2} (Tahmin/Forecast: {s2:.2f})"
        pdf_bytes = generate_pdf_report(pdf_title, pdf_text, fig)
        st.download_button(
            label="📥 PDF Olarak İndir / Download PDF",
            data=pdf_bytes,
            file_name=f"Karsilastirma_{c1}_{c2}.pdf",
            mime="application/pdf",
            key="dl_t2"
        )


# SEKME 3: HEDEF VE SHAP
with t3:
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
            fig_shap = plt.figure(figsize=(9, 6))
            shap.plots.waterfall(shap_values[0], max_display=10, show=False)
            plt.tight_layout()
            st.pyplot(fig_shap)
            
        # PDF İNDİRME BUTONU
        pdf_title = "Hedef ve SHAP Analizi" if lang=="tr" else "Target and SHAP Analysis"
        pdf_text = (target_text + "\n" + shap_text).replace("**", "")
        pdf_bytes = generate_pdf_report(pdf_title, pdf_text, fig_shap)
        st.download_button(
            label="📥 PDF Olarak İndir / Download PDF",
            data=pdf_bytes,
            file_name=f"SHAP_Analizi_{d4}.pdf",
            mime="application/pdf",
            key="dl_t3"
        )


# SEKME 4: TREND ANALİZİ
with t4:
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
            fig_trend, ax = plt.subplots(figsize=(9, 5))
            ax.plot(x, y, marker='o', color='#0f766e', linewidth=2.5)
            ax.set_title(f"{d5} - {feat_dropdown}")
            st.pyplot(fig_trend)
            
            # PDF İNDİRME BUTONU
            pdf_title = "Trend Analizi" if lang=="tr" else "Trend Analysis"
            pdf_text = f"Ulke/Country: {d5}\nIncelenen Degisken/Variable: {feat_dropdown}"
            pdf_bytes = generate_pdf_report(pdf_title, pdf_text, fig_trend)
            st.download_button(
                label="📥 PDF Olarak İndir / Download PDF",
                data=pdf_bytes,
                file_name=f"Trend_Analizi_{d5}.pdf",
                mime="application/pdf",
                key="dl_t4"
            )
        else:
            st.error("Veri bulunamadı.")


# ============================================================
# SEKME 5: DUYARLILIK ANALİZİ (ESKİ SEKME 1)
# ============================================================
with t5:
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
        
        # PDF İNDİRME BUTONU
        clean_report = report.replace("**", "") 
        pdf_bytes = generate_pdf_report("Duyarlilik Analizi Raporu" if lang=="tr" else "Sensitivity Analysis Report", clean_report)
        st.download_button(
            label="📥 PDF Olarak İndir / Download PDF",
            data=pdf_bytes,
            file_name=f"Duyarlilik_Analizi_{adv_country}.pdf",
            mime="application/pdf",
            key="dl_t5_new"
        )

st.markdown("---")
st.markdown(f"<p style='text-align: center; color: gray;'>{TARGET_YEAR} Strategic Decision Support System</p>", unsafe_allow_html=True)
