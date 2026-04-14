# ============================================================================== 
# GII 2025 STRATEGY DASHBOARD (BILINGUAL CLEAN VERSION)
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
from fpdf import FPDF 
import plotly.express as px 

st.set_page_config(page_title="GII 2025 Strategic Prediction Interface", page_icon="📊", layout="wide")

# ============================================================
# PROFESSIONAL THEME AND UI CSS INJECTION
# ============================================================
custom_css = """
<style>
.main-title {
    text-align: center;
    background: -webkit-linear-gradient(45deg, #1A5276, #5DADE2);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 900;
    font-size: 2.8em;
    margin-bottom: 20px;
}
div[data-testid="stMetric"] {
    background-color: #f0f7ff;
    border: 1px solid #cce3ff;
    padding: 15px 20px;
    border-radius: 10px;
}
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    font-size: 16px;
    font-weight: 600;
    color: #2874A6;
}
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================
# PDF GENERATION HELPER FUNCTION
# ============================================================
def generate_pdf_report(title, text_content="", fig=None):
    tr_map = str.maketrans("ğüşiöçĞÜŞİÖÇıI", "gusiocGUSIOCiI")
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Helvetica", 'B', 16)
    safe_title = str(title).translate(tr_map).encode('latin-1', 'ignore').decode('latin-1')
    pdf.cell(0, 10, txt=safe_title, ln=True, align='C')
    pdf.ln(8)
    if text_content:
        pdf.set_font("Helvetica", size=11)
        safe_text = str(text_content).translate(tr_map).encode('latin-1', 'ignore').decode('latin-1')
        pdf.write(h=8, txt=safe_text)
        pdf.ln(10)
    if fig:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
            fig.savefig(tmpfile.name, format='png', bbox_inches='tight')
            pdf.image(tmpfile.name, w=170)
    return bytes(pdf.output())

# ============================================================
# 1. FILE PATHS AND LOADING 
# ============================================================
lang_choice = st.sidebar.radio("Language / Dil", ["🇹🇷 Türkçe", "🇬🇧 English"])
lang = "tr" if "Türkçe" in lang_choice else "en"

load_msg = "Dosyalar yükleniyor..." if lang=="tr" else "Loading files..."
@st.cache_resource(show_spinner=load_msg)
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
        err_msg = f"Dosya Yükleme Hatası: {e}" if lang=="tr" else f"File Load Error: {e}"
        st.error(err_msg)
        st.stop()

df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features = load_system_files()

# ============================================================
# 2. NAME MAPPINGS & DATA PREPARATION
# ============================================================
def sanitize_name(name):
    return re.sub(r"[^A-Za-z0-9_]", "", name)

sanitized_to_original = {sanitize_name(orig): orig for orig in scaler_cols}
country_col = [c for c in df_raw.columns if "country" in c.lower() or "economy" in c.lower()][0]
year_col = "year"

TARGET_YEAR = 2025
INPUT_YEAR  = TARGET_YEAR - 2 # 2023

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
# 3. CORE CALCULATION LOGIC
# ============================================================
def get_raw_values(country_name):
    if country_name not in latest_data_raw.index: return [0.0] * len(ui_input_names)
    row = latest_data_raw.loc[country_name]
    return [float(row.get(col, 0.0) if not pd.isna(row.get(col, 0.0)) else 0.0) for col in ui_input_names]

def calculate_score_engine(country_name, raw_inputs_list):
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
    except:
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
    except: pass
    return actual_val_str

# ============================================================
# 4. STREAMLIT INTERFACE
# ============================================================
col1, col2, col3 = st.columns([1,2,1])
with col2:
    try: st.image("logo.png", use_container_width=True)
    except: pass 

title_text = "GII 2025 Tahmin ve Karar Destek Sistemi" if lang=="tr" else "GII 2025 Forecast & Decision Support System"
st.markdown(f"<h1 class='main-title'>{title_text}</h1>", unsafe_allow_html=True)

expander_title = "Metodoloji Hakkında" if lang=="tr" else "About Methodology"
with st.expander(expander_title):
    if lang == "tr":
        st.markdown("**1. Klasik GII Hesaplaması:** 78 göstergenin ağırlıklı ortalamasıdır. \n\n**2. Yapay Zeka Modeli:** En kritik 24 değişkeni kullanarak tahmin üretir.")
    else:
        st.markdown("**1. Classical GII Calculation:** Weighted average of 78 indicators. \n\n**2. AI Model:** Generates forecasts using 24 critical variables.")

t1, t2, t3, t4, t5, t6, t7 = st.tabs([
    "Senaryo Simülatörü", "Karşılaştırmalı Analiz", "Hedef ve SHAP", "Trend Analizi", "Duyarlılık Analizi", "Küresel Sıralama ve Harita", "Veri Keşfi ve Korelasyon"
] if lang=="tr" else [
    "Scenario Simulator", "Comparative Analysis", "Target & SHAP", "Trend Analysis", "Sensitivity Analysis", "Global Leaderboard & Map", "Data Explorer & Correlation"
])

# TAB 1: SCENARIO SIMULATOR
with t1:
    st.markdown("### " + ("GII 2025 Senaryo Simülatörü" if lang=="tr" else "GII 2025 Scenario Simulator"))
    info_t1 = "Bu modül üzerinden ülke verilerini inceleyebilir ve senaryolar oluşturabilirsiniz." if lang=="tr" else "Explore country data and create scenarios in this module."
    st.info("💡 " + info_t1)

    label_country = "Simülasyon İçin Ülke Seç" if lang=="tr" else "Select Country for Simulation"
    sim_country = st.selectbox(label_country, country_list, key="sim_country_box")
    base_raw_values = get_raw_values(sim_country)
    base_pred, _ = calculate_score_engine(sim_country, base_raw_values)
    
    st.divider()
    col_input, col_res = st.columns([2, 1])
    new_sim_values = []
    
    with col_input:
        sub_title_input = "Değişken Ham Değerleri" if lang=="tr" else "Raw Variable Values"
        st.subheader("📝 " + sub_title_input)
        sub_c1, sub_c2 = st.columns(2)
        for i, feat_name in enumerate(ui_input_names):
            target_sub_col = sub_c1 if i % 2 == 0 else sub_c2
            u_val = target_sub_col.number_input(label=f"{feat_name}", value=float(base_raw_values[i]), format="%.3f", key=f"input_{sim_country}_{i}")
            new_sim_values.append(u_val)

    with col_res:
        st.subheader("🎯 " + ("Simülasyon Çıktısı" if lang=="tr" else "Simulation Output"))
        sim_pred, _ = calculate_score_engine(sim_country, new_sim_values)
        diff = sim_pred - base_pred
        label_sim_score = "Simüle Edilen Skor" if lang=="tr" else "Simulated Score"
        label_delta = "Puan Değişimi" if lang=="tr" else "Point Change"
        st.metric(label=f"{TARGET_YEAR} {label_sim_score}", value=f"{sim_pred:.2f}", delta=f"{diff:.2f} {label_delta}")
        
        st.write("---")
        orig_label = "Orijinal Tahmin:" if lang=="tr" else "Original Forecast:"
        st.write(f"**{sim_country} ({INPUT_YEAR})** {orig_label}")
        st.success(f"**{base_pred:.2f}**")
        
        reset_label = "Değerleri Sıfırla" if lang=="tr" else "Reset Values"
        if st.button(reset_label):
            for key in list(st.session_state.keys()):
                if key.startswith(f"input_{sim_country}_"): del st.session_state[key]
            st.rerun()

    expander_diff = "Değişim Detaylarını Gör" if lang=="tr" else "See Change Details"
    with st.expander(expander_diff):
        comparison_df = pd.DataFrame({"Değişken": ui_input_names, "Baz": base_raw_values, "Simülasyon": new_sim_values})
        st.table(comparison_df[comparison_df["Baz"] != comparison_df["Simülasyon"]])
        dl_label = "📥 Senaryo Raporunu İndir" if lang=="tr" else "📥 Download Scenario Report"
        pdf_bytes = generate_pdf_report("Report", "Simulation Content")
        st.download_button(label=dl_label, data=pdf_bytes, file_name=f"Senaryo_{sim_country}.pdf", mime="application/pdf")

# TAB 2: COMPARATIVE ANALYSIS
with t2:
    st.markdown("### " + ("Karşılaştırmalı Performans Profili" if lang=="tr" else "Comparative Performance Profile"))
    label_c1 = "Ülke A" if lang=="tr" else "Country A"
    label_c2 = "Ülke B" if lang=="tr" else "Country B"
    c1_col, c2_col = st.columns(2)
    with c1_col: c1 = st.selectbox(label_c1, country_list, key="bench_c1")
    with c2_col: c2 = st.selectbox(label_c2, country_list, key="bench_c2", index=1)
    
    btn_chart = "Grafiği Oluştur" if lang=="tr" else "Generate Chart"
    if st.button(btn_chart, type="primary", key="bench_btn"):
        v1, v2 = get_raw_values(c1), get_raw_values(c2)
        s1, _ = calculate_score_engine(c1, v1)
        s2, _ = calculate_score_engine(c2, v2)
        row_proc_1, row_proc_2 = latest_data_proc.loc[c1], latest_data_proc.loc[c2]
        z1, z2, lbls = [], [], [f + " (-)" if f.lower().strip() in cost_cols else f for f in ui_input_names]
        for f in ui_input_names:
            z1.append(row_proc_1[f]); z2.append(row_proc_2[f])
            
        fig_height = max(5, len(lbls) * 0.25) 
        fig, ax = plt.subplots(figsize=(6, fig_height))
        y = np.arange(len(lbls))
        ax.barh(y - 0.175, z1, 0.35, label=f"{c1}", color="#0f766e")
        ax.barh(y + 0.175, z2, 0.35, label=f"{c2}", color="#64748b")
        ax.set_yticks(y); ax.set_yticklabels(lbls, fontsize=7)
        ax.legend(fontsize=7); ax.axvline(0, color='black', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        
        pdf_dl_t2 = "📥 PDF Olarak İndir" if lang=="tr" else "📥 Download PDF"
        pdf_bytes = generate_pdf_report("Comparison", f"{c1} vs {c2}", fig)
        st.download_button(label=pdf_dl_t2, data=pdf_bytes, file_name="Karsilastirma.pdf", key="dl_t2")

# TAB 3: TARGET AND SHAP
with t3:
    st.markdown("### " + ("Model Açıklanabilirliği ve Hedef Takibi" if lang=="tr" else "Model Explainability & Target Tracking"))
    shap_c, target_c = st.columns([2,1])
    with shap_c: 
        label_select = "Ülke Seç" if lang=="tr" else "Select Country"
        d4 = st.selectbox(label_select, country_list, key="shap_c")
    with target_c: 
        label_target = "Hedeflenen GII Skoru" if lang=="tr" else "Targeted GII Score"
        target_score = st.number_input(label_target, value=0.0)
    
    btn_analysis = "Analizi Başlat" if lang=="tr" else "Start Analysis"
    if st.button(btn_analysis, type="primary", key="shap_btn"):
        raw_vals = get_raw_values(d4)
        pred, model_input = calculate_score_engine(d4, raw_vals)
        explainer = shap.Explainer(model)
        shap_values = explainer(model_input)
        fig_shap = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(shap_values[0], max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig_shap)

# TAB 4: TREND ANALYSIS
with t4:
    st.markdown("### " + ("Trend Analizi" if lang=="tr" else "Trend Analysis"))
    d5_c, f5_c = st.columns(2)
    with d5_c: 
        label_select_t4 = "Ülke Seç" if lang=="tr" else "Select Country"
        d5 = st.selectbox(label_select_t4, country_list, key="trend_c")
    with f5_c: 
        label_var = "İncelenecek Değişken" if lang=="tr" else "Variable to Examine"
        feat_dropdown = st.selectbox(label_var, trend_features_tr if lang=="tr" else trend_features_en)
    
    btn_plot = "Trendi Çiz" if lang=="tr" else "Plot Trend"
    if st.button(btn_plot, type="primary", key="trend_btn"):
        country_data = df_raw[df_raw[country_col] == d5].sort_values(by=year_col)
        st.write(f"Trend for {d5}") # Basit gösterim

# TAB 5: SENSITIVITY ANALYSIS 
with t5:
    st.markdown("### " + ("Duyarlılık Analizi" if lang=="tr" else "Sensitivity Analysis"))
    label_select_t5 = "Ülke Seç" if lang=="tr" else "Select Country"
    adv_country = st.selectbox(label_select_t5, country_list, key="adv_country")
    btn_start_t5 = "Analizi Başlat" if lang=="tr" else "Start Analysis"
    if st.button(btn_start_t5, type="primary"):
        st.write("Sensitivity calculation...")

# TAB 6: GLOBAL LEADERBOARD & MAP
with t6:
    st.markdown("### " + ("Küresel Sıralama ve Harita" if lang=="tr" else "Global Leaderboard & Map"))
    btn_load_map = "Harita ve Sıralamayı Yükle" if lang=="tr" else "Load Map & Leaderboard"
    if st.button(btn_load_map):
        st.write("Map Loading...")

# TAB 7: DATA EXPLORER
with t7:
    st.markdown("### " + ("Veri Keşfi ve Korelasyon" if lang=="tr" else "Data Explorer & Correlation"))
    btn_corr = "Korelasyon Matrisini Çiz" if lang=="tr" else "Plot Correlation Matrix"
    if st.button(btn_corr):
        st.write("Correlation Heatmap...")

# FOOTER
st.markdown("---")
footer_text = f"{TARGET_YEAR} Karar Destek Sistemi" if lang=="tr" else f"{TARGET_YEAR} Decision Support System"
st.markdown(f"<div style='text-align: center; color: gray;'>{footer_text}</div>", unsafe_allow_html=True)
