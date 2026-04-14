# ============================================================================== 
# GII 2025 STRATEGY DASHBOARD (FULL LOGIC - CLEAN UI VERSION)
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
# CSS INJECTION
# ============================================================
custom_css = """
<style>
.main-title { text-align: center; background: -webkit-linear-gradient(45deg, #1A5276, #5DADE2); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-weight: 900; font-size: 2.8em; margin-bottom: 20px; }
div[data-testid="stMetric"] { background-color: #f0f7ff; border: 1px solid #cce3ff; padding: 15px 20px; border-radius: 10px; }
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p { font-size: 16px; font-weight: 600; color: #2874A6; }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# ============================================================
# PDF HELPER
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
# 1. LOADING & PREP
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
        st.error(f"Error: {e}")
        st.stop()

df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features = load_system_files()

def sanitize_name(name): return re.sub(r"[^A-Za-z0-9_]", "", name)
sanitized_to_original = {sanitize_name(orig): orig for orig in scaler_cols}
country_col = [c for c in df_raw.columns if "country" in c.lower() or "economy" in c.lower()][0]
year_col = "year"
TARGET_YEAR = 2025
INPUT_YEAR = 2023

latest_data_raw = df_raw[df_raw[year_col] == INPUT_YEAR].set_index(country_col)
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
if gii_col_exact and gii_col_exact[0] in trend_candidates: trend_candidates.remove(gii_col_exact[0])
trend_features = ["GII Skoru (Gerçekleşen)" if lang=="tr" else "GII Score (Actual)"] + trend_candidates

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
                if feat_ui.lower().strip() in cost_cols: new_scaled = -new_scaled
                final_scaled_feat = new_scaled
            model_input.at[0, feature_map[feat_ui]] = final_scaled_feat
        pred = model.predict(model_input)[0]
        return max(0, min(100, pred)), model_input
    except: return 0.0, None

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
# UI RENDER
# ============================================================
title_text = "GII 2025 Tahmin ve Karar Destek Sistemi" if lang=="tr" else "GII 2025 Forecast & Decision Support System"
st.markdown(f"<h1 class='main-title'>{title_text}</h1>", unsafe_allow_html=True)

with st.expander("Metodoloji Hakkında" if lang=="tr" else "About Methodology"):
    if lang == "tr":
        st.markdown("**1. Klasik GII Hesaplaması:** WIPO tarafından belirlenen 78 farklı göstergenin ortalamasıdır. \n\n**2. Yapay Zeka Modeli:** En kritik 24 belirleyici üzerinden tahmin yapar.")
    else:
        st.markdown("**1. Classical GII Calculation:** Weighted average of 78 indicators. \n\n**2. AI Model:** Predictive analysis based on 24 critical determinants.")

tabs = ["Senaryo Simülatörü", "Karşılaştırmalı Analiz", "Hedef ve SHAP", "Trend Analizi", "Duyarlılık Analizi", "Küresel Sıralama ve Harita", "Veri Keşfi ve Korelasyon"] if lang=="tr" else ["Scenario Simulator", "Comparative Analysis", "Target & SHAP", "Trend Analysis", "Sensitivity Analysis", "Global Leaderboard & Map", "Data Explorer & Correlation"]
t1, t2, t3, t4, t5, t6, t7 = st.tabs(tabs)

# TAB 1: SCENARIO
with t1:
    st.markdown("### " + ("Senaryo Simülatörü" if lang=="tr" else "Scenario Simulator"))
    st.info("💡 " + ("Parametreleri değiştirerek 2025 öngörü skorunu analiz edin." if lang=="tr" else "Analyze 2025 forecast by modifying parameters."))
    sim_country = st.selectbox("Ülke Seç" if lang=="tr" else "Select Country", country_list, key="sim_c")
    base_raw = get_raw_values(sim_country)
    base_pred, _ = calculate_score_engine(sim_country, base_raw)
    
    st.divider()
    col_in, col_res = st.columns([2, 1])
    new_vals = []
    with col_in:
        st.subheader("📝 " + ("Ham Değerler" if lang=="tr" else "Raw Values"))
        sc1, sc2 = st.columns(2)
        for i, f_name in enumerate(ui_input_names):
            t_col = sc1 if i % 2 == 0 else sc2
            v = t_col.number_input(label=f_name, value=float(base_raw[i]), format="%.3f", key=f"in_{sim_country}_{i}")
            new_vals.append(v)
    with col_res:
        st.subheader("🎯 " + ("Çıktı" if lang=="tr" else "Output"))
        sim_pred, _ = calculate_score_engine(sim_country, new_vals)
        diff = sim_pred - base_pred
        st.metric("2025 " + ("Simüle Skor" if lang=="tr" else "Simulated Score"), f"{sim_pred:.2f}", delta=f"{diff:.2f}")
        if st.button("Sıfırla" if lang=="tr" else "Reset"): 
            for k in list(st.session_state.keys()):
                if k.startswith(f"in_{sim_country}_"): del st.session_state[k]
            st.rerun()

# TAB 2: COMPARATIVE (FIXED CHART SIZE)
with t2:
    st.markdown("### " + ("Karşılaştırmalı Analiz" if lang=="tr" else "Comparative Analysis"))
    c1_c, c2_c = st.columns(2)
    with c1_c: c1 = st.selectbox("Ülke A" if lang=="tr" else "Country A", country_list, key="b1")
    with c2_c: c2 = st.selectbox("Ülke B" if lang=="tr" else "Country B", country_list, key="b2", index=1)
    
    if st.button("Grafiği Oluştur" if lang=="tr" else "Generate Chart", type="primary"):
        v1, v2 = get_raw_values(c1), get_raw_values(c2)
        row1, row2 = latest_data_proc.loc[c1], latest_data_proc.loc[c2]
        z1, z2, lbls = [], [], [f + " (-)" if f.lower().strip() in cost_cols else f for f in ui_input_names]
        for f in ui_input_names:
            z1.append(row1[f]); z2.append(row2[f])
        
        # Grafik Küçültme Ayarları
        f_h = max(5, len(lbls) * 0.25)
        fig, ax = plt.subplots(figsize=(6, f_h))
        y_pos = np.arange(len(lbls))
        ax.barh(y_pos - 0.175, z1, 0.35, label=c1, color="#0f766e")
        ax.barh(y_pos + 0.175, z2, 0.35, label=c2, color="#64748b")
        ax.set_yticks(y_pos); ax.set_yticklabels(lbls, fontsize=7)
        ax.legend(fontsize=7); ax.axvline(0, color='black', linestyle='--')
        plt.tight_layout()
        st.pyplot(fig, use_container_width=False)
        
        pdf_b = generate_pdf_report("Comparison", f"{c1} vs {c2}", fig)
        st.download_button("📥 PDF İndir" if lang=="tr" else "📥 Download PDF", pdf_b, file_name="Karsilastirma.pdf")

# TAB 3: SHAP
with t3:
    st.markdown("### " + ("Model Açıklanabilirliği (SHAP)" if lang=="tr" else "Model Explainability (SHAP)"))
    sc, tc = st.columns([2,1])
    with sc: d4 = st.selectbox("Ülke Seç" if lang=="tr" else "Select Country", country_list, key="sh_c")
    with tc: target_s = st.number_input("Hedef Skor" if lang=="tr" else "Target Score", value=0.0)
    
    if st.button("Analizi Başlat" if lang=="tr" else "Start Analysis", key="sh_b"):
        rv = get_raw_values(d4)
        pred, m_in = calculate_score_engine(d4, rv)
        explainer = shap.Explainer(model)
        shap_v = explainer(m_in)
        fig_sh = plt.figure(figsize=(9, 6))
        shap.plots.waterfall(shap_v[0], max_display=10, show=False)
        plt.tight_layout()
        st.pyplot(fig_sh)

# TAB 4: TREND (RESTORED FULL LOGIC)
with t4:
    st.markdown("### " + ("Trend Analizi" if lang=="tr" else "Trend Analysis"))
    d5_c, f5_c = st.columns(2)
    with d5_c: d5 = st.selectbox("Ülke Seç" if lang=="tr" else "Select Country", country_list, key="tr_c")
    with f5_c: f_drop = st.selectbox("İncelenecek Değişken" if lang=="tr" else "Variable", trend_features)
    
    if st.button("Trendi Çiz" if lang=="tr" else "Plot Trend"):
        c_data = df_raw[df_raw[country_col] == d5].sort_values(by=year_col)
        act_col = None
        if f_drop in ["GII Skoru (Gerçekleşen)", "GII Score (Actual)"]:
            g_cols = [c for c in df_raw.columns if "global innovation index" in c.lower()]
            if g_cols: act_col = g_cols[0]
        else: act_col = f_drop
        
        if act_col and act_col in c_data.columns:
            fig_t, ax_t = plt.subplots(figsize=(7, 4))
            ax_t.plot(c_data[year_col].astype(int), c_data[act_col], marker='o', color='#0f766e')
            ax_t.set_title(f"{d5} - {f_drop}")
            st.pyplot(fig_t)
        else: st.error("Data Not Found")

# TAB 5: SENSITIVITY (RESTORED FULL LOGIC)
with t5:
    st.markdown("### " + ("Duyarlılık Analizi" if lang=="tr" else "Sensitivity Analysis"))
    adv_c = st.selectbox("Ülke Seç" if lang=="tr" else "Select Country", country_list, key="adv_c")
    if st.button("Analizi Başlat" if lang=="tr" else "Start Analysis", key="adv_b"):
        adv_in = get_raw_values(adv_c)
        cur_s, _ = calculate_score_engine(adv_c, adv_in)
        impacts = []
        for i, o_name in enumerate(ui_input_names):
            val = adv_in[i]
            is_cost = o_name.lower().strip() in cost_cols
            new_v = val * 0.90 if is_cost else val * 1.10
            temp = adv_in.copy()
            temp[i] = new_v
            ns, _ = calculate_score_engine(adv_c, temp)
            gain = ns - cur_s
            if gain > 0.01: impacts.append({"Feat": o_name, "Gain": gain, "Val": new_v})
        impacts.sort(key=lambda x: x["Gain"], reverse=True)
        if impacts:
            st.write("#### " + ("Stratejik Müdahale Alanları" if lang=="tr" else "Strategic Intervention Areas"))
            st.table(pd.DataFrame(impacts[:5]))
        else: st.warning("No impact detected.")

# TAB 6: MAP (RESTORED FULL LOGIC)
with t6:
    st.markdown("### " + ("Küresel Harita" if lang=="tr" else "Global Map"))
    if st.button("Yükle" if lang=="tr" else "Load", key="map_l"):
        with st.spinner("..."):
            m_data = []
            for c in country_list:
                p, _ = calculate_score_engine(c, get_raw_values(c))
                m_data.append({"Country": c, "Forecast": p})
            df_m = pd.DataFrame(m_data)
            fig_m = px.choropleth(df_m, locations="Country", locationmode="country names", color="Forecast", color_continuous_scale="Viridis")
            st.plotly_chart(fig_m, use_container_width=True)

# TAB 7: DATA
with t7:
    st.markdown("### " + ("Veri Keşfi" if lang=="tr" else "Data Explorer"))
    st.dataframe(latest_data_raw[ui_input_names])
    if st.button("Korelasyon" if lang=="tr" else "Correlation"):
        corr = latest_data_raw[ui_input_names].corr()
        st.plotly_chart(px.imshow(corr, color_continuous_scale="RdBu_r"))

# FOOTER
st.markdown("---")
st.markdown(f"<div style='text-align: center; color: gray;'>{TARGET_YEAR} Strategic Dashboard</div>", unsafe_allow_html=True)
