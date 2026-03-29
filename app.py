import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import re
import math
import shap

# ============================================================
# 1. SAYFA YAPISI VE KURUMSAL TEMA
# ============================================================
st.set_page_config(page_title="D-LOGII Dashboard", layout="wide")

# Teal ve Slate Renk Teması İçin CSS
st.markdown("""
    <style>
    .main { background-color: #f8fafc; }
    .stTabs [data-baseweb="tab-list"] { gap: 24px; }
    .stTabs [data-baseweb="tab"] { height: 50px; white-space: pre-wrap; background-color: #f1f5f9; border-radius: 4px 4px 0px 0px; gap: 1px; }
    .stTabs [aria-selected="true"] { background-color: #0f766e !important; color: white !important; }
    h1 { color: #0f766e; font-weight: 800; text-align: center; margin-bottom: 0px; }
    .subtitle { color: #0f172a; text-align: center; font-size: 1.2rem; margin-bottom: 2rem; font-weight: 500; }
    .footer { text-align: center; margin-top: 50px; color: #64748b; border-top: 1px solid #e2e8f0; padding-top: 20px; font-weight: bold; }
    </style>
""", unsafe_allow_html=True)

# ============================================================
# 2. VERİ VE MODEL YÜKLEME (Cache Mekanizması)
# ============================================================
@st.cache_resource
def load_all_assets():
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
        st.error(f"⚠️ Dosya yükleme hatası! Lütfen tüm dosyaların GitHub'da olduğundan emin olun. Hata: {e}")
        return None

assets = load_all_assets()
if assets:
    df_raw, df_proc, scaler, scaler_cols, cost_cols, model, model_features = assets

    # --- Ön Hazırlık ---
    def sanitize_name(name): return re.sub(r"[^A-Za-z0-9_]", "", name)
    sanitized_to_original = {sanitize_name(c): c for c in scaler_cols}

    country_col = [c for c in df_raw.columns if "country" in c.lower() or "economy" in c.lower()][0]
    year_col = "year"
    TARGET_YEAR, INPUT_YEAR = 2025, 2023 # Lag 2 ayarı

    latest_data_raw = df_raw[df_raw[year_col] == INPUT_YEAR].set_index(country_col)
    latest_data_proc = df_proc[df_proc[year_col] == INPUT_YEAR].set_index(country_col)
    country_list = sorted(latest_data_raw.index.tolist())

    feature_map = {}
    ui_input_names = []
    for feat in model_features:
        base_clean = re.sub(r"_lag\d+$", "", feat)
        if base_clean in sanitized_to_original:
            orig = sanitized_to_original[base_clean]
            feature_map[orig] = feat
            ui_input_names.append(orig)
    reverse_feature_map = {v: k for k, v in feature_map.items()}

    # --- ÇEKİRDEK HESAPLAMA MOTORU ---
    def calculate_score_engine(country_name, user_input_values):
        row_raw = latest_data_raw.loc[country_name]
        row_proc = latest_data_proc.loc[country_name]
        model_input = pd.DataFrame(0.0, index=[0], columns=model_features)

        for feat_ui in ui_input_names:
            user_val = user_input_values[feat_ui]
            base_raw_val = row_raw.get(feat_ui, np.nan)
            displayed_base = 0.0 if pd.isna(base_raw_val) else float(base_raw_val)

            # Kullanıcı değeri değiştirmediyse proc verisini kullan (Birebir aynı sonuç için)
            if math.isclose(user_val, displayed_base, rel_tol=1e-5):
                final_val = row_proc[feat_ui]
            else:
                idx = scaler_cols.index(feat_ui)
                scaled = (user_val - scaler.mean_[idx]) / scaler.scale_[idx]
                if feat_ui.lower().strip() in cost_cols: scaled = -scaled
                final_val = scaled
            
            model_input.at[0, feature_map[feat_ui]] = final_val

        pred = model.predict(model_input)[0]
        return max(0, min(100, pred))

    # ============================================================
    # 3. ARAYÜZ BİLEŞENLERİ
    # ============================================================
    st.markdown("<h1>D-LOGII</h1>", unsafe_allow_html=True)
    st.markdown("<p class='subtitle'>Dynamic Lasso-Optimized Global Innovation Index</p>", unsafe_allow_html=True)

    with st.expander("ℹ️ Metodoloji Hakkında: WIPO Standartları vs. Yapay Zeka Modeli"):
        st.write("Bu sistem, GII skorunu en çok etkileyen 'Kritik Belirleyicileri' tespit ederek tahmin üretir.")

    tab1, tab2, tab3, tab4 = st.tabs([
        "🎯 Senaryo Simülatörü", 
        "📊 Duyarlılık Analizi", 
        "⚖️ Karşılaştırmalı Analiz", 
        "🧠 SHAP Analizi"
    ])

    # --- TAB 1: SİMÜLATÖR ---
    with tab1:
        st.subheader("What-If Analizi (Senaryo Bazlı Tahmin)")
        selected_c1 = st.selectbox("Analiz Edilecek Ülke", country_list, key="c1")
        
        raw_vals = latest_data_raw.loc[selected_c1]
        user_inputs = {}
        
        with st.expander("Değişkenleri Düzenle (Ham Veriler)", expanded=False):
            col_l, col_r = st.columns(2)
            for i, name in enumerate(ui_input_names):
                val = float(raw_vals.get(name, 0.0)) if not pd.isna(raw_vals.get(name, 0.0)) else 0.0
                target_col = col_l if i % 2 == 0 else col_r
                user_inputs[name] = target_col.number_input(name, value=val, key=f"sim_{name}")

        if st.button("Hesapla", type="primary"):
            res = calculate_score_engine(selected_c1, user_inputs)
            st.metric(label=f"{TARGET_YEAR} Tahmini GII Skoru", value=f"{res:.2f}")
            
            # Gerçekleşen Değer Kontrolü
            mask = (df_raw[country_col] == selected_c1) & (df_raw[year_col] == TARGET_YEAR)
            if mask.any():
                gii_col = [c for c in df_raw.columns if "global innovation index" in c.lower()]
                if gii_col:
                    act = df_raw.loc[mask, gii_col[0]].values[0]
                    st.write(f"**{TARGET_YEAR} Gerçekleşen Değeri:** {act:.2f}")

    # --- TAB 2: DUYARLILIK ---
    with tab2:
        st.subheader("Öncelikli Gelişim Alanları Raporu")
        selected_c2 = st.selectbox("Ülke Seç", country_list, key="c2")
        if st.button("Analizi Başlat"):
            current_inputs = {n: float(latest_data_raw.loc[selected_c2].get(n, 0.0)) for n in ui_input_names}
            base_s = calculate_score_engine(selected_c2, current_inputs)
            
            impacts = []
            for name in ui_input_names:
                is_cost = name.lower().strip() in cost_cols
                new_val = current_inputs[name] * (0.9 if is_cost else 1.1)
                temp = current_inputs.copy()
                temp[name] = new_val
                gain = calculate_score_engine(selected_c2, temp) - base_s
                if gain > 0.01:
                    impacts.append((name, gain, "AZALTILIRSA" if is_cost else "ARTIRILIRSA", new_val))
            
            impacts.sort(key=lambda x: x[1], reverse=True)
            for n, g, a, v in impacts[:5]:
                st.info(f"**[{n}]** %10 {a} -> **+{g:.3f}** puan artış (Hedef Değer: {v:.2f})")

    # --- TAB 3: KIYASLAMA ---
    with tab3:
        st.subheader("Performans Karşılaştırma Matrisi (Z-Skor)")
        ca, cb = st.columns(2)
        u1 = ca.selectbox("Ülke A", country_list, index=0)
        u2 = cb.selectbox("Ülke B", country_list, index=1)
        
        if st.button("Grafiği Oluştur"):
            row1, row2 = latest_data_proc.loc[u1], latest_data_proc.loc[u2]
            lbls = [f + " (-)" if f.lower().strip() in cost_cols else f for f in ui_input_names]
            
            fig, ax = plt.subplots(figsize=(10, len(lbls)*0.4))
            y = np.arange(len(lbls))
            ax.barh(y - 0.2, [row1[f] for f in ui_input_names], 0.4, label=u1, color="#0f766e")
            ax.barh(y + 0.2, [row2[f] for f in ui_input_names], 0.4, label=u2, color="#64748b")
            ax.set_yticks(y); ax.set_yticklabels(lbls, fontsize=9)
            ax.axvline(0, color='black', linewidth=0.8, linestyle='--')
            ax.legend(); st.pyplot(fig)

    # --- TAB 4: SHAP ---
    with tab4:
        st.subheader("Yapay Zeka Karar Açıklanabilirliği")
        selected_c4 = st.selectbox("Ülke Seç", country_list, key="c4")
        if st.button("SHAP Analizini Görüntüle"):
            row_p = latest_data_proc.loc[selected_c4]
            input_df = pd.DataFrame([row_p[ui_input_names].values], columns=model_features)
            
            explainer = shap.Explainer(model)
            shap_v = explainer(input_df)
            
            st.write("Kırmızı: Skoru artıranlar | Mavi: Skoru düşürenler")
            fig, ax = plt.subplots()
            shap.plots.waterfall(shap_v[0], max_display=10, show=False)
            st.pyplot(plt.gcf())

    st.markdown("<div class='footer'>2025 Stratejik Tahmin ve Karar Destek Sistemi</div>", unsafe_allow_html=True)
