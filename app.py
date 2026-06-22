import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import time

st.set_page_config(page_title="Kaze Yuuji Simulator", layout="wide", page_icon="🌀")

# ============================================================
# DARK THEME — KAZE YUUJI SIGNATURE
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    .stApp {
        background: #0a0e17;
        background-image:
            radial-gradient(ellipse at 20% 50%, rgba(102,126,234,0.05) 0%, transparent 50%),
            radial-gradient(ellipse at 80% 50%, rgba(118,75,162,0.05) 0%, transparent 50%);
    }

    .block-container { padding: 1.5rem 3rem !important; }

    h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown {
        color: #e8eaf0 !important;
    }

    .main-header {
        text-align: center;
        padding: 1.2rem 0 0.8rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(102,126,234,0.15);
        position: relative;
    }

    .main-header h1 {
        font-size: 2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #e8eaf0 0%, #667eea 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin: 0;
        letter-spacing: -0.5px;
    }

    .main-header p {
        color: rgba(232,234,240,0.5) !important;
        font-size: 0.85rem;
        font-weight: 300;
        letter-spacing: 2px;
        text-transform: uppercase;
        margin-top: 4px;
    }

    .ky-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: #fff !important;
        padding: 3px 14px;
        border-radius: 20px;
        font-size: 0.7rem;
        font-weight: 600;
        margin-left: 10px;
        vertical-align: middle;
        letter-spacing: 0.5px;
        -webkit-text-fill-color: #fff;
    }

    .kaze-signature {
        position: fixed;
        bottom: 12px;
        right: 16px;
        background: rgba(102,126,234,0.1);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(102,126,234,0.15);
        color: rgba(232,234,240,0.6) !important;
        padding: 6px 16px;
        border-radius: 24px;
        font-size: 0.7rem;
        font-weight: 500;
        z-index: 999;
        letter-spacing: 0.5px;
    }

    .kaze-watermark {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%) rotate(-30deg);
        font-size: 140px;
        font-weight: 900;
        color: rgba(102,126,234,0.02);
        z-index: -1;
        pointer-events: none;
        white-space: nowrap;
        user-select: none;
        letter-spacing: 12px;
    }

    .metric-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 12px;
        padding: 1rem 1.2rem;
        backdrop-filter: blur(8px);
        transition: border-color 0.2s;
    }

    .metric-card:hover { border-color: rgba(102,126,234,0.25); }

    .opt-result {
        background: rgba(102,126,234,0.06);
        border: 1px solid rgba(102,126,234,0.12);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        text-align: center;
    }

    .opt-value {
        font-size: 1.1rem;
        font-weight: 700;
        color: #667eea !important;
    }

    footer { display: none !important; }
    #MainMenu { visibility: hidden; }

    .stTabs [data-baseweb="tab-list"] {
        gap: 2px;
        background: rgba(255,255,255,0.03);
        border-radius: 10px;
        padding: 3px;
        border: 1px solid rgba(255,255,255,0.06);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 0.4rem 1.2rem;
        color: rgba(232,234,240,0.5) !important;
        font-size: 0.8rem;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: linear-gradient(135deg, rgba(102,126,234,0.15), rgba(118,75,162,0.15));
        color: #e8eaf0 !important;
    }

    .stSlider [data-baseweb="slider"] {
        background: rgba(255,255,255,0.08);
    }

    .stSlider [data-baseweb="slider"] div {
        background: #667eea !important;
    }

    .stButton button {
        border-radius: 8px;
        font-weight: 600;
        font-size: 0.8rem;
        transition: all 0.2s;
    }

    .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #667eea, #764ba2) !important;
        border: none !important;
        color: #fff !important;
    }

    .stButton button[kind="primary"]:hover {
        transform: translateY(-1px);
        box-shadow: 0 4px 20px rgba(102,126,234,0.3);
    }

    .stSidebar {
        background: rgba(10,14,23,0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.04);
    }

    .stSidebar .sidebar-content {
        background: transparent !important;
    }

    .stSidebar h2, .stSidebar h3 {
        font-size: 0.85rem !important;
        font-weight: 600;
        letter-spacing: 0.5px;
        text-transform: uppercase;
        color: rgba(232,234,240,0.4) !important;
    }

    .stSidebar .stMarkdown p {
        color: rgba(232,234,240,0.75) !important;
        font-size: 0.8rem;
    }

    .stMetric {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 10px;
        padding: 0.8rem 1rem;
    }

    .stMetric label {
        color: rgba(232,234,240,0.4) !important;
        font-size: 0.7rem !important;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.4rem !important;
        font-weight: 700 !important;
        color: #e8eaf0 !important;
    }

    .stMetric [data-testid="stMetricDelta"] {
        font-size: 0.8rem !important;
    }

    .stAlert {
        background: rgba(255,255,255,0.03) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
        backdrop-filter: blur(8px);
    }

    .stAlert [data-testid="stAlertContainer"] {
        color: #e8eaf0 !important;
    }

    .stSpinner > div {
        border-color: #667eea !important;
        border-top-color: transparent !important;
    }

    hr {
        border-color: rgba(255,255,255,0.06) !important;
        margin: 1rem 0;
    }

    .caption-text {
        color: rgba(232,234,240,0.3) !important;
        font-size: 0.7rem;
        text-align: center;
        letter-spacing: 1px;
    }

    .footer-text {
        text-align: center;
        color: rgba(232,234,240,0.25) !important;
        font-size: 0.7rem;
        line-height: 1.8;
    }

    .footer-text strong {
        color: rgba(232,234,240,0.4) !important;
    }

    .footer-brand {
        color: rgba(102,126,234,0.5) !important;
    }

    .stDataFrame {
        background: transparent !important;
    }

    .stDataFrame [data-testid="stTable"] {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 10px !important;
    }
</style>
<div class="kaze-watermark">KAZE YUUJI</div>
<div class="kaze-signature">✦ Kaze Yuuji Simulator v1.0 ✦</div>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header"><h1>🌀 Nusantara Profit Simulator <span class="ky-badge">by Kaze Yuuji</span></h1><p>Interactive What-If Analysis &bull; Policy Simulation Engine</p></div>', unsafe_allow_html=True)

# ============================================================
# SESSION STATE
# ============================================================
if 'saved_scenarios' not in st.session_state:
    st.session_state.saved_scenarios = []
if 'optimizer_running' not in st.session_state:
    st.session_state.optimizer_running = False

# ============================================================
# MODEL & BASELINE (from 15b material)
# ============================================================
X_train = np.array([[5, 10], [10, 20], [15, 5], [20, 25], [25, 15]])
y_train = np.array([50, 80, 110, 90, 150])

model = LinearRegression().fit(X_train, y_train)

baseline_input = np.array([[10, 10]])
baseline_pred = model.predict(baseline_input)[0]

# ============================================================
# SIMULATION ENGINE (from 15b material)
# ============================================================
def run_simulation(new_iklan, new_diskon):
    intervention_input = np.array([[new_iklan, new_diskon]])
    prediction = model.predict(intervention_input)[0]
    delta_y = prediction - baseline_pred
    return prediction, delta_y

# ============================================================
# SIDEBAR: CONTROLLABLE VARIABLES
# ============================================================
st.sidebar.markdown("## Tuas Kebijakan")
st.sidebar.markdown("*Variabel Kontrol*")

iklan_slider = st.sidebar.slider("📺 Anggaran Iklan (Juta Rp)", 0, 50, 10, help="Biaya iklan per periode")
diskon_slider = st.sidebar.slider("🏷️ Besaran Diskon (%)", 0, 50, 10, help="Persentase diskon produk")

st.sidebar.markdown("---")
st.sidebar.markdown("### Aksi Cerdas")

if st.sidebar.button("🤖 Optimasi Keuntungan", type="primary", use_container_width=True):
    st.session_state.optimizer_running = True
else:
    st.session_state.optimizer_running = False

st.sidebar.markdown("---")
st.sidebar.markdown("### Simpan Skenario")
if st.sidebar.button("📌 Simpan skenario ini", use_container_width=True):
    st.session_state.saved_scenarios.append({
        'iklan': iklan_slider,
        'diskon': diskon_slider,
        'prediksi': run_simulation(iklan_slider, diskon_slider)[0],
        'delta': run_simulation(iklan_slider, diskon_slider)[1]
    })

st.sidebar.markdown("---")
st.sidebar.markdown("### Info")
st.sidebar.markdown(f"**Baseline:** Iklan 10 Juta, Diskon 10%")
st.sidebar.markdown(f"**Prediksi Baseline:** Rp {baseline_pred:.2f} Juta")
st.sidebar.markdown("---")
st.sidebar.markdown('<p class="caption-text">✦ Kaze Yuuji v1.0 ✦</p>', unsafe_allow_html=True)

# ============================================================
# MAIN CONTENT
# ============================================================

hasil_pred, delta = run_simulation(iklan_slider, diskon_slider)

delta_color = "normal" if delta >= 0 else "inverse"
delta_symbol = "▲" if delta >= 0 else "▼"

# --- TOP METRICS ---
col_met1, col_met2, col_met3, col_met4 = st.columns(4)
with col_met1:
    st.metric("💰 Prediksi Keuntungan", f"Rp {hasil_pred:.2f} Jt", f"{delta_symbol} {delta:.2f} Jt", delta_color=delta_color)
with col_met2:
    st.metric("📊 Baseline", f"Rp {baseline_pred:.2f} Jt", None)
with col_met3:
    efektivitas = (hasil_pred / baseline_pred) * 100 if baseline_pred > 0 else 0
    st.metric("📈 Efektivitas", f"{efektivitas:.1f}%", f"{efektivitas - 100:+.1f}%", delta_color=delta_color)
with col_met4:
    perubahan = "NAIK" if delta > 0 else ("TURUN" if delta < 0 else "STABIL")
    st.metric("🔄 Status", perubahan, f"Rp {abs(delta):.2f} Jt", delta_color=delta_color)

# --- OPTIMIZER ---
if st.session_state.optimizer_running:
    with st.spinner("Mencari kombinasi optimal..."):
        time.sleep(0.5)
        def neg_profit(x):
            return -model.predict([[x[0], x[1]]])[0]

        bounds = [(0, 50), (0, 50)]
        result = minimize(neg_profit, x0=[25, 25], bounds=bounds, method='L-BFGS-B')
        opt_iklan = round(result.x[0], 1)
        opt_diskon = round(result.x[1], 1)
        opt_profit = model.predict([[opt_iklan, opt_diskon]])[0]

        st.markdown("### Hasil Optimasi")
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            st.markdown(f'<div class="opt-result">📺 Iklan Optimal<br><span class="opt-value">{opt_iklan} Juta Rp</span></div>', unsafe_allow_html=True)
        with col_opt2:
            st.markdown(f'<div class="opt-result">🏷️ Diskon Optimal<br><span class="opt-value">{opt_diskon}%</span></div>', unsafe_allow_html=True)
        with col_opt3:
            st.markdown(f'<div class="opt-result">💰 Profit Maksimal<br><span class="opt-value">Rp {opt_profit:.2f} Juta</span></div>', unsafe_allow_html=True)

        st.markdown(
            f'<div style="background:rgba(102,126,234,0.06);border:1px solid rgba(102,126,234,0.1);border-radius:10px;padding:1rem;margin-top:0.5rem;">'
            f'💡 <strong>Rekomendasi Kaze Yuuji:</strong> Dengan anggaran iklan <strong>Rp {opt_iklan} Juta</strong> dan diskon '
            f'<strong>{opt_diskon}%</strong>, keuntungan diprediksi <strong>Rp {opt_profit:.2f} Juta</strong> '
            f'(kenaikan <strong>{(opt_profit / baseline_pred - 1) * 100:.1f}%</strong> dari baseline Rp {baseline_pred:.2f} Juta).'
            f'</div>',
            unsafe_allow_html=True
        )

# --- VISUALIZATION ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Perbandingan", "🗺️ 3D Landscape", "📉 Sensitivitas", "📋 Skenario"])

DARK_PLOTLY = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='rgba(232,234,240,0.7)', size=11),
    xaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.06)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.04)', zerolinecolor='rgba(255,255,255,0.06)'),
)

with tab1:
    st.markdown("### Perbandingan Baseline vs Intervensi")

    data_plot = pd.DataFrame({
        'Skenario': ['Baseline', 'Intervensi'],
        'Keuntungan (Juta Rp)': [baseline_pred, hasil_pred]
    })

    fig_bar = px.bar(
        data_plot, x='Skenario', y='Keuntungan (Juta Rp)',
        color='Skenario',
        text=data_plot['Keuntungan (Juta Rp)'].apply(lambda x: f'Rp {x:.2f} Jt'),
        color_discrete_map={'Baseline': 'rgba(102,126,234,0.6)', 'Intervensi': 'rgba(239,85,59,0.7)'},
    )
    fig_bar.update_layout(**DARK_PLOTLY, height=400, showlegend=False)
    fig_bar.update_traces(textposition='outside', textfont=dict(color='rgba(232,234,240,0.7)'))

    if abs(delta) > 0.01:
        fig_bar.add_annotation(
            x=1, y=hasil_pred,
            text=f"Delta: {delta_symbol} {abs(delta):.2f} Juta",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2,
            ax=0, ay=-50,
            font=dict(color='#00CC96' if delta > 0 else '#EF553B', size=13, weight=600)
        )

    st.plotly_chart(fig_bar, use_container_width=True)

    if abs(delta) > 0.01:
        insight = (f"**Analisis Kaze Yuuji:** Skenario ini memberikan dampak **{'positif' if delta > 0 else 'negatif'}** "
                   f"sebesar **Rp {abs(delta):.2f} Juta**. Dengan iklan Rp {iklan_slider} Juta dan diskon {diskon_slider}%, "
                   f"keuntungan {'naik' if delta > 0 else 'turun'} menjadi **Rp {hasil_pred:.2f} Juta**.")
    else:
        insight = "**Analisis Kaze Yuuji:** Skenario ini tidak menghasilkan perubahan signifikan terhadap baseline."
    st.info(insight)

with tab2:
    st.markdown("### 3D Profit Landscape")

    iklan_range = np.linspace(0, 50, 30)
    diskon_range = np.linspace(0, 50, 30)
    I, D = np.meshgrid(iklan_range, diskon_range)
    Z = model.predict(np.column_stack((I.ravel(), D.ravel()))).reshape(I.shape)

    fig_3d = go.Figure(data=[
        go.Surface(z=Z, x=I, y=D, colorscale='Viridis', opacity=0.85, showscale=True,
                   colorbar=dict(title="Profit (Juta Rp)", tickfont=dict(color='rgba(232,234,240,0.7)')))
    ])
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title="Iklan (Juta Rp)", gridcolor='rgba(255,255,255,0.04)', color='rgba(232,234,240,0.7)'),
            yaxis=dict(title="Diskon (%)", gridcolor='rgba(255,255,255,0.04)', color='rgba(232,234,240,0.7)'),
            zaxis=dict(title="Profit (Juta Rp)", gridcolor='rgba(255,255,255,0.04)', color='rgba(232,234,240,0.7)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=500, margin=dict(l=0, r=0, t=10, b=0)
    )

    fig_3d.add_trace(go.Scatter3d(
        x=[iklan_slider], y=[diskon_slider], z=[hasil_pred],
        mode='markers+text',
        marker=dict(size=7, color='#EF553B', symbol='diamond', line=dict(color='#fff', width=1)),
        text=['Skenario'], textposition='top center', textfont=dict(color='rgba(232,234,240,0.9)', size=11),
        name='Skenario'
    ))
    fig_3d.add_trace(go.Scatter3d(
        x=[10], y=[10], z=[baseline_pred],
        mode='markers+text',
        marker=dict(size=7, color='#FFD700', symbol='circle', line=dict(color='#fff', width=1)),
        text=['Baseline'], textposition='top center', textfont=dict(color='rgba(232,234,240,0.9)', size=11),
        name='Baseline'
    ))

    st.plotly_chart(fig_3d, use_container_width=True)
    st.markdown('<p class="caption-text">Putar grafik 3D dengan mouse untuk menjelajahi permukaan profit dari berbagai sudut</p>', unsafe_allow_html=True)

with tab3:
    st.markdown("### Analisis Sensitivitas — Cari Tuas Paling Efektif")

    step = 2
    iklan_up = run_simulation(iklan_slider + step, diskon_slider)[0]
    iklan_down = run_simulation(max(0, iklan_slider - step), diskon_slider)[0]
    diskon_up = run_simulation(iklan_slider, min(50, diskon_slider + step))[0]
    diskon_down = run_simulation(iklan_slider, max(0, diskon_slider - step))[0]

    sens_data = pd.DataFrame({
        'Variabel': ['Iklan ▲ +2', 'Iklan ▼ −2', 'Diskon ▲ +2', 'Diskon ▼ −2'],
        'Δ Profit (Juta Rp)': [
            iklan_up - hasil_pred,
            iklan_down - hasil_pred,
            diskon_up - hasil_pred,
            diskon_down - hasil_pred
        ]
    })

    fig_tornado = px.bar(
        sens_data, x='Δ Profit (Juta Rp)', y='Variabel',
        orientation='h',
        color='Δ Profit (Juta Rp)',
        color_continuous_scale=['#EF553B', '#2b2f3a', '#00CC96'],
        text=sens_data['Δ Profit (Juta Rp)'].apply(lambda x: f'{x:+.2f} Jt')
    )
    fig_tornado.update_layout(**DARK_PLOTLY, height=350, xaxis_title="Perubahan Profit (Juta Rp)")
    fig_tornado.update_traces(textposition='outside', textfont=dict(color='rgba(232,234,240,0.7)'))
    fig_tornado.update_coloraxes(showscale=False)

    st.plotly_chart(fig_tornado, use_container_width=True)

    dampak_iklan = abs(iklan_up - iklan_down)
    dampak_diskon = abs(diskon_up - diskon_down)
    variabel_sensitif = "📺 **Anggaran Iklan**" if dampak_iklan > dampak_diskon else "🏷️ **Besaran Diskon**"
    rasio = max(dampak_iklan, dampak_diskon) / min(dampak_iklan, dampak_diskon) if min(dampak_iklan, dampak_diskon) > 0 else 99

    st.warning(
        f"**Insight Kaze Yuuji:** {variabel_sensitif} adalah variabel paling sensitif — "
        f"dampaknya **{rasio:.1f}x** lebih besar. Fokuskan kebijakan pada variabel ini untuk hasil maksimal."
    )

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.markdown("#### Sensitivitas Iklan")
        fig_sens_i = px.line(
            x=np.linspace(0, 50, 20),
            y=[run_simulation(i, diskon_slider)[0] for i in np.linspace(0, 50, 20)],
            labels={'x': 'Iklan (Juta Rp)', 'y': 'Profit (Juta Rp)'}
        )
        fig_sens_i.update_layout(**DARK_PLOTLY, height=250, showlegend=False)
        fig_sens_i.update_traces(line=dict(color='#667eea', width=2.5))
        fig_sens_i.add_vline(x=iklan_slider, line_dash="dot", line_color="rgba(232,234,240,0.2)")
        st.plotly_chart(fig_sens_i, use_container_width=True)

    with col_h2:
        st.markdown("#### Sensitivitas Diskon")
        fig_sens_d = px.line(
            x=np.linspace(0, 50, 20),
            y=[run_simulation(iklan_slider, d)[0] for d in np.linspace(0, 50, 20)],
            labels={'x': 'Diskon (%)', 'y': 'Profit (Juta Rp)'}
        )
        fig_sens_d.update_layout(**DARK_PLOTLY, height=250, showlegend=False)
        fig_sens_d.update_traces(line=dict(color='#764ba2', width=2.5))
        fig_sens_d.add_vline(x=diskon_slider, line_dash="dot", line_color="rgba(232,234,240,0.2)")
        st.plotly_chart(fig_sens_d, use_container_width=True)

with tab4:
    st.markdown("### Skenario Tersimpan")

    if not st.session_state.saved_scenarios:
        st.info("Belum ada skenario tersimpan. Gunakan tombol **📌 Simpan skenario ini** di sidebar.")
    else:
        df_scenarios = pd.DataFrame(st.session_state.saved_scenarios)
        df_scenarios.index = [f"Skenario {i+1}" for i in range(len(df_scenarios))]
        df_scenarios['Status'] = df_scenarios['delta'].apply(
            lambda d: '✅ Lebih Baik' if d > 0 else ('❌ Lebih Buruk' if d < 0 else '➡️ Sama')
        )
        st.dataframe(
            df_scenarios[['iklan', 'diskon', 'prediksi', 'delta', 'Status']]
            .rename(columns={'iklan': 'Iklan (Jt)', 'diskon': 'Diskon (%)', 'prediksi': 'Profit (Jt)', 'delta': 'Δ Profit (Jt)'}),
            use_container_width=True
        )

        if len(df_scenarios) > 1:
            fig_compare = px.line(
                df_scenarios, y='prediksi', markers=True,
            )
            fig_compare.update_layout(**DARK_PLOTLY, height=350,
                yaxis_title="Keuntungan (Juta Rp)", xaxis_title="Skenario", showlegend=False)
            fig_compare.add_hline(y=baseline_pred, line_dash="dash", line_color="rgba(102,126,234,0.4)",
                                  annotation_text=f"Baseline: Rp {baseline_pred:.2f} Jt",
                                  annotation_font_color='rgba(232,234,240,0.5)')
            fig_compare.update_traces(line=dict(color='#667eea', width=2.5), marker=dict(color='#667eea', size=8))
            st.plotly_chart(fig_compare, use_container_width=True)

        if st.button("🗑️ Hapus semua skenario"):
            st.session_state.saved_scenarios = []
            st.rerun()

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
st.markdown("""
<div class="footer-text">
    ✦ <strong>Kaze Yuuji Simulator v1.0</strong> ✦<br>
    Built with Streamlit &bull; Linear Regression Engine &bull; Interactive What-If Analysis<br>
    <span class="footer-brand">&copy; 2026 Kaze Yuuji — Ciri Khas Tiada Duanya</span>
</div>
""", unsafe_allow_html=True)
