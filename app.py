import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import time

st.set_page_config(page_title="Nusantara Profit Simulator", layout="wide", page_icon="📈")

# ============================================================
# PROFESSIONAL DARK THEME
# ============================================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }

    .stApp {
        background: #0b0f1a;
        background-image: radial-gradient(ellipse at 20% 50%, rgba(59,130,246,0.04) 0%, transparent 50%),
                          radial-gradient(ellipse at 80% 50%, rgba(99,102,241,0.03) 0%, transparent 50%);
    }

    .block-container { padding: 1.5rem 2.5rem !important; }

    h1, h2, h3, h4, h5, h6, p, span, div, label, .stMarkdown { color: #e2e8f0 !important; }

    .main-header {
        padding: 0.8rem 0 0.6rem 0;
        margin-bottom: 1.5rem;
        border-bottom: 1px solid rgba(255,255,255,0.05);
    }

    .main-header h1 {
        font-size: 1.5rem;
        font-weight: 600;
        color: #f1f5f9 !important;
        margin: 0;
        letter-spacing: -0.3px;
    }

    .main-header p {
        color: rgba(226,232,240,0.35) !important;
        font-size: 0.75rem;
        font-weight: 400;
        margin-top: 2px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 1px;
        background: rgba(255,255,255,0.02);
        border-radius: 8px;
        padding: 2px;
        border: 1px solid rgba(255,255,255,0.05);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 6px;
        padding: 0.3rem 1rem;
        color: rgba(226,232,240,0.4) !important;
        font-size: 0.75rem;
        font-weight: 500;
    }

    .stTabs [data-baseweb="tab"][aria-selected="true"] {
        background: rgba(59,130,246,0.12);
        color: #e2e8f0 !important;
    }

    .stSidebar {
        background: rgba(11,15,26,0.96) !important;
        border-right: 1px solid rgba(255,255,255,0.03);
    }

    .stSidebar h2, .stSidebar h3 {
        font-size: 0.7rem !important;
        font-weight: 600;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        color: rgba(226,232,240,0.3) !important;
    }

    .stSidebar .stMarkdown p {
        color: rgba(226,232,240,0.7) !important;
        font-size: 0.8rem;
    }

    .stSlider [data-baseweb="slider"] { background: rgba(255,255,255,0.06); }
    .slider-ads .stSlider [data-baseweb="slider"] div { background: #f97316 !important; }
    .slider-disc .stSlider [data-baseweb="slider"] div { background: #8b5cf6 !important; }

    .stButton button {
        border-radius: 6px;
        font-weight: 500;
        font-size: 0.75rem;
        transition: all 0.15s;
    }

    .stButton button[kind="primary"] {
        background: #3b82f6 !important;
        border: none !important;
        color: #fff !important;
    }

    .stButton button[kind="primary"]:hover {
        background: #2563eb !important;
        box-shadow: 0 2px 12px rgba(59,130,246,0.25);
    }

    .stMetric {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 8px;
        padding: 0.6rem 0.8rem;
    }

    .stMetric label {
        color: rgba(226,232,240,0.35) !important;
        font-size: 0.65rem !important;
        font-weight: 500;
        letter-spacing: 0.5px;
    }

    .stMetric [data-testid="stMetricValue"] {
        font-size: 1.3rem !important;
        font-weight: 600 !important;
        color: #f1f5f9 !important;
    }

    .stMetric [data-testid="stMetricDelta"] {
        font-size: 0.75rem !important;
    }

    .stAlert {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.05) !important;
        border-radius: 8px !important;
    }

    .stInfo { border-left-color: #3b82f6 !important; }
    .stWarning { border-left-color: #f59e0b !important; }

    .stSpinner > div {
        border-color: #3b82f6 !important;
        border-top-color: transparent !important;
    }

    hr { border-color: rgba(255,255,255,0.04) !important; margin: 1rem 0; }

    footer { display: none !important; }
    #MainMenu { visibility: hidden; }

    .opt-card {
        background: rgba(59,130,246,0.05);
        border: 1px solid rgba(59,130,246,0.1);
        border-radius: 8px;
        padding: 0.75rem 1rem;
        text-align: center;
    }

    .opt-card .label {
        font-size: 0.65rem;
        color: rgba(226,232,240,0.4) !important;
        letter-spacing: 0.5px;
    }

    .opt-card .value {
        font-size: 1.1rem;
        font-weight: 600;
        color: #60a5fa !important;
    }

    .rec-card {
        background: rgba(59,130,246,0.04);
        border: 1px solid rgba(59,130,246,0.08);
        border-radius: 8px;
        padding: 1rem;
        margin-top: 0.5rem;
        font-size: 0.85rem;
        color: rgba(226,232,240,0.85) !important;
    }

    .sidebar-info {
        font-size: 0.75rem;
        color: rgba(226,232,240,0.4) !important;
    }

    .coef-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 0.5rem 1rem;
        text-align: center;
        flex: 1;
        min-width: 120px;
    }

    .coef-label {
        display: block;
        font-size: 0.6rem;
        color: rgba(226,232,240,0.3) !important;
        letter-spacing: 0.8px;
        text-transform: uppercase;
        margin-bottom: 2px;
    }

    .coef-val {
        display: block;
        font-size: 1rem;
        font-weight: 600;
    }

    .delta-card {
        background: rgba(255,255,255,0.02);
        border: 1px solid rgba(255,255,255,0.05);
        border-radius: 8px;
        padding: 0.5rem 0.75rem;
        text-align: center;
    }

    .stExpander {
        background: rgba(255,255,255,0.01);
        border: 1px solid rgba(255,255,255,0.04);
        border-radius: 8px;
        margin-bottom: 1rem;
    }

    .stExpander summary {
        font-weight: 500;
        color: #e2e8f0 !important;
    }

    .explanation {
        font-size: 0.8rem;
        line-height: 1.7;
        color: rgba(226,232,240,0.8) !important;
    }

    .explanation h4 {
        font-size: 0.85rem;
        font-weight: 600;
        color: #e2e8f0 !important;
        margin-top: 1rem;
        margin-bottom: 0.3rem;
    }

    .explanation h4:first-child {
        margin-top: 0;
    }

    .explanation ul {
        padding-left: 1.2rem;
        margin-top: 0.2rem;
        margin-bottom: 0.5rem;
    }

    .explanation li {
        margin-bottom: 0.2rem;
        color: rgba(226,232,240,0.75) !important;
    }

    .explanation strong {
        color: #e2e8f0 !important;
    }

    .stDataFrame { background: transparent !important; }

    .stDataFrame [data-testid="stTable"] {
        background: rgba(255,255,255,0.02) !important;
        border: 1px solid rgba(255,255,255,0.04) !important;
        border-radius: 8px !important;
    }

    .caption-text {
        color: rgba(226,232,240,0.2) !important;
        font-size: 0.65rem;
        text-align: center;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header"><h1>📈 Nusantara Profit Simulator</h1><p>What-If Analysis &bull; Policy Simulation</p></div>', unsafe_allow_html=True)

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
# SIDEBAR
# ============================================================
st.sidebar.markdown("### Control Variables")

st.sidebar.markdown('<div class="slider-ads">', unsafe_allow_html=True)
iklan_slider = st.sidebar.slider("Anggaran Iklan (Juta Rp)", 0, 50, 10, help="Biaya iklan per periode")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown('<div class="slider-disc">', unsafe_allow_html=True)
diskon_slider = st.sidebar.slider("Besaran Diskon (%)", 0, 50, 10, help="Persentase diskon produk")
st.sidebar.markdown('</div>', unsafe_allow_html=True)

st.sidebar.markdown("---")

if st.sidebar.button("Run Optimizer", type="primary", width='stretch'):
    st.session_state.optimizer_running = True
else:
    st.session_state.optimizer_running = False

st.sidebar.markdown("---")

if st.sidebar.button("Save Scenario", width='stretch'):
    st.session_state.saved_scenarios.append({
        'iklan': iklan_slider,
        'diskon': diskon_slider,
        'prediksi': run_simulation(iklan_slider, diskon_slider)[0],
        'delta': run_simulation(iklan_slider, diskon_slider)[1]
    })

st.sidebar.markdown("---")
st.sidebar.markdown('<p class="sidebar-info">Baseline: Iklan 10 Juta, Diskon 10%</p>', unsafe_allow_html=True)
st.sidebar.markdown(f'<p class="sidebar-info">Prediksi Baseline: Rp {baseline_pred:.2f} Juta</p>', unsafe_allow_html=True)

# ============================================================
# MAIN CONTENT
# ============================================================

hasil_pred, delta = run_simulation(iklan_slider, diskon_slider)

delta_color = "normal" if delta >= 0 else "inverse"
delta_arrow = "▲" if delta >= 0 else "▼"

# --- METRICS ---
col_m1, col_m2, col_m3, col_m4 = st.columns(4)
with col_m1:
    st.metric("Predicted Profit", f"Rp {hasil_pred:.2f} Jt", f"{delta_arrow} {delta:.2f} Jt", delta_color=delta_color)
with col_m2:
    st.metric("Baseline", f"Rp {baseline_pred:.2f} Jt", None)
with col_m3:
    efektivitas = (hasil_pred / baseline_pred) * 100 if baseline_pred > 0 else 0
    st.metric("Effectiveness", f"{efektivitas:.1f}%", f"{efektivitas - 100:+.1f}%", delta_color=delta_color)
with col_m4:
    label = "Increase" if delta > 0 else ("Decline" if delta < 0 else "Stable")
    st.metric("Status", label, f"Rp {abs(delta):.2f} Jt", delta_color=delta_color)

# ============================================================
# DETAIL PERHITUNGAN
# ============================================================
coef_iklan = model.coef_[0]
coef_diskon = model.coef_[1]
intercept = model.intercept_

kontribusi_iklan_baseline = coef_iklan * 10
kontribusi_diskon_baseline = coef_diskon * 10
kontribusi_iklan_now = coef_iklan * iklan_slider
kontribusi_diskon_now = coef_diskon * diskon_slider

with st.expander("📐 Lihat Detail Perhitungan", expanded=False):
    st.markdown("#### Model Persamaan Regresi Linear")
    st.latex(r"Profit = \beta_1 \times \text{Iklan} + \beta_2 \times \text{Diskon} + \beta_0")

    st.markdown(
        f'<div style="display:flex;gap:0.75rem;flex-wrap:wrap;margin-top:0.5rem">'
        f'<div class="coef-card"><span class="coef-label">β₁ (Iklan)</span><span class="coef-val" style="color:#f97316">{coef_iklan:+.4f}</span></div>'
        f'<div class="coef-card"><span class="coef-label">β₂ (Diskon)</span><span class="coef-val" style="color:#8b5cf6">{coef_diskon:+.4f}</span></div>'
        f'<div class="coef-card"><span class="coef-label">β₀ (Intercept)</span><span class="coef-val" style="color:#60a5fa">{intercept:+.4f}</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    st.markdown("#### Perhitungan Baseline")
    st.markdown(
        f'Profit = ({coef_iklan:+.4f} × 10) + ({coef_diskon:+.4f} × 10) + ({intercept:+.4f})'
    )
    st.markdown(
        f'Profit = {kontribusi_iklan_baseline:+.4f} + {kontribusi_diskon_baseline:+.4f} + {intercept:+.4f}'
    )
    st.markdown(f'**Profit Baseline = Rp {baseline_pred:.2f} Juta**')

    st.markdown("#### Perhitungan Skenario Saat Ini")
    st.markdown(
        f'Profit = ({coef_iklan:+.4f} × {iklan_slider}) + ({coef_diskon:+.4f} × {diskon_slider}) + ({intercept:+.4f})'
    )
    st.markdown(
        f'Profit = {kontribusi_iklan_now:+.4f} + {kontribusi_diskon_now:+.4f} + {intercept:+.4f}'
    )
    st.markdown(f'**Profit Skenario = Rp {hasil_pred:.2f} Juta**')

    st.markdown("#### Analisis Delta")
    pct = (delta / baseline_pred) * 100 if baseline_pred > 0 else 0
    kontribusi_iklan_delta = kontribusi_iklan_now - kontribusi_iklan_baseline
    kontribusi_diskon_delta = kontribusi_diskon_now - kontribusi_diskon_baseline

    st.markdown(f'Δ = Profit Skenario − Profit Baseline')
    st.markdown(f'Δ = ({hasil_pred:.2f}) − ({baseline_pred:.2f})')
    st.markdown(f'**Δ = Rp {delta:+.2f} Juta ({pct:+.1f}%)**')

    st.markdown("##### Rincian Sumber Perubahan:")
    col_d1, col_d2, col_d3 = st.columns(3)
    with col_d1:
        st.markdown(
            f'<div class="delta-card"><span class="coef-label">Dari Iklan</span>'
            f'<span class="coef-val" style="color:#f97316">Rp {kontribusi_iklan_delta:+.2f} Jt</span>'
            f'<br><span style="font-size:0.6rem;color:rgba(226,232,240,0.25)">({coef_iklan:+.4f} × Δ{iklan_slider - 10:+d})</span></div>',
            unsafe_allow_html=True,
        )
    with col_d2:
        st.markdown(
            f'<div class="delta-card"><span class="coef-label">Dari Diskon</span>'
            f'<span class="coef-val" style="color:#8b5cf6">Rp {kontribusi_diskon_delta:+.2f} Jt</span>'
            f'<br><span style="font-size:0.6rem;color:rgba(226,232,240,0.25)">({coef_diskon:+.4f} × Δ{diskon_slider - 10:+d})</span></div>',
            unsafe_allow_html=True,
        )
    with col_d3:
        st.markdown(
            f'<div class="delta-card"><span class="coef-label">Total Δ</span>'
            f'<span class="coef-val" style="color:{"#34d399" if delta >= 0 else "#f87171"}">Rp {delta:+.2f} Jt</span>'
            f'<br><span style="font-size:0.6rem;color:rgba(226,232,240,0.25)">({pct:+.1f}% dari baseline)</span></div>',
            unsafe_allow_html=True,
        )

    akurasi_rmse = 10  # from soal no.9 in 15b material
    st.markdown("#### Tingkat Kepercayaan")
    st.markdown(
        f'<div class="rec-card" style="margin-top:0">'
        f'⚠️ Model memiliki RMSE sekitar <strong>Rp {akurasi_rmse} Juta</strong>. '
        f'Artinya, hasil prediksi memiliki rentang ketidakpastian ±Rp {akurasi_rmse} Juta. '
        f'Untuk skenario ini, profit diperkirakan antara <strong>Rp {max(0, hasil_pred - akurasi_rmse):.1f} Juta</strong> '
        f'sampai <strong>Rp {hasil_pred + akurasi_rmse:.1f} Juta</strong>.'
        f'</div>',
        unsafe_allow_html=True,
    )

# --- OPTIMIZER ---
if st.session_state.optimizer_running:
    with st.spinner("Computing optimal combination..."):
        time.sleep(0.5)
        def neg_profit(x):
            return -model.predict([[x[0], x[1]]])[0]

        result = minimize(neg_profit, x0=[25, 25], bounds=[(0, 50), (0, 50)], method='L-BFGS-B')
        opt_iklan = round(result.x[0], 1)
        opt_diskon = round(result.x[1], 1)
        opt_profit = model.predict([[opt_iklan, opt_diskon]])[0]

        col_o1, col_o2, col_o3 = st.columns(3)
        with col_o1:
            st.markdown(f'<div class="opt-card"><div class="label">Optimal Ads</div><div class="value">{opt_iklan} Juta Rp</div></div>', unsafe_allow_html=True)
        with col_o2:
            st.markdown(f'<div class="opt-card"><div class="label">Optimal Discount</div><div class="value">{opt_diskon}%</div></div>', unsafe_allow_html=True)
        with col_o3:
            st.markdown(f'<div class="opt-card"><div class="label">Max Profit</div><div class="value">Rp {opt_profit:.2f} Juta</div></div>', unsafe_allow_html=True)

        st.markdown(
            f'<div class="rec-card">'
            f'💡 Optimal configuration: Ads <strong>Rp {opt_iklan} Juta</strong>, Discount <strong>{opt_diskon}%</strong> '
            f'→ Predicted Profit <strong>Rp {opt_profit:.2f} Juta</strong> '
            f'(+<strong>{(opt_profit / baseline_pred - 1) * 100:.1f}%</strong> from baseline).'
            f'</div>',
            unsafe_allow_html=True
        )

# ============================================================
# CHARTS
# ============================================================
tab1, tab2, tab3, tab4 = st.tabs(["Comparison", "3D Surface", "Sensitivity", "Scenarios"])

DARK_BASE = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='rgba(226,232,240,0.65)', size=10, family='Inter, sans-serif'),
)

def dark_axis(title=None):
    return dict(
        title=title or '',
        gridcolor='rgba(255,255,255,0.04)',
        zerolinecolor='rgba(255,255,255,0.06)',
        tickfont=dict(color='rgba(226,232,240,0.4)', size=9),
        title_font=dict(color='rgba(226,232,240,0.5)', size=10),
        showline=True, linecolor='rgba(255,255,255,0.05)',
    )

# ============================================================
# TAB 1 — COMPARISON BAR CHART
# ============================================================
with tab1:
    st.markdown("### Profit Comparison: Baseline vs Intervention")
    st.markdown('<p style="color:rgba(226,232,240,0.3) !important;font-size:0.7rem;margin-top:-8px">Bar heights show predicted profit for each scenario. Delta = Intervention − Baseline.</p>', unsafe_allow_html=True)

    colors_bar = {'Baseline': '#3b82f6', 'Intervensi': '#f87171'}
    data_bar = pd.DataFrame({
        'Scenario': ['Baseline', 'Intervensi'],
        'Profit (Juta Rp)': [baseline_pred, hasil_pred],
        'Type': ['Baseline', 'Intervensi']
    })

    fig1 = go.Figure()
    for i, row in data_bar.iterrows():
        color = colors_bar[row['Scenario']]
        fig1.add_trace(go.Bar(
            x=[row['Scenario']],
            y=[row['Profit (Juta Rp)']],
            name=row['Scenario'],
            marker_color=color,
            text=f"Rp {row['Profit (Juta Rp)']:.2f} Jt",
            textposition='outside',
            textfont=dict(color='rgba(226,232,240,0.8)', size=12, weight=600),
            hovertemplate='<b>%{x}</b><br>Profit: Rp %{y:.2f} Juta<extra></extra>',
            width=0.45,
        ))

    fig1.update_layout(
        **DARK_BASE,
        height=420,
        showlegend=False,
        yaxis=dict(
            **dark_axis('Profit (Juta Rp)'),
            range=[0, max(baseline_pred, hasil_pred) * 1.35],
            tickprefix='Rp ',
            ticksuffix=' Jt',
        ),
        xaxis=dict(**dark_axis(), type='category'),
        bargap=0.5,
    )

    if abs(delta) > 0.01:
        fig1.add_annotation(
            x='Intervensi', y=hasil_pred,
            text=f"Δ = {delta_arrow} {abs(delta):.2f} Juta ({'+' if delta > 0 else ''}{delta:.2f} Jt)",
            showarrow=True, arrowhead=2, arrowsize=1.2, arrowwidth=2, ax=0, ay=-55,
            font=dict(color='#34d399' if delta > 0 else '#f87171', size=12, weight=600),
            bgcolor='rgba(0,0,0,0.5)', bordercolor='rgba(255,255,255,0.1)', borderwidth=1, borderpad=4,
        )

    fig1.add_hline(
        y=baseline_pred, line_dash='dot', line_color='rgba(59,130,246,0.3)', line_width=1.5,
        annotation_text=f'Baseline: Rp {baseline_pred:.2f} Jt',
        annotation_font_color='rgba(226,232,240,0.3)', annotation_font_size=10,
        annotation_position='bottom right',
    )

    st.plotly_chart(fig1, width='stretch')

    delta_pct = (delta / baseline_pred) * 100 if baseline_pred > 0 else 0
    if abs(delta) > 0.01:
        st.info(
            f"**Insight:** Intervention yields a **{'positive' if delta > 0 else 'negative'}** impact of "
            f"**Rp {abs(delta):.2f} Juta** ({delta_pct:+.1f}% vs baseline). "
            f"With Ads = Rp {iklan_slider} Juta and Discount = {diskon_slider}%, "
            f"profit {'rises' if delta > 0 else 'falls'} to **Rp {hasil_pred:.2f} Juta**."
        )
    else:
        st.info("**Insight:** No meaningful change from baseline.")

    with st.expander("📖 Penjelasan: Comparison Chart"):
        st.markdown("""
        <div class="explanation">
        <h4>Apa yang ditampilkan?</h4>
        <p>Grafik batang ini membandingkan <strong>Profit Baseline</strong> (kondisi saat ini) dengan 
        <strong>Profit Intervensi</strong> (skenario yang Anda pilih melalui slider).</p>

        <h4>Komponen Utama:</h4>
        <ul>
        <li><strong>Batang Biru (Baseline):</strong> Profit jika Iklan = 10 Juta dan Diskon = 10% — ini adalah kondisi awal/titik acuan.</li>
        <li><strong>Batang Merah (Intervensi):</strong> Profit berdasarkan nilai slider Iklan dan Diskon yang Anda atur.</li>
        <li><strong>Garis Putus-putus:</strong> Nilai baseline sebagai referensi visual.</li>
        <li><strong>Label Δ (Delta):</strong> Selisih antara Intervensi dan Baseline — menunjukkan apakah skenario Anda lebih baik atau lebih buruk.</li>
        </ul>

        <h4>Cara Membaca:</h4>
        <ul>
        <li>Jika batang merah <strong>lebih tinggi</strong> dari batang biru → skenario Anda <strong>menguntungkan</strong> (profit naik).</li>
        <li>Jika batang merah <strong>lebih rendah</strong> → skenario <strong>merugikan</strong> (profit turun).</li>
        <li>Δ positif (▲) = keuntungan tambahan; Δ negatif (▼) = pengurangan keuntungan.</li>
        </ul>

        <h4>Konsep dari 15b:</h4>
        <p>Ini adalah implementasi <strong>Digital Twin</strong> — model ML bertindak sebagai replika sistem nyata. 
        Anda bisa menguji berbagai skenario tanpa risiko di dunia nyata. Grafik ini menjawab pertanyaan: 
        <em>"Apa yang terjadi pada profit jika saya mengubah anggaran iklan dan diskon?"</em></p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 2 — 3D SURFACE (with contour and detail)
# ============================================================
with tab2:
    st.markdown("### 3D Profit Surface — Full Decision Space")
    st.markdown('<p style="color:rgba(226,232,240,0.3) !important;font-size:0.7rem;margin-top:-8px">Surface shows profit across all Ads & Discount combinations. Drag to rotate, scroll to zoom.</p>', unsafe_allow_html=True)

    n_grid = 40
    iklan_grid = np.linspace(0, 50, n_grid)
    diskon_grid = np.linspace(0, 50, n_grid)
    I, D = np.meshgrid(iklan_grid, diskon_grid)
    Z = model.predict(np.column_stack((I.ravel(), D.ravel()))).reshape(I.shape)

    # Optimal point
    opt_res = minimize(lambda x: -model.predict([[x[0], x[1]]])[0], x0=[25, 25], bounds=[(0, 50), (0, 50)], method='L-BFGS-B')
    opt_i, opt_d = opt_res.x[0], opt_res.x[1]
    opt_p = model.predict([[opt_i, opt_d]])[0]

    fig2 = go.Figure()

    fig2.add_trace(go.Surface(
        z=Z, x=I, y=D,
        colorscale='Viridis',
        opacity=0.92,
        hovertemplate='Iklan: Rp %{x:.1f} Jt<br>Diskon: %{y:.1f}%%<br>Profit: Rp %{z:.2f} Jt<extra></extra>',
        colorbar=dict(
            title=dict(text='Profit<br>(Juta Rp)', font=dict(color='rgba(226,232,240,0.6)', size=10)),
            tickfont=dict(color='rgba(226,232,240,0.4)', size=9),
            len=0.85,
        ),
        contours=dict(z=dict(show=True, width=1, color='rgba(255,255,255,0.1)', project=dict(z=True))),
    ))

    # Baseline
    fig2.add_trace(go.Scatter3d(
        x=[10], y=[10], z=[baseline_pred],
        mode='markers+text',
        marker=dict(size=8, color='#3b82f6', symbol='circle', line=dict(color='#fff', width=1.5)),
        text=['Baseline'], textposition='top center',
        textfont=dict(color='rgba(226,232,240,0.8)', size=10),
        hovertemplate='<b>Baseline</b><br>Iklan: Rp 10 Jt<br>Diskon: 10%<br>Profit: Rp %{z:.2f} Jt<extra></extra>',
        name='Baseline',
    ))

    # Current intervention
    fig2.add_trace(go.Scatter3d(
        x=[iklan_slider], y=[diskon_slider], z=[hasil_pred],
        mode='markers+text',
        marker=dict(size=8, color='#f87171', symbol='diamond', line=dict(color='#fff', width=1.5)),
        text=['Current'], textposition='top center',
        textfont=dict(color='rgba(226,232,240,0.8)', size=10),
        hovertemplate='<b>Current</b><br>Iklan: Rp %{x:.1f} Jt<br>Diskon: %{y:.1f}%%<br>Profit: Rp %{z:.2f} Jt<extra></extra>',
        name='Current',
    ))

    # Optimal point
    fig2.add_trace(go.Scatter3d(
        x=[opt_i], y=[opt_d], z=[opt_p],
        mode='markers+text',
        marker=dict(size=10, color='#34d399', symbol='diamond', line=dict(color='#fff', width=1.5)),
        text=['★ Optimal'], textposition='top center',
        textfont=dict(color='rgba(52,211,153,0.9)', size=10),
        hovertemplate='<b>★ Optimal</b><br>Iklan: Rp %{x:.1f} Jt<br>Diskon: %{y:.1f}%%<br>Profit: Rp %{z:.2f} Jt<extra></extra>',
        name='Optimal',
    ))

    fig2.update_layout(
        scene=dict(
            xaxis=dict(
                title='Iklan (Juta Rp)', range=[0, 50],
                gridcolor='rgba(255,255,255,0.04)', color='rgba(226,232,240,0.6)',
                tickfont=dict(color='rgba(226,232,240,0.4)', size=9),
                title_font=dict(color='rgba(226,232,240,0.5)', size=10),
                dtick=10,
            ),
            yaxis=dict(
                title='Diskon (%)', range=[0, 50],
                gridcolor='rgba(255,255,255,0.04)', color='rgba(226,232,240,0.6)',
                tickfont=dict(color='rgba(226,232,240,0.4)', size=9),
                title_font=dict(color='rgba(226,232,240,0.5)', size=10),
                dtick=10,
            ),
            zaxis=dict(
                title='Profit (Juta Rp)',
                gridcolor='rgba(255,255,255,0.04)', color='rgba(226,232,240,0.6)',
                tickfont=dict(color='rgba(226,232,240,0.4)', size=9),
                title_font=dict(color='rgba(226,232,240,0.5)', size=10),
            ),
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
            bgcolor='rgba(0,0,0,0)',
            aspectmode='cube',
        ),
        **DARK_BASE,
        height=520,
        margin=dict(l=0, r=0, t=10, b=0),
        legend=dict(
            orientation='h', y=1.02, x=0.5, xanchor='center',
            font=dict(color='rgba(226,232,240,0.5)', size=9),
            bgcolor='rgba(0,0,0,0)',
        ),
    )

    st.plotly_chart(fig2, width='stretch')

    # Summary table below 3D
    col_t1, col_t2, col_t3 = st.columns(3)
    with col_t1:
        st.markdown(f'<div class="opt-card"><div class="label">Baseline</div><div class="value" style="color:#3b82f6!important">Rp {baseline_pred:.2f} Jt</div></div>', unsafe_allow_html=True)
    with col_t2:
        st.markdown(f'<div class="opt-card"><div class="label">Current Scenario</div><div class="value" style="color:#f87171!important">Rp {hasil_pred:.2f} Jt</div></div>', unsafe_allow_html=True)
    with col_t3:
        st.markdown(f'<div class="opt-card"><div class="label">★ Global Optimum</div><div class="value" style="color:#34d399!important">Rp {opt_p:.2f} Jt</div></div>', unsafe_allow_html=True)

    with st.expander("📖 Penjelasan: 3D Profit Surface"):
        st.markdown("""
        <div class="explanation">
        <h4>Apa yang ditampilkan?</h4>
        <p>Grafik 3D ini menunjukkan <strong>seluruh ruang keputusan</strong> — bagaimana profit berubah untuk 
        <em>setiap</em> kombinasi Iklan (0–50 Juta) dan Diskon (0–50%). Permukaan (surface) berwarna 
        mewakili nilai profit: <span style="color:#440154">unggu tua</span> (profit rendah) hingga 
        <span style="color:#fde725">kuning</span> (profit tinggi).</p>

        <h4>Tiga Titik Penting:</h4>
        <ul>
        <li><strong>🔵 Baseline (biru):</strong> Iklan=10, Diskon=10 — titik acuan Anda.</li>
        <li><strong>🔴 Current (merah):</strong> Posisi skenario yang sedang Anda uji.</li>
        <li><strong>🟢 ★ Optimal (hijau):</strong> Kombinasi Iklan & Diskon yang menghasilkan profit tertinggi di seluruh ruang — 
        solusi optimal global yang ditemukan oleh algoritma optimasi.</li>
        </ul>

        <h4>Cara Membaca:</h4>
        <ul>
        <li><strong>Drag</strong> dengan mouse untuk memutar grafik dan melihat dari berbagai sudut.</li>
        <li><strong>Scroll</strong> untuk zoom in/out.</li>
        <li>Perhatikan bahwa permukaan <strong>naik</strong> ke arah Iklan tinggi & Diskon rendah — artinya model memprediksi 
        profit terbesar saat iklan dinaikkan dan diskon diturunkan.</li>
        <li>Garis kontur tipis di permukaan membantu membaca perubahan ketinggian (profit).</li>
        </ul>

        <h4>Konsep dari 15b:</h4>
        <p>Grafik ini megimplementasikan <strong>Analisis Sensitivitas Visual</strong> (Bab X, Subbab 10.3). 
        Dengan melihat keseluruhan permukaan, Anda bisa mengidentifikasi <em>"policy levers"</em> — 
        variabel mana yang paling efektif. Kemiringan (slope) permukaan menunjukkan seberapa sensitif 
        profit terhadap perubahan masing-masing variabel.</p>
        </div>
        """, unsafe_allow_html=True)

# ============================================================
# TAB 3 — SENSITIVITY (tornado + line plots)
# ============================================================
with tab3:
    st.markdown("### Sensitivity Analysis — Identifying Key Leverage Points")
    st.markdown('<p style="color:rgba(226,232,240,0.3) !important;font-size:0.7rem;margin-top:-8px">Tornado chart shows how ±2 unit change in each variable affects profit. Line charts show profit response across full range.</p>', unsafe_allow_html=True)

    step = 2
    i_up = run_simulation(iklan_slider + step, diskon_slider)[0]
    i_dn = run_simulation(max(0, iklan_slider - step), diskon_slider)[0]
    d_up = run_simulation(iklan_slider, min(50, diskon_slider + step))[0]
    d_dn = run_simulation(iklan_slider, max(0, diskon_slider - step))[0]

    # --- TORNADO ---
    sens_df = pd.DataFrame({
        'Label': ['Iklan +2', 'Iklan −2', 'Diskon +2', 'Diskon −2'],
        'Δ': [i_up - hasil_pred, i_dn - hasil_pred, d_up - hasil_pred, d_dn - hasil_pred],
    })
    sens_df['Color'] = sens_df['Δ'].apply(lambda x: '#34d399' if x > 0 else '#f87171')
    sens_df['Abs'] = sens_df['Δ'].abs()

    fig3a = go.Figure()
    for _, row in sens_df.iterrows():
        fig3a.add_trace(go.Bar(
            y=[row['Label']],
            x=[row['Δ']],
            orientation='h',
            marker_color=row['Color'],
            width=0.5,
            text=f"{row['Δ']:+.2f} Jt",
            textposition='outside',
            textfont=dict(color='rgba(226,232,240,0.7)', size=11, weight=500),
            hovertemplate='<b>%{y}</b><br>Δ Profit: %{x:+.2f} Juta<extra></extra>',
            showlegend=False,
        ))

    fig3a.add_vline(x=0, line_color='rgba(226,232,240,0.2)', line_width=1)
    fig3a.update_layout(
        **DARK_BASE,
        height=260,
        xaxis=dict(**dark_axis('Δ Profit (Juta Rp)'), zeroline=False),
        yaxis=dict(**dark_axis(), autorange='reversed'),
        margin=dict(l=0, r=0, t=5, b=0),
    )

    st.plotly_chart(fig3a, width='stretch')

    dampak_i = abs(i_up - i_dn)
    dampak_d = abs(d_up - d_dn)
    var_terkuat = '**Anggaran Iklan**' if dampak_i > dampak_d else '**Besaran Diskon**'
    rasio_sens = max(dampak_i, dampak_d) / max(min(dampak_i, dampak_d), 0.01)

    st.warning(
        f"**Insight:** {var_terkuat} is the most sensitive policy lever — its impact is "
        f"**{rasio_sens:.1f}×** larger than the other variable. Prioritize tuning this variable for maximum effect."
    )

    # --- SENSITIVITY LINE CHARTS ---
    col_s1, col_s2 = st.columns(2)

    x_iklan = np.linspace(0, 50, 50)
    y_iklan = [run_simulation(v, diskon_slider)[0] for v in x_iklan]

    with col_s1:
        st.markdown("#### Ads Sensitivity")
        fig_i = go.Figure()
        fig_i.add_trace(go.Scatter(
            x=x_iklan, y=y_iklan,
            mode='lines+markers',
            line=dict(color='#f97316', width=3),
            marker=dict(size=4, color='#f97316', symbol='circle'),
            fill='tozeroy',
            fillcolor='rgba(249,115,22,0.06)',
            hovertemplate='Iklan: Rp %{x:.1f} Jt<br>Profit: Rp %{y:.2f} Jt<extra></extra>',
            name='',
            showlegend=False,
        ))
        fig_i.add_vline(x=iklan_slider, line_dash='dot', line_color='rgba(226,232,240,0.2)',
                        annotation_text=f'Current: {iklan_slider}',
                        annotation_font_color='rgba(226,232,240,0.3)', annotation_position='top')
        fig_i.add_hline(y=hasil_pred, line_dash='dot', line_color='rgba(226,232,240,0.08)')

        slope_i = np.polyfit(x_iklan, y_iklan, 1)[0]
        fig_i.add_annotation(
            x=45, y=max(y_iklan) * 0.85,
            text=f'Slope: {slope_i:.2f} Jt / unit',
            font=dict(color='rgba(226,232,240,0.3)', size=9),
            showarrow=False,
        )

        fig_i.update_layout(**DARK_BASE, height=280,
            xaxis=dict(**dark_axis('Iklan (Juta Rp)'), range=[0, 52]),
            yaxis=dict(**dark_axis('Profit (Juta Rp)')),
            margin=dict(l=0, r=0, t=5, b=0),
        )
        st.plotly_chart(fig_i, width='stretch')

    with col_s2:
        st.markdown("#### Discount Sensitivity")
        x_diskon = np.linspace(0, 50, 50)
        y_diskon = [run_simulation(iklan_slider, d)[0] for d in x_diskon]

        fig_d = go.Figure()
        fig_d.add_trace(go.Scatter(
            x=x_diskon, y=y_diskon,
            mode='lines+markers',
            line=dict(color='#8b5cf6', width=3),
            marker=dict(size=4, color='#8b5cf6', symbol='circle'),
            fill='tozeroy',
            fillcolor='rgba(139,92,246,0.06)',
            hovertemplate='Diskon: %{x:.1f}%<br>Profit: Rp %{y:.2f} Jt<extra></extra>',
            name='',
            showlegend=False,
        ))
        fig_d.add_vline(x=diskon_slider, line_dash='dot', line_color='rgba(226,232,240,0.2)',
                        annotation_text=f'Current: {diskon_slider}%',
                        annotation_font_color='rgba(226,232,240,0.3)', annotation_position='top')
        fig_d.add_hline(y=hasil_pred, line_dash='dot', line_color='rgba(226,232,240,0.08)')

        slope_d = np.polyfit(x_diskon, y_diskon, 1)[0]
        fig_d.add_annotation(
            x=45, y=max(y_diskon) * 0.85,
            text=f'Slope: {slope_d:.2f} Jt / %',
            font=dict(color='rgba(226,232,240,0.3)', size=9),
            showarrow=False,
        )

        fig_d.update_layout(**DARK_BASE, height=280,
            xaxis=dict(**dark_axis('Diskon (%)'), range=[0, 52]),
            yaxis=dict(**dark_axis('Profit (Juta Rp)')),
            margin=dict(l=0, r=0, t=5, b=0),
        )
        st.plotly_chart(fig_d, width='stretch')

    # Sensitivity summary
    st.markdown(
        f'<div style="display:flex;gap:1rem;justify-content:center;margin-top:0.5rem">'
        f'<div style="background:rgba(249,115,22,0.08);border:1px solid rgba(249,115,22,0.12);border-radius:8px;padding:0.5rem 1.5rem;text-align:center">'
        f'<span style="color:rgba(226,232,240,0.3);font-size:0.6rem;letter-spacing:1px">ADS SLOPE</span><br>'
        f'<span style="color:#f97316;font-weight:600;font-size:1rem">{slope_i:+.2f} Jt/unit</span></div>'
        f'<div style="background:rgba(139,92,246,0.08);border:1px solid rgba(139,92,246,0.12);border-radius:8px;padding:0.5rem 1.5rem;text-align:center">'
        f'<span style="color:rgba(226,232,240,0.3);font-size:0.6rem;letter-spacing:1px">DISCOUNT SLOPE</span><br>'
        f'<span style="color:#8b5cf6;font-weight:600;font-size:1rem">{slope_d:+.2f} Jt/%</span></div>'
        f'<div style="background:rgba(255,255,255,0.02);border:1px solid rgba(255,255,255,0.04);border-radius:8px;padding:0.5rem 1.5rem;text-align:center">'
        f'<span style="color:rgba(226,232,240,0.3);font-size:0.6rem;letter-spacing:1px">DOMINANT VARIABLE</span><br>'
        f'<span style="color:#e2e8f0;font-weight:600;font-size:1rem">{max(["Ads","Discount"], key=lambda v: dampak_i if v=="Ads" else dampak_d)} ({rasio_sens:.1f}×)</span></div>'
        f'</div>',
        unsafe_allow_html=True,
    )

    with st.expander("📖 Penjelasan: Sensitivity Analysis"):
        st.markdown("""
        <div class="explanation">
        <h4>Apa yang ditampilkan?</h4>
        <p>Analisis sensitivitas mengukur <strong>seberapa besar dampak perubahan</strong> pada setiap variabel 
        terhadap profit. Ada tiga komponen visual:</p>

        <h4>1. Tornado Chart 🔄</h4>
        <p>Menunjukkan perubahan profit jika Iklan atau Diskon digeser ±2 unit dari posisi slider saat ini.</p>
        <ul>
        <li><strong>Batang hijau →</strong> perubahan menaikkan profit (positif).</li>
        <li><strong>Batang merah →</strong> perubahan menurunkan profit (negatif).</li>
        <li>Semakin panjang batang, semakin besar dampak variabel tersebut.</li>
        </ul>

        <h4>2. Grafik Sensitivitas 📈</h4>
        <p>Menunjukkan <strong>hubungan linear</strong> antara masing-masing variabel dengan profit:</p>
        <ul>
        <li><strong>Ads Sensitivity (orange):</strong> Garis naik → semakin besar iklan, semakin tinggi profit. 
        Slope = <strong>{slope_i:+.2f} Jt/unit</strong>, artinya setiap tambahan Rp 1 Juta iklan meningkatkan profit Rp {abs(slope_i):.2f} Juta.</li>
        <li><strong>Discount Sensitivity (ungu):</strong> Garis turun → semakin besar diskon, semakin rendah profit. 
        Slope = <strong>{slope_d:+.2f} Jt/%</strong>, artinya setiap tambahan 1% diskon menurunkan profit Rp {abs(slope_d):.2f} Juta.</li>
        <li><strong>Garis putus-putus:</strong> Posisi slider Anda saat ini sebagai referensi.</li>
        </ul>

        <h4>Cara Membaca:</h4>
        <ul>
        <li>Variabel dengan <strong>slope absolut lebih besar</strong> adalah variabel yang lebih sensitif — perubahan kecil saja sudah berdampak besar.</li>
        <li>Gunakan informasi ini untuk memprioritaskan variabel mana yang perlu dioptimalkan terlebih dahulu.</li>
        </ul>

        <h4>Konsep dari 15b:</h4>
        <p>Ini adalah implementasi <strong>Delta Chart</strong> dan <strong>Sensitivity Map</strong> (Bab X, Subbab 10.3). 
        Tujuannya: menemukan <em>"policy levers"</em> atau tuas kebijakan yang paling efektif. 
        Dalam soal umpan balik nomor 3, Anda diminta mengidentifikasi variabel mana yang menyebabkan perubahan lebih besar — 
        inilah jawaban visualnya.</p>
        </div>
        """.format(slope_i=slope_i, slope_d=slope_d),
        unsafe_allow_html=True,
    )

# ============================================================
# TAB 4 — SAVED SCENARIOS
# ============================================================
with tab4:
    st.markdown("### Scenario Comparison")
    st.markdown('<p style="color:rgba(226,232,240,0.3) !important;font-size:0.7rem;margin-top:-8px">Saved scenarios are shown relative to baseline. Color = Better (green) / Worse (red).</p>', unsafe_allow_html=True)

    if not st.session_state.saved_scenarios:
        st.info("No saved scenarios yet. Use **Save Scenario** in the sidebar to save the current slider configuration.")
    else:
        df = pd.DataFrame(st.session_state.saved_scenarios)
        df.index = [f"S{i+1}" for i in range(len(df))]
        df['Δ%'] = (df['delta'] / baseline_pred * 100)
        colors_sc = df['delta'].apply(lambda d: '#34d399' if d > 0 else ('#f87171' if d < 0 else '#94a3b8'))

        fig4 = go.Figure()

        fig4.add_trace(go.Scatter(
            x=df.index, y=df['prediksi'],
            mode='lines+markers+text',
            line=dict(color='#3b82f6', width=2.5),
            marker=dict(size=10, color=colors_sc, line=dict(color='#fff', width=1)),
            text=df['delta'].apply(lambda d: f"{'+' if d > 0 else ''}{d:.1f} Jt"),
            textposition='top center',
            textfont=dict(color='rgba(226,232,240,0.7)', size=10, weight=500),
            hovertemplate=(
                '<b>%{x}</b><br>'
                'Ads: Rp %{customdata[0]:.0f} Jt<br>'
                'Discount: %{customdata[1]:.0f}%%<br>'
                'Profit: Rp %{y:.2f} Jt<br>'
                'Δ: %{customdata[2]:+.2f} Jt (%{customdata[3]:+.1f}%%)<extra></extra>'
            ),
            customdata=np.column_stack([df['iklan'], df['diskon'], df['delta'], df['Δ%']]),
            showlegend=False,
        ))

        fig4.add_hline(
            y=baseline_pred, line_dash='dash', line_color='rgba(59,130,246,0.3)', line_width=1.5,
            annotation_text=f'Baseline: Rp {baseline_pred:.2f} Jt',
            annotation_font_color='rgba(226,232,240,0.3)', annotation_font_size=10,
            annotation_position='bottom left',
        )

        fig4.update_layout(
            **DARK_BASE,
            height=360,
            xaxis=dict(**dark_axis('Scenario'), type='category'),
            yaxis=dict(**dark_axis('Profit (Juta Rp)'), tickprefix='Rp ', ticksuffix=' Jt'),
            margin=dict(l=0, r=0, t=5, b=0),
            hovermode='x unified',
        )

        st.plotly_chart(fig4, width='stretch')

        display_df = df[['iklan', 'diskon', 'prediksi', 'delta', 'Δ%']].rename(columns={
            'iklan': 'Ads (Jt)', 'diskon': 'Disc (%)', 'prediksi': 'Profit (Jt)',
            'delta': 'Δ (Jt)', 'Δ%': 'Δ (%)'
        })
        display_df['Δ (%)'] = display_df['Δ (%)'].apply(lambda x: f"{x:+.1f}%")
        display_df['Δ (Jt)'] = display_df['Δ (Jt)'].apply(lambda x: f"{x:+.2f}")

        styled = display_df.style.map(
            lambda v: 'color: #34d399' if isinstance(v, str) and v.startswith('+') else (
                'color: #f87171' if isinstance(v, str) and v.startswith('-') else ''),
            subset=['Δ (Jt)', 'Δ (%)']
        )
        st.dataframe(styled, width='stretch')

        if st.button("Clear All Scenarios"):
            st.session_state.saved_scenarios = []
            st.rerun()

    with st.expander("📖 Penjelasan: Scenario Comparison"):
        st.markdown("""
        <div class="explanation">
        <h4>Apa yang ditampilkan?</h4>
        <p>Tab ini menyimpan dan membandingkan <strong>skenario-skenario</strong> yang pernah Anda simpan 
        menggunakan tombol <strong>"Save Scenario"</strong> di sidebar.</p>

        <h4>Komponen:</h4>
        <ul>
        <li><strong>Grafik Garis:</strong> Menunjukkan tren profit antar skenario yang disimpan.</li>
        <li><strong>Garis putus-putus biru:</strong> Baseline (Iklan=10, Diskon=10) sebagai referensi tetap.</li>
        <li><strong>Titik hijau/merah:</strong> Setiap skenario diwarnai hijau jika profitnya di atas baseline (lebih baik) 
        dan merah jika di bawah baseline (lebih buruk).</li>
        <li><strong>Tabel:</strong> Menampilkan detail setiap skenario — Iklan, Diskon, Profit, dan Δ (selisih dari baseline).</li>
        </ul>

        <h4>Cara Membaca:</h4>
        <ul>
        <li>Bandingkan antar skenario untuk menemukan kombinasi Iklan & Diskon terbaik.</li>
        <li>Perhatikan kolom <strong>Δ%</strong> — menunjukkan persentase perubahan dari baseline.</li>
        <li>Gunakan fitur ini untuk <strong>eksperimen What-If</strong>: simpan beberapa skenario, lalu bandingkan hasilnya secara berdampingan.</li>
        </ul>

        <h4>Konsep dari 15b:</h4>
        <p>Ini adalah implementasi <strong>Comparative Visualization</strong> yang memungkinkan pengambil keputusan 
        membandingkan beberapa skenario kebijakan sekaligus. Fungsinya seperti <em>"A/B testing"</em> untuk kebijakan bisnis — 
        Anda bisa melihat mana skenario yang paling optimal sebelum benar-benar menerapkannya.</p>
        </div>
        """, unsafe_allow_html=True)
