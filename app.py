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
    .stSlider [data-baseweb="slider"] div { background: #3b82f6 !important; }

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

    .stDataFrame {
        background: transparent !important;
    }

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

iklan_slider = st.sidebar.slider("Anggaran Iklan (Juta Rp)", 0, 50, 10, help="Biaya iklan per periode")
diskon_slider = st.sidebar.slider("Besaran Diskon (%)", 0, 50, 10, help="Persentase diskon produk")

st.sidebar.markdown("---")

if st.sidebar.button("Run Optimizer", type="primary", use_container_width=True):
    st.session_state.optimizer_running = True
else:
    st.session_state.optimizer_running = False

st.sidebar.markdown("---")

if st.sidebar.button("Save Scenario", use_container_width=True):
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

# --- TABS ---
tab1, tab2, tab3, tab4 = st.tabs(["Comparison", "3D Surface", "Sensitivity", "Scenarios"])

PLOTLY_DARK = dict(
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='rgba(226,232,240,0.6)', size=10),
    xaxis=dict(gridcolor='rgba(255,255,255,0.03)', zerolinecolor='rgba(255,255,255,0.04)'),
    yaxis=dict(gridcolor='rgba(255,255,255,0.03)', zerolinecolor='rgba(255,255,255,0.04)'),
)

with tab1:
    data_plot = pd.DataFrame({
        'Skenario': ['Baseline', 'Intervensi'],
        'Keuntungan (Juta Rp)': [baseline_pred, hasil_pred]
    })

    fig_bar = px.bar(
        data_plot, x='Skenario', y='Keuntungan (Juta Rp)',
        color='Skenario',
        text=data_plot['Keuntungan (Juta Rp)'].apply(lambda x: f'Rp {x:.2f} Jt'),
        color_discrete_map={'Baseline': 'rgba(59,130,246,0.5)', 'Intervensi': 'rgba(239,68,68,0.6)'},
    )
    fig_bar.update_layout(**PLOTLY_DARK, height=400, showlegend=False)
    fig_bar.update_traces(textposition='outside', textfont=dict(color='rgba(226,232,240,0.7)'))

    if abs(delta) > 0.01:
        fig_bar.add_annotation(
            x=1, y=hasil_pred,
            text=f"Delta: {delta_arrow} {abs(delta):.2f} Juta",
            showarrow=True, arrowhead=2, arrowsize=1, arrowwidth=2,
            ax=0, ay=-50,
            font=dict(color='#34d399' if delta > 0 else '#f87171', size=12, weight=600)
        )

    st.plotly_chart(fig_bar, use_container_width=True)

    if abs(delta) > 0.01:
        st.info(f"**Insight:** Skenario ini memberikan dampak **{'positif' if delta > 0 else 'negatif'}** sebesar **Rp {abs(delta):.2f} Juta**. "
                f"Dengan iklan Rp {iklan_slider} Juta dan diskon {diskon_slider}%, "
                f"keuntungan {'naik' if delta > 0 else 'turun'} menjadi **Rp {hasil_pred:.2f} Juta**.")
    else:
        st.info("**Insight:** Tidak ada perubahan signifikan dari baseline.")

with tab2:
    iklan_range = np.linspace(0, 50, 25)
    diskon_range = np.linspace(0, 50, 25)
    I, D = np.meshgrid(iklan_range, diskon_range)
    Z = model.predict(np.column_stack((I.ravel(), D.ravel()))).reshape(I.shape)

    fig_3d = go.Figure(data=[
        go.Surface(z=Z, x=I, y=D, colorscale='Blues', opacity=0.8, showscale=True,
                   colorbar=dict(title="Profit (Juta Rp)", tickfont=dict(color='rgba(226,232,240,0.6)')))
    ])
    fig_3d.update_layout(
        scene=dict(
            xaxis=dict(title="Iklan (Juta Rp)", gridcolor='rgba(255,255,255,0.03)', color='rgba(226,232,240,0.6)'),
            yaxis=dict(title="Diskon (%)", gridcolor='rgba(255,255,255,0.03)', color='rgba(226,232,240,0.6)'),
            zaxis=dict(title="Profit (Juta Rp)", gridcolor='rgba(255,255,255,0.03)', color='rgba(226,232,240,0.6)'),
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
            bgcolor='rgba(0,0,0,0)'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        height=500, margin=dict(l=0, r=0, t=10, b=0)
    )

    fig_3d.add_trace(go.Scatter3d(
        x=[iklan_slider], y=[diskon_slider], z=[hasil_pred],
        mode='markers',
        marker=dict(size=6, color='#f87171', symbol='diamond', line=dict(color='#fff', width=1)),
        name='Current'
    ))
    fig_3d.add_trace(go.Scatter3d(
        x=[10], y=[10], z=[baseline_pred],
        mode='markers',
        marker=dict(size=6, color='#fbbf24', symbol='circle', line=dict(color='#fff', width=1)),
        name='Baseline'
    ))

    st.plotly_chart(fig_3d, use_container_width=True)

with tab3:
    step = 2
    iklan_up = run_simulation(iklan_slider + step, diskon_slider)[0]
    iklan_down = run_simulation(max(0, iklan_slider - step), diskon_slider)[0]
    diskon_up = run_simulation(iklan_slider, min(50, diskon_slider + step))[0]
    diskon_down = run_simulation(iklan_slider, max(0, diskon_slider - step))[0]

    sens_data = pd.DataFrame({
        'Variabel': ['Iklan +2', 'Iklan -2', 'Diskon +2', 'Diskon -2'],
        'Δ Profit (Juta Rp)': [
            iklan_up - hasil_pred, iklan_down - hasil_pred,
            diskon_up - hasil_pred, diskon_down - hasil_pred
        ]
    })

    fig_tornado = px.bar(
        sens_data, x='Δ Profit (Juta Rp)', y='Variabel',
        orientation='h', color='Δ Profit (Juta Rp)',
        color_continuous_scale=['#f87171', '#1e293b', '#34d399'],
        text=sens_data['Δ Profit (Juta Rp)'].apply(lambda x: f'{x:+.2f} Jt')
    )
    fig_tornado.update_layout(**PLOTLY_DARK, height=300, xaxis_title="Δ Profit (Juta Rp)")
    fig_tornado.update_traces(textposition='outside', textfont=dict(color='rgba(226,232,240,0.6)'))
    fig_tornado.update_coloraxes(showscale=False)
    st.plotly_chart(fig_tornado, use_container_width=True)

    dampak_iklan = abs(iklan_up - iklan_down)
    dampak_diskon = abs(diskon_up - diskon_down)
    var_sens = "**Anggaran Iklan**" if dampak_iklan > dampak_diskon else "**Besaran Diskon**"
    rasio = max(dampak_iklan, dampak_diskon) / max(min(dampak_iklan, dampak_diskon), 0.01)

    st.warning(f"**Insight:** {var_sens} adalah variabel paling sensitif — dampaknya **{rasio:.1f}x** lebih besar. Fokuskan kebijakan pada variabel ini.")

    col_l1, col_l2 = st.columns(2)
    with col_l1:
        fig_i = px.line(x=np.linspace(0, 50, 20),
                        y=[run_simulation(i, diskon_slider)[0] for i in np.linspace(0, 50, 20)],
                        labels={'x': 'Iklan (Juta Rp)', 'y': 'Profit (Juta Rp)'})
        fig_i.update_layout(**PLOTLY_DARK, height=220, showlegend=False)
        fig_i.update_traces(line=dict(color='#3b82f6', width=2))
        fig_i.add_vline(x=iklan_slider, line_dash="dot", line_color="rgba(226,232,240,0.15)")
        st.plotly_chart(fig_i, use_container_width=True)

    with col_l2:
        fig_d = px.line(x=np.linspace(0, 50, 20),
                        y=[run_simulation(iklan_slider, d)[0] for d in np.linspace(0, 50, 20)],
                        labels={'x': 'Diskon (%)', 'y': 'Profit (Juta Rp)'})
        fig_d.update_layout(**PLOTLY_DARK, height=220, showlegend=False)
        fig_d.update_traces(line=dict(color='#8b5cf6', width=2))
        fig_d.add_vline(x=diskon_slider, line_dash="dot", line_color="rgba(226,232,240,0.15)")
        st.plotly_chart(fig_d, use_container_width=True)

with tab4:
    if not st.session_state.saved_scenarios:
        st.info("Belum ada skenario tersimpan. Gunakan **Save Scenario** di sidebar.")
    else:
        df = pd.DataFrame(st.session_state.saved_scenarios)
        df.index = [f"S{i+1}" for i in range(len(df))]
        df['Status'] = df['delta'].apply(lambda d: 'Better' if d > 0 else ('Worse' if d < 0 else 'Same'))
        st.dataframe(
            df[['iklan', 'diskon', 'prediksi', 'delta', 'Status']]
            .rename(columns={'iklan': 'Ads', 'diskon': 'Disc%', 'prediksi': 'Profit', 'delta': 'Δ'}),
            use_container_width=True
        )

        if len(df) > 1:
            fig_c = px.line(df, y='prediksi', markers=True)
            fig_c.update_layout(**PLOTLY_DARK, height=300,
                yaxis_title="Profit (Juta Rp)", xaxis_title="Scenario", showlegend=False)
            fig_c.add_hline(y=baseline_pred, line_dash="dash", line_color="rgba(59,130,246,0.3)",
                            annotation_text=f"Baseline: Rp {baseline_pred:.2f} Jt",
                            annotation_font_color='rgba(226,232,240,0.4)')
            fig_c.update_traces(line=dict(color='#3b82f6', width=2), marker=dict(color='#3b82f6', size=6))
            st.plotly_chart(fig_c, use_container_width=True)

        if st.button("Clear Scenarios"):
            st.session_state.saved_scenarios = []
            st.rerun()
