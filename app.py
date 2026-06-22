import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from sklearn.linear_model import LinearRegression
from scipy.optimize import minimize
import time

st.set_page_config(page_title="Kaze Yuuji Simulator", layout="wide")

# ============================================================
# CIRI KHAS: KAZE YUUJI SIGNATURE
# ============================================================
st.markdown("""
<style>
    .kaze-signature {
        position: fixed;
        bottom: 10px;
        right: 10px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 8px 16px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: bold;
        z-index: 999;
        box-shadow: 0 2px 10px rgba(102,126,234,0.4);
        letter-spacing: 1px;
    }
    .kaze-watermark {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, -50%) rotate(-30deg);
        font-size: 120px;
        font-weight: 900;
        color: rgba(102,126,234,0.04);
        z-index: -1;
        pointer-events: none;
        white-space: nowrap;
        user-select: none;
    }
    .main-header {
        text-align: center;
        padding: 1rem 0;
        border-bottom: 3px solid #667eea;
        margin-bottom: 2rem;
    }
    .ky-badge {
        display: inline-block;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2px 12px;
        border-radius: 12px;
        font-size: 11px;
        font-weight: 600;
        margin-left: 8px;
        vertical-align: middle;
    }
    .st-emotion-cache-1v7f65g { border-color: #667eea; }
</style>
<div class="kaze-watermark">KAZE YUUJI</div>
<div class="kaze-signature">✦ Kaze Yuuji Simulator v1.0 ✦</div>
""", unsafe_allow_html=True)

# ============================================================
# HEADER
# ============================================================
st.markdown('<div class="main-header"><h1>🌀 Nusantara Profit Simulator <span class="ky-badge">by Kaze Yuuji</span></h1><p style="color: #666; margin-top: 5px;">Interactive What-If Analysis & Policy Simulation Engine</p></div>', unsafe_allow_html=True)

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
st.sidebar.markdown("## 🎛️ Tuas Kebijakan")
st.sidebar.markdown("*Variabel Kontrol — Intervensi*")

iklan_slider = st.sidebar.slider("📺 Anggaran Iklan (Juta Rp)", 0, 50, 10, help="Biaya iklan per periode")
diskon_slider = st.sidebar.slider("🏷️ Besaran Diskon (%)", 0, 50, 10, help="Persentase diskon produk")

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Aksi Cerdas")

if st.sidebar.button("🤖 Optimasi Keuntungan", type="primary", use_container_width=True):
    st.session_state.optimizer_running = True
else:
    st.session_state.optimizer_running = False

st.sidebar.markdown("---")
st.sidebar.markdown("### 💾 Simpan Skenario")
if st.sidebar.button("📌 Simpan skenario ini", use_container_width=True):
    st.session_state.saved_scenarios.append({
        'iklan': iklan_slider,
        'diskon': diskon_slider,
        'prediksi': run_simulation(iklan_slider, diskon_slider)[0],
        'delta': run_simulation(iklan_slider, diskon_slider)[1]
    })

st.sidebar.markdown("---")
st.sidebar.markdown("### ℹ️ Info")
st.sidebar.markdown("**Baseline:** Iklan 10 Juta, Diskon 10%")
st.sidebar.markdown(f"**Prediksi Baseline:** Rp {baseline_pred:.2f} Juta")
st.sidebar.markdown("---")
st.sidebar.caption("✦ Kaze Yuuji Simulator v1.0 ✦")

# ============================================================
# MAIN CONTENT
# ============================================================

# Run simulation
hasil_pred, delta = run_simulation(iklan_slider, diskon_slider)

delta_color = "normal" if delta >= 0 else "inverse"
delta_symbol = "▲" if delta >= 0 else "▼"

# --- TOP METRICS ---
col_met1, col_met2, col_met3, col_met4 = st.columns(4)
with col_met1:
    st.metric("💰 Prediksi Keuntungan", f"Rp {hasil_pred:.2f} Jt", f"{delta_symbol} {delta:.2f} Jt", delta_color=delta_color)
with col_met2:
    st.metric("📊 Baseline (Saat Ini)", f"Rp {baseline_pred:.2f} Jt", None)
with col_met3:
    efektivitas = (hasil_pred / baseline_pred) * 100 if baseline_pred > 0 else 0
    st.metric("📈 Efektivitas", f"{efektivitas:.1f}%", f"{efektivitas - 100:+.1f}%", delta_color=delta_color)
with col_met4:
    perubahan = "NAIK" if delta > 0 else ("TURUN" if delta < 0 else "STABIL")
    st.metric("🔄 Status", perubahan, f"Rp {abs(delta):.2f} Jt", delta_color=delta_color)

# --- OPTIMIZER ---
if st.session_state.optimizer_running:
    with st.spinner("🧠 Mencari kombinasi optimal..."):
        time.sleep(0.5)
        def neg_profit(x):
            return -model.predict([[x[0], x[1]]])[0]

        bounds = [(0, 50), (0, 50)]
        result = minimize(neg_profit, x0=[25, 25], bounds=bounds, method='L-BFGS-B')
        opt_iklan = round(result.x[0], 1)
        opt_diskon = round(result.x[1], 1)
        opt_profit = model.predict([[opt_iklan, opt_diskon]])[0]

        st.success("### ✅ Optimasi Selesai!")
        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            st.info(f"**📺 Iklan Optimal:** {opt_iklan} Juta Rp")
        with col_opt2:
            st.info(f"**🏷️ Diskon Optimal:** {opt_diskon}%")
        with col_opt3:
            st.success(f"**💰 Profit Maksimal:** Rp {opt_profit:.2f} Juta")

        st.markdown(f"""
        > 💡 **Rekomendasi Kaze Yuuji:** Dengan menetapkan anggaran iklan sebesar **Rp {opt_iklan} Juta** 
        > dan diskon **{opt_diskon}%**, keuntungan yang diprediksi mencapai **Rp {opt_profit:.2f} Juta**.
        > Bandingkan dengan baseline Rp {baseline_pred:.2f} Juta — potensi kenaikan **{(opt_profit / baseline_pred - 1) * 100:.1f}%**.
        """)

# --- VISUALIZATION ---
tab1, tab2, tab3, tab4 = st.tabs(["📊 Perbandingan", "🗺️ 3D Profit Landscape", "📉 Analisis Sensitivitas", "📋 Skenario Tersimpan"])

with tab1:
    st.subheader("Perbandingan Baseline vs Intervensi")

    data_plot = pd.DataFrame({
        'Skenario': ['Baseline\n(Saat Ini)', 'Intervensi\n(Skenario Baru)'],
        'Keuntungan (Juta Rp)': [baseline_pred, hasil_pred]
    })

    fig_bar = px.bar(
        data_plot, x='Skenario', y='Keuntungan (Juta Rp)',
        color='Skenario', text='Keuntungan (Juta Rp)',
        color_discrete_map={'Baseline\n(Saat Ini)': '#636EFA', 'Intervensi\n(Skenario Baru)': '#EF553B'},
        template='plotly_white'
    )
    fig_bar.update_traces(texttemplate='Rp %{text:.2f} Jt', textposition='outside')
    fig_bar.update_layout(height=400, showlegend=False)

    delta_label = f"{delta_symbol} {delta:.2f} Juta" if abs(delta) > 0.01 else "Tidak ada perubahan"
    fig_bar.add_annotation(
        x=1, y=hasil_pred,
        text=f"Delta: {delta_label}",
        showarrow=True, arrowhead=2,
        ax=0, ay=-40, font=dict(color="#EF553B" if delta < 0 else "#00CC96", size=14)
    )
    st.plotly_chart(fig_bar, use_container_width=True)

    # Auto-narrative
    if abs(delta) > 0.01:
        dampak = "positif 📈" if delta > 0 else "negatif 📉"
        st.info(f"**Analisis Kaze Yuuji:** Skenario ini menghasilkan dampak **{dampak}** sebesar **Rp {abs(delta):.2f} Juta** terhadap keuntungan. "
                f"Dengan anggaran iklan Rp {iklan_slider} Juta dan diskon {diskon_slider}%, "
                f"keuntungan diprediksi {('naik' if delta > 0 else 'turun')} menjadi Rp {hasil_pred:.2f} Juta.")
    else:
        st.info("**Analisis Kaze Yuuji:** Skenario ini tidak menghasilkan perubahan signifikan terhadap baseline.")

with tab2:
    st.subheader("🗺️ 3D Profit Landscape — Jelajahi Semua Kemungkinan")

    iklan_range = np.linspace(0, 50, 30)
    diskon_range = np.linspace(0, 50, 30)
    I, D = np.meshgrid(iklan_range, diskon_range)
    Z = model.predict(np.column_stack((I.ravel(), D.ravel()))).reshape(I.shape)

    fig_3d = go.Figure(data=[
        go.Surface(z=Z, x=I, y=D, colorscale='Viridis', opacity=0.9, showscale=True,
                   colorbar=dict(title="Profit (Juta Rp)"))
    ])
    fig_3d.update_layout(
        scene=dict(
            xaxis_title="Anggaran Iklan (Juta Rp)",
            yaxis_title="Besaran Diskon (%)",
            zaxis_title="Keuntungan (Juta Rp)",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
        ),
        height=500, margin=dict(l=0, r=0, t=20, b=0)
    )
    fig_3d.add_trace(go.Scatter3d(
        x=[iklan_slider], y=[diskon_slider], z=[hasil_pred],
        mode='markers+text',
        marker=dict(size=8, color='red', symbol='diamond'),
        text=['Skenario Saat Ini'],
        textposition='top center',
        name='Posisi Anda'
    ))
    fig_3d.add_trace(go.Scatter3d(
        x=[10], y=[10], z=[baseline_pred],
        mode='markers+text',
        marker=dict(size=8, color='yellow', symbol='circle'),
        text=['Baseline'],
        textposition='top center',
        name='Baseline'
    ))

    st.plotly_chart(fig_3d, use_container_width=True)
    st.caption("💡 Putar grafik 3D dengan mouse untuk menjelajahi permukaan profit dari berbagai sudut.")

with tab3:
    st.subheader("📉 Analisis Sensitivitas — Cari Tuas Paling Efektif")

    step = 2
    iklan_up = run_simulation(iklan_slider + step, diskon_slider)[0]
    iklan_down = run_simulation(max(0, iklan_slider - step), diskon_slider)[0]
    diskon_up = run_simulation(iklan_slider, min(50, diskon_slider + step))[0]
    diskon_down = run_simulation(iklan_slider, max(0, diskon_slider - step))[0]

    sens_data = pd.DataFrame({
        'Variabel': ['Iklan +2', 'Iklan -2', 'Diskon +2', 'Diskon -2'],
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
        color_continuous_scale=['#EF553B', '#636EFA', '#00CC96'],
        template='plotly_white',
        text='Δ Profit (Juta Rp)'
    )
    fig_tornado.update_traces(texttemplate='%{text:.2f} Jt', textposition='outside')
    fig_tornado.update_layout(height=350, xaxis_title="Perubahan Profit (Juta Rp)")

    st.plotly_chart(fig_tornado, use_container_width=True)

    dampak_iklan = abs(iklan_up - iklan_down)
    dampak_diskon = abs(diskon_up - diskon_down)

    if dampak_iklan > dampak_diskon:
        variabel_sensitif = "📺 **Anggaran Iklan**"
        rasio = dampak_iklan / dampak_diskon if dampak_diskon > 0 else 99
    else:
        variabel_sensitif = "🏷️ **Besaran Diskon**"
        rasio = dampak_diskon / dampak_iklan if dampak_iklan > 0 else 99

    st.warning(f"**🔍 Insight Kaze Yuuji:** {variabel_sensitif} adalah variabel yang paling sensitif! "
               f"Perubahan kecil pada variabel ini berdampak {rasio:.1f}x lebih besar dibanding variabel lainnya. "
               f"Fokuskan kebijakan pada variabel ini untuk hasil maksimal.")

    # Sensitivity map
    st.markdown("#### Peta Sensitivitas")
    iklan_test = np.linspace(0, 50, 20)
    diskon_test = np.linspace(0, 50, 20)

    profits = []
    for d in diskon_test:
        row = []
        for i in iklan_test:
            row.append(run_simulation(i, d)[0])
        profits.append(row)

    fig_heat = px.imshow(
        profits,
        x=[f"{i:.0f}" for i in iklan_test],
        y=[f"{d:.0f}" for d in diskon_test],
        labels=dict(x="Iklan (Juta Rp)", y="Diskon (%)", color="Profit"),
        color_continuous_scale='Viridis',
        template='plotly_white',
        aspect='auto'
    )
    fig_heat.update_layout(height=400, xaxis_title="Anggaran Iklan (Juta Rp)", yaxis_title="Besaran Diskon (%)")
    st.plotly_chart(fig_heat, use_container_width=True)

with tab4:
    st.subheader("📋 Skenario Tersimpan")

    if not st.session_state.saved_scenarios:
        st.info("Belum ada skenario yang disimpan. Gunakan tombol **📌 Simpan skenario ini** di sidebar untuk menyimpan skenario favorit Anda.")
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
                df_scenarios, y='prediksi',
                markers=True, template='plotly_white'
            )
            fig_compare.add_hline(y=baseline_pred, line_dash="dash", line_color="gray",
                                  annotation_text=f"Baseline: Rp {baseline_pred:.2f} Jt")
            fig_compare.update_layout(
                height=350,
                yaxis_title="Keuntungan (Juta Rp)",
                xaxis_title="Skenario"
            )
            st.plotly_chart(fig_compare, use_container_width=True)

        if st.button("🗑️ Hapus semua skenario"):
            st.session_state.saved_scenarios = []
            st.rerun()

# ============================================================
# FOOTER
# ============================================================
st.markdown("---")
col_f1, col_f2, col_f3 = st.columns([1, 2, 1])
with col_f2:
    st.markdown("""
    <div style='text-align: center; color: #888; font-size: 12px;'>
        <p style='margin: 0;'>
            ✦ <strong>Kaze Yuuji Simulator v1.0</strong> ✦<br>
            Built with Streamlit • Linear Regression Engine • Interactive What-If Analysis<br>
            <span style='color: #667eea;'>© 2026 Kaze Yuuji — Ciri Khas Tiada Duanya</span>
        </p>
    </div>
    """, unsafe_allow_html=True)

# Hide default Streamlit elements
st.markdown("""
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    .stApp {background: #fafafa;}
</style>
""", unsafe_allow_html=True)
