import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# 1. KONFIGURASI HALAMAN
st.set_page_config(
    page_title="Sistem Prediksi Pemeliharaan Jalan",
    layout="wide"
)

# Custom CSS: Tema Monokrom (Hitam, Putih, Abu-abu)
st.markdown("""
    <style>
    /* Mengubah font global ke Sans-Serif */
    html, body, [data-testid="stSidebar"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Desain Kotak Hasil */
    .card-status {
        padding: 20px;
        border-radius: 4px;
        border: 1px solid #d1d5db;
        margin-bottom: 20px;
    }
    .status-aman {
        background-color: #f9fafb;
        color: #111827;
        border-left: 6px solid #4b5563;
    }
    .status-rusak {
        background-color: #f3f4f6;
        color: #111827;
        border-left: 6px solid #111827;
    }
    .title-status {
        margin: 0;
        font-size: 18px;
        font-weight: 700;
        letter-spacing: 0.5px;
    }
    .text-status {
        margin: 5px 0 0 0;
        font-size: 14px;
        color: #374151;
    }
    /* Merapikan visual tombol */
    div.stButton > button:first-child {
        background-color: #111827 !important;
        color: #ffffff !important;
        border-radius: 4px !important;
        border: none !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #374151 !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. LOAD ASSETS
@st.cache_resource
def load_assets():
    model = joblib.load('model_xgboost_jalan.joblib')
    fitur_desain = joblib.load('daftar_fitur_jalan.joblib')
    return model, fitur_desain

try:
    model, fitur_desain = load_assets()
except Exception as e:
    st.error(f"Gagal memuat aset model: {e}")

# 3. HEADER UTAMA
st.title("Sistem Evaluasi Kelayakan Jalan Raya")
st.caption("Data Analytics Division — PPKD Jakarta Selatan")
st.markdown("---")

# 4. SIDEBAR INPUT PARAMETER
st.sidebar.header("Parameter Operasional")

st.sidebar.subheader("Volume Lalu Lintas")
aadt_vol = st.sidebar.number_input("Volume Harian Rata-rata (AADT)", min_value=0, max_value=100000, value=5400)
traffic_intensity = st.sidebar.number_input("Akumulasi Beban per Tahun", min_value=0, max_value=250000, value=15000)

st.sidebar.subheader("Kondisi Lingkungan")
curah_hujan = st.sidebar.slider("Curah Hujan Rata-rata (mm)", min_value=0.0, max_value=200.0, value=75.5)
rain_impact = st.sidebar.number_input("Dampak Akumulasi Hujan", min_value=-50000, max_value=50000, value=0)

st.sidebar.subheader("Spesifikasi Struktur")
tipe_jalan = st.sidebar.selectbox("Klasifikasi Kelas Jalan", ["Nasional", "Sekunder", "Tersier"])
jenis_aspal = st.sidebar.selectbox("Jenis Perkerasan", ["Asphalt Konvensional", "Beton (Concrete)"])

tombol_analisis = st.sidebar.button("Jalankan Analisis", use_container_width=True)

# 5. PEMROSESAN & OUTPUT
if tombol_analisis:
    # Mapping Dummy Variable
    road_sec = 1 if tipe_jalan == "Sekunder" else 0
    road_ter = 1 if tipe_jalan == "Tersier" else 0
    asphalt_concrete = 1 if jenis_aspal == "Beton (Concrete)" else 0
    
    # Mapping nama kolom sesuai training
    input_kamus = {}
    for col in fitur_desain:
        c = col.lower().strip()
        if 'aadt' in c: input_kamus[col] = aadt_vol
        elif 'impact' in c or 'dampak' in c: input_kamus[col] = rain_impact
        elif 'rain' in c or 'hujan' in c: input_kamus[col] = curah_hujan
        elif 'intensity' in c or 'beban' in c: input_kamus[col] = traffic_intensity
        elif 'secondary' in c or 'sekunder' in c: input_kamus[col] = road_sec
        elif 'tertiary' in c or 'tersier' in c: input_kamus[col] = road_ter
        elif 'concrete' in c or 'beton' in c: input_kamus[col] = asphalt_concrete
        else: input_kamus[col] = 0

    # DataFrame Input
    df_input = pd.DataFrame([input_kamus])[fitur_desain]
    
    # Prediksi Model
    prediksi = model.predict(df_input.values)[0]
    probabilitas = model.predict_proba(df_input.values)[0]
    skor_yakin = np.max(probabilitas) * 100
    
    # Layout Output
    col_kiri, col_kanan = st.columns([6, 4])
    
    with col_kiri:
        st.subheader("Hasil Diagnosis")
        if prediksi == 1:
            st.markdown("""
                <div class="card-status status-rusak">
                    <div class="title-status">KLASIFIKASI: BERPOTENSI RUSAK</div>
                    <div class="text-status">Indikator operasional menunjukkan deviasi kritis. Penjadwalan pemeliharaan preventif direkomendasikan.</div>
                </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
                <div class="card-status status-aman">
                    <div class="title-status">KLASIFIKASI: AMAN / STABIL</div>
                    <div class="text-status">Seluruh parameter berada dalam batas toleransi standar pelayanan minimum jalan.</div>
                </div>
            """, unsafe_allow_html=True)

    with col_kanan:
        st.subheader("Metrik Keyakinan")
        
       # Grafik Gauge dengan skema warna monokrom (Abu-abu ke Hitam)
        warna_bar = "#111827" if prediksi == 1 else "#6b7280"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = skor_yakin,
            number = {'suffix': "%", 'font': {'size': 32, 'color': '#111827'}},
            gauge = {
                'axis': {'range': [0, 100], 'dtick': 20, 'tickcolor': '#6b7280'}, # Selesai diperbaiki: 'ticktick' diganti ke 'dtick'
                'bar': {'color': warna_bar},
                'bgcolor': "#f3f4f6",
                'borderwidth': 1,
                'bordercolor': "#d1d5db"
            }
        ))
        fig.update_layout(margin=dict(l=20, r=20, t=10, b=10), height=180, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.write("Silakan tentukan parameter pada panel kiri dan klik 'Jalankan Analisis' untuk memulai proses evaluasi.")
