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

# Custom CSS: Tema Monokrom dengan Aksen Merah Khusus untuk Status Rusak
st.markdown("""
    <style>
    html, body, [data-testid="stSidebar"] {
        font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    }
    /* Kotak Hasil */
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
    /* Berubah menjadi Merah Gelap saat Rusak agar Kontras dan Formal */
    .status-rusak {
        background-color: #fdf2f2;
        color: #9b1c1c;
        border-left: 6px solid #e02424;
        border-color: #f8b4b4;
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
    }
    .status-aman .text-status {
        color: #374151;
    }
    .status-rusak .text-status {
        color: #7f1d1d;
    }
    /* Tombol Analisis */
    div.stButton > button:first-child {
        background-color: #111827 !important;
        color: #ffffff !important;
        border-radius: 4px !important;
        border: none !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #374151 !important;
    }
    /* Memperjelas Teks Deskripsi Parameter di Sidebar agar Lebih Terbaca */
    .stMarkdown p {
        font-size: 13px !important;
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
st.sidebar.markdown("<small style='color: #9ca3af; display: block; margin-top: -10px; margin-bottom: 10px;'><b>Satuan: Kendaraan/Hari</b><br>AADT (Annual Average Daily Traffic) adalah total volume lalu lintas kendaraan yang melewati satu titik jalan dalam satu tahun dibagi 365 hari.</small>", unsafe_allow_html=True)

traffic_intensity = st.sidebar.number_input("Akumulasi Beban per Tahun", min_value=0, max_value=250000, value=15000)
st.sidebar.markdown("<small style='color: #9ca3af; display: block; margin-top: -10px; margin-bottom: 10px;'><b>Satuan: Ton/Tahun</b><br>Total proyeksi berat muatan kendaraan (tonase) kumulatif yang melintasi struktur perkerasan jalan dalam periode satu tahun.</small>", unsafe_allow_html=True)

st.sidebar.subheader("Kondisi Lingkungan")
curah_hujan = st.sidebar.slider("Curah Hujan Rata-rata (mm)", min_value=0.0, max_value=200.0, value=75.5)
st.sidebar.markdown("<small style='color: #9ca3af; display: block; margin-top: -10px; margin-bottom: 10px;'><b>Satuan: mm (Milimeter)</b><br>Ketinggian air hujan yang terkumpul di permukaan datar dalam durasi pengamatan tertentu.</small>", unsafe_allow_html=True)

rain_impact = st.sidebar.number_input("Dampak Akumulasi Hujan", min_value=-50000, max_value=50000, value=0)
st.sidebar.markdown("<small style='color: #9ca3af; display: block; margin-top: -10px; margin-bottom: 15px;'><b>Satuan: Indeks Skalar (Angka)</b><br>Variabel interaksi matematika (Rain × Traffic) untuk mengukur tingkat kerawanan struktur jalan ketika dilewati muatan berat saat kondisi jenuh air.</small>", unsafe_allow_html=True)

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

    df_input = pd.DataFrame([input_kamus])[fitur_desain]
    
    prediksi = model.predict(df_input.values)[0]
    probabilitas = model.predict_proba(df_input.values)[0]
    skor_yakin = np.max(probabilitas) * 100
    
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
        
        # Penyesuaian Warna Gauge (Merah jika Rusak, Abu-abu jika Aman)
        warna_bar = "#e02424" if prediksi == 1 else "#4b5563"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = skor_yakin,
            number = {'suffix': "%", 'font': {'size': 32, 'color': '#ffffff'}}, # Memperbaiki teks persentase menjadi PUTIH agar kontras dengan dark theme
            gauge = {
                'axis': {'range': [0, 100], 'dtick': 20, 'tickcolor': '#9ca3af'},
                'bar': {'color': warna_bar},
                'bgcolor': "#1f2937", # Latar gauge abu-abu gelap agar selaras dengan dark theme Streamlit
                'borderwidth': 1,
                'bordercolor': "#4b5563"
            }
        ))
        fig.update_layout(margin=dict(l=20, r=20, t=10, b=10), height=180, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

else:
    st.write("Silakan tentukan parameter pada panel kiri dan klik 'Jalankan Analisis' untuk memulai proses evaluasi.")
