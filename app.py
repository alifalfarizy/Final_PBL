import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go

# 1. KONFIGURASI HALAMAN PREMIUM
st.set_page_config(
    page_title="Sistem Prediksi Pemeliharaan Jalan",
    page_icon="🛣️",
    layout="wide"
)

# Custom CSS untuk mempercantik tampilan kartu dan font
st.markdown("""
    <style>
    .reportview-container { background: #f5f7f8; }
    .card-aman {
        background-color: #d4edda;
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid #28a745;
        color: #155724;
        margin-bottom: 20px;
    }
    .card-rusak {
        background-color: #f8d7da;
        padding: 20px;
        border-radius: 10px;
        border-left: 8px solid #dc3545;
        color: #721c24;
        margin-bottom: 20px;
    }
    .insight-box {
        background-color: #e2e3e5;
        padding: 15px;
        border-radius: 8px;
        color: #383d41;
        border-left: 5px solid #6c757d;
    }
    </style>
""", unsafe_allow_html=True)

# 2. CACHING ASSETS DATA SCIENCE
@st.cache_resource
def load_all_assets():
    model = joblib.load('model_logistic_regression_jalan.joblib')
    scaler = joblib.load('scaler_fitur_jalan.joblib')
    fitur_desain = joblib.load('daftar_fitur_jalan.joblib')
    return model, scaler, fitur_desain

try:
    model, scaler, fitur_desain = load_all_assets()
except Exception as e:
    st.error(f"Gagal memuat file pemodelan. Error: {e}")

# 3. AREA HEADER UTAMA WEB
st.write("""
# 🛣️ Sistem Cerdas Evaluasi & Prediksi Kelayakan Jalan Raya
##### Kemitraan Strategis Pusat Pelatihan Kerja Daerah (PPKD) Jakarta Selatan — Data Analytics Division
---
""")

# 4. PANEL INPUT UTAMA (SIDEBAR KIRI)
st.sidebar.markdown("## 📊 Parameter Operasional Jalan")
st.sidebar.markdown("Silakan sesuaikan variabel di bawah ini sesuai kondisi lapangan:")

# -- Sub-seksi 1: Lalu Lintas
st.sidebar.markdown("### 🚗 Karakteristik Lalu Lintas")
aadt_vol = st.sidebar.number_input(
    "Volume Harian Rata-rata (AADT)", 
    min_value=100, max_value=100000, value=5400, step=100,
    help="Volume kendaraan harian yang melintasi ruas jalan terkait."
)
traffic_intensity = st.sidebar.number_input(
    "Akumulasi Beban Lalin per Tahun", 
    min_value=100, max_value=250000, value=15000, step=500,
    help="Total akumulasi beban geser kendaraan dalam satu tahun berjalan."
)

# -- Sub-seksi 2: Lingkungan & Struktur
st.sidebar.markdown("### 🌧️ Faktor Lingkungan & Konstruksi")
curah_hujan = st.sidebar.slider(
    "Curah Hujan Rata-rata (mm)", 
    min_value=0.0, max_value=200.0, value=75.5, step=0.1
)
rain_impact = st.sidebar.number_input(
    "Dampak Akumulasi Hujan & Lalin (Rain Traffic Impact)", 
    min_value=-50000, max_value=50000, value=1200, step=100,
    help="Variabel interaksi gabungan antara volume hujan dengan kepadatan arus lalin harian."
)

st.sidebar.markdown("### 🏗️ Spesifikasi Fisik Jalan")
tipe_jalan = st.sidebar.selectbox("Klasifikasi Klas Jalan", ["Nasional", "Sekunder", "Tersier"])
jenis_aspal = st.sidebar.selectbox("Jenis Perkerasan (Material)", ["Asphalt Konvensional", "Beton (Concrete)"])

# Tombol Eksekusi
tombol_analisis = st.sidebar.button("🔮 Analisis Potensi Kerusakan Jalan", type="primary", use_container_width=True)


# 5. DASHBOARD HASIL PREDIKSI (HALAMAN UTAMA KANAN)
if tombol_analisis:
    
    # --- Proses Mapping Dummy Variable (One-Hot Encoding) ---
    road_sec = 1 if tipe_jalan == "Sekunder" else 0
    road_ter = 1 if tipe_jalan == "Tersier" else 0
    asphalt_concrete = 1 if jenis_aspal == "Beton (Concrete)" else 0
    
    # Satukan ke Dictionary sesuai struktur X_train asli
    raw_input = {
        'AADT': aadt_vol,
        'Average_Rainfall': curah_hujan,
        'Traffic_Intensity_per_Year': traffic_intensity,
        'Rain_Traffic_Impact': rain_impact,
        'Road_Type_Secondary': road_sec,
        'Road_Type_Tertiary': road_ter,
        'Asphalt_Type_Concrete': asphalt_concrete
    }
    
    df_input = pd.DataFrame([raw_input])[fitur_desain]
    
    # --- Proses Standardisasi Data Baru ---
    df_scaled = df_input.copy()
    kolom_numerik = ['AADT', 'Average_Rainfall', 'Traffic_Intensity_per_Year', 'Rain_Traffic_Impact']
    df_scaled[kolom_numerik] = scaler.transform(df_input[kolom_numerik])
    
    # --- Eksekusi Otak Machine Learning ---
    prediksi = model.predict(df_scaled)[0]
    probabilitas = model.predict_proba(df_scaled)[0]
    skor_yakin = np.max(probabilitas) * 100
    
    st.markdown("### 📋 Hasil Diagnosis Kecerdasan Buatan (Real-Time)")
    
    # Membuat layout kolom: Kiri untuk Status & Insight, Kanan untuk Gauge Chart
    col_kiri, col_kanan = st.columns([6, 4])
    
    with col_kiri:
        # Tampilan Bagian Atas - Status Utama (Prediction Card) menggunakan HTML premium
        if prediksi == 1:
            st.markdown(f"""
                <div class="card-rusak">
                    <h2 style='margin:0; font-size:24px;'>🚨 PERINGATAN STRUKTURAL!</h2>
                    <p style='margin:5px 0 0 0; font-size:18px; font-weight:bold;'>KONDISI JALAN: BERPOTENSI RUSAK / BUTUH MAINTENANCE</p>
                </div>
            """, unsafe_allow_html=True)
            
            # Tampilan Bagian Bawah - Faktor Rekomendasi (Actionable Insights Dinamis Berdasarkan Model LR)
            st.markdown("#### 💡 Dokumen Rekomendasi Teknis (Actionable Insights):")
            insight_text = "Analisis model mendeteksi adanya anomali beban batas pada jalan ini. "
            if curah_hujan > 80:
                insight_text += f"Faktor pemicu utama dipengaruhi oleh tingginya volume <b>Curah Hujan ({curah_hujan} mm)</b> yang menurunkan daya ikat aspal. "
            if tipe_jalan == "Tersier" or tipe_jalan == "Sekunder":
                insight_text += f"Kerentanan ini diperparah oleh status jalan sebagai jalan <b>{tipe_jalan}</b> yang secara struktural memiliki daya tahan pondasi lebih rendah dibandingkan jalan Nasional. "
            if jenis_aspal == "Asphalt Konvensional":
                insight_text += "Disarankan untuk melakukan pelapisan ulang menggunakan zat aditif anti-air atau mempertimbangkan konversi material beton pada siklus pemeliharaan berikutnya."
            
            st.markdown(f"<div class='insight-box'>{insight_text}</div>", unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
                <div class="card-aman">
                    <h2 style='margin:0; font-size:24px;'>✅ KONDISI PRIMA</h2>
                    <p style='margin:5px 0 0 0; font-size:18px; font-weight:bold;'>KONDISI JALAN: AMAN / TIDAK BUTUH PERBAIKAN STRUKTURAL</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 💡 Dokumen Rekomendasi Teknis (Actionable Insights):")
            insight_text = "Struktur jalan diproyeksikan berada dalam performa terbaiknya. "
            if jenis_aspal == "Beton (Concrete)":
                insight_text += "Penggunaan jenis perkerasan <b>Beton (Concrete)</b> berhasil bertindak sebagai tameng proteksi utama yang mereduksi risiko deformasi akibat beban kendaraan harian."
            else:
                insight_text += "Kombinasi antara intensitas lalu lintas harian dan kondisi lingkungan saat ini masih berada di bawah ambang batas kritis material jalan."
            
            st.markdown(f"<div class='insight-box'>{insight_text}</div>", unsafe_allow_html=True)

    with col_kanan:
        # Tampilan Bagian Tengah - Skor Keyakinan menggunakan Gauge Chart Interaktif Plotly
        st.markdown("<h4 style='text-align: center;'>🎯 Skor Keyakinan Model</h4>", unsafe_allow_html=True)
        
        warna_gauge = "#dc3545" if prediksi == 1 else "#28a745"
        
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = skor_yakin,
            domain = {'x': [0, 1], 'y': [0, 1]},
            number = {'suffix': "%", 'font': {'size': 36}},
            gauge = {
                'axis': {'range': [0, 100], 'tickwidth': 1, 'tickcolor': "black"},
                'bar': {'color': warna_gauge},
                'bgcolor': "white",
                'borderwidth': 2,
                'bordercolor': "#6c757d",
                'steps': [
                    {'range': [0, 60], 'color': '#f8f9fa'},
                    {'range': [60, 85], 'color': '#e9ecef'},
                    {'range': [85, 100], 'color': '#dee2e6'}
                ],
            }
        ))
        
        fig.update_layout(
            margin=dict(l=20, r=20, t=10, b=10),
            height=220,
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)'
        )
        st.plotly_chart(fig, use_container_width=True)

    # Menampilkan tabel data yang diinput di paling bawah untuk audit data
    st.markdown("---")
    st.markdown("##### Metadata Input Audit Teknis:")
    st.dataframe(df_input, hide_index=True)

else:
    # Tampilan Dashboard Utama Standby
    st.info("💡 **Petunjuk Penggunaan:** Silakan sesuaikan parameter teknis, operasional lalin, dan lingkungan jalan pada **Panel Sidebar Sebelah Kiri**, lalu klik tombol **Analisis Potensi Kerusakan Jalan** untuk mengeksekusi sistem pakar digital.")
