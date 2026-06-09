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

# Custom CSS Premium untuk Antarmuka Dashboard
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
    st.error(f"Gagal memuat file pemodelan .joblib. Error: {e}")

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
    min_value=0, max_value=100000, value=5400, step=100
)
traffic_intensity = st.sidebar.number_input(
    "Akumulasi Beban Lalin per Tahun", 
    min_value=0, max_value=250000, value=15000, step=500
)

# -- Sub-seksi 2: Lingkungan & Struktur
st.sidebar.markdown("### 🌧️ Faktor Lingkungan & Konstruksi")
curah_hujan = st.sidebar.slider(
    "Curah Hujan Rata-rata (mm)", 
    min_value=0.0, max_value=200.0, value=75.5, step=0.1
)
rain_impact = st.sidebar.number_input(
    "Dampak Akumulasi Hujan & Lalin (Rain Traffic Impact)", 
    min_value=-50000, max_value=50000, value=0, step=100,
    help="Isi 0 jika tidak ada curah hujan atau lalin melintas."
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
    
    # Satukan ke Dictionary awal sesuai nama kolom asli
    raw_input = {
        'AADT': aadt_vol,
        'Average_Rainfall': curah_hujan,
        'Traffic_Intensity_per_Year': traffic_intensity,
        'Rain_Traffic_Impact': rain_impact,
        'Road_Type_Secondary': road_sec,
        'Road_Type_Tertiary': road_ter,
        'Asphalt_Type_Concrete': asphalt_concrete
    }
    
    # Buat Dataframe dan paksa urutan kolomnya 100% sama dengan matriks model training
    df_input = pd.DataFrame([raw_input])[fitur_desain]
    df_scaled = df_input.copy()
    
    # --- PROSES STANDARDISASI OTOMATIS BERDASARKAN STRUKTUR SCALER ---
    try:
        if hasattr(scaler, 'feature_names_in_'):
            kolom_scaler = list(scaler.feature_names_in_)
            df_scaled[kolom_scaler] = scaler.transform(df_input[kolom_scaler])
        else:
            kolom_numerik_dasar = ['Average_Rainfall', 'AADT', 'Traffic_Intensity_per_Year', 'Rain_Traffic_Impact']
            kolom_numerik_urut = [c for c in fitur_desain if c in kolom_numerik_dasar]
            jumlah_fitur_scaler = len(scaler.mean_)
            
            if jumlah_fitur_scaler == len(kolom_numerik_urut):
                scaled_values = scaler.transform(df_input[kolom_numerik_urut])
                for i, col in enumerate(kolom_numerik_urut):
                    df_scaled[col] = scaled_values[:, i]
            elif jumlah_fitur_scaler == len(fitur_desain):
                df_scaled[fitur_desain] = scaler.transform(df_input[fitur_desain])
            else:
                kolom_numerik_alt = ['AADT', 'Average_Rainfall', 'Traffic_Intensity_per_Year', 'Rain_Traffic_Impact']
                scaled_values = scaler.transform(df_input[kolom_numerik_alt])
                for i, col in enumerate(kolom_numerik_alt):
                    df_scaled[col] = scaled_values[:, i]
    except Exception as e:
        st.warning(f"Metode standardisasi otomatis mendeteksi variasi dimensi: {e}. Menggunakan fallback pemrosesan langsung.")
        df_scaled[fitur_desain] = scaler.transform(df_input[fitur_desain])

    # --- Eksekusi Otak Machine Learning ---
    prediksi = model.predict(df_scaled)[0]
    probabilitas = model.predict_proba(df_scaled)[0]
    skor_yakin = np.max(probabilitas) * 100
    
    st.markdown("### 📋 Hasil Diagnosis Kecerdasan Buatan (Real-Time)")
    
    col_kiri, col_kanan = st.columns([6, 4])
    
    with col_kiri:
        if prediksi == 1:
            st.markdown(f"""
                <div class="card-rusak">
                    <h2 style='margin:0; font-size:24px;'>🚨 PERINGATAN STRUKTURAL!</h2>
                    <p style='margin:5px 0 0 0; font-size:18px; font-weight:bold;'>KONDISI JALAN: BERPOTENSI RUSAK / BUTUH MAINTENANCE</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 💡 Dokumen Rekomendasi Teknis (Actionable Insights):")
            insight_text = "Analisis model mendeteksi adanya akumulasi beban batas kritis pada jalan ini. "
            if curah_hujan > 50:
                insight_text += f"Faktor risiko dipicu oleh parameter <b>Curah Hujan ({curah_hujan} mm)</b> yang berpotensi melunakkan ikatan agregat aspal. "
            if tipe_jalan != "Nasional":
                insight_text += f"Struktur pondasi sebagai kelas jalan <b>{tipe_jalan}</b> memerlukan pemantauan berkala guna mengantisipasi deformasi permukaan."
            
            st.markdown(f"<div class='insight-box'>{insight_text}</div>", unsafe_allow_html=True)
            
        else:
            st.markdown(f"""
                <div class="card-aman">
                    <h2 style='margin:0; font-size:24px;'>✅ KONDISI PRIMA</h2>
                    <p style='margin:5px 0 0 0; font-size:18px; font-weight:bold;'>KONDISI JALAN: AMAN / TIDAK BUTUH PERBAIKAN STRUKTURAL</p>
                </div>
            """, unsafe_allow_html=True)
            
            st.markdown("#### 💡 Dokumen Rekomendasi Teknis (Actionable Insights):")
            insight_text = "Struktur geometris dan fungsional jalan diproyeksikan berada dalam performa pelayanan terbaiknya (Ambang batas aman). "
            if jenis_aspal == "Beton (Concrete)":
                insight_text += "Penggunaan material tipe <b>Beton (Concrete)</b> terbukti efektif memberikan daya tahan optimal terhadap tekanan roda kendaraan."
            
            st.markdown(f"<div class='insight-box'>{insight_text}</div>", unsafe_allow_html=True)

    with col_kanan:
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
            }
        ))
        fig.update_layout(margin=dict(l=20, r=20, t=10, b=10), height=220, paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
        st.plotly_chart(fig, use_container_width=True)

    # PUSAT DIAGNOSTIK UNTUK MENCARI KEBENARAN
    with st.expander("🔍 PUSAT AUDIT INTERNALS MODEL (Uji Kebenaran Sidang)"):
        col_diag1, col_diag2 = st.columns(2)
        with col_diag1:
            st.write("**1. Urutan Fitur Ekspektasi Sistem (`daftar_fitur_jalan.joblib`):**")
            st.json(fitur_desain)
            if hasattr(scaler, 'mean_'):
                st.write("**2. Nilai Mean pada Scaler (`scaler.mean_`):**")
                st.write(scaler.mean_)
        with col_diag2:
            if hasattr(model, 'coef_'):
                st.write("**3. Koefisien Matematika Model (`model.coef_`):**")
                st.write(model.coef_[0])
                st.write("**4. Nilai Intercept Model (`model.intercept_`):**")
                st.write(model.intercept_)
        
        st.markdown("---")
        st.markdown("**5. Matriks Data setelah Standardisasi (Scaled Data):**")
        st.dataframe(df_scaled, hide_index=True)

else:
    st.info("💡 **Petunjuk Penggunaan:** Silakan sesuaikan parameter operasional pada **Panel Sidebar Sebelah Kiri**, lalu klik tombol **Analisis Potensi Kerusakan Jalan** untuk memulai kalkulasi.")
