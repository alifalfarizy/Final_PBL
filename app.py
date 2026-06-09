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

# -- Sub-seksi 1: Lalu Lintas
st.sidebar.markdown("### 🚗 Karakteristik Lalu Lintas")
aadt_vol = st.sidebar.number_input("Volume Harian Rata-rata (AADT)", min_value=0, max_value=100000, value=5400, step=100)
traffic_intensity = st.sidebar.number_input("Akumulasi Beban Lalin per Tahun", min_value=0, max_value=250000, value=15000, step=500)

# -- Sub-seksi 2: Lingkungan & Struktur
st.sidebar.markdown("### 🌧️ Faktor Lingkungan & Konstruksi")
curah_hujan = st.sidebar.slider("Curah Hujan Rata-rata (mm)", min_value=0.0, max_value=200.0, value=75.5, step=0.1)
rain_impact = st.sidebar.number_input("Dampak Akumulasi Hujan & Lalin (Rain Traffic Impact)", min_value=-50000, max_value=50000, value=0, step=100)

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
    
    # Pemetakan otomatis agar nama kolom sinkron dengan isi joblib lo
    input_kamus = {}
    for col in fitur_desain:
        col_clean = col.lower().strip()
        if 'aadt' in col_clean:
            input_kamus[col] = aadt_vol
        elif 'rain' in col_clean and 'impact' in col_clean:
            input_kamus[col] = rain_impact
        elif 'rain' in col_clean or 'hujan' in col_clean:
            input_kamus[col] = curah_hujan
        elif 'intensity' in col_clean or 'beban' in col_clean:
            input_kamus[col] = traffic_intensity
        elif 'secondary' in col_clean or 'sekunder' in col_clean:
            input_kamus[col] = road_sec
        elif 'tertiary' in col_clean or 'tersier' in col_clean:
            input_kamus[col] = road_ter
        elif 'concrete' in col_clean or 'beton' in col_clean:
            input_kamus[col] = asphalt_concrete
        else:
            input_kamus[col] = 0

    # Buat DataFrame awal dengan urutan kolom saklek
    df_input = pd.DataFrame([input_kamus])[fitur_desain]
    df_scaled = df_input.copy()
    
    # --- JALUR PINTAR PERBAIKAN SCALER (ANTI OVERFLOW) ---
    status_scaler = "Mulai Analisis..."
    try:
        # Cek apakah scaler lo menggunakan penamaan kolom otomatis (Scikit-Learn modern)
        if hasattr(scaler, 'feature_names_in_'):
            kolom_scaler = list(scaler.feature_names_in_)
            df_scaled[kolom_scaler] = scaler.transform(df_input[kolom_scaler])
            status_scaler = "Penyelarasan Sempurna (Berbasis Nama Kolom)."
        else:
            # Cek panjang array rata-rata di dalam scaler asli lo
            jumlah_fitur_scaler = len(scaler.mean_)
            
            if jumlah_fitur_scaler == 4:
                # Fiks, scaler lo di Colab dulu cuma memproses 4 kolom numerik dasar ini!
                kolom_numerik = ['Average_Rainfall', 'AADT', 'Traffic_Intensity_per_Year', 'Rain_Traffic_Impact']
                # Urutkan posisinya agar presisi sesuai template joblib lo
                kolom_numerik_urut = [c for c in fitur_desain if c in kolom_numerik]
                
                if len(kolom_numerik_urut) == 4:
                    df_scaled[kolom_numerik_urut] = scaler.transform(df_input[kolom_numerik_urut])
                    status_scaler = "Berhasil: Hanya 4 Fitur Numerik Dasar yang di-Scale (Dummy Diabaikan)."
                else:
                    # Alternatif jika ada typo nama kolom, ambil 4 kolom pertama yang bukan dummy
                    kolom_alt = [c for c in fitur_desain if not any(x in c.lower() for x in ['road', 'asphalt', 'type', 'concrete', 'secondary', 'tertiary'])]
                    df_scaled[kolom_alt] = scaler.transform(df_input[kolom_alt])
                    status_scaler = "Berhasil dengan Fallback Pemetaan 4 Fitur Utama."
            elif jumlah_fitur_scaler == len(fitur_desain):
                df_scaled[fitur_desain] = scaler.transform(df_input[fitur_desain])
                status_scaler = "Berhasil: Seluruh 7 Fitur di-Scale Sekaligus."
            else:
                # Skenario paksa array urutan depan
                scaled_values = scaler.transform(df_input.values[:, :jumlah_fitur_scaler])
                for i in range(jumlah_fitur_scaler):
                    df_scaled.iloc[0, i] = scaled_values[0, i]
                status_scaler = f"Partial Scaling Aktif untuk {jumlah_fitur_scaler} Fitur Pertama."
    except Exception as e:
        status_scaler = f"Mode Darurat Aktif. Gagal scaling karena: {e}"

    # --- Eksekusi Utama Prediksi Model ---
    prediksi = model.predict(df_scaled)[0]
    probabilitas = model.predict_proba(df_scaled)[0]
    skor_yakin = np.max(probabilitas) * 100
    
    st.markdown("### 📋 Hasil Diagnosis Kecerdasan Buatan (Real-Time)")
    
    # Notifikasi sistem pipeline di web
    st.caption(f"🔧 **Status Pipeline Data:** {status_scaler}")
    
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
            insight_text = "Analisis model mendeteksi adanya akumulasi beban batas kritis pada jalan ini. Diperlukan inspeksi visual berkala oleh tim teknis lapangan PPKD Jaksel untuk mencegah penyebaran deformasi struktural."
            st.markdown(f"<div class='insight-box'>{insight_text}</div>", unsafe_allow_html=True)
        else:
            st.markdown(f"""
                <div class="card-aman">
                    <h2 style='margin:0; font-size:24px;'>✅ KONDISI PRIMA</h2>
                    <p style='margin:5px 0 0 0; font-size:18px; font-weight:bold;'>KONDISI JALAN: AMAN / TIDAK BUTUH PERBAIKAN STRUKTURAL</p>
                </div>
            """, unsafe_allow_html=True)
            st.markdown("#### 💡 Dokumen Rekomendasi Teknis (Actionable Insights):")
            insight_text = "Struktur geometris dan fungsional jalan diproyeksikan berada dalam performa pelayanan terbaiknya. Nilai parameter operasional masih berada di bawah ambang batas kritis kerusakan."
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

    # AUDIT PANEL
    with st.expander("🔍 PUSAT AUDIT INTERNALS MODEL (Cek Kebenaran Data)"):
        st.markdown("**1. Data Sebelum Masuk Scaler (Nilai UI Asli):**")
        st.dataframe(df_input)
        st.markdown("**2. Data Setelah Melewati Filter Scaler Pintar (Scaled Data):**")
        st.dataframe(df_scaled)

else:
    st.info("💡 **Petunjuk Penggunaan:** Silakan sesuaikan parameter operasional pada **Panel Sidebar Sebelah Kiri**, lalu klik tombol **Analisis Potensi Kerusakan Jalan** untuk memulai kalkulasi.")
