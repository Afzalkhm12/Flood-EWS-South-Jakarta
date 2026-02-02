import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import random
from datetime import datetime

# ==========================================
# 1. KONFIGURASI HALAMAN
# ==========================================
st.set_page_config(
    page_title="FloodGuard AI | Early Warning System",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS agar tampilan lebih modern (Mirip Dashboard Profesional)
st.markdown("""
    <style>
    .stMetric {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    .stButton button {
        width: 100%;
        border-radius: 5px;
        height: 50px;
        font-weight: bold;
    }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 2. LOAD MODEL & ASSETS
# ==========================================
@st.cache_resource
def load_resources():
    try:
        # Load Model
        model = tf.keras.models.load_model('model_flood_final.h5', compile=False)
        # Load Scaler
        scaler = joblib.load('scaler_final.pkl')
        return model, scaler
    except Exception as e:
        st.error(f"Gagal memuat model/scaler. Pastikan file .h5 dan .pkl ada di folder yang sama. Error: {e}")
        return None, None

model, scaler = load_resources()

# ==========================================
# 3. FUNGSI UTILITAS
# ==========================================
# Fungsi untuk menghitung fitur turunan (Physics-based)
def calculate_derived_features(suhu, kelembaban, tekanan, dew_point, hujan_kemarin, hujan_2hari, mean_7d, std_7d, tanggal):
    # Hitung Dew Point Depression
    dew_depression = suhu - dew_point
    
    # Hitung Sin Time (Siklus Tahunan)
    timestamp = pd.Timestamp(tanggal).timestamp()
    sin_time = np.sin(timestamp * (2 * np.pi / (365.25 * 24 * 3600)))
    
    # Delta (Kita asumsikan perubahan dari kemarin kecil untuk simulasi manual)
    delta_ps = 0.0 # Simplifikasi
    delta_rh = 0.0 # Simplifikasi
    
    # Susun Array Fitur sesuai urutan training (12 Fitur)
    # ['Suhu', 'Kelembaban', 'Angin', 'Tekanan', 'Dew_Depression', 'Delta_PS', 'Delta_RH', 'Mean_7d', 'Std_7d', 'Lag1', 'Lag2', 'Sin_Time']
    # Angin kita set default/random kecil karena user jarang tahu kecepatan angin
    angin = 2.5 
    
    features = np.array([[suhu, kelembaban, angin, tekanan, dew_depression, delta_ps, delta_rh, mean_7d, std_7d, hujan_kemarin, hujan_2hari, sin_time]])
    return features

# Inisialisasi Session State untuk Nilai Input (Agar bisa di-random)
if 'input_data' not in st.session_state:
    st.session_state.input_data = {
        'suhu': 28.0, 'kelembaban': 80.0, 'tekanan': 1010.0, 'dew_point': 24.0,
        'lag1': 0.0, 'lag2': 0.0, 'mean7d': 5.0, 'std7d': 2.0
    }

# Fungsi Tombol Random
def randomize_inputs():
    # Randomize dengan logika iklim Jakarta (Tropis)
    st.session_state.input_data['suhu'] = round(random.uniform(24.0, 34.0), 1)
    st.session_state.input_data['kelembaban'] = round(random.uniform(60.0, 98.0), 1)
    st.session_state.input_data['tekanan'] = round(random.uniform(1000.0, 1015.0), 1)
    st.session_state.input_data['dew_point'] = round(st.session_state.input_data['suhu'] - random.uniform(1.0, 5.0), 1)
    
    # Randomize Hujan (Seringnya 0, sesekali hujan deras)
    is_rainy = random.choice([True, False, False]) # 1/3 peluang hujan
    if is_rainy:
        st.session_state.input_data['lag1'] = round(random.uniform(20.0, 100.0), 1) # Hujan Deras Kemarin
        st.session_state.input_data['lag2'] = round(random.uniform(0.0, 50.0), 1)
        st.session_state.input_data['mean7d'] = round(random.uniform(10.0, 60.0), 1)
        st.session_state.input_data['std7d'] = round(random.uniform(5.0, 30.0), 1)
    else:
        st.session_state.input_data['lag1'] = round(random.uniform(0.0, 10.0), 1) # Hujan Ringan/Nihil
        st.session_state.input_data['lag2'] = round(random.uniform(0.0, 10.0), 1)
        st.session_state.input_data['mean7d'] = round(random.uniform(0.0, 15.0), 1)
        st.session_state.input_data['std7d'] = round(random.uniform(0.0, 5.0), 1)

# ==========================================
# 4. SIDEBAR INPUT
# ==========================================
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/4721/4721528.png", width=80)
    st.title("Control Panel")
    
    # Tombol Random
    if st.button("🎲 ISI ACAK (RANDOM SCENARIO)", type="primary"):
        randomize_inputs()
    
    st.markdown("---")
    st.subheader("1. Kondisi Atmosfer Hari Ini")
    in_suhu = st.slider("🌡️ Suhu (°C)", 20.0, 40.0, st.session_state.input_data['suhu'])
    in_lemb = st.slider("💧 Kelembaban (%)", 40.0, 100.0, st.session_state.input_data['kelembaban'])
    in_tek = st.slider("🎈 Tekanan Udara (hPa)", 990.0, 1020.0, st.session_state.input_data['tekanan'])
    in_dew = st.slider("🌫️ Titik Embun (°C)", 20.0, 30.0, st.session_state.input_data['dew_point'])
    
    st.markdown("---")
    st.subheader("2. Histori Hujan (Antecedent Rain)")
    in_lag1 = st.number_input("🌧️ Hujan Kemarin (mm)", 0.0, 200.0, st.session_state.input_data['lag1'])
    in_lag2 = st.number_input("🌧️ Hujan 2 Hari Lalu (mm)", 0.0, 200.0, st.session_state.input_data['lag2'])
    in_mean = st.number_input("📊 Rata-rata Hujan 7 Hari (mm)", 0.0, 100.0, st.session_state.input_data['mean7d'])
    in_std = st.number_input("📉 Variasi Hujan 7 Hari (Std)", 0.0, 50.0, st.session_state.input_data['std7d'])
    
    st.markdown("---")
    in_tgl = st.date_input("📆 Tanggal Prediksi", datetime.today())

# ==========================================
# 5. HALAMAN UTAMA (DASHBOARD)
# ==========================================
st.title("🌊 FloodGuard AI: Sistem Peringatan Dini Banjir Jakarta Selatan")
st.markdown("Model berbasis **Ensemble Bi-LSTM** dengan data satelit **CHIRPS & NASA POWER**.")

# Main Action Button
if st.button("🔍 JALANKAN ANALISIS PREDIKSI", use_container_width=True):
    if model is not None:
        # 1. Siapkan Data
        raw_input = calculate_derived_features(in_suhu, in_lemb, in_tek, in_dew, in_lag1, in_lag2, in_mean, in_std, in_tgl)
        
        # 2. Preprocessing (Scaling)
        scaled_input = scaler.transform(raw_input)
        
        # 3. Reshape untuk LSTM (Samples, TimeSteps, Features)
        # Karena model dilatih dengan window 14 hari, untuk prediksi "single point" manual, 
        # kita harus menduplikasi input ini sebanyak 14 kali (Simplifikasi untuk demo)
        # ATAU (Lebih baik) kita padding dengan nol jika tidak ada sequence history.
        # Strategy Demo: Repeat vector 14 times
        model_input = np.repeat(scaled_input, 14, axis=0).reshape(1, 14, 12)
        
        # 4. Prediksi
        prediction_prob = model.predict(model_input)[0][0]
        
        # Threshold (Gunakan Threshold Paper Anda)
        THRESHOLD_PAPER = 0.22 
        status = "BANJIR" if prediction_prob > THRESHOLD_PAPER else "AMAN"
        
        # ==========================================
        # TAMPILAN HASIL (RESULT SECTION)
        # ==========================================
        st.markdown("---")
        col_result1, col_result2 = st.columns([2, 1])
        
        with col_result1:
            st.subheader("📢 HASIL PREDIKSI")
            if status == "BANJIR":
                st.error(f"### 🚨 PERINGATAN: POTENSI BANJIR TINGGI")
                st.markdown("Sistem mendeteksi pola anomali cuaca yang mirip dengan kejadian banjir historis.")
            else:
                st.success(f"### ✅ STATUS: AMAN")
                st.markdown("Kondisi cuaca dan hidrologi terpantau dalam batas normal.")
                
        with col_result2:
            st.subheader("🎯 KEYAKINAN AI (ACCURACY)")
            
            # Tampilan Gauge Sederhana dengan Progress Bar
            st.write("Probabilitas Banjir:")
            st.progress(float(prediction_prob))
            
            confidence_display = f"{prediction_prob*100:.1f}%"
            st.metric(label="Tingkat Keyakinan Model", value=confidence_display)
            
            if prediction_prob > THRESHOLD_PAPER:
                st.caption(f"Melewati ambang batas sensitivitas ({THRESHOLD_PAPER})")
            else:
                st.caption(f"Di bawah ambang batas sensitivitas ({THRESHOLD_PAPER})")

        # ==========================================
        # INTERPRETABILITY (KENAPA HASILNYA BEGINI?)
        # ==========================================
        st.markdown("---")
        st.subheader("🧐 Analisis Faktor Penyebab (Why?)")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Curah Hujan Kemarin", f"{in_lag1} mm", "Tinggi" if in_lag1 > 50 else "Normal")
        with col2:
            dew_dep = in_suhu - in_dew
            st.metric("Saturasi Udara (Dew Dep.)", f"{dew_dep:.1f} °C", "Jenuh" if dew_dep < 2.0 else "Kering", delta_color="inverse")
        with col3:
            st.metric("Kelembaban", f"{in_lemb}%", "Basah" if in_lemb > 90 else "Normal")
        with col4:
            st.metric("Akumulasi Mingguan", f"{in_mean*7:.1f} mm", "Bahaya" if in_mean*7 > 200 else "Aman", delta_color="inverse")

    else:
        st.error("Model belum dimuat.")
else:
    st.info("👈 Masukkan data cuaca di panel sebelah kiri atau klik 'ISI ACAK' untuk simulasi.")

# Footer
st.markdown("---")
st.caption("Developed for Research Purpose | Sinta 3 Publication Target")