import streamlit as st
import joblib
import numpy as np

# Load model dan scaler
model = joblib.load("logistic_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("Analisis Kelayakan Peminjaman Anggota Koperasi")

# Input dari user
status_anggota = st.selectbox("Status Anggota", ["Anggota Biasa", "Anggota Luar Biasa"])
status_pinjaman = st.selectbox("Status Pinjaman", ["New Order", "Repeat Order"])
jumlah_simpanan = st.number_input("Jumlah Simpanan", min_value=0.0, format="%.2f")
jumlah_pinjam = st.number_input("Jumlah Pinjam", min_value=0.0, format="%.2f")
jangka_waktu = st.number_input("Jangka Waktu (bulan)", min_value=1)

# Ubah input ke format model
if st.button("Cek Kelayakan"):
    input_data = np.array([
        1 if status_anggota == "Anggota Biasa" else 0,
        1 if status_pinjaman == "New Order" else 0,
        jumlah_simpanan,
        jumlah_pinjam,
        jangka_waktu
    ]).reshape(1, -1)

    input_scaled = scaler.transform(input_data)
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]

    hasil = "Layak" if pred == 1 else "Tidak Layak"
    st.success(f"Hasil Prediksi: {hasil}")
    st.info(f"Probabilitas Kelulusan: {prob:.2f}")
    persentase = round(prob * 100, 2)
    st.markdown(f"📌 Berdasarkan hasil prediksi, **ada {persentase}% kemungkinan anggota ini layak diberi pinjaman**.")
