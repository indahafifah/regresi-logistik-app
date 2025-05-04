import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc
import matplotlib.pyplot as plt

# ---------- LOAD MODEL & SCALER ----------
MODEL_PATH = 'model/logistic_model.pkl'
SCALER_PATH = 'model/scaler.pkl'
DATA_PATH = 'data/dataset.xlsx'

# Cek file model & scaler
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model atau Scaler tidak ditemukan. Pastikan file logistic_model.pkl dan scaler.pkl ada di folder 'model/'.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

st.title("📊 Prediksi Kelayakan Peminjaman (Koperasi) - Regresi Logistik Biner")

# ---------- FORM INPUT ----------
st.header("📝 Input Data Calon Peminjam")
status_anggota = st.selectbox("Status Anggota", ["biasa", "luar biasa"])
status_peminjaman = st.selectbox("Status Peminjaman", ["new order", "repeat order"])
jumlah_simpanan = st.number_input("Jumlah Simpanan", min_value=0)
jumlah_pinjaman = st.number_input("Jumlah Pinjaman", min_value=0)
jangka_waktu = st.number_input("Jangka Waktu (bulan)", min_value=1)

# Mapping categorical to numeric
status_anggota_num = 1 if status_anggota == "luar biasa" else 0
status_peminjaman_num = 1 if status_peminjaman == "repeat order" else 0

# Input vector
input_data = np.array([[status_anggota_num, jumlah_simpanan, status_peminjaman_num, jumlah_pinjaman, jangka_waktu]])
input_scaled = scaler.transform(input_data)

# ---------- PREDIKSI ----------
if st.button("🔍 Prediksi"):
    prediction = model.predict(input_scaled)[0]
    proba = model.predict_proba(input_scaled)[0][1]
    if prediction == 1:
        st.success(f"✅ Hasil: **Layak** untuk pinjaman (Probabilitas: {proba:.2f})")
    else:
        st.error(f"❌ Hasil: **Tidak Layak** untuk pinjaman (Probabilitas: {proba:.2f})")

# ---------- VISUALISASI EVALUASI ----------
st.markdown("---")
st.header("📈 Evaluasi Model (Confusion Matrix & ROC Curve)")

if os.path.exists(DATA_PATH):
    # Load data
    df = pd.read_excel(DATA_PATH, sheet_name="testing")
    
    # Mapping dan scaling
    df["status_anggota"] = df["status_anggota"].map({"biasa": 0, "luar biasa": 1})
    df["status_peminjaman"] = df["status_peminjaman"].map({"new order": 0, "repeat order": 1})

    X_test = df[["status_anggota", "jumlah_simpanan", "status_peminjaman", "jumlah_pinjaman", "jangka_waktu"]]
    y_test = df["kelayakan"]

    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    y_proba = model.predict_proba(X_test_scaled)[:, 1]

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    st.subheader("📊 Confusion Matrix")
    st.write(pd.DataFrame(cm, columns=["Pred: Tidak Layak", "Pred: Layak"], index=["Actual: Tidak Layak", "Actual: Layak"]))

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_test, y_proba)
    roc_auc = auc(fpr, tpr)
    st.subheader("🔵 ROC Curve")
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, color='blue', label=f'ROC curve (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', linestyle='--')
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Receiver Operating Characteristic (ROC)")
    ax.legend()
    st.pyplot(fig)
else:
    st.warning("File dataset.xlsx tidak ditemukan di folder 'data/'. Visualisasi tidak bisa ditampilkan.")
