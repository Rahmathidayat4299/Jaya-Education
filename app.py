import streamlit as st
import joblib
import pandas as pd
import numpy as np
import os

# Load model dari folder 'model'
model_path = os.path.join("model", "student_status_model.pkl")
model = joblib.load(model_path)

# Judul aplikasi
st.title("Prediksi Status Mahasiswa (Dropout, Enrolled, Graduate)")

# Input fitur dari pengguna
st.header("Masukkan Data Mahasiswa")
curricular_units_2nd_sem_grade = st.number_input("Nilai Semester 2", min_value=0.0, max_value=20.0, step=0.1)
curricular_units_2nd_sem_approved = st.number_input("Jumlah Mata Kuliah Semester 2 yang Lulus", min_value=0, max_value=20, step=1)
curricular_units_1st_sem_grade = st.number_input("Nilai Semester 1", min_value=0.0, max_value=20.0, step=0.1)
tuition_fees_up_to_date = st.selectbox("Pembayaran Uang Kuliah Tepat Waktu (0 = Tidak, 1 = Ya)", [0, 1])
curricular_units_1st_sem_approved = st.number_input("Jumlah Mata Kuliah Semester 1 yang Lulus", min_value=0, max_value=20, step=1)
age_at_enrollment = st.number_input("Umur Saat Mendaftar", min_value=15, max_value=50, step=1)

# Buat DataFrame dari input pengguna
input_data_raw = pd.DataFrame({
    'Curricular_units_2nd_sem_grade': [curricular_units_2nd_sem_grade],
    'Curricular_units_2nd_sem_approved': [curricular_units_2nd_sem_approved],
    'Curricular_units_1st_sem_grade': [curricular_units_1st_sem_grade],
    'Tuition_fees_up_to_date': [tuition_fees_up_to_date],
    'Curricular_units_1st_sem_approved': [curricular_units_1st_sem_approved],
    'Age_at_enrollment': [age_at_enrollment]
})

# Encode data input menggunakan pd.get_dummies
input_data_encoded = pd.get_dummies(input_data_raw)

# Samakan kolom input dengan kolom model training
X_encoded_columns = model.feature_names_in_  # Kolom yang digunakan model saat training
input_data_encoded = input_data_encoded.reindex(columns=X_encoded_columns, fill_value=0)

# Lakukan prediksi
if st.button("Prediksi"):
    prediction = model.predict(input_data_encoded)

    # Mapping hasil prediksi ke label
    status_mapping = {0: "Dropout", 1: "Enrolled", 2: "Graduate"}  # Sesuaikan dengan encoding target y
    predicted_label = status_mapping[prediction[0]]

    # Tampilkan hasil dengan warna khusus untuk Dropout
    if predicted_label == "Dropout":
        st.markdown(f"<h4 style='color:red;'>Prediksi Status Mahasiswa: {predicted_label}</h4>", unsafe_allow_html=True)
    else:
        st.success(f"Prediksi Status Mahasiswa: {predicted_label}")

