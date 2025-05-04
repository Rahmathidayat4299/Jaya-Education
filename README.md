# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech
## Deskripsi Proyek
Proyek ini bertujuan untuk membantu perusahaan Edutech dalam memprediksi status mahasiswa (Dropout, Enrolled, Graduate) berdasarkan data akademik dan demografis. Dengan menggunakan model machine learning, perusahaan dapat mengidentifikasi mahasiswa yang berisiko dropout dan memberikan intervensi dini untuk meningkatkan tingkat kelulusan.
Business Understanding
## Latar Belakang Bisnis
Perusahaan Edutech menghadapi tantangan dalam meningkatkan tingkat kelulusan mahasiswa. Dengan memanfaatkan data historis mahasiswa, perusahaan ingin memahami pola yang memengaruhi status mahasiswa dan mengambil langkah-langkah strategis untuk mengurangi tingkat dropout.

## Permasalahan Bisnis
* Bagaimana cara memprediksi status mahasiswa secara akurat?
* Faktor apa saja yang memengaruhi status mahasiswa?
* Bagaimana perusahaan dapat mengurangi tingkat dropout mahasiswa?
## Cakupan Proyek
* Mengolah data mahasiswa untuk memahami pola dan distribusi data.
* Melatih model machine learning untuk memprediksi status mahasiswa.
* Mengevaluasi performa model dan memberikan rekomendasi untuk meningkatkan tingkat kelulusan.
## Persiapan
## Sumber Data
Dataset yang digunakan berasal dari file data.csv, yang berisi informasi akademik dan demografis mahasiswa.

## Setup Environment
Python Version: 3.9 atau lebih baru.
Library yang Digunakan:
pandas
numpy
scikit-learn
matplotlib
seaborn
streamlit
joblib
Cara Menginstal Dependensi
Jalankan perintah berikut untuk menginstal semua dependensi:
`` pip install -r requirements.txt ``
##  Business Dashboard
##  Deskripsi Dashboard
Dashboard interaktif dibuat menggunakan Streamlit untuk memvisualisasikan data dan hasil prediksi status mahasiswa. Dashboard ini memungkinkan pengguna untuk:

1.  Melihat distribusi data berdasarkan fitur tertentu.
2.  Melakukan prediksi status mahasiswa berdasarkan input data.
## Cara Menjalankan Dashboard
1.  Pastikan semua dependensi telah diinstal.
2.  Jalankan perintah berikut di terminal:<br>
streamlit run app.py
3.  Masukkan data mahasiswa pada form yang tersedia untuk mendapatkan prediksi status.
## Link Dashboard
Jika dashboard telah di-deploy, sertakan link di sini. Contoh:
Dashboard Prediksi Mahasiswa

## Menjalankan Sistem Machine Learning
## Cara Menjalankan Prototype
* Pastikan semua dependensi telah diinstal menggunakan requirements.txt:
``` pip install -r requirements.txt ```
* Jalankan aplikasi Streamlit:
``` streamlit run app.py ```
* Masukkan data mahasiswa pada form yang tersedia untuk mendapatkan prediksi status.
* Link Prototype
Jika prototype telah di-deploy, sertakan link di sini. Contoh:
Prototype Prediksi Mahasiswa

Hasil dan Evaluasi
* Akurasi Model: Model machine learning berhasil dilatih dengan akurasi 76.72% pada data uji.
* Faktor Penting:
Nilai semester 1 dan 2.
Jumlah mata kuliah yang lulus.
Pembayaran uang kuliah tepat waktu.
* Visualisasi:
Distribusi data dan fitur penting divisualisasikan menggunakan matplotlib dan seaborn.
## Kesimpulan
* Model machine learning dapat memprediksi status mahasiswa dengan akurasi yang cukup baik.
* Faktor-faktor penting yang memengaruhi status mahasiswa telah diidentifikasi.
* Dashboard interaktif memungkinkan pengguna untuk memvisualisasikan data dan melakukan prediksi status mahasiswa.
## Rekomendasi Action Items
* Meningkatkan Kualitas Data

Pastikan data yang dikumpulkan lengkap dan akurat, terutama data akademik dan demografis mahasiswa.
* Intervensi Dini

Gunakan hasil prediksi untuk mengidentifikasi mahasiswa yang berisiko dropout dan berikan intervensi dini, seperti konseling atau bantuan akademik.
* Pengembangan Dashboard

Tambahkan fitur analitik lanjutan pada dashboard untuk membantu manajemen dalam pengambilan keputusan.
* Evaluasi Model Secara Berkala

Lakukan evaluasi dan retraining model secara berkala untuk memastikan performa tetap optimal dengan data terbaru.
