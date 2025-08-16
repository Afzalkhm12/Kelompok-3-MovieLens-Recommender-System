🎬 MovieLens Recommender System (NCF vs MF)

🚀 Ujian Akhir Semester – Deep Learning
📚 Fakultas Teknologi Komunikasi & Informatika (Universitas  Nasional)
👥 Kelompok 3: Afzal , Morando, Pipit, Salsa

📌 Deskripsi Proyek

Proyek ini membangun Sistem Rekomendasi Film berbasis interaksi pengguna–item menggunakan dataset MovieLens 20M.
Kami membandingkan pendekatan Matrix Factorization (MF) klasik dengan Neural Collaborative Filtering (NCF) untuk memprediksi rating/klik film.

📊 Dataset

Dataset: MovieLens 20M

📌 Statistik Dataset

👤 n_users: 138,493

🎞️ n_items: 26,744

🔀 Split data:

Train: 19,723,277

Validation: 138,493

Test: 138,493

📈 Exploratory Data Analysis (EDA)
🔹 Distribusi Rating

Mayoritas rating berada di kisaran 3–4, menunjukkan kecenderungan user memberi rating positif.

🔹 Distribusi Interaksi per User

Mayoritas user hanya melakukan sedikit interaksi, hanya sebagian kecil user yang sangat aktif.

🔹 Distribusi Interaksi per Item

Sebagian besar film hanya mendapat sedikit rating, tetapi ada film populer yang mendapat puluhan ribu interaksi (popularity bias).

🔹 Grafik Tambahan






Grafik tambahan memperkuat insight tentang skew distribution baik dari sisi user maupun item.

⚙️ Metodologi

Matrix Factorization (MF) – baseline model dengan latent factor.

Neural Collaborative Filtering (NCF) – embedding user & item, dilanjutkan MLP dengan dropout & regularisasi.

Negative Sampling untuk implicit feedback.

Optimizer: Adam, loss: MSE.

Evaluasi dengan RMSE & MAE.

📊 Hasil Evaluasi
Model	Test RMSE	Test MAE
MF	0.9870	0.7582
NCF	0.8759	0.6682

👉 Kesimpulan: NCF lebih akurat daripada MF dalam memprediksi rating.

🖼️ Screenshots Aplikasi (Streamlit)
🔹 Halaman Input User ID

(contoh screenshot UI aplikasi, bisa ditambahkan dari hasil run Streamlit)

🔹 Rekomendasi Film untuk User

(contoh screenshot daftar film yang direkomendasikan dengan judul + genre)

🔹 Visualisasi Popularitas Film

(contoh grafik dari rekomendasi yang ditampilkan di UI)

👩‍💻 Cara Menjalankan
# Clone repository
git clone https://github.com/username/MovieLens-Recommender-System-NCF-vs-MF.git
cd MovieLens-Recommender-System-NCF-vs-MF

# Install dependencies
pip install -r requirements.txt

# Jalankan Streamlit
streamlit run app.py

✨ Kontributor

👥 Kelompok 3 – Deep Learning

Afzal

Morando

Pipit

Salsa
