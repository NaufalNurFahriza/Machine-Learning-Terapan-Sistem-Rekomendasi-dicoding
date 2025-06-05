
# Laporan Proyek Machine Learning - Sistem Rekomendasi Anime

## Project Overview

Sistem rekomendasi telah menjadi bagian penting dari berbagai platform digital, termasuk layanan streaming anime. Dalam proyek ini, kami mengembangkan sistem rekomendasi anime berbasis *Content-Based Filtering* untuk membantu pengguna menemukan anime baru yang relevan dengan preferensi mereka.

Sumber dataset: [Kaggle - Top Anime Dataset 2024](https://www.kaggle.com/datasets/bhavyadhingra00020/top-anime-dataset-2024)

## Business Understanding

### Problem Statements

- Bagaimana sistem dapat merekomendasikan anime yang mirip dengan anime yang disukai pengguna?
- Bagaimana meningkatkan pengalaman pengguna melalui personalisasi konten berbasis konten anime?

### Goals

- Mengembangkan sistem rekomendasi menggunakan pendekatan *Content-Based Filtering*.
- Memberikan rekomendasi top-N anime berdasarkan kemiripan konten seperti genre, tipe, dan sinopsis.

### Solution Approach

Pendekatan utama yang digunakan adalah *Content-Based Filtering*, dengan teknik:
- Ekstraksi fitur teks menggunakan TF-IDF dari kombinasi `Genres`, `Type`, dan `Description`
- Penghitungan kemiripan antar anime menggunakan *Cosine Similarity*

## Data Understanding

Dataset terdiri dari lebih dari 300 anime, dengan fitur-fitur seperti:
- `English`: Judul anime dalam bahasa Inggris
- `Genres`: Genre dari anime (misalnya Action, Comedy)
- `Type`: Jenis media (TV, Movie, OVA, dll.)
- `Episodes`: Jumlah episode
- `Studios`: Studio produksi
- `Score`: Skor rating pengguna
- `Description`: Ringkasan cerita anime

Distribusi skor rata-rata menunjukkan mayoritas anime berada dalam kisaran 7–9. Top anime seperti *Attack on Titan*, *Fullmetal Alchemist*, dan *Steins;Gate* mendapatkan skor di atas 9.

## Data Preparation

- Menghapus nilai kosong dari kolom `Description`, `Genres`, dan `Type` dengan pengisian default.
- Menggabungkan kolom `Genres`, `Type`, dan `Description` menjadi satu kolom teks untuk ekstraksi fitur.
- Konversi teks menjadi huruf kecil untuk normalisasi.

## Modeling

Model menggunakan pendekatan *Content-Based Filtering*:

1. Menggunakan `TfidfVectorizer` untuk mengubah teks gabungan menjadi vektor numerik.
2. Menghitung *cosine similarity* antar anime berdasarkan vektor TF-IDF.
3. Menghasilkan rekomendasi top-10 berdasarkan skor kemiripan tertinggi.

Contoh fungsi utama:
```python
def get_enhanced_recommendations(title, cosine_sim, df, top_n=10):
    idx = df[df['English'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx].toarray().flatten()))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    anime_indices = [i[0] for i in sim_scores]
    recommendations = df.iloc[anime_indices]
    return recommendations
```

## Evaluation

Model dievaluasi secara kualitatif dan kuantitatif:

- **Evaluasi Kualitatif**: Rekomendasi untuk "Attack on Titan Season 3 Part 2" menghasilkan anime seperti *Fullmetal Alchemist* dan *Demon Slayer*, yang memang memiliki genre dan tema serupa.

- **Evaluasi Kuantitatif**: Menggunakan *Precision@10* berbasis kemiripan genre:
  - Precision@10 Attack on Titan: 0.80
  - Precision@10 Fullmetal Alchemist: 0.90
  - Rata-rata Precision@10: 0.85

Metrik Precision@K mengevaluasi seberapa relevan hasil rekomendasi terhadap preferensi konten dari anime asli.

---

_Proyek ini berhasil menunjukkan bahwa pendekatan content-based filtering dapat memberikan hasil rekomendasi yang relevan secara kontekstual. Untuk pengembangan lebih lanjut, sistem dapat dikembangkan menjadi hybrid dengan collaborative filtering._
