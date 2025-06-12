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

### Sumber Data

Dataset yang digunakan dalam proyek ini bersumber dari: [Kaggle - Top Anime Dataset 2024](https://www.kaggle.com/datasets/bhavyadhingra00020/top-anime-dataset-2024)

### 📊 Visualisasi Data

#### 1. Distribusi Rating Anime

![Distribusi Rating Anime](distribusi_rating_anime.jpg)   

**Insight**:
- Mayoritas anime memiliki skor rating antara 7.8 hingga 8.5.
- Hanya sebagian kecil yang mencapai skor di atas 9.0.

#### 2. Top 10 Anime Berdasarkan Rating

![Top 10 Anime Berdasarkan Rating](top_10_anime.jpg)  

**Insight**:
- Anime seperti *Frieren: Beyond Journey's End*, *Fullmetal Alchemist: Brotherhood*, dan *Steins;Gate* termasuk yang tertinggi dengan skor hampir sempurna.
- Franchise *Gintama* mendominasi beberapa posisi dalam 10 besar.

Dataset terdiri dari 1000 baris dan 22 kolom.

### Informasi Jumlah Data dan Missing Values

Berdasarkan output .info() dari data awal (sebelum preprocessing):
- **Total data**: 1000 entries, 22 columns
- **Fitur numerik**: Score (float64), Popularity, Rank, Members (int64)
- **Fitur teks**: 18 kolom bertipe object

**Missing values pada data awal**:
- `Score`: 1000 non-null, 0 missing
- `Popularity`: 1000 non-null, 0 missing  
- `Rank`: 1000 non-null, 0 missing
- `Members`: 1000 non-null, 0 missing
- `Description`: 1000 non-null, 0 missing
- `Type`: 1000 non-null, 0 missing
- `Episodes`: 1000 non-null, 0 missing
- `Status`: 1000 non-null, 0 missing
- `Aired`: 1000 non-null, 0 missing
- `Producers`: 1000 non-null, 0 missing
- `Licensors`: 1000 non-null, 0 missing
- `Studios`: 1000 non-null, 0 missing
- `Source`: 1000 non-null, 0 missing
- `Duration`: 1000 non-null, 0 missing
- `Rating`: 1000 non-null, 0 missing
- `Synonyms`: 709 non-null, 291 missing
- `Japanese`: 999 non-null, 1 missing
- `English`: 859 non-null, 141 missing
- `Premiered`: 569 non-null, 431 missing
- `Broadcast`: 569 non-null, 431 missing
- `Genres`: 771 non-null, 229 missing
- `Demographic`: 521 non-null, 479 missing

### Fitur-fitur yang Tersedia

**Fitur yang tersedia dalam dataset** (berdasarkan kolom aktual dalam dataframe):
1. `Score`: Skor rating pengguna
2. `Popularity`: Peringkat popularitas 
3. `Rank`: Peringkat keseluruhan
4. `Members`: Jumlah anggota yang menambahkan ke daftar
5. `Description`: Ringkasan cerita anime
6. `Synonyms`: Nama alternatif anime
7. `Japanese`: Judul dalam bahasa Jepang
8. `English`: Judul anime dalam bahasa Inggris
9. `Type`: Jenis media (TV, Movie, OVA, dll)
10. `Episodes`: Jumlah episode
11. `Status`: Status penayangan
12. `Aired`: Tanggal tayang
13. `Premiered`: Musim tayang premiere
14. `Broadcast`: Jadwal siaran
15. `Producers`: Daftar produser
16. `Licensors`: Daftar lisensi
17. `Studios`: Studio produksi
18. `Source`: Sumber cerita (manga, original, novel)
19. `Genres`: Genre dari anime
20. `Demographic`: Target demografi (Shounen, Seinen, dll)
21. `Duration`: Durasi per episode
22. `Rating`: Rating usia penonton

## Data Preparation

### Pembersihan Data
- **Pengecekan duplikat**: Tidak ditemukan data duplikat dalam dataset
- **Handling missing values**:
  - `Description`: Diisi dengan 'No description available'
  - `Genres`: Diisi dengan 'Unknown'
  - `Type`: Diisi dengan 'Unknown'
  - `Demographic`: Diisi dengan 'Unknown'
  - `Source`: Diisi dengan 'Unknown'
  - `Synonyms`: Diisi dengan 'No synonyms'
  - `Broadcast`: Diisi dengan 'Unknown'

### Feature Engineering
- Membuat kolom `content_features` yang menggabungkan:
  - `Genres` + `Type` + `Description`, semua dikonversi ke huruf kecil (lowercase)
  - Contoh hasil gabungan: `"adventureadventure, dramadrama, fantasyfantasy..."`

### Preprocessing Khusus untuk TF-IDF
- **Penanganan nilai NaN pada content_features**: Dilakukan langkah `anime_df['content_features'] = anime_df['content_features'].fillna('').astype(str)` sebelum melakukan TF-IDF Vectorization untuk memastikan tidak ada nilai NaN yang dapat menyebabkan error.

### TF-IDF Vectorization
- Menggunakan `TfidfVectorizer` dengan parameter:
  - `stop_words='english'`
  - `ngram_range=(1,2)`
  - `max_features=5000`
- Hasil: matriks vektor sparse berukuran 1000 x 5000

## Modeling

### Pendekatan Content-Based Filtering

1. **Representasi Konten**: Setiap anime direpresentasikan sebagai vektor hasil TF-IDF dari fitur `content_features`
2. **Similarity Computation**: Menggunakan *cosine similarity* untuk menghitung kedekatan antar anime
3. **Matriks Similarity**: Berukuran (1000, 1000) yang menunjukkan skor kemiripan antara setiap pasang anime
4. **Fungsi Rekomendasi**: Mengambil Top-N anime dengan similarity tertinggi berdasarkan cosine similarity

### Contoh Output Rekomendasi

**Input**: "Attack on Titan Season 3 Part 2"

**Output Rekomendasi** (sesuai hasil notebook):
1. Attack on Titan: The Roar of Awakening (Similarity: 0.4797)
2. Attack on Titan: Final Season (Similarity: 0.4659)
3. Attack on Titan Season 2 (Similarity: 0.4496)
4. Attack on Titan: Final Season Part 2 (Similarity: 0.3379)
5. Attack on Titan (Similarity: 0.3122)
6. Attack on Titan Season 3 (Similarity: 0.2770)
7. Attack on Titan: Final Season - The Final Chapters (Similarity: 0.2404)
8. Attack on Titan: No Regrets (Similarity: 0.2255)
9. Classroom of the Elite III (Similarity: 0.1774)
10. Attack on Titan: Lost Girls (Similarity: 0.1742)

## Evaluation

### Metrik Evaluasi

**Precision@10**: Mengukur seberapa banyak anime yang direkomendasikan memiliki genre yang relevan (dengan threshold kesamaan genre > 60%).

### Hasil Evaluasi

Berdasarkan hasil evaluasi di notebook:
- **Attack on Titan Season 3 Part 2**: Precision@10 = 0.90
- **Fullmetal Alchemist: Brotherhood**: Precision@10 = 0.40
- **Steins;Gate**: Precision@10 = 0.70

**Rata-rata Precision@10**: 0.67

### Interpretasi Hasil
- Sistem menunjukkan performa yang baik dengan rata-rata precision 0.67
- Precision tertinggi pada anime dengan genre yang khas dan konsisten seperti Attack on Titan
- Variasi precision menunjukkan bahwa sistem bekerja lebih baik untuk anime dengan karakteristik genre yang jelas

## Conclusion

Sistem rekomendasi berhasil dibangun dengan spesifikasi sebagai berikut:

* **Dataset**: 1000 anime dengan 22 fitur termasuk *Title*, *Genres*, *Description*, *Score*, dan lainnya.
* **Pendekatan**: Content-Based Filtering menggunakan TF-IDF dan Cosine Similarity
* **Evaluasi**: Rata-rata Precision@10 sebesar **0.67**
* **Fungsi utama**: Memberikan rekomendasi berdasarkan kemiripan konten antara anime

#### Kelebihan Sistem:

* Tidak memerlukan data interaksi pengguna (seperti rating atau klik).
* Dapat merekomendasikan anime baru yang belum pernah ditonton oleh pengguna lain.
* Alasan rekomendasi dapat ditelusuri karena berbasis fitur konten.

#### Keterbatasan Sistem:

* Bergantung penuh pada kualitas dan kelengkapan fitur konten yang tersedia.
* Tidak dapat memahami preferensi pengguna yang kompleks atau kontekstual.
* Rentan terhadap over-specialization, yaitu memberikan rekomendasi yang terlalu mirip satu sama lain.

---