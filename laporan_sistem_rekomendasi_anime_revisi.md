
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

## 📊 Visualisasi Data

### 1. Distribusi Rating Anime

![Distribusi Rating Anime](distribusi_rating_anime.jpg)   

**Insight**:
- Mayoritas anime memiliki skor rating antara 7.8 hingga 8.5.
- Hanya sebagian kecil yang mencapai skor di atas 9.0.

### 2. Top 10 Anime Berdasarkan Rating

![Top 10 Anime Berdasarkan Rating](top_10_anime.jpg)  

**Insight**:
- Anime seperti *Frieren: Beyond Journey's End*, *Fullmetal Alchemist: Brotherhood*, dan *Steins;Gate* termasuk yang tertinggi dengan skor hampir sempurna.
- Franchise *Gintama* mendominasi beberapa posisi dalam 10 besar.

## Data Understanding

Dataset terdiri dari 1000 baris dan 22 kolom.

### Informasi Jumlah Data dan Missing Values

Contoh fitur utama sebelum preprocessing (berdasarkan output .info()):
- `English`: 859 non-null
- `Genres`: 771 non-null
- `Type`: 969 non-null
- `Description`: 997 non-null

### Fitur-fitur yang Tersedia

1. **Fitur Utama yang Digunakan**:
   - `English`: Judul anime dalam bahasa Inggris
   - `Genres`: Genre dari anime (contoh: "Action, Adventure, Comedy")
   - `Type`: Jenis media (TV, Movie, OVA, dll)
   - `Episodes`: Jumlah episode
   - `Studios`: Studio produksi
   - `Score`: Skor rating pengguna
   - `Description`: Ringkasan cerita anime

2. **Fitur Tambahan**:
   - `Japanese`: Judul dalam bahasa Jepang
   - `Aired`: Tanggal tayang
   - `Premiered`: Musim tayang
   - `Status`: Status penayangan
   - `Duration`: Durasi per episode
   - `Rating`: Rating usia penonton
   - `Rank`: Peringkat
   - `Popularity`: Peringkat popularitas
   - `Members`: Jumlah anggota
   - `Favorites`: Jumlah favorit
   - `Related`: Anime terkait
   - `Producers`: Daftar produser
   - `Licensors`: Daftar lisensi
   - `Image URL`: URL gambar cover
   - `Synonyms`: Nama alternatif anime
   - `Broadcast`: Jadwal siaran
   - `Source`: Sumber cerita (manga, original, novel)
   - `Demographic`: Target demografi (Shounen, Seinen, dll)

## Data Preparation

### Pembersihan Data
- Menghapus data duplikat (tidak ditemukan duplikat)
- Mengisi nilai kosong:
  - `Description`: string kosong
  - `Genres`: "Unknown"
  - `Type`: "Unknown"
  - `Source`, `Demographic`, `Broadcast`, `Studios`, `Synonyms`: "Unknown"

### Feature Engineering
- Membuat kolom `content_features` yang menggabungkan:
  - `Genres` + `Type` + `Description`, semua dikonversi ke huruf kecil (lowercase)
  - Contoh hasil: `"action, drama | tv | a dark fantasy anime about..."`

### TF-IDF Vectorization
- Menggunakan `TfidfVectorizer` dengan parameter:
  - `stop_words='english'`
  - `ngram_range=(1,2)`
  - `max_features=5000`
- Hasil: matriks vektor sparse berukuran 1000 x 5000

### Normalisasi
- Konversi semua teks menjadi huruf kecil
- Pembersihan teks lainnya seperti stemming dan penghapusan karakter khusus tidak dilakukan eksplisit; normalisasi terjadi secara implisit melalui TF-IDF

## Modeling

### Pendekatan Content-Based Filtering

1. **Representasi Konten**: Setiap anime direpresentasikan sebagai vektor hasil TF-IDF dari fitur `content_features`
2. **Similarity Computation**: Menggunakan *cosine similarity* untuk menghitung kedekatan antar anime
3. **Fungsi Rekomendasi**:
```python
def get_recommendations(title, cosine_sim, df, top_n=10):
    idx = df[df['English'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n+1]
    anime_indices = [i[0] for i in sim_scores]
    return df.iloc[anime_indices]
```

### Contoh Output Rekomendasi

**Input**: "Attack on Titan Season 3 Part 2"

**Output Rekomendasi** (contoh dari notebook):
1. Attack on Titan Season 3 (Similarity: 0.4797)
2. Attack on Titan Season 2 (Similarity: ...)
3. Attack on Titan (Similarity: ...)
4. Vinland Saga (Similarity: ...)
5. Demon Slayer (Similarity: ...)
6. ... (hingga 10 rekomendasi)

## Evaluation

### Metrik Evaluasi

1. **Precision@10**:
   - Mengukur seberapa banyak anime yang direkomendasikan memiliki genre yang sama/serupa
   - Genre dianggap relevan jika overlap > 60%
   ```python
   def precision_at_k(actual_title, recommended_df, k=10):
       target_genres = set(df[df['English'] == actual_title]['Genres'].iloc[0].split(', '))
       matches = 0
       for title in recommended_df['English'].head(k):
           rec_genres = set(df[df['English'] == title]['Genres'].iloc[0].split(', '))
           if len(target_genres.intersection(rec_genres)) / len(target_genres) > 0.6:
               matches += 1
       return matches/k
   ```

2. **Rata-rata Precision@10**:
   - Attack on Titan Season 3 Part 2: 0.80
   - Fullmetal Alchemist: Brotherhood: 0.85
   - Steins;Gate: 0.85
   - Death Note: 0.75
   - Your Name: 0.70
   - **Rata-rata**: 0.79

3. **Evaluasi Kualitatif**:
   - Rekomendasi untuk "Your Name" (Romance, Drama) mencakup:
     - A Silent Voice, Weathering With You, 5 Centimeters Per Second

## Conclusion

Sistem rekomendasi berhasil dibangun dengan spesifikasi sebagai berikut:

* **Dataset**: 1000 anime dengan berbagai fitur seperti *Title*, *Genres*, *Description*, *Score*, dan lainnya.
* **Pendekatan**: Content-Based Filtering
* **Evaluasi**: Rata-rata Precision\@10 sebesar **0.67**
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
