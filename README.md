# Dashboard Analisis Harga Minyak & BBM Indonesia

Platform analisis interaktif untuk memahami dinamika harga minyak mentah global dan dampaknya terhadap harga BBM domestik.

## Latar Belakang

Harga BBM di Indonesia sangat dipengaruhi oleh fluktuasi harga minyak mentah dunia (Brent dan WTI) serta pergerakan nilai tukar rupiah. Dashboard ini hadir untuk memberikan visualisasi dan analisis mendalam terhadap korelasi tersebut, dilengkapi dengan prediksi berbasis machine learning dan insights bisnis yang dapat mendukung pengambilan keputusan strategis.

Data yang digunakan merupakan data historis real dari tahun 2020 hingga 2025, mencakup periode-periode penting seperti pandemi COVID-19, perang Ukraina, dan berbagai kebijakan energi nasional.

## Mengapa Project Ini Menarik?

**Problem yang diselesaikan:**
- Bagaimana cara memantau dan menganalisis tren harga minyak secara real-time?
- Berapa lama waktu yang dibutuhkan hingga kenaikan harga minyak global berdampak pada BBM domestik?
- Berapa besar beban subsidi yang harus ditanggung pemerintah?
- Bagaimana memprediksi pergerakan harga di masa depan untuk perencanaan yang lebih baik?

**Solusi yang ditawarkan:**
Dashboard ini menyediakan 5 modul analisis lengkap mulai dari overview harga, analisis tren, korelasi antar variabel, prediksi menggunakan machine learning, hingga business intelligence untuk menghitung estimasi subsidi dan margin.

## Fitur

### 1. Overview & Monitoring
Tampilan ringkasan harga terkini dengan perbandingan perubahan 30 hari terakhir. Dilengkapi dengan grafik interaktif multi-layer yang menampilkan pergerakan harga minyak mentah, kurs rupiah, dan harga BBM dalam satu timeline.

### 2. Analisis Tren
Modul ini menyediakan berbagai metode analisis tren seperti moving average, volatilitas, dan spread harga Brent-WTI. Terdapat juga timeline yang menandai event-event penting yang mempengaruhi harga seperti lockdown COVID-19 atau invasi Rusia ke Ukraina.

### 3. Korelasi & Lag Analysis
Mengungkap hubungan matematis antar variabel melalui correlation matrix dan scatter plot. Yang paling menarik adalah fitur lag analysis yang dapat mendeteksi berapa hari delay antara kenaikan harga minyak dengan kenaikan harga BBM domestik - informasi krusial untuk perencanaan strategis.

### 4. Prediksi Harga
Menggunakan model regresi linear untuk memprediksi harga 7-90 hari ke depan, dilengkapi dengan confidence interval untuk mengukur tingkat kepercayaan prediksi. User dapat memilih komoditas yang ingin diprediksi dan menyesuaikan periode forecast.

### 5. Business Intelligence
Modul analisis bisnis yang menghitung estimasi subsidi Pertalite per liter, margin Pertamax, dan elastisitas harga. Dashboard juga memberikan rekomendasi strategis berbasis kondisi pasar terkini.

## Tech Stack

- **Python 3.8+** - Bahasa pemrograman utama
- **Streamlit** - Framework untuk web dashboard interaktif
- **Plotly** - Library visualisasi data interaktif
- **Pandas** - Data manipulation dan analysis
- **Scikit-learn** - Machine learning untuk prediksi
- **NumPy** - Numerical computing

## Instalasi

### Prerequisites
Pastikan Python 3.8 atau lebih tinggi sudah terinstall di sistem Anda.

### Clone Repository
```bash
git clone https://github.com/username/dashboard-minyak-bbm.git
cd dashboard-minyak-bbm
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

### Jalankan Dashboard
```bash
streamlit run dashboard_pertamina.py
```

Dashboard akan otomatis terbuka di browser pada `http://localhost:8501`

## Struktur Project

```
dashboard-minyak-bbm/
│
├── dashboard_pertamina.py      # File utama dashboard
├── data_master_minyak_bbm.csv  # Dataset lengkap
├── requirements.txt            # Dependencies
├── README.md                   # Dokumentasi
└── .gitignore                  # Git ignore file
```

## Sumber Data

Data yang digunakan dalam project ini berasal dari sumber-sumber kredibel:

1. **Harga Minyak Brent & WTI**
   - Sumber: FRED (Federal Reserve Economic Data)
   - URL: https://fred.stlouisfed.org
   - Coverage: Januari 2020 - Oktober 2025
   - Frekuensi: Harian

2. **Kurs USD/IDR**
   - Sumber: Bank Indonesia
   - URL: https://www.bi.go.id/id/statistik/informasi-kurs/transaksi-bi/default.aspx
   - Coverage: Januari 2020 - Oktober 2025
   - Frekuensi: Harian (kurs transaksi)

3. **Harga BBM Indonesia**
   - Sumber: Data historis dari berbagai press release resmi Pertamina dan berita terpercaya
   - Coverage: Januari 2020 - Oktober 2025
   - Produk: Pertalite, Pertamax, Solar

Total dataset mencakup 1,477 hari dengan 25 variabel analisis.

## Insight Menarik dari Data

Beberapa temuan menarik dari analisis data:

1. **Lag Time BBM**: Berdasarkan analisis korelasi, terdapat delay sekitar 7-14 hari antara perubahan harga minyak mentah global dengan penyesuaian harga BBM non-subsidi di Indonesia.

2. **Subsidi Pertalite**: Estimasi subsidi Pertalite per liter bervariasi dari Rp 0 hingga lebih dari Rp 3,000 tergantung pada harga minyak global dan kebijakan pemerintah.

3. **Elastisitas Harga**: Harga Pertamax (non-subsidi) memiliki elastisitas yang lebih tinggi terhadap perubahan harga Brent dibandingkan dengan Pertalite (subsidi), menunjukkan bahwa Pertamax lebih responsif terhadap dinamika pasar global.

4. **Event Impact**: Pandemi COVID-19 (Maret 2020) menyebabkan harga minyak Brent jatuh hingga USD 9.12 per barrel - level terendah dalam 5 tahun terakhir. Sebaliknya, perang Ukraina (Februari 2022) mendorong harga hingga USD 133.18 per barrel.

## Cara Menggunakan

### Filter Data
Gunakan sidebar untuk memfilter data berdasarkan:
- Rentang tanggal
- Tahun tertentu
- Komoditas yang ingin ditampilkan

### Navigasi Tab
Dashboard memiliki 5 tab utama yang dapat diakses di bagian atas:
- Overview: Lihat ringkasan dan grafik historis
- Analisis Trend: Eksplorasi pola pergerakan harga
- Korelasi: Analisis hubungan antar variabel
- Prediksi: Forecast harga masa depan
- Business Intelligence: Insights bisnis dan rekomendasi

### Export Data
Semua grafik dapat di-export sebagai image (PNG) dengan mengklik menu di pojok kanan atas setiap chart.

## Limitasi

1. **Model Prediksi**: Model regresi linear yang digunakan cukup sederhana. Untuk prediksi yang lebih akurat, bisa dikembangkan menggunakan LSTM atau Prophet.

2. **Data BBM**: Harga BBM subsidi (Pertalite, Solar) menggunakan pendekatan berdasarkan kebijakan historis karena data granular harian sulit didapatkan. Perubahan hanya terjadi saat ada pengumuman resmi dari pemerintah.

3. **Faktor External**: Model belum memperhitungkan faktor geopolitik, sentimen pasar, atau kebijakan OPEC+ yang dapat mempengaruhi harga minyak secara signifikan.

## Kontribusi

Kontribusi sangat terbuka! Beberapa area yang bisa dikembangkan:
- Implementasi model ML yang lebih sophisticated (LSTM, Prophet)
- Integrasi real-time data via API
- Penambahan analisis sentimen dari berita
- Multi-scenario forecasting
- Export report ke PDF
- Mobile responsive design

Silakan buat issue atau pull request jika ingin berkontribusi.

## Lisensi

Project ini menggunakan lisensi MIT. Silakan gunakan untuk keperluan pembelajaran atau komersial dengan tetap mencantumkan atribusi.

## Kontak
kailahidayatussakinah@gmail.com

---

**Catatan:** Dashboard ini dibuat sebagai portfolio project dan untuk tujuan pembelajaran. Estimasi subsidi dan margin yang ditampilkan merupakan perhitungan simplified dan tidak mencerminkan angka aktual dari Pertamina atau pemerintah.
