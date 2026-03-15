# 📊 Personal Stock Analyzer IHSG

> **Value Investing Dashboard** untuk Pasar Saham Indonesia — Powered by DQC Logic

Dashboard analisis saham Indonesia (IHSG) yang dibangun dengan filosofi **Value Investing** (Benjamin Graham & Warren Buffett) menggunakan **zero-cost stack**.

---

## 🏗️ Tech Stack

| Komponen | Teknologi | Biaya |
|----------|-----------|-------|
| Data Harga & Fundamental | `yfinance` | ✅ Gratis |
| Analisis AI | Google Gemini API (Free Tier) | ✅ Gratis (15 RPM) |
| Analisis Teknikal | `pandas_ta` | ✅ Gratis |
| Visualisasi | `plotly` | ✅ Gratis |
| Interface | `streamlit` | ✅ Gratis |

---

## 📁 Struktur Proyek

```
Investing/
├── .env.example          # Template API Key (JANGAN commit .env asli!)
├── .gitignore            # Proteksi file sensitif
├── requirements.txt      # Dependensi Python
├── config.py             # Konfigurasi terpusat
├── app.py                # Dashboard utama (Streamlit)
├── modules/
│   ├── __init__.py
│   ├── fundamental.py    # Modul 1: Analisis Fundamental
│   ├── ai_assistant.py   # Modul 2: AI Gemini PDF Analyzer
│   └── technical.py      # Modul 3: Sinyal Teknikal (RSI, MA)
└── README.md             # File ini
```

---

## 🚀 Cara Install & Menjalankan

### 1. Install Dependencies

```bash
# Pastikan Python 3.9+ terinstall
pip install -r requirements.txt
```

### 2. Setup Gemini API Key (GRATIS)

Langkah mendapatkan API Key:

1. Buka **[Google AI Studio](https://aistudio.google.com/app/apikey)**
2. Login dengan **akun Google** (Gmail biasa, gratis)
3. Klik **"Create API Key"**
4. Pilih project atau buat baru, lalu klik **"Create API Key in new project"**
5. **Copy** API Key yang muncul

Cara memasang di proyek (pilih salah satu):

**Opsi A — File `.env` (Rekomendasi untuk Development):**
```bash
# Copy template
cp .env.example .env

# Edit file .env, ganti dengan API Key asli
GEMINI_API_KEY=AIzaSy....your_actual_key_here
```

**Opsi B — Langsung di Dashboard:**
- Jalankan aplikasi, lalu masukkan API Key di **sidebar** → kolom "Gemini API Key"
- API Key hanya tersimpan selama sesi browser berjalan (lebih aman, tidak tersimpan di file)

> ⚠️ **PENTING**: Jangan pernah commit file `.env` ke Git! File `.gitignore` sudah dikonfigurasi untuk mengabaikannya.

### 3. Jalankan Aplikasi

```bash
streamlit run app.py
```

Dashboard akan terbuka otomatis di browser: `http://localhost:8501`

---

## 📊 Fitur Dashboard

### Tab 1 — Overview
- Informasi perusahaan (sektor, industri, market cap)
- Grafik candlestick interaktif dengan MA20 overlay
- Volume perdagangan

### Tab 2 — Fundamental
- Rasio keuangan + penilaian Graham (PBV, PE, D/E, ROE)
- Tabel laporan keuangan annual
- Chart Revenue vs Net Income

### Tab 3 — Technical
- RSI (14) dengan zona oversold/overbought
- Moving Average (20) dengan sinyal uptrend/downtrend
- Overall signal (kombinasi RSI + MA)

### Tab 4 — AI Analysis
- **Quick Analysis**: Analisis cepat dari data yfinance (tanpa PDF)
- **PDF Analysis**: Upload laporan keuangan → Gemini meringkas 3 Alasan Beli & 3 Risiko

---

## 🎓 Penjelasan DQC (Data Quality Control)

Sistem ini didesain dengan mindset **audit kualitas data**:

| Prinsip DQC | Implementasi di Kode |
|-------------|---------------------|
| **Completeness** | Setiap field dicek `None`/`N/A` sebelum ditampilkan |
| **Accuracy** | Kalkulasi rasio divalidasi (persentase × 100, pembulatan) |
| **Consistency** | Format angka konsisten (Triliun/Miliar/Juta) |
| **Traceability** | Setiap fungsi didokumentasi sumber datanya |
| **Timeliness** | Cache 5 menit (`@st.cache_data(ttl=300)`) untuk menghindari request berlebihan |

---

## 📝 Contoh Ticker IHSG

| Ticker | Perusahaan |
|--------|-----------|
| `BBCA.JK` | Bank Central Asia |
| `BBRI.JK` | Bank Rakyat Indonesia |
| `TLKM.JK` | Telkom Indonesia |
| `LSIP.JK` | London Sumatra |
| `TAPG.JK` | Triputra Agro Persada |
| `ASII.JK` | Astra International |

---

## ⚠️ Disclaimer

Aplikasi ini bersifat **informatif** dan **BUKAN rekomendasi investasi**. Selalu lakukan riset mandiri (*Do Your Own Research / DYOR*) sebelum mengambil keputusan investasi.

---

*Built with ❤️ using Python, Streamlit, and the DQC Mindset*
