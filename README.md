# Refactor Streamlit Dashboard Sederhana

Versi ini mempertahankan gaya dashboard lama, tapi logic-nya diubah supaya:

- admin upload dan memproses dataset dari raw text
- hasil terakhir disimpan ke `models/latest_result.json`
- user publik hanya melihat hasil terakhir
- semua orang bisa mencoba analisis 1 komentar tunggal
- model tidak dilatih ulang di dashboard

## Struktur penting

- `streamlit_app.py` → app utama
- `src/preprocessing.py` → pipeline preprocessing
- `src/inference.py` → load model, predict single text, process dataset, LDA sederhana
- `artifacts/` → isi dengan `model.pkl`, `tfidf_vectorizer.pkl`, dan file JSON preprocessing
- `models/latest_result.json` → penyimpanan hasil dashboard terakhir

## Yang harus kamu siapkan dulu

Dari notebook training, simpan:
- `model.pkl`
- `tfidf_vectorizer.pkl`
- `slang_dict.json`
- `custom_stopwords.json`
- `protected_words.json`
- `special_cases.json`

Lihat contoh di `export_artifacts_example.py`.

## Jalankan app

```bash
pip install -r requirements.txt
streamlit run streamlit_app.py
```

## Login admin

- username: `admin`
- password: `admin123`

## Format dataset yang paling aman

Minimal punya 1 kolom teks mentah, misalnya:
- `komentar`
- `comment`

Kalau ada label asli, boleh tambah kolom:
- `Sentiment`

Label yang didukung:
- `Positif` / `Negatif`
- `1` / `0`

## 🚀 Deployment

Untuk deploy ke Streamlit Cloud, lihat [DEPLOYMENT_GUIDE.md](DEPLOYMENT_GUIDE.md)

**Quick Start Deployment:**
1. Pastikan semua files di artifacts/ sudah ada (model.pkl, vectorizer, JSON files)
2. Push ke GitHub: `git push`
3. Ke https://streamlit.io/cloud → Deploy dari GitHub repo
4. App akan live di: `https://share.streamlit.io/genowwz/dashboard-sentiment-analysis/main/streamlit_app.py`

## Catatan

Kalau kamu belum sempat ekspor `slang_dict.json`, app tetap bisa jalan, tapi bagian normalisasi slang jadi tidak maksimal.
