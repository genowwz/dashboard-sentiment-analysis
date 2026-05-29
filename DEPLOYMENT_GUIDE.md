# Deployment Guide - Dashboard Sentiment Analysis

## 📋 Checklist Persiapan Deployment

### ✅ 1. Local Verification
- [ ] Test app secara local: `streamlit run streamlit_app.py`
- [ ] Pastikan semua dependencies di `requirements.txt` sudah correct
- [ ] Test login admin dengan username: `admin`, password: `admin123`

### ✅ 2. Repository Preparation
- [ ] Push semua files ke GitHub ke repository: https://github.com/genowwz/dashboard-sentiment-analysis
- [ ] Pastikan `.gitignore` sudah dibuat (sudah dibuat ✓)
- [ ] Exclude: `venv/`, `__pycache__/`, `.streamlit/secrets.toml`, `uploads/`, model files (.pkl)

### ✅ 3. Critical Files Check
Pastikan files ini sudah ada di repository:
- [ ] `streamlit_app.py` - main app
- [ ] `requirements.txt` - dependencies
- [ ] `src/` folder dengan:
  - [ ] `preprocessing.py`
  - [ ] `inference.py`
  - [ ] `topic_modeling.py`
- [ ] `artifacts/` folder dengan:
  - [ ] `model.pkl` - trained model
  - [ ] `tfidf_vectorizer.pkl` - vectorizer
  - [ ] `slang_dict.json`
  - [ ] `custom_stopwords.json`
  - [ ] `protected_words.json`
  - [ ] `special_cases.json`

⚠️ **PENTING**: Files `.pkl` dan `.json` di artifacts/ HARUS ada, karena app membutuhkannya untuk berjalan!

### ✅ 4. Deploy ke Streamlit Cloud

1. **Sign Up / Login ke Streamlit Cloud**
   - Buka https://streamlit.io/cloud
   - Login dengan GitHub account

2. **Deploy App**
   - Klik "New app"
   - Pilih repository: `dashboard-sentiment-analysis`
   - Pilih branch: `main`
   - Set main file: `streamlit_app.py`
   - Klik "Deploy"

3. **Tunggu Deployment**
   - Process bisa 2-5 menit
   - Monitor logs di Streamlit Cloud dashboard

### ✅ 5. Post-Deployment Checks
- [ ] App berhasil di-deploy (status "Healthy")
- [ ] Coba akses public URL
- [ ] Test login admin
- [ ] Test upload CSV
- [ ] Test analisis single text

## 🔐 Security Notes

### Ganti Admin Password (Opsional)
Jika ingin ganti password, edit di `streamlit_app.py`:
```python
ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"  # ← Ganti ini
```

### Menggunakan Environment Variables (Recommended)
Untuk production yang lebih aman, gunakan Streamlit Secrets:

1. Buka https://share.streamlit.io/genowwz/dashboard-sentiment-analysis/main/streamlit_app.py (setelah deploy)
2. Settings → Secrets → Tambahkan:
```toml
admin_username = "admin"
admin_password = "your_secure_password"
```

3. Update code di `streamlit_app.py`:
```python
ADMIN_USERNAME = st.secrets["admin_username"]
ADMIN_PASSWORD = st.secrets["admin_password"]
```

## 📊 Deployment Options Comparison

| Platform | Cost | Setup | Ease | Best For |
|----------|------|-------|------|----------|
| **Streamlit Cloud** ⭐ | Free | Simple | Easy | Public Streamlit apps |
| AWS | $5-50/mo | Medium | Medium | Large-scale apps |
| Google Cloud | $0-50/mo | Medium | Medium | Scalable apps |
| Railway | $5/mo | Simple | Easy | Simple deployment |
| Render | Free/Paid | Simple | Easy | Simple deployment |

**Rekomendasi untuk Anda**: Gunakan **Streamlit Cloud** karena:
- ✅ Gratis untuk public repo
- ✅ Paling mudah setup
- ✅ Automatic deployment saat push ke GitHub
- ✅ Built-in monitoring & logs

## 🚨 Common Issues & Solutions

### Error: "ModuleNotFoundError: No module named 'src'"
**Solusi**: Pastikan structure folder sudah benar dan `.gitignore` tidak mengeclude folder `src/`

### Error: "No such file or directory: 'artifacts/model.pkl'"
**Solusi**: Pastikan semua model files ada di folder `artifacts/` dan sudah di-push ke GitHub

### Error: "Upload size exceeds max allowed size"
**Solusi**: Sudah dikonfigurasi max 200MB di `.streamlit/config.toml`

### App berjalan lambat
**Solusi**: 
- Model loading bisa di-cache dengan `@st.cache_resource`
- Sudah ada di `src/inference.py`

## 📝 Next Steps

1. **Push ke GitHub**
   ```bash
   git add .
   git commit -m "Prepare for deployment"
   git push origin main
   ```

2. **Deploy di Streamlit Cloud**
   - Ikuti langkah di section "Deploy ke Streamlit Cloud"

3. **Share dengan orang lain**
   - URL: `https://share.streamlit.io/genowwz/dashboard-sentiment-analysis/main/streamlit_app.py`

## 📞 Support

Jika ada masalah:
1. Check Streamlit Cloud logs
2. Verify `requirements.txt` sudah updated
3. Check GitHub repo ada semua files yang diperlukan

---

**Good luck with your deployment! 🚀**
