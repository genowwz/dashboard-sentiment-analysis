"""
Jalankan script ini SETELAH notebook training selesai.
Ubah variabel `best_model`, `vectorizer`, `kamus_tidak_baku`, dll agar sesuai nama variabel di notebook kamu.
"""

import json
import os
import joblib

os.makedirs("artifacts", exist_ok=True)

# Ganti nama variabel ini sesuai notebook kamu
# best_model = soft_voting_improved_no_smote
# vectorizer = vectorizer
# kamus_tidak_baku = {...}
# custom_stopwords = [...]
# protected_words = [...]
# special_cases = {...}

joblib.dump(best_model, "artifacts/model.pkl")
joblib.dump(vectorizer, "artifacts/tfidf_vectorizer.pkl")

with open("artifacts/slang_dict.json", "w", encoding="utf-8") as f:
    json.dump(kamus_tidak_baku, f, ensure_ascii=False, indent=2)

with open("artifacts/custom_stopwords.json", "w", encoding="utf-8") as f:
    json.dump(custom_stopwords, f, ensure_ascii=False, indent=2)

with open("artifacts/protected_words.json", "w", encoding="utf-8") as f:
    json.dump(protected_words, f, ensure_ascii=False, indent=2)

with open("artifacts/special_cases.json", "w", encoding="utf-8") as f:
    json.dump(special_cases, f, ensure_ascii=False, indent=2)

print("Semua artifact berhasil disimpan ke folder artifacts/")
