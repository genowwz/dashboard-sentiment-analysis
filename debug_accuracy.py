#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Debug script untuk check akurasi SV Improved model"""

import pandas as pd
from pathlib import Path
from src.preprocessing import preprocess_text
from sklearn.metrics import accuracy_score
import joblib

# Load data
df = pd.read_csv(Path("artifacts/test_set_skripsi.csv"))
print(f"Total rows: {len(df)}")

# Preprocess texts
print("Preprocessing texts...")
processed = []
for text in df['text'].fillna("").astype(str):
    stemmed_text = preprocess_text(text)["stemmed_text"]
    processed.append(stemmed_text)

# Load vectorizer and model
print("Loading vectorizer and model...")
vectorizer = joblib.load(Path("artifacts/new_artifacts/tfidf_vectorizer.pkl"))
model = joblib.load(Path("artifacts/new_artifacts/sv_tuned_no_smote.pkl"))

# Transform
print("Transforming data...")
X_eval = vectorizer.transform(processed)

# Labels
y_true = df['label'].values

# Predict
print("Making predictions...")
preds = model.predict(X_eval)

# Calculate accuracy
acc = accuracy_score(y_true, preds)

print(f"\n=== HASIL ===")
print(f"Akurasi (full precision): {acc}")
print(f"Akurasi (10 decimal): {round(acc, 10)}")
print(f"Akurasi (6 decimal): {round(acc, 6)}")
print(f"Akurasi (4 decimal): {round(acc, 4)}")
print(f"\nTotal predictions: {len(preds)}")
print(f"Correct predictions: {(preds == y_true).sum()}")
print(f"Accuracy % : {(preds == y_true).sum() / len(preds) * 100:.4f}%")
