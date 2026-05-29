#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script untuk find perbedaan 1 data antara dashboard dan colab result
"""

import pandas as pd
from pathlib import Path
from src.preprocessing import preprocess_text
from sklearn.metrics import confusion_matrix
import joblib

# Load data
df = pd.read_csv(Path("artifacts/test_set_skripsi.csv"))

# Preprocess
print("Preprocessing texts...")
processed = []
for text in df['text'].fillna("").astype(str):
    stemmed_text = preprocess_text(text)["stemmed_text"]
    processed.append(stemmed_text)

# Load vectorizer dan model
print("Loading model and vectorizer...")
vectorizer = joblib.load(Path("artifacts/new_artifacts/tfidf_vectorizer.pkl"))
model = joblib.load(Path("artifacts/new_artifacts/sv_tuned_smote.pkl"))

# Transform
X_eval = vectorizer.transform(processed)

# Labels
y_true = df['label'].values

# Predictions
print("Making predictions...")
y_pred = model.predict(X_eval)

# Find mismatches
mismatches = y_pred != y_true
mismatch_indices = [i for i, m in enumerate(mismatches) if m]

print(f"\n=== HASIL ===")
print(f"Total: {len(y_true)}")
print(f"Benar: {(y_pred == y_true).sum()}")
print(f"Salah: {mismatches.sum()}")
print(f"\nConfusion Matrix:")
cm = confusion_matrix(y_true, y_pred)
print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")

# Bandingkan dengan expected dari dashboard
# Expected dari screenshot: TN=427, FP=80, FN=74, TP=215
expected_fn = 74
actual_fn = cm[1, 0]  # False Negatives
difference = actual_fn - expected_fn

print(f"\nExpected FN (Colab): {expected_fn}")
print(f"Actual FN (Local): {actual_fn}")
print(f"Difference: {difference}")

if difference > 0:
    print(f"\n❌ {difference} Positive data yang diprediksi sebagai Negative (tambahan FN)")
elif difference < 0:
    print(f"\n⚠️ {abs(difference)} Positive data yang seharusnya Negative (kurang FN)")
else:
    print("\n✅ Match! Tidak ada perbedaan")

# Save prediction results untuk analisis
results_df = df.copy()
results_df['predicted'] = y_pred
results_df['correct'] = y_pred == y_true
results_df['preprocessed'] = processed

# Save mismatches
if mismatch_indices:
    print(f"\n📍 Detail data yang BERBEDA:")
    for idx in mismatch_indices:
        print(f"\nIndex {idx}:")
        print(f"  Text: {df.iloc[idx]['text'][:100]}...")
        print(f"  True: {y_true[idx]} (Positif={1}, Negatif={0})")
        print(f"  Pred: {y_pred[idx]}")
        print(f"  Stemmed: {processed[idx][:100]}...")
