#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Verification script untuk check apakah pkl local match dengan Colab
"""

import pandas as pd
from pathlib import Path
from src.preprocessing import preprocess_text
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

print("="*60)
print("VERIFIKASI PKL FILE LOCAL VS COLAB")
print("="*60)

# Load data
df = pd.read_csv(Path("artifacts/test_set_skripsi.csv"))
print(f"\nTotal data: {len(df)}")

# Preprocess
print("\n📊 Preprocessing texts...")
processed = []
for text in df['text'].fillna("").astype(str):
    stemmed_text = preprocess_text(text)["stemmed_text"]
    processed.append(stemmed_text)
print(f"✅ Selesai preprocessing {len(processed)} texts")

# Load vectorizer dan model
print("\n🔧 Loading pkl files...")
try:
    model = joblib.load(Path("artifacts/new_artifacts/sv_tuned_smote.pkl"))
    print(f"✅ sv_tuned_smote.pkl loaded")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

try:
    vectorizer = joblib.load(Path("artifacts/new_artifacts/tfidf_vectorizer.pkl"))
    print(f"✅ tfidf_vectorizer.pkl loaded")
except Exception as e:
    print(f"❌ Error loading vectorizer: {e}")
    exit(1)

# Transform
print("\n🔄 Transforming data...")
X_eval = vectorizer.transform(processed)
print(f"✅ Data transformed: {X_eval.shape}")

# Labels
y_true = df['label'].values

# Predict
print("\n🤖 Making predictions...")
y_pred = model.predict(X_eval)

# Calculate metrics
acc = accuracy_score(y_true, y_pred)
cm = confusion_matrix(y_true, y_pred)

print(f"\n" + "="*60)
print(f"HASIL PREDIKSI LOCAL")
print(f"="*60)
print(f"Accuracy: {acc:.6f}")
print(f"\nConfusion Matrix:")
print(f"┌─────────────────┬──────────┬──────────┐")
print(f"│                 │ Pred Neg │ Pred Pos │")
print(f"├─────────────────┼──────────┼──────────┤")
print(f"│ Actual Negative │   {cm[0,0]:3d}    │   {cm[0,1]:3d}    │")
print(f"│ Actual Positive │   {cm[1,0]:3d}    │   {cm[1,1]:3d}    │")
print(f"└─────────────────┴──────────┴──────────┘")

# Compare dengan expected dari Colab
print(f"\n" + "="*60)
print(f"EXPECTED (DARI COLAB)")
print(f"="*60)
expected_cm = {
    'TN': 427,
    'FP': 80,
    'FN': 75,
    'TP': 214
}

print(f"┌─────────────────┬──────────┬──────────┐")
print(f"│                 │ Pred Neg │ Pred Pos │")
print(f"├─────────────────┼──────────┼──────────┤")
print(f"│ Actual Negative │   {expected_cm['TN']:3d}    │   {expected_cm['FP']:3d}    │")
print(f"│ Actual Positive │   {expected_cm['FN']:3d}    │   {expected_cm['TP']:3d}    │")
print(f"└─────────────────┴──────────┴──────────┘")

# Check differences
print(f"\n" + "="*60)
print(f"PERBANDINGAN")
print(f"="*60)
tn_diff = cm[0,0] - expected_cm['TN']
fp_diff = cm[0,1] - expected_cm['FP']
fn_diff = cm[1,0] - expected_cm['FN']
tp_diff = cm[1,1] - expected_cm['TP']

print(f"TN: {cm[0,0]} vs {expected_cm['TN']} (diff: {tn_diff:+d})")
print(f"FP: {cm[0,1]} vs {expected_cm['FP']} (diff: {fp_diff:+d})")
print(f"FN: {cm[1,0]} vs {expected_cm['FN']} (diff: {fn_diff:+d})")
print(f"TP: {cm[1,1]} vs {expected_cm['TP']} (diff: {tp_diff:+d})")

total_diff = abs(tn_diff) + abs(fp_diff) + abs(fn_diff) + abs(tp_diff)

print(f"\n" + "="*60)
if total_diff == 0:
    print(f"✅ PERFECT MATCH! Pkl file sama dengan Colab")
else:
    print(f"⚠️ MISMATCH! Ada {total_diff//2} data perbedaan")
    print(f"\nSolusi:")
    print(f"1. Pastikan pkl dari Colab adalah hasil retrain DENGAN preprocessing baru")
    print(f"2. Download ulang sv_tuned_smote.pkl dari Colab")
    print(f"3. Replace di local /artifacts/new_artifacts/sv_tuned_smote.pkl")
    print(f"4. Klik 'Reload Models' di dashboard")
print(f"="*60)
