import json
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from joblib import Parallel, delayed

import joblib
import pandas as pd
import streamlit as st
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score

from src.preprocessing import preprocess_text

ARTIFACTS_DIR = Path("artifacts") / "new_artifacts"


# =========================
# UTIL
# =========================
def _preprocess_texts_batch(texts: list) -> list:
    """
    Preprocess multiple texts dengan parallel processing untuk performa optimal.
    Menggunakan joblib.Parallel untuk multi-core processing.
    Safe untuk Streamlit dengan fallback ke sequential processing.
    """
    if len(texts) < 10:
        # Untuk dataset kecil, gunakan sequential processing (overhead not worth)
        return [preprocess_text(text)["stemmed_text"] for text in texts]
    
    # Untuk dataset medium-besar, coba gunakan parallel processing
    try:
        # Gunakan threading backend untuk Windows compatibility
        # n_jobs=-1 artinya gunakan semua cores, tapi capped at 4 untuk safety
        n_jobs = min(4, -1)
        
        results = Parallel(n_jobs=n_jobs, backend="threading", verbose=0)(
            delayed(preprocess_text)(text) for text in texts
        )
        return [r["stemmed_text"] for r in results]
    except Exception:
        # Fallback ke sequential processing jika parallel fails
        # (bisa karena threading issues di Streamlit atau env issues)
        return [preprocess_text(text)["stemmed_text"] for text in texts]


def _normalize_label(x):
    s = str(x).strip().lower()
    if s in ["1", "positif", "positive", "pos"]:
        return 1
    if s in ["0", "negatif", "negative", "neg"]:
        return 0
    return None


def _get_file_hash(filepath: Path) -> str:
    """Get file modification time as cache key to clear cache when file changes"""
    if filepath.exists():
        return str(os.path.getmtime(filepath))
    return "0"


# =========================
# SINGLE MODEL (BEST MODEL)
# =========================
@st.cache_resource
def load_model_and_vectorizer():
    # Cache busting: use file modification time as key
    # This ensures cache is cleared when pkl files are updated
    _ = _get_file_hash(ARTIFACTS_DIR / "sv_tuned_no_smote.pkl")
    _ = _get_file_hash(ARTIFACTS_DIR / "tfidf_vectorizer.pkl")
    
    model = joblib.load(ARTIFACTS_DIR / "sv_tuned_no_smote.pkl")
    vectorizer = joblib.load(ARTIFACTS_DIR / "tfidf_vectorizer.pkl")
    return model, vectorizer


def predict_single_text(text: str) -> Dict[str, object]:
    model, vectorizer = load_model_and_vectorizer()
    prep = preprocess_text(text)

    X = vectorizer.transform([prep["stemmed_text"]])
    pred = int(model.predict(X)[0])

    prob_neg = None
    prob_pos = None

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        prob_neg = float(probs[0])
        prob_pos = float(probs[1])

    return {
        "preprocessing": prep,
        "prediction": pred,
        "label": "Positif" if pred == 1 else "Negatif",
        "prob_neg": prob_neg,
        "prob_pos": prob_pos,
    }


# =========================
# MULTI MODEL (5 MODEL 🔥)
# =========================
@st.cache_resource
def load_models_multi():
    # Cache busting: use file modification time as key
    # This ensures cache is cleared when pkl files are updated
    _ = _get_file_hash(ARTIFACTS_DIR / "tfidf_vectorizer.pkl")
    _ = _get_file_hash(ARTIFACTS_DIR / "sv_baseline_smote.pkl")
    _ = _get_file_hash(ARTIFACTS_DIR / "sv_optimized_smote.pkl")
    _ = _get_file_hash(ARTIFACTS_DIR / "sv_tuned_no_smote.pkl")
    _ = _get_file_hash(ARTIFACTS_DIR / "lr_compare_tuned.pkl")
    _ = _get_file_hash(ARTIFACTS_DIR / "rf_compare_tuned.pkl")
    
    models = {
        "Soft Voting Baseline": joblib.load(ARTIFACTS_DIR / "sv_baseline_smote.pkl"),
        "Soft Voting Optimasi Awal": joblib.load(ARTIFACTS_DIR / "sv_optimized_smote.pkl"),
        "Soft Voting Tuned": joblib.load(ARTIFACTS_DIR / "sv_tuned_no_smote.pkl"),
        "Logistic Regression Tuned": joblib.load(ARTIFACTS_DIR / "lr_compare_tuned.pkl"),
        "Random Forest Tuned": joblib.load(ARTIFACTS_DIR / "rf_compare_tuned.pkl"),
    }

    vectorizer = joblib.load(ARTIFACTS_DIR / "tfidf_vectorizer.pkl")
    return models, vectorizer


def predict_single_text_multi(text: str):
    models, vectorizer = load_models_multi()
    prep = preprocess_text(text)

    X = vectorizer.transform([prep["stemmed_text"]])

    results = {}

    for name, model in models.items():
        pred = int(model.predict(X)[0])

        prob_neg = None
        prob_pos = None

        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)[0]
            prob_neg = float(probs[0])
            prob_pos = float(probs[1])

        results[name] = {
            "prediction": pred,
            "label": "Positif" if pred == 1 else "Negatif",
            "prob_neg": prob_neg,
            "prob_pos": prob_pos,
        }

    return prep, results


# =========================
# DATASET SINGLE MODEL
# =========================
def process_dataset(df: pd.DataFrame, text_col: str, label_col: Optional[str] = None) -> Tuple[pd.DataFrame, Dict[str, object]]:
    working_df = df.copy()
    working_df[text_col] = working_df[text_col].fillna("").astype(str)

    rows = []
    for text in working_df[text_col].tolist():
        result = predict_single_text(text)
        prep = result["preprocessing"]

        rows.append({
            "original_text": text,
            "cleaned_text": prep["cleaned"],
            "normalized_text": prep["normalized"],
            "stemmed_text": prep["stemmed_text"],
            "predicted_label": result["label"],
            "predicted_num": result["prediction"],
            "prob_neg": result["prob_neg"],
            "prob_pos": result["prob_pos"],
        })

    result_df = pd.DataFrame(rows)

    metrics = {
        "total_data": int(len(result_df)),
        "pred_positif": int((result_df["predicted_num"] == 1).sum()),
        "pred_negatif": int((result_df["predicted_num"] == 0).sum()),
        "has_ground_truth": False,
        "accuracy": None,
        "precision": None,
        "recall": None,
        "f1": None,
        "confusion_matrix": None,
        "classification_report": None,
    }

    if label_col and label_col in working_df.columns:
        y_true = working_df[label_col].apply(_normalize_label)
        valid_mask = y_true.notna()

        if valid_mask.any():
            y_true_valid = y_true[valid_mask].astype(int)
            y_pred_valid = result_df.loc[valid_mask, "predicted_num"].astype(int)

            metrics.update({
                "has_ground_truth": True,
                "accuracy": float(accuracy_score(y_true_valid, y_pred_valid)),
                "precision": float(precision_score(y_true_valid, y_pred_valid, zero_division=0)),
                "recall": float(recall_score(y_true_valid, y_pred_valid, zero_division=0)),
                "f1": float(f1_score(y_true_valid, y_pred_valid, zero_division=0)),
                "confusion_matrix": confusion_matrix(y_true_valid, y_pred_valid, labels=[0, 1]).tolist(),
                "classification_report": classification_report(
                    y_true_valid,
                    y_pred_valid,
                    target_names=["Negatif", "Positif"],
                    output_dict=True,
                    zero_division=0,
                ),
            })

    return result_df, metrics


# =========================
# DATASET MULTI MODEL 🔥
# =========================
def process_dataset_multi(df: pd.DataFrame, text_col: str):
    models, vectorizer = load_models_multi()

    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)

    processed_texts = []
    for text in df[text_col]:
        prep = preprocess_text(text)
        processed_texts.append(prep["stemmed_text"])

    X = vectorizer.transform(processed_texts)

    for name, model in models.items():
        preds = model.predict(X)
        df[f"{name}"] = ["Positif" if p == 1 else "Negatif" for p in preds]

    return df


# =========================
# DATASET PREDICTION (NO EVALUATION) 🔥
# =========================
def predict_dataset_multi(df: pd.DataFrame, text_col: str, model_dict: Dict = None, vectorizer_obj = None) -> Tuple[pd.DataFrame, list, list]:
    """
    Prediksi dataset menggunakan selected models tanpa memerlukan ground truth labels.
    Optimized dengan batch processing + parallel preprocessing untuk kecepatan maksimal.
    
    Args:
        df: DataFrame dengan data untuk diprediksi
        text_col: Nama kolom teks
        model_dict: Dictionary of models {name: model_obj}. Jika None, gunakan semua model
        vectorizer_obj: Vectorizer object. Jika None, load dari artifacts
    
    Returns: (result_df, processed_texts, all_predictions_ensemble)
    """
    if model_dict is None:
        models, vectorizer = load_models_multi()
    else:
        models = model_dict
        vectorizer = vectorizer_obj if vectorizer_obj is not None else load_models_multi()[1]
    
    df = df.copy()
    df[text_col] = df[text_col].fillna("").astype(str)
    
    # ===== BATCH PREPROCESSING (PARALLELIZED) =====
    # Preprocess semua text sekaligus dengan parallel processing
    processed_texts = _preprocess_texts_batch(df[text_col].tolist())
    
    # ===== BATCH VECTORIZATION =====
    # Vectorize semua text sekaligus (bukan per-row)
    X = vectorizer.transform(processed_texts)
    
    # ===== BATCH PREDICTION UNTUK SEMUA MODEL =====
    # Prediksi semua data untuk setiap model sekaligus
    model_predictions = {}
    model_probabilities = {}
    
    for model_name, model in models.items():
        preds = model.predict(X)
        model_predictions[model_name] = preds
        
        if hasattr(model, "predict_proba"):
            probs = model.predict_proba(X)
            model_probabilities[model_name] = probs[:, 1]  # Probabilitas kelas positif
        else:
            model_probabilities[model_name] = None
    
    # ===== BUILD RESULT DATAFRAME =====
    rows = []
    all_predictions = []
    
    # Check if single model atau multiple models
    is_single_model = len(models) == 1
    
    for idx, text in enumerate(df[text_col]):
        row = {
            "No": idx + 1,
            "Original Text": text,
            "Stemmed Text": processed_texts[idx]
        }
        
        predictions = []
        
        # Collect predictions dari selected model(s)
        for model_name in models.keys():
            pred = int(model_predictions[model_name][idx])
            predictions.append(pred)
            
            # Get probability
            prob_pos = model_probabilities[model_name][idx] if model_probabilities[model_name] is not None else None
            
            # Jika single model, gunakan format yang lebih simple (seperti komentar tunggal)
            if is_single_model:
                row["Prediction"] = "Positif" if pred == 1 else "Negatif"
                row["Probability (Positif)"] = f"{prob_pos:.2%}" if prob_pos is not None else "N/A"
                if prob_pos is not None:
                    row["Probability (Negatif)"] = f"{(1-prob_pos):.2%}"
            else:
                row[f"{model_name}"] = "Positif" if pred == 1 else "Negatif"
                row[f"{model_name} (Prob)"] = f"{prob_pos:.2%}" if prob_pos is not None else "N/A"
        
        # Ensemble prediction hanya untuk multiple models
        if not is_single_model:
            ensemble_pred = 1 if sum(predictions) >= len(predictions) / 2 else 0
            row["Ensemble Prediction"] = "Positif" if ensemble_pred == 1 else "Negatif"
            row["Ensemble Voting Count"] = sum(predictions)
            all_predictions.append(ensemble_pred)
        else:
            # Single model - gunakan prediksi dari model tersebut
            all_predictions.append(predictions[0])
        
        rows.append(row)
    
    result_df = pd.DataFrame(rows)
    return result_df, processed_texts, all_predictions


# =========================
# LDA (TETAP)
# =========================
def build_lda_topics(text_series: pd.Series, n_topics: int = 5, n_top_words: int = 10):
    text_series = text_series.fillna("").astype(str)
    text_series = text_series[text_series.str.strip() != ""]
    if len(text_series) < 5:
        return [], []

    vectorizer = CountVectorizer(max_features=2000, ngram_range=(1, 1), min_df=2, max_df=0.95)
    X = vectorizer.fit_transform(text_series)

    if X.shape[0] == 0 or X.shape[1] == 0:
        return [], []

    lda = LatentDirichletAllocation(n_components=n_topics, random_state=42, learning_method="batch")
    lda.fit(X)

    feature_names = vectorizer.get_feature_names_out()

    topics = []
    for i, comp in enumerate(lda.components_):
        top_idx = comp.argsort()[:-n_top_words - 1:-1]
        top_words = [feature_names[j] for j in top_idx]
        topics.append({"topic": i + 1, "words": ", ".join(top_words)})

    dominant_topics = lda.transform(X).argmax(axis=1)
    counts = pd.Series(dominant_topics).value_counts().sort_index()

    distribution = [{"topic": i + 1, "count": int(counts.get(i, 0))} for i in range(n_topics)]

    return topics, distribution