# =========================
# IMPORT
# =========================
import json
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
from wordcloud import WordCloud

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from src.inference import load_models_multi, predict_single_text_multi, predict_dataset_multi, _preprocess_texts_batch
from src.preprocessing import preprocess_text
from src.topic_modeling import generate_lda_topics_from_data, extract_top_words_by_sentiment

# =========================
# CONFIG
# =========================
BASE_DIR = Path(".")
ARTIFACTS = BASE_DIR / "artifacts"
UPLOAD_FOLDER = BASE_DIR / "uploads"
RESULT_FILE = ARTIFACTS / "latest_result.json"

ADMIN_USERNAME = "admin"
ADMIN_PASSWORD = "admin123"

ARTIFACTS.mkdir(exist_ok=True)
UPLOAD_FOLDER.mkdir(exist_ok=True)

st.set_page_config(
    page_title="Dashboard Analisis Sentimen",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =========================
# STYLE
# =========================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@300;400;500;600;700&display=swap');

* {
    font-family: 'Poppins', sans-serif;
}

.block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}
.metric-card {
    background: #667eea;
    color: white;
    border: none;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.08);
}
.metric-card-light {
    background: #f8fafc;
    border: 1px solid #e5e7eb;
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
}
.small-muted {
    color: #6b7280;
    font-size: 0.92rem;
}
.result-card {
    background: #ffffff;
    border-left: 4px solid #667eea;
    padding: 16px;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
    margin: 8px 0;
}
.prob-container {
    background: #f3f4f6;
    padding: 16px;
    border-radius: 8px;
    margin: 12px 0;
}
.sentiment-positive {
    color: #10b981;
    font-weight: 600;
}
.sentiment-negative {
    color: #ef4444;
    font-weight: 600;
}
.header-section {
    border-bottom: 1px solid #ddd;
    padding-bottom: 10px;
    margin-bottom: 20px;
}
</style>
""", unsafe_allow_html=True)

# =========================
# SESSION
# =========================
if "admin" not in st.session_state:
    st.session_state.admin = False

# 🔥 tambahan anti spam klik
if "processing" not in st.session_state:
    st.session_state.processing = False

# Cache latest result di session state supaya tidak selalu reload dari disk
if "latest_result_cache" not in st.session_state:
    st.session_state.latest_result_cache = None

# =========================
# HELPERS
# =========================
def save_latest_result(payload: dict):
    """Save latest result ke disk dan cache di session state"""
    # Ensure all data is JSON-serializable
    payload = _make_json_serializable(payload)
    with open(RESULT_FILE, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    # Update cache di session state
    st.session_state.latest_result_cache = payload

def _make_json_serializable(obj):
    """Convert numpy types dan objects lainnya ke format JSON-serializable"""
    if isinstance(obj, dict):
        return {k: _make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [_make_json_serializable(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    else:
        return obj

def load_latest_result():
    """Load latest result dari cache session state, atau dari disk jika cache kosong"""
    if st.session_state.latest_result_cache is None:
        if RESULT_FILE.exists():
            with open(RESULT_FILE, "r", encoding="utf-8") as f:
                st.session_state.latest_result_cache = json.load(f)
    return st.session_state.latest_result_cache

def normalize_labels(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().map({
        "positif": 1,
        "positive": 1,
        "1": 1,
        "negatif": 0,
        "negative": 0,
        "0": 0
    })

@st.cache_resource
def load_lda_models():
    """Load LDA models for positive and negative sentiments"""
    try:
        lda_pos_dict = joblib.load(ARTIFACTS / "lda_positif.pkl")
        lda_neg_dict = joblib.load(ARTIFACTS / "lda_negatif.pkl")
        
        # Extract LDA model and feature names from dictionary
        lda_pos = lda_pos_dict.get("lda_model") if isinstance(lda_pos_dict, dict) else lda_pos_dict
        lda_neg = lda_neg_dict.get("lda_model") if isinstance(lda_neg_dict, dict) else lda_neg_dict
        
        feature_names_pos = lda_pos_dict.get("feature_names") if isinstance(lda_pos_dict, dict) else None
        feature_names_neg = lda_neg_dict.get("feature_names") if isinstance(lda_neg_dict, dict) else None
        
        return lda_pos, lda_neg, feature_names_pos, feature_names_neg
    except Exception as e:
        st.error(f"Error loading LDA models: {e}")
        return None, None, None, None

def get_lda_topics(lda_model, feature_names, n_words=10):
    """Extract top words for each topic from LDA model"""
    if lda_model is None or feature_names is None:
        return []
    
    topics = []
    
    for topic_idx, topic in enumerate(lda_model.components_):
        top_indices = topic.argsort()[-n_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        top_weights = [topic[i] for i in top_indices]
        topics.append({
            "topic_id": topic_idx,
            "words": top_words,
            "weights": top_weights
        })
    
    return topics

def create_wordcloud_from_topic(words, weights, title="", sentiment_color="blue"):
    """Create wordcloud visualization from topic words and weights dengan style lebih baik"""
    if not words:
        return None
    
    # Create word frequency dictionary
    word_freq = {word: weight for word, weight in zip(words, weights)}
    
    # Generate wordcloud dengan parameter yang lebih baik
    if sentiment_color == "positive":
        # Green colormap untuk positif
        wordcloud = WordCloud(
            width=800,
            height=500,
            background_color="white",
            colormap="Greens",
            relative_scaling=0.5,
            min_font_size=20,
            max_font_size=120,
            prefer_horizontal=0.7
        ).generate_from_frequencies(word_freq)
    elif sentiment_color == "negative":
        # Red colormap untuk negatif
        wordcloud = WordCloud(
            width=800,
            height=500,
            background_color="white",
            colormap="Reds",
            relative_scaling=0.5,
            min_font_size=20,
            max_font_size=120,
            prefer_horizontal=0.7
        ).generate_from_frequencies(word_freq)
    else:
        # Blue default
        wordcloud = WordCloud(
            width=800,
            height=500,
            background_color="white",
            colormap="Blues",
            relative_scaling=0.5,
            min_font_size=20,
            max_font_size=120,
            prefer_horizontal=0.7
        ).generate_from_frequencies(word_freq)
    
    # Create matplotlib figure dengan ukuran lebih besar
    fig, ax = plt.subplots(figsize=(12, 7), dpi=100)
    ax.imshow(wordcloud, interpolation="bilinear")
    ax.axis("off")
    if title:
        ax.set_title(title, fontsize=14, fontweight="bold", pad=15)
    plt.tight_layout(pad=0)
    
    return fig

def extract_words_from_preprocessed_dataset(csv_path):
    """
    Extract word frequency dari dataset preprocessing yang sudah ada
    CSV harus memiliki kolom 'Sentiment' dan 'stemming'
    """
    try:
        df = pd.read_csv(csv_path)
        
        # Validasi kolom yang diperlukan
        if 'Sentiment' not in df.columns or 'stemming' not in df.columns:
            st.error("File CSV harus memiliki kolom 'Sentiment' dan 'stemming'")
            return {}, {}
        
        word_freq_neg = {}
        word_freq_pos = {}
        
        for idx, row in df.iterrows():
            sentiment = row['Sentiment']
            stemmed = row['stemming']
            
            # Parse stemmed text (bisa string atau list)
            if isinstance(stemmed, str):
                # Jika berupa string representasi list, parse dengan ast
                try:
                    import ast
                    words = ast.literal_eval(stemmed) if stemmed.startswith('[') else stemmed.split()
                except:
                    words = stemmed.split()
            else:
                words = str(stemmed).split()
            
            # Hitung frekuensi berdasarkan sentimen
            if sentiment == "Negatif" or sentiment == "negatif" or sentiment == "Negative":
                for word in words:
                    word_freq_neg[word] = word_freq_neg.get(word, 0) + 1
            elif sentiment == "Positif" or sentiment == "positif" or sentiment == "Positive":
                for word in words:
                    word_freq_pos[word] = word_freq_pos.get(word, 0) + 1
        
        # Sort dan ambil top words
        top_words_neg = dict(sorted(word_freq_neg.items(), key=lambda x: x[1], reverse=True)[:50])
        top_words_pos = dict(sorted(word_freq_pos.items(), key=lambda x: x[1], reverse=True)[:50])
        
        return top_words_pos, top_words_neg
    
    except Exception as e:
        st.error(f"Error membaca dataset preprocessing: {e}")
        return {}, {}

def extract_words_by_sentiment(processed_texts, predictions):
    """Extract word frequency dari stemmed text berdasarkan sentimen (0=negatif, 1=positif)"""
    word_freq_neg = {}
    word_freq_pos = {}
    
    for text, pred in zip(processed_texts, predictions):
        words = text.split()
        
        if pred == 0:  # Negatif
            for word in words:
                word_freq_neg[word] = word_freq_neg.get(word, 0) + 1
        else:  # Positif
            for word in words:
                word_freq_pos[word] = word_freq_pos.get(word, 0) + 1
    
    # Sort dan ambil top words
    top_words_neg = dict(sorted(word_freq_neg.items(), key=lambda x: x[1], reverse=True)[:50])
    top_words_pos = dict(sorted(word_freq_pos.items(), key=lambda x: x[1], reverse=True)[:50])
    
    return top_words_pos, top_words_neg

def extract_words_from_testset():
    """Extract word frequency dari test_set_skripsi berdasarkan label (0=negatif, 1=positif)"""
    try:
        df = pd.read_csv(ARTIFACTS / "test_set_skripsi.csv")
        word_freq_neg = {}
        word_freq_pos = {}
        
        for _, row in df.iterrows():
            text = str(row.get('text', '')).strip()
            label = row.get('label')
            
            if not text:
                continue
            
            words = text.split()
            
            if label == 0:  # Negatif
                for word in words:
                    word = word.strip()
                    if word:
                        word_freq_neg[word] = word_freq_neg.get(word, 0) + 1
            elif label == 1:  # Positif
                for word in words:
                    word = word.strip()
                    if word:
                        word_freq_pos[word] = word_freq_pos.get(word, 0) + 1
        
        # Sort dan ambil top 50 words
        top_words_neg = dict(sorted(word_freq_neg.items(), key=lambda x: x[1], reverse=True)[:50])
        top_words_pos = dict(sorted(word_freq_pos.items(), key=lambda x: x[1], reverse=True)[:50])
        
        return top_words_pos, top_words_neg
    except Exception as e:
        st.error(f"Error membaca test set: {e}")
        return {}, {}

def show_lda_visualization(lda_topics_pos, lda_topics_neg):
    """Tampilkan visualisasi LDA topics dalam format tabel"""
    st.markdown("<div class='header-section'><h3>Analisis Topik (LDA)</h3></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Topik Positif
    with col1:
        st.markdown("<h4 style='color: #333;'>Topik Sentimen Positif</h4>", unsafe_allow_html=True)
        if lda_topics_pos:
            topics_data_pos = []
            for topic in lda_topics_pos[:5]:
                words_str = ", ".join(topic.get('words', []) if isinstance(topic.get('words'), list) else topic.get('words', '').split(", "))
                # Handle weights - might be missing or in different format
                weights = topic.get('weights', [])
                if weights and isinstance(weights, (list, tuple)):
                    weights_str = ", ".join([f"{float(w):.2f}" for w in weights])
                else:
                    weights_str = "N/A"
                topics_data_pos.append({
                    "Topik": topic.get('topic_id', 0) + 1,
                    "Kata Kunci": words_str,
                    "Bobot": weights_str
                })
            
            if topics_data_pos:
                df_pos = pd.DataFrame(topics_data_pos)
                st.dataframe(df_pos, use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada data topik positif")
    
    # Topik Negatif
    with col2:
        st.markdown("<h4 style='color: #333;'>Topik Sentimen Negatif</h4>", unsafe_allow_html=True)
        if lda_topics_neg:
            topics_data_neg = []
            for topic in lda_topics_neg[:5]:
                words_str = ", ".join(topic.get('words', []) if isinstance(topic.get('words'), list) else topic.get('words', '').split(", "))
                # Handle weights - might be missing or in different format
                weights = topic.get('weights', [])
                if weights and isinstance(weights, (list, tuple)):
                    weights_str = ", ".join([f"{float(w):.2f}" for w in weights])
                else:
                    weights_str = "N/A"
                topics_data_neg.append({
                    "Topik": topic.get('topic_id', 0) + 1,
                    "Kata Kunci": words_str,
                    "Bobot": weights_str
                })
            
            if topics_data_neg:
                df_neg = pd.DataFrame(topics_data_neg)
                st.dataframe(df_neg, use_container_width=True, hide_index=True)
        else:
            st.info("Tidak ada data topik negatif")

def show_sentiment_wordcloud(words_pos, words_neg):
    """Tampilkan wordcloud dari stemmed text per sentiment"""
    st.markdown("<div class='header-section'><h3>Visualisasi Kata Sentimen</h3></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    # Wordcloud Positif
    with col1:
        st.markdown("<h4 style='color: #333;'>WordCloud Sentimen Positif</h4>", unsafe_allow_html=True)
        if words_pos:
            fig = create_wordcloud_from_topic(
                list(words_pos.keys()),
                list(words_pos.values()),
                title="WordCloud Sentimen Positif",
                sentiment_color="positive"
            )
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            st.info("Tidak ada data sentimen positif")
    
    # Wordcloud Negatif
    with col2:
        st.markdown("<h4 style='color: #333;'>WordCloud Sentimen Negatif</h4>", unsafe_allow_html=True)
        if words_neg:
            fig = create_wordcloud_from_topic(
                list(words_neg.keys()),
                list(words_neg.values()),
                title="WordCloud Sentimen Negatif",
                sentiment_color="negative"
            )
            if fig:
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)
        else:
            st.info("Tidak ada data sentimen negatif")

def show_sentiment_analysis_stages():
    """Tampilkan tahapan analisis sentimen secara interaktif dan elegan"""
    
    # Header section
    st.markdown("<div class='header-section'><h3>📊 Tahapan Analisis Sentimen</h3></div>", unsafe_allow_html=True)
    st.markdown("<p style='color: #6b7280; text-align: center; margin-bottom: 20px;'>Metodologi Analisis Sentimen Terhadap Perencanaan Terhadap Perencanaan Pemilu Elektronik (E-Voting) di Indonesia</p>", unsafe_allow_html=True)
    
    # Stages data
    stages = [
        {
            "number": "1",
            "title": "Seleksi Data",
            "icon": "📥",
            "color": "#667eea",
            "bg_color": "#eef2ff",
            "border_color": "#667eea",
            "details": [
                "Data Tweet X dan Komentar TikTok",
                "Total ±3500 data"
            ]
        },
        {
            "number": "2",
            "title": "Labeling",
            "icon": "🏷️",
            "color": "#8b5cf6",
            "bg_color": "#f3e8ff",
            "border_color": "#8b5cf6",
            "details": [
                "Klasifikasi sentimen",
                "Positif dan Negatif"
            ]
        },
        {
            "number": "3",
            "title": "Preprocessing",
            "icon": "🔧",
            "color": "#ec4899",
            "bg_color": "#fce7f3",
            "border_color": "#ec4899",
            "details": [
                "Cleaning & Case Folding",
                "Normalization (Slang)",
                "Tokenizing & Stopword Removal",
                "Stemming (Sastrawi)"
            ]
        },
        {
            "number": "4",
            "title": "Split Data",
            "icon": "✂️",
            "color": "#10b981",
            "bg_color": "#dcfce7",
            "border_color": "#10b981",
            "details": [
                "Pembagian data 80:20",
                "80% training, 20% testing"
            ]
        },
        {
            "number": "5",
            "title": "Vektorisasi",
            "icon": "🔢",
            "color": "#f59e0b",
            "bg_color": "#fef3c7",
            "border_color": "#f59e0b",
            "details": [
                "TF-IDF → Klasifikasi Sentimen",
                "CountVectorizer → Topic Modeling"
            ]
        },
        {
            "number": "6",
            "title": "Imbalance Handling",
            "icon": "⚖️",
            "color": "#06b6d4",
            "bg_color": "#cffafe",
            "border_color": "#06b6d4",
            "details": [
                "BorderlineSMOTE pada training",
                "Mengatasi ketidakseimbangan data"
            ]
        },
        {
            "number": "7",
            "title": "Model Klasifikasi",
            "icon": "🤖",
            "color": "#f97316",
            "bg_color": "#ffedd5",
            "border_color": "#f97316",
            "details": [
                "Logistic Regression",
                "Random Forest",
                "Soft Voting Ensemble"
            ]
        },
        {
            "number": "8",
            "title": "Evaluasi",
            "icon": "📈",
            "color": "#8b5cf6",
            "bg_color": "#f3e8ff",
            "border_color": "#8b5cf6",
            "details": [
                "Confusion Matrix",
                "Accuracy, Precision, Recall, F1-score",
                "Stratified K-Fold"
            ]
        },
        {
            "number": "9",
            "title": "Topic Modeling",
            "icon": "🗂️",
            "color": "#ef4444",
            "bg_color": "#fee2e2",
            "border_color": "#ef4444",
            "details": [
                "LDA (Latent Dirichlet Allocation)",
                "Menemukan topik utama dari data"
            ]
        }
    ]
    
    # Display stages vertically
    for stage in stages:
        details_text = " • ".join(stage["details"])
        html = f"""
        <div style='
            background: {stage["bg_color"]};
            border-left: 4px solid {stage["border_color"]};
            padding: 16px;
            border-radius: 8px;
            margin: 12px 0;
        '>
            <div style='display: flex; align-items: center; gap: 12px; margin-bottom: 8px;'>
                <span style='font-size: 1.5em;'>{stage["icon"]}</span>
                <div style='
                    width: 32px;
                    height: 32px;
                    background: {stage["color"]};
                    color: white;
                    border-radius: 50%;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-weight: 700;
                    font-size: 1em;
                '>
                    {stage["number"]}
                </div>
                <strong style='color: {stage["color"]}; font-size: 1.05em;'>{stage["title"]}</strong>
            </div>
            <div style='color: #666; font-size: 0.9rem; margin-left: 44px;'>
                {details_text}
            </div>
        </div>
        """
        st.markdown(html, unsafe_allow_html=True)
    
    # Flow visualization
    st.markdown("<div style='margin-top: 30px; margin-bottom: 20px;'></div>", unsafe_allow_html=True)
    
    st.markdown("""
    <div style='
        background: linear-gradient(90deg, #667eea15 0%, #f59e0b15 50%, #ef444415 100%);
        border-radius: 16px;
        padding: 24px;
        border-left: 4px solid #667eea;
    '>
        <h4 style='color: #1f2937; font-weight: 700; margin-top: 0;'>🔄 Alur Lengkap Analisis</h4>
        <p style='color: #6b7280; line-height: 1.8; margin: 16px 0;'>
            <strong style='color: #667eea;'>Input Data</strong> 
            → <strong style='color: #8b5cf6;'>Labeling</strong> 
            → <strong style='color: #ec4899;'>Preprocessing</strong> 
            → <strong style='color: #10b981;'>Split Data (80:20)</strong>
            → <strong style='color: #f59e0b;'>Vektorisasi</strong>
            → <strong style='color: #06b6d4;'>Imbalance Handling</strong>
            → <strong style='color: #f97316;'>Training Model</strong>
            → <strong style='color: #8b5cf6;'>Evaluasi</strong> 
            → <strong style='color: #ef4444;'>Topic Modeling & Insights</strong>
        </p>
    </div>
    """, unsafe_allow_html=True)

def show_metric_row(df_res: pd.DataFrame):
    if df_res.empty:
        return

    best_acc_row = df_res.sort_values("Accuracy", ascending=False).iloc[0]
    best_f1_row = df_res.sort_values("F1", ascending=False).iloc[0]
    best_prec_row = df_res.sort_values("Precision", ascending=False).iloc[0]
    best_rec_row = df_res.sort_values("Recall", ascending=False).iloc[0]

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("📊 Jumlah Model", len(df_res), delta=None)
    c2.metric("🏆 Akurasi Tertinggi", f"{best_acc_row['Accuracy']:.4f}", delta=f"{best_acc_row['Model']}")
    c3.metric("📈 Precision Tertinggi", f"{best_prec_row['Precision']:.4f}", delta=f"{best_prec_row['Model']}")
    c4.metric("🎯 F1 Tertinggi", f"{best_f1_row['F1']:.4f}", delta=f"{best_f1_row['Model']}")

def show_result_visuals(df_res: pd.DataFrame):
    if df_res.empty:
        st.info("Belum ada hasil untuk ditampilkan.")
        return

    st.markdown("<div class='header-section'><h3>Perbandingan Kinerja Model</h3></div>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([3, 1])
    with col1:
        st.dataframe(df_res, width=800, hide_index=True)
    
    with col2:
        best_row = df_res.sort_values(["Accuracy", "F1"], ascending=False).iloc[0]
        best_model = best_row["Model"]
        best_accuracy = best_row["Accuracy"]
        best_f1 = best_row["F1"]

        st.markdown(f"""
        <div style='background: #f5f5f5; border-left: 4px solid #667eea; padding: 12px; border-radius: 6px;'>
            <strong>Best Model</strong><br/>
            {best_model}<br/>
            <span style='font-size: 0.85rem; color: #6b7280;'>
                Akurasi {best_accuracy:.4f} • F1-score {best_f1:.4f}
            </span>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("""
        <div style='background: #f5f5f5; border-left: 4px solid #10b981; padding: 12px; border-radius: 6px; margin-top: 12px;'>
            <strong>Model Stabil Terbaik</strong><br/>
            SV Tuning Tanpa SMOTE<br/>
            <span style='font-size: 0.85rem; color: #6b7280;'>
                Akurasi 0.8003 • Gap 0.0706
            </span>
        </div>
        """, unsafe_allow_html=True)  

    st.bar_chart(df_res.set_index("Model")[["Accuracy", "Precision", "Recall", "F1"]])
        
    st.markdown("---")

def shorten_model_name(name: str) -> str:
    """Singkat nama model untuk tampilan confusion matrix dan k-fold"""
    replacements = {
        "Soft Voting Baseline + SMOTE": "SV Baseline + SMOTE",
        "Soft Voting Baseline Tanpa SMOTE": "SV Baseline Tanpa SMOTE",

        "Soft Voting Baseline + BorderlineSMOTE": "SV Baseline + B-SMOTE",
        "Soft Voting Baseline Tanpa BorderlineSMOTE": "SV Baseline Tanpa B-SMOTE",

        "Soft Voting Optimasi Awal + SMOTE": "SV Optimasi Awal + SMOTE",
        "Soft Voting Optimasi Awal Tanpa SMOTE": "SV Optimasi Awal Tanpa SMOTE",

        "Soft Voting Optimasi Awal + BorderlineSMOTE": "SV Optimasi Awal + B-SMOTE",
        "Soft Voting Optimasi Awal Tanpa BorderlineSMOTE": "SV Optimasi Awal Tanpa B-SMOTE",

        "Soft Voting Tuning + SMOTE": "SV Tuning + SMOTE",
        "Soft Voting Tuning Tanpa SMOTE": "SV Tuning Tanpa SMOTE",

        "Soft Voting Tuning + BorderlineSMOTE": "SV Tuning + B-SMOTE",
        "Soft Voting Tuning Tanpa BorderlineSMOTE": "SV Tuning Tanpa B-SMOTE",

        "Logistic Regression": "Log. Regression",
        "Logistic Regression Tuning": "Log. Regression Tuning",
        "Random Forest Tuning": "RF Tuning",
    }
    return replacements.get(name, name)

def show_confusion_matrices(confusions: dict):
    if not confusions:
        return

    st.markdown("<div class='header-section'><h3>Confusion Matrix</h3></div>", unsafe_allow_html=True)
    
    # Display confusion matrices dalam grid
    cols_per_row = 3
    total_cols = len(confusions)
    
    rows = (total_cols + cols_per_row - 1) // cols_per_row
    
    for row in range(rows):
        cols = st.columns(cols_per_row)
        for col_idx in range(cols_per_row):
            model_idx = row * cols_per_row + col_idx
            
            if model_idx >= total_cols:
                break
            
            model_name, cm = list(confusions.items())[model_idx]
            short_name = shorten_model_name(model_name)
            
            with cols[col_idx]:
                # Header dengan styling konsisten
                st.markdown(f"<div style='padding: 12px 16px; background: #f3f4f6; border: 1px solid #d1d5db; border-radius: 8px 8px 0 0;'><strong style='font-size: 0.95em; color: #374151;'>{short_name}</strong></div>", unsafe_allow_html=True)
                
                # Format confusion matrix dengan styling minimal
                tn, fp, fn, tp = cm[0][0], cm[0][1], cm[1][0], cm[1][1]
                total = tn + fp + fn + tp
                
                cm_html = f"""
                <div style='border: 1px solid #d1d5db; border-top: none; border-radius: 0 0 8px 8px; overflow: hidden;'>
                <table style='width: 100%; border-collapse: collapse; text-align: center; font-size: 0.9em; font-family: system-ui, -apple-system, sans-serif;'>
                    <tr style='background: #f9fafb; border-bottom: 1px solid #d1d5db;'>
                        <td style='padding: 10px 8px; font-weight: 600; color: #374151;'>-</td>
                        <td style='padding: 10px 8px; font-weight: 600; color: #374151;'>Negatif</td>
                        <td style='padding: 10px 8px; font-weight: 600; color: #374151;'>Positif</td>
                    </tr>
                    <tr style='border-bottom: 1px solid #d1d5db;'>
                        <td style='padding: 10px 8px; font-weight: 600; color: #374151; background: #f9fafb;'>Negatif</td>
                        <td style='padding: 10px 8px; color: #374151;'>{tn}</td>
                        <td style='padding: 10px 8px; color: #374151;'>{fp}</td>
                    </tr>
                    <tr>
                        <td style='padding: 10px 8px; font-weight: 600; color: #374151; background: #f9fafb;'>Positif</td>
                        <td style='padding: 10px 8px; color: #374151;'>{fn}</td>
                        <td style='padding: 10px 8px; color: #374151;'>{tp}</td>
                    </tr>
                </table>
                </div>
                """
                st.markdown(cm_html, unsafe_allow_html=True)

def show_kfold_cross_validation(kfold_data: dict):
    """
    Tampilkan visualisasi K-Fold Cross Validation untuk setiap model
    
    Format data:
    {
        "model_name": {
            "folds": [
                {"Training Accuracy": 0.903, "Validation Accuracy": 0.765},
                ...
            ],
            "stats": {  # Optional - jika tidak ada, akan dihitung otomatis
                "avg_train": 0.899,
                "avg_val": 0.779,
                "train_std": 0.0XX,
                "val_std": 0.0XX,
                "gap": 0.120
            }
        }
    }
    """
    if not kfold_data:
        return
    
    st.markdown("<div class='header-section'><h3>📊 K-Fold Cross Validation</h3></div>", unsafe_allow_html=True)
    
    # Buat tab untuk setiap model
    model_names = list(kfold_data.keys())
    tabs = st.tabs([shorten_model_name(name) for name in model_names])
    
    for tab, model_name in zip(tabs, model_names):
        with tab:
            # Support both old format (list) and new format (dict with folds/stats)
            model_data = kfold_data[model_name]
            
            if isinstance(model_data, list):
                # Old format: direct list of fold results
                fold_results = model_data
                stats = None
            else:
                # New format: dict with folds and optional stats
                fold_results = model_data.get("folds", model_data)
                stats = model_data.get("stats", None)
            
            # Prepare DataFrame untuk chart
            df_folds = pd.DataFrame(fold_results)
            df_folds['Fold'] = [f'Fold {i+1}' for i in range(len(df_folds))]
            df_folds_plot = df_folds.set_index('Fold')[['Training Accuracy', 'Validation Accuracy']]
            
            # Line chart
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.plot(df_folds['Fold'], df_folds['Training Accuracy'], marker='o', linewidth=2, label='Training Accuracy', color='#2563eb')
            ax.plot(df_folds['Fold'], df_folds['Validation Accuracy'], marker='s', linewidth=2, label='Validation Accuracy', color='#f97316')
            
            # Styling
            ax.set_title(f'K-Fold Cross Validation - {model_name}', fontsize=14, fontweight='bold', pad=20)
            ax.set_xlabel('Fold', fontsize=11)
            ax.set_ylabel('Accuracy', fontsize=11)
            ax.set_ylim(min(0.6, df_folds['Training Accuracy'].min() - 0.05), 1.0)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='best', fontsize=10)
            
            # Add value labels on points
            for i, (fold, train_acc, val_acc) in enumerate(zip(df_folds['Fold'], df_folds['Training Accuracy'], df_folds['Validation Accuracy'])):
                ax.text(i, train_acc, f'{train_acc:.3f}', ha='center', va='bottom', fontsize=9, color='#2563eb')
                ax.text(i, val_acc, f'{val_acc:.3f}', ha='center', va='bottom', fontsize=9, color='#f97316')
            
            plt.tight_layout()
            st.pyplot(fig, use_container_width=True)
            
            # Statistics table
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            
            # Gunakan hardcoded stats jika tersedia, atau hitung otomatis
            if stats:
                train_mean = stats.get("avg_train")
                val_mean = stats.get("avg_val")
                val_std = stats.get("val_std")
                gap = stats.get("gap")
            else:
                train_mean = df_folds['Training Accuracy'].mean()
                val_mean = df_folds['Validation Accuracy'].mean()
                val_std = df_folds['Validation Accuracy'].std()
                gap = train_mean - val_mean
            
            st.markdown("### Statistik K-Fold")

            col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

            with col_stats1:
                st.metric("Avg Training Accuracy", f"{train_mean:.4f}")

            with col_stats2:
                st.metric("Avg Validation Accuracy", f"{val_mean:.4f}")

            with col_stats3:
                st.metric("Validation Std", f"{val_std:.4f}")

            with col_stats4:
                st.metric("Gap", f"{gap:.4f}")            
            


def show_preprocessing_steps(prep: dict):
    st.markdown("<div class='header-section'><h3>Tahapan Preprocessing</h3></div>", unsafe_allow_html=True)
    
    steps = [
        ("1️⃣ Original Text", "Teks asli dari input", prep.get("original", ""), "#f3f4f6", "#d1d5db"),
        ("2️⃣ Cleaning", "Hapus URL, @mention, #hashtag, angka, dan punctuation", prep.get("cleaned", ""), "#fef3c7", "#fbbf24"),
        ("3️⃣ Case Folding", "Ubah semua karakter ke lowercase", prep.get("case_folded", ""), "#dbeafe", "#60a5fa"),
        ("4️⃣ Normalize", "Normalisasi kata (slang ke kata baku)", prep.get("normalized", ""), "#cffafe", "#22d3ee"),
        ("5️⃣ Tokenize", "Pisahkan teks ke token individual", " → ".join(prep.get("tokens", [])), "#e0e7ff", "#818cf8"),
        ("6️⃣ Stopword Removal", "Hapus kata umum yang tidak penting", " → ".join(prep.get("filtered_tokens", [])), "#f5d4f4", "#d946ef"),
        ("7️⃣ Stemming", "Ubah kata ke bentuk dasar (stem)", " → ".join(prep.get("stemmed_tokens", [])), "#dcfce7", "#22c55e"),
    ]
    
    for title, description, content, bg_color, border_color in steps:
        display_content = content if content else "-"
        html = f"<div style='background: {bg_color}; border-left: 4px solid {border_color}; padding: 16px; border-radius: 8px; margin: 12px 0;'>"
        html += f"<strong>{title}</strong><br/>"
        html += f"<span style='color: #666; font-size: 0.9rem;'>{description}</span><br/>"
        html += f"<div style='background: white; padding: 12px; border-radius: 6px; margin-top: 10px; font-family: monospace; font-size: 0.95rem; word-wrap: break-word;'>{display_content}</div>"
        html += "</div>"
        st.markdown(html, unsafe_allow_html=True)
    
    st.markdown("")

def show_probability_visualization(results: dict):
    """Tampilkan visualisasi probabilitas untuk setiap model"""
    st.markdown("<div class='header-section'><h3>Probabilitas Prediksi per Model</h3></div>", unsafe_allow_html=True)
    
    # Prepare data for visualization
    prob_data = []
    for model_name, result in results.items():
        if result["prob_neg"] is not None and result["prob_pos"] is not None:
            prob_data.append({
                "Model": model_name,
                "Negatif": result["prob_neg"],
                "Positif": result["prob_pos"]
            })
    
    if not prob_data:
        st.warning("Model tidak memiliki probabilitas prediksi")
        return
    
    df_prob = pd.DataFrame(prob_data)
    df_prob_set = df_prob.set_index("Model")
    
    # Horizontal bar chart
    st.bar_chart(df_prob_set)
    
    # Individual model cards
    cols = st.columns(min(3, len(results)))
    for idx, (model_name, result) in enumerate(results.items()):
        with cols[idx % len(cols)]:
            if result["prob_neg"] is not None:
                pred_label = "✅ Positif" if result["label"] == "Positif" else "❌ Negatif"
                color = "#10b981" if result["label"] == "Positif" else "#ef4444"
                
                st.markdown(f"""
                <div class='result-card'>
                    <strong>{model_name}</strong><br/>
                    <div style='color: {color}; font-size: 1.1rem; margin: 8px 0;'>{pred_label}</div>
                    <div style='font-size: 0.9rem; color: #666;'>
                        Positif: <strong>{result["prob_pos"]:.2%}</strong><br/>
                        Negatif: <strong>{result["prob_neg"]:.2%}</strong>
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"<div class='result-card'><strong>{model_name}</strong><br/>{result['label']}</div>", unsafe_allow_html=True)

def evaluate_models_on_dataset(df_eval: pd.DataFrame, text_col: str, label_col: str, models, vectorizer):
    # ===== BATCH PREPROCESSING (PARALLELIZED) =====
    processed = _preprocess_texts_batch(df_eval[text_col].fillna("").astype(str).tolist())
    X_eval = vectorizer.transform(processed)

    y_true = normalize_labels(df_eval[label_col])
    valid_idx = y_true.notna().to_numpy()

    y_true = y_true[valid_idx]
    X_eval = X_eval[valid_idx]
    processed_valid = [p for p, v in zip(processed, valid_idx) if v]

    results = []
    confusions = {}
    all_predictions = None  # For prediction tracking

    for idx, (model_name, model) in enumerate(models.items()):
        preds = model.predict(X_eval)
        
        # Store first model's predictions for wordcloud
        if idx == 0:
            all_predictions = preds

        acc = accuracy_score(y_true, preds)
        prec = precision_score(y_true, preds, zero_division=0)
        rec = recall_score(y_true, preds, zero_division=0)
        f1 = f1_score(y_true, preds, zero_division=0)
        cm = confusion_matrix(y_true, preds).tolist()

        results.append({
            "Model": model_name,
            "Accuracy": round(acc, 6),
            "Precision": round(prec, 6),
            "Recall": round(rec, 6),
            "F1": round(f1, 6),
        })
        confusions[model_name] = cm

    return pd.DataFrame(results), confusions, all_predictions, processed_valid, y_true

# =========================
# LOAD MODEL
# =========================
models, vectorizer = load_models_multi()

# =========================
# HEADER
# =========================
st.title("🎯 Dashboard Analisis Sentimen")
st.caption("Visualisasi hasil analisis sentimen terhadap perencanaan pemilu elektronik (e-voting) di Indonesia")
st.markdown("---")

# =========================
# SINGLE COMMENT
# =========================
st.markdown("<div class='header-section'><h2>Analisis Komentar Tunggal</h2></div>", unsafe_allow_html=True)

text_input = st.text_area("Masukkan komentar", placeholder="Ketik komentar Anda di sini...")

if st.button("🔍 Analisis Komentar", use_container_width=False):
    if text_input.strip():
        with st.spinner("⏳ Menganalisis komentar..."):
            prep, results = predict_single_text_multi(text_input)
            
            # Tampilkan semua tahapan preprocessing
            show_preprocessing_steps(prep)
            
            # Results visualization
            show_probability_visualization(results)
    else:
        st.warning("Mohon masukkan komentar terlebih dahulu!")

st.markdown("---")

# =========================
# USER VIEW (FIXED 🔥)
# =========================
latest = load_latest_result()

if latest is None:
    st.markdown("<div class='header-section'><h2>Visualisasi Hasil Terakhir</h2></div>", unsafe_allow_html=True)
    st.info("ℹ️ Belum ada hasil analisis. Silakan login sebagai admin untuk memulai analisis.")
else:
    meta = latest.get("meta", {})
    
    # 🔥 JUDUL DINAMIS
    st.markdown(f"<div class='header-section'><h2>📊 {meta.get('title', 'Visualisasi Hasil Terakhir')}</h2></div>", unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown(f"<div class='metric-card-light'><strong>⏰ Waktu Analisis</strong><br/><span style='font-size: 1.2rem; color: #667eea;'>{meta.get('timestamp', '-')}</span></div>", unsafe_allow_html=True)
    with col2:
        st.markdown(f"<div class='metric-card-light'><strong>📊 Total Data</strong><br/><span style='font-size: 1.2rem; color: #667eea;'>{meta.get('n_rows', '-')}</span></div>", unsafe_allow_html=True)
    
    # =========================
    # PREDICTION MODE
    # =========================
    if meta.get("mode") == "prediction":
        st.markdown("")
        
        # Show predictions table
        st.markdown("<div class='header-section'><h3>Hasil Prediksi Sentimen</h3></div>", unsafe_allow_html=True)
        
        df_predictions = pd.DataFrame(latest.get("predictions_df", []))
        
        if len(df_predictions) > 0:
            st.dataframe(df_predictions, use_container_width=True, height=400)
            
            # Download CSV button
            st.markdown("")
            col_download, col_space = st.columns([1, 4])
            with col_download:
                csv_data = df_predictions.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label="📥 Download CSV",
                    data=csv_data,
                    file_name=f"prediksi_sentimen_{meta.get('timestamp', 'result').replace(':', '-')}.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            
            # Show sentiment distribution
            st.markdown("")
            st.markdown("<div class='header-section'><h3>💡 Statistik Distribusi Sentimen</h3></div>", unsafe_allow_html=True)
            
            # Determine prediction column name (single vs multiple models)
            pred_col = "Prediction" if "Prediction" in df_predictions.columns else "Ensemble Prediction"
            
            col_stat1, col_stat2 = st.columns(2)
            
            with col_stat1:
                positif_count = (df_predictions[pred_col] == 'Positif').sum()
                positif_pct = positif_count / len(df_predictions) * 100
                st.markdown(f"""
                <div class='metric-card' style='background: #10b981;'>
                    <strong>😊 Sentimen Positif</strong><br/>
                    <span style='font-size: 2rem; font-weight: bold;'>{positif_count}</span><br/>
                    <span style='font-size: 1.1rem;'>{positif_pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
            
            with col_stat2:
                negatif_count = (df_predictions[pred_col] == 'Negatif').sum()
                negatif_pct = negatif_count / len(df_predictions) * 100
                st.markdown(f"""
                <div class='metric-card' style='background: #ef4444;'>
                    <strong>😠 Sentimen Negatif</strong><br/>
                    <span style='font-size: 2rem; font-weight: bold;'>{negatif_count}</span><br/>
                    <span style='font-size: 1.1rem;'>{negatif_pct:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
        
        # Show LDA topics if available
        lda_topics_pos = latest.get("lda_topics_pos", [])
        lda_topics_neg = latest.get("lda_topics_neg", [])
        if lda_topics_pos or lda_topics_neg:
            st.markdown("")
            show_lda_visualization(lda_topics_pos, lda_topics_neg)
        
        # Show sentiment wordcloud if available
        words_pos = latest.get("words_pos", {})
        words_neg = latest.get("words_neg", {})
        if words_pos or words_neg:
            st.markdown("")
            show_sentiment_wordcloud(words_pos, words_neg)
    
    # =========================
    # EVALUATION MODE
    # =========================
    else:
        df_res = pd.DataFrame(latest["results"])
        confusions = latest.get("confusions", {})

        with col3:
            valid_rows = meta.get('n_valid_rows', meta.get('n_rows', '-'))
            st.markdown(f"<div class='metric-card-light'><strong>✅ Data Valid</strong><br/><span style='font-size: 1.2rem; color: #667eea;'>{valid_rows}</span></div>", unsafe_allow_html=True)
        with col4:
            st.markdown(f"<div class='metric-card-light'><strong>🤖 Jumlah Model</strong><br/><span style='font-size: 1.2rem; color: #667eea;'>{len(df_res)} model</span></div>", unsafe_allow_html=True)

        st.markdown("")
        show_metric_row(df_res)
        st.markdown("")
        show_result_visuals(df_res)
        show_confusion_matrices(confusions)
        
        # Show K-Fold Cross Validation if available
        kfold_data = latest.get("kfold", {})
        if kfold_data:
            st.markdown("")
            show_kfold_cross_validation(kfold_data)
        
        # Show LDA topics if available
        lda_topics_pos = latest.get("lda_topics_pos", [])
        lda_topics_neg = latest.get("lda_topics_neg", [])
        if lda_topics_pos or lda_topics_neg:
            st.markdown("")
            show_lda_visualization(lda_topics_pos, lda_topics_neg)
        
        # Show sentiment wordcloud if available
        words_pos = latest.get("words_pos", {})
        words_neg = latest.get("words_neg", {})
        if words_pos or words_neg:
            st.markdown("")
            show_sentiment_wordcloud(words_pos, words_neg)
    
    # Show sentiment analysis stages
    st.markdown("")
    show_sentiment_analysis_stages()

st.markdown("---")

# =========================
# SIDEBAR HEADER
# =========================
with st.sidebar:
    # Logo section
    col_left, col_right = st.columns(2)
    
    with col_left:
        try:
            from PIL import Image
            img_si = Image.open("assets/sistem_informasi.png")
            st.image(img_si, width=80)
        except:
            pass
    
    with col_right:
        try:
            img_undiksha = Image.open("assets/undiksha.png")
            st.image(img_undiksha, width=80)
        except:
            pass
    
    st.markdown("---")



# =========================
# ADMIN LOGIN
# =========================
st.sidebar.markdown("<h4 style='margin-top: 10px; margin-bottom: 15px;'>Admin Panel</h4>", unsafe_allow_html=True)

if not st.session_state.admin:
    u = st.sidebar.text_input("Username")
    p = st.sidebar.text_input("Password", type="password")

    if st.sidebar.button("Login", use_container_width=True):
        if u == ADMIN_USERNAME and p == ADMIN_PASSWORD:
            st.session_state.admin = True
            st.rerun()
else:
    st.sidebar.markdown("<div style='background: #f5f5f5; padding: 10px; border-radius: 6px; text-align: center; font-size: 13px;'><strong>Status: Admin</strong></div>", unsafe_allow_html=True)
    
    if st.sidebar.button("🗑️ Clear Cache", use_container_width=True):
        if RESULT_FILE.exists():
            RESULT_FILE.unlink()
        st.session_state.latest_result_cache = None
        st.rerun()
    
    if st.sidebar.button("Logout", use_container_width=True):
        st.session_state.admin = False
        st.rerun()

# =========================
# ADMIN PANEL
# =========================
if st.session_state.admin:

    st.subheader("Panel Admin")

    mode = st.radio(
        "Mode",
        [
            "Visualisasi Analisis Sentimen E-Voting",
            "Uji Data Baru (Prediksi Sentimen)"
        ]
    )

    # =========================
    # MODE SKRIPSI - VISUALISASI E-VOTING
    # =========================
    if mode == "Visualisasi Analisis Sentimen E-Voting":
        
        if st.button("Proses", disabled=st.session_state.processing):
            
            st.session_state.processing = True
            
            with st.spinner("⏳ Memproses data skripsi..."):
                
                # =====================================================
                # HARDCODED RESULTS FROM COLAB (Soft Voting Models)
                # =====================================================
                evoting_results = [
                    {
                        "Model": "Soft Voting Baseline + SMOTE",
                        "Accuracy": 0.7990,
                        "Precision": 0.8018,
                        "Recall": 0.7990,
                        "F1": 0.8001,
                    },
                    {
                        "Model": "Soft Voting Baseline Tanpa SMOTE",
                        "Accuracy": 0.7739,
                        "Precision": 0.7793,
                        "Recall": 0.7739,
                        "F1": 0.7584,
                    },
                    {
                        "Model": "Soft Voting Optimasi Awal + SMOTE",
                        "Accuracy": 0.8078,
                        "Precision": 0.8092,
                        "Recall": 0.8078,
                        "F1": 0.8084,
                    },
                    {
                        "Model": "Soft Voting Optimasi Awal Tanpa SMOTE",
                        "Accuracy": 0.7940,
                        "Precision": 0.7919,
                        "Recall": 0.7940,
                        "F1": 0.7875,
                    },
                    {
                        "Model": "Soft Voting Tuning + SMOTE",
                        "Accuracy": 0.7902,
                        "Precision": 0.7929,
                        "Recall": 0.7902,
                        "F1": 0.7913,
                    },
                    {
                        "Model": "Soft Voting Tuning Tanpa SMOTE",
                        "Accuracy": 0.8003,
                        "Precision": 0.8015,
                        "Recall": 0.8003,
                        "F1": 0.7917,
                    },
                    {
                        "Model": "Logistic Regression Tuning",
                        "Accuracy": 0.7965,
                        "Precision": 0.7945,
                        "Recall": 0.7965,
                        "F1": 0.7904,
                    },
                    {
                        "Model": "Random Forest Tuning",
                        "Accuracy": 0.7651,
                        "Precision": 0.7801,
                        "Recall": 0.7651,
                        "F1": 0.7426,
                    }
                ]
                
                evoting_confusions = {
                    "Soft Voting Baseline Tanpa SMOTE": [[419, 88], [72, 217]],
                    "Soft Voting Baseline + SMOTE": [[472, 35], [145, 144]],
                    "Soft Voting Optimasi Awal + SMOTE": [[426, 81], [72, 217]],
                    "Soft Voting Optimasi Awal Tanpa SMOTE": [[456, 51], [113, 176]],
                    "Soft Voting Tuning + SMOTE": [[416, 91], [76, 213]],
                    "Soft Voting Tuning Tanpa SMOTE": [[467, 40], [119, 170]],
                    "Logistic Regression Tuning": [[456, 51], [111, 178]],
                    "Random Forest Tuning": [[482, 25], [162, 127]],
                }
                
                # =====================================================
                # HARDCODED K-FOLD CROSS VALIDATION RESULTS
                # =====================================================
                evoting_kfold = {
                    "Soft Voting Baseline + SMOTE": {
                        "folds": [
                            {"Training Accuracy": 0.871, "Validation Accuracy": 0.750},
                            {"Training Accuracy": 0.866, "Validation Accuracy": 0.783},
                            {"Training Accuracy": 0.863, "Validation Accuracy": 0.777},
                            {"Training Accuracy": 0.863, "Validation Accuracy": 0.797},
                            {"Training Accuracy": 0.866, "Validation Accuracy": 0.802},
                        ],
                        "stats": {
                            "avg_train": 0.8659,
                            "avg_val": 0.7820,
                            "val_std": 0.0182,
                            "gap": 0.0839
                        }
                    },
                    "Soft Voting Baseline Tanpa SMOTE": {
                        "folds": [
                            {"Training Accuracy": 0.819, "Validation Accuracy": 0.763},
                            {"Training Accuracy": 0.818, "Validation Accuracy": 0.796},
                            {"Training Accuracy": 0.824, "Validation Accuracy": 0.760},
                            {"Training Accuracy": 0.820, "Validation Accuracy": 0.756},
                            {"Training Accuracy": 0.817, "Validation Accuracy": 0.800},
                        ],
                        "stats": {
                            "avg_train": 0.8195,
                            "avg_val": 0.7751,
                            "val_std": 0.0190,
                            "gap": 0.0445
                        }
                    },
                    "Soft Voting Optimasi Awal + SMOTE": {
                        "folds": [
                            {"Training Accuracy": 0.903, "Validation Accuracy": 0.755},
                            {"Training Accuracy": 0.899, "Validation Accuracy": 0.794},
                            {"Training Accuracy": 0.899, "Validation Accuracy": 0.765},
                            {"Training Accuracy": 0.901, "Validation Accuracy": 0.792},
                            {"Training Accuracy": 0.902, "Validation Accuracy": 0.803},
                        ],
                        "stats": {
                            "avg_train": 0.9008,
                            "avg_val": 0.7820,
                            "val_std": 0.1188,
                            "gap": 0.0187
                        }
                    },
                    "Soft Voting Optimasi Awal Tanpa SMOTE": {
                        "folds": [
                            {"Training Accuracy": 0.888, "Validation Accuracy": 0.769},
                            {"Training Accuracy": 0.888, "Validation Accuracy": 0.805},
                            {"Training Accuracy": 0.887, "Validation Accuracy": 0.786},
                            {"Training Accuracy": 0.885, "Validation Accuracy": 0.800},
                            {"Training Accuracy": 0.881, "Validation Accuracy": 0.816},
                        ],
                        "stats": {
                            "avg_train": 0.8856,
                            "avg_val": 0.7955,
                            "val_std": 0.0162,
                            "gap": 0.0901
                        }
                    },
                    "Soft Voting Tuning + SMOTE": {
                        "folds": [
                            {"Training Accuracy": 0.866, "Validation Accuracy": 0.749},
                            {"Training Accuracy": 0.860, "Validation Accuracy": 0.779},
                            {"Training Accuracy": 0.855, "Validation Accuracy": 0.783},
                            {"Training Accuracy": 0.858, "Validation Accuracy": 0.794},
                            {"Training Accuracy": 0.859, "Validation Accuracy": 0.800},
                        ],
                        "stats": {
                            "avg_train": 0.8596,
                            "avg_val": 0.7810,
                            "val_std": 0.0178,
                            "gap": 0.0786
                        }
                    },
                    "Soft Voting Tuning Tanpa SMOTE": {
                        "folds": [
                            {"Training Accuracy": 0.858, "Validation Accuracy": 0.772},
                            {"Training Accuracy": 0.860, "Validation Accuracy": 0.794},
                            {"Training Accuracy": 0.864, "Validation Accuracy": 0.777},
                            {"Training Accuracy": 0.856, "Validation Accuracy": 0.781},
                            {"Training Accuracy": 0.852, "Validation Accuracy": 0.811},
                        ],
                        "stats": {
                            "avg_train": 0.8579,
                            "avg_val": 0.7873,
                            "val_std": 0.0141,
                            "gap": 0.0706
                        }
                    },
                    "Logistic Regression Tuning": {
                        "folds": [
                            {"Training Accuracy": 0.870, "Validation Accuracy": 0.774},
                            {"Training Accuracy": 0.872, "Validation Accuracy": 0.812},
                            {"Training Accuracy": 0.873, "Validation Accuracy": 0.783},
                            {"Training Accuracy": 0.871, "Validation Accuracy": 0.799},
                            {"Training Accuracy": 0.872, "Validation Accuracy": 0.811},
                        ],
                        "stats": {
                            "avg_train": 0.8717,
                            "avg_val": 0.7958,
                            "val_std": 0.0150,
                            "gap": 0.0759
                        }
                    },
                    "Logistic Regression Tuning": {
                        "folds": [
                            {"Training Accuracy": 0.809, "Validation Accuracy": 0.750},
                            {"Training Accuracy": 0.815, "Validation Accuracy": 0.785},
                            {"Training Accuracy": 0.810, "Validation Accuracy": 0.744},
                            {"Training Accuracy": 0.812, "Validation Accuracy": 0.747},
                            {"Training Accuracy": 0.804, "Validation Accuracy": 0.786},
                        ],
                        "stats": {
                            "avg_train": 0.8100,
                            "avg_val": 0.7625,
                            "val_std": 0.0189,
                            "gap": 0.0475
                        }
                    },
                }
                
                # =====================================================
                # HARDCODED LDA TOPICS (Negatif & Positif)
                # =====================================================
                lda_topics_neg = [
                    {
                        "topic_id": 0,
                        "words": ["online", "hacker", "irit", "percaya", "banget", "lucu", "bahaya", "pemilu", "takut", "perintah"],
                        "weights": [0.1, 0.09, 0.08, 0.07, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05]
                    },
                    {
                        "topic_id": 1,
                        "words": ["tuju", "main", "lawak", "tau", "pemilu", "negara", "hasil", "pegang", "no", "ku"],
                        "weights": [0.1, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05]
                    },
                    {
                        "topic_id": 2,
                        "words": ["data", "bocor", "biar", "manipulasi", "gampang", "mudah", "curang", "digital", "pemilu", "pakai"],
                        "weights": [0.11, 0.09, 0.08, 0.07, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05]
                    },
                    {
                        "topic_id": 3,
                        "words": ["presiden", "orang", "pilih", "akun", "langsung", "digital", "pakai", "tau", "hp", "tebak"],
                        "weights": [0.1, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05]
                    },
                    {
                        "topic_id": 4,
                        "words": ["menang", "digital", "pemilu", "suara", "indonesia", "tau", "pilih", "langsung", "curang", "negeri"],
                        "weights": [0.09, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05]
                    }
                ]
                
                lda_topics_pos = [
                    {
                        "topic_id": 0,
                        "words": ["pemilu", "pilih", "pakai", "elektronik", "sistem", "suara", "voting", "ktp", "data", "nik"],
                        "weights": [0.11, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05]
                    },
                    {
                        "topic_id": 1,
                        "words": ["pemilu", "digital", "negara", "online", "konoha", "banteng", "menang", "rakyat", "hemat", "data"],
                        "weights": [0.1, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05]
                    },
                    {
                        "topic_id": 2,
                        "words": ["pakai", "sistem", "voting", "pemilu", "akun", "pilih", "teknologi", "hati", "kadang", "digital"],
                        "weights": [0.1, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05]
                    },
                    {
                        "topic_id": 3,
                        "words": ["kardus", "gembok", "menang", "pakai", "langsung", "kotak", "kemarin", "tebak", "digital", "paham"],
                        "weights": [0.09, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.05, 0.05]
                    },
                    {
                        "topic_id": 4,
                        "words": ["biar", "irit", "menang", "mudah", "rakyat", "no", "hemat", "langsung", "curang", "cerdas"],
                        "weights": [0.09, 0.09, 0.08, 0.08, 0.07, 0.07, 0.06, 0.06, 0.06, 0.05]
                    }
                ]
                
                # Save to latest_result
                df_res_evoting = pd.DataFrame(evoting_results)
                
                # Extract words dari test set berdasarkan label
                words_pos, words_neg = extract_words_from_testset()
                
                save_latest_result({
                    "meta": {
                        "title": "Visualisasi Analisis Sentimen E-Voting",
                        "timestamp": datetime.now().strftime("%H:%M:%S"),
                        "n_rows": 796,
                        "n_valid_rows": 796,
                        "note": "Soft Voting ensemble models dari Colab training"
                    },
                    "results": df_res_evoting.to_dict(orient="records"),
                    "confusions": evoting_confusions,
                    "kfold": evoting_kfold,
                    "predictions": [],
                    "processed_texts": [],
                    "lda_topics_pos": lda_topics_pos,
                    "lda_topics_neg": lda_topics_neg,
                    "words_pos": words_pos,
                    "words_neg": words_neg
                })
            
            st.session_state.processing = False
            st.success("Selesai")
            st.rerun()

    # =========================
    # MODE DEMO
    # =========================
    else:

        file = st.file_uploader("Upload CSV")

        if file:
            df = pd.read_csv(file)

            text_col = st.selectbox("Kolom teks", df.columns)
            
            # 🔥 pilih model
            model_mode = st.radio("Model", ["Semua Model", "Satu Model"])

            selected_model = None
            if model_mode == "Satu Model":
                selected_model = st.selectbox("Pilih Model", list(models.keys()))

            if st.button("Proses", disabled=st.session_state.processing):

                st.session_state.processing = True

                with st.spinner("⏳ Memproses data..."):
                    
                    # =========================
                    # PREDICTION MODE (Only)
                    # =========================
                    # Use new prediction function with selected model(s)
                    model_list = models if model_mode == "Semua Model" else {selected_model: models[selected_model]}
                    
                    df_predictions, processed_texts, all_predictions_ensemble = predict_dataset_multi(
                        df, text_col, model_dict=model_list, vectorizer_obj=vectorizer
                    )
                    
                    # Generate LDA topics dari uploaded data (bukan pre-trained)
                    lda_topics_pos, lda_topics_neg = generate_lda_topics_from_data(
                        processed_texts, 
                        all_predictions_ensemble,
                        n_topics=5,
                        n_top_words=10
                    )
                    
                    # Extract word frequencies dari ensemble predictions
                    words_pos, words_neg = extract_top_words_by_sentiment(processed_texts, all_predictions_ensemble, n_top_words=50)

                    save_latest_result({
                        "meta": {
                            "title": "Uji Data Baru - Prediksi Dataset",
                            "timestamp": datetime.now().strftime("%H:%M:%S"),
                            "n_rows": len(df),
                            "mode": "prediction"
                        },
                        "predictions_df": df_predictions.to_dict(orient="records"),
                        "processed_texts": processed_texts if processed_texts else [],
                        "lda_topics_pos": lda_topics_pos,
                        "lda_topics_neg": lda_topics_neg,
                        "words_pos": words_pos,
                        "words_neg": words_neg
                    })

                st.session_state.processing = False
                st.success("Selesai")
                st.rerun()

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; color: #666; font-size: 13px;'>
    <p style='margin: 5px 0;'><strong>I Putu Dennis Prana Arta</strong></p>
    <p style='margin: 5px 0;'>Program Studi Sistem Informasi</p>
    <p style='margin: 5px 0; font-size: 12px; color: #999;'>Universitas Pendidikan Ganesha © 2026</p>
</div>
""", unsafe_allow_html=True)