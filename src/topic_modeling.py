# =========================
# TOPIC MODELING UTILITIES
# =========================
from typing import List, Tuple, Dict
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


def generate_lda_topics_from_data(
    processed_texts: List[str],
    predictions: List[int],
    n_topics: int = 5,
    n_top_words: int = 10,
    min_df: int = 1,
    max_df: float = 1.0
) -> Tuple[List[Dict], List[Dict]]:
    """
    Generate LDA topics dari processed text berdasarkan sentimen prediction.
    
    Args:
        processed_texts: List of stemmed/preprocessed texts
        predictions: List of predictions (0 atau 1 untuk negative/positive)
        n_topics: Jumlah topics per sentimen
        n_top_words: Jumlah top words per topic
        min_df: Minimum document frequency untuk words
        max_df: Maximum document frequency untuk words
    
    Returns:
        (lda_topics_pos, lda_topics_neg) - List of topic dictionaries
    """
    
    # Separate texts by sentiment
    texts_pos = [text for text, pred in zip(processed_texts, predictions) if pred == 1]
    texts_neg = [text for text, pred in zip(processed_texts, predictions) if pred == 0]
    
    lda_topics_pos = []
    lda_topics_neg = []
    
    # Generate topics untuk Positif
    if texts_pos:
        lda_topics_pos = _generate_lda_for_sentiment(
            texts_pos, 
            n_topics=n_topics, 
            n_top_words=n_top_words,
            min_df=min_df,
            max_df=max_df,
            sentiment="positive"
        )
    
    # Generate topics untuk Negatif
    if texts_neg:
        lda_topics_neg = _generate_lda_for_sentiment(
            texts_neg, 
            n_topics=n_topics, 
            n_top_words=n_top_words,
            min_df=min_df,
            max_df=max_df,
            sentiment="negative"
        )
    
    return lda_topics_pos, lda_topics_neg


def _generate_lda_for_sentiment(
    texts: List[str],
    n_topics: int = 5,
    n_top_words: int = 10,
    min_df: int = 1,
    max_df: float = 1.0,
    sentiment: str = "positive"
) -> List[Dict]:
    """
    Internal function untuk generate LDA topics dari text list tertentu.
    
    Args:
        texts: List of texts untuk analisis
        n_topics: Jumlah topics
        n_top_words: Jumlah top words per topic
        min_df: Min document frequency
        max_df: Max document frequency
        sentiment: "positive" atau "negative" untuk reference
    
    Returns:
        List of topic dictionaries dengan format:
        [
            {
                "topic_id": 0,
                "words": ["word1", "word2", ...],
                "weights": [0.1, 0.09, ...]
            },
            ...
        ]
    """
    if not texts or len(texts) < 2:
        return []
    
    try:
        # Create CountVectorizer
        vectorizer = CountVectorizer(
            min_df=min_df,
            max_df=max_df,
            max_features=100,
            lowercase=True,
            analyzer='word',
            ngram_range=(1, 1)
        )
        
        # Fit dan transform
        X = vectorizer.fit_transform(texts)
        
        # Check if we have enough features
        if X.shape[1] == 0:
            return []
        
        # Adjust n_topics jika lebih besar dari doc count
        n_topics_actual = min(n_topics, X.shape[0] - 1) if X.shape[0] > 1 else 1
        n_topics_actual = max(1, n_topics_actual)
        
        # Fit LDA
        lda = LatentDirichletAllocation(
            n_components=n_topics_actual,
            random_state=42,
            learning_method='batch',
            max_iter=10
        )
        lda.fit(X)
        
        # Get feature names
        feature_names = vectorizer.get_feature_names_out()
        
        # Extract topics
        topics = []
        for topic_idx, topic in enumerate(lda.components_):
            # Get top words indices
            top_indices = topic.argsort()[-n_top_words:][::-1]
            top_words = [str(feature_names[i]) for i in top_indices]
            top_weights = [float(topic[i]) for i in top_indices]
            
            topics.append({
                "topic_id": topic_idx,
                "words": top_words,
                "weights": top_weights
            })
        
        return topics
        
    except Exception as e:
        print(f"Error generating LDA for {sentiment}: {e}")
        return []


def extract_top_words_by_sentiment(
    processed_texts: List[str],
    predictions: List[int],
    n_top_words: int = 50,
    min_df: int = 1,
    max_df: float = 1.0
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Extract top words untuk positive dan negative sentimen.
    
    Args:
        processed_texts: List of stemmed texts
        predictions: List of predictions (0 atau 1)
        n_top_words: Jumlah top words untuk extract
        min_df: Minimum document frequency
        max_df: Maximum document frequency
    
    Returns:
        (top_words_pos, top_words_neg) - Dict dengan word: frequency
    """
    
    top_words_pos = {}
    top_words_neg = {}
    
    for text, pred in zip(processed_texts, predictions):
        words = text.split()
        
        if pred == 1:  # Positive
            for word in words:
                word = word.strip()
                if word:
                    top_words_pos[word] = top_words_pos.get(word, 0) + 1
        else:  # Negative
            for word in words:
                word = word.strip()
                if word:
                    top_words_neg[word] = top_words_neg.get(word, 0) + 1
    
    # Sort dan take top N
    top_words_pos = dict(sorted(top_words_pos.items(), key=lambda x: x[1], reverse=True)[:n_top_words])
    top_words_neg = dict(sorted(top_words_neg.items(), key=lambda x: x[1], reverse=True)[:n_top_words])
    
    return top_words_pos, top_words_neg
