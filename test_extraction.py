import os
import re
from collections import Counter
import pandas as pd

def _tokenize_for_wordcloud(text):
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    if not text:
        return []
    tokens = re.findall(r"[a-z0-9]+", text)
    return tokens

ARTIFACTS = "artifacts"
test_path = os.path.join(os.getcwd(), ARTIFACTS, "test_set_skripsi.csv")
df = pd.read_csv(test_path)

word_freq_neg = Counter()
word_freq_pos = Counter()

for _, row in df.iterrows():
    text = str(row.get('text', '')).strip()
    label = row.get('label')
    if not text:
        continue
    
    tokens = _tokenize_for_wordcloud(text)
    if not tokens:
        continue
    
    ngrams = []
    for n in range(1, 3):
        if len(tokens) < n:
            continue
        for start in range(len(tokens) - n + 1):
            ngrams.append(" ".join(tokens[start:start + n]))
    
    if label == 0:
        word_freq_neg.update(ngrams)
    elif label == 1:
        word_freq_pos.update(ngrams)

top_words_neg = dict(word_freq_neg.most_common(80))
top_words_pos = dict(word_freq_pos.most_common(80))

print("Pos word count:", len(top_words_pos))
print("Neg word count:", len(top_words_neg))
print("Top 10 pos:", list(top_words_pos.items())[:10])
print("Top 10 neg:", list(top_words_neg.items())[:10])
