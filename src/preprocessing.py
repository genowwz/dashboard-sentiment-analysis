import json
import os
import re
import string
from pathlib import Path
from typing import Dict, List

from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

ARTIFACTS_DIR = Path("artifacts")

DEFAULT_CUSTOM_STOPWORDS = [
    "ya","tok","deh","wok","we","the","wi","sih","roy","not","oh","entar","mah",
    "kayak","auto","nih","cs","y","abang","bang","gue","gwe","gua","guaa","wkwk",
    "wkwkwk","wk","hehe","hehee","yg","aja","nya","dong","nih","nihh","lah",
    "jokowi","puan","banteng"
]
DEFAULT_PROTECTED_WORDS = ["pemilu", "elektronik", "evoting", "e-voting"]
DEFAULT_SPECIAL_CASES = {
    "milih": "pilih",
    "milihnya": "pilih",
    "pemilihan": "pilih",
    "kecurangan": "curang",
    "hasilnya": "hasil",
    "dihilangkan": "hilang",
    "dimanipulasi": "manipulasi",
    "datanya": "data",
    "memilih": "pilih",
    "mengurangi": "kurang",
    "peratusan": "ratus",
    "melaksanakan": "laksana",
    "tingkatkan": "tingkat",
    "saatnya": "saat",
    "kelelahan": "lelah",
    "mengundi": "undi",
    "dilakukan": "laku"
}
DEFAULT_BASIC_STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "untuk", "pada", "adalah", "itu", "ini",
    "atau", "juga", "dengan", "karena", "dalam", "agar", "sudah", "belum", "akan",
    "jadi", "lebih", "masih", "ada", "saja", "bisa", "tidak", "tak", "nya"
}


def _load_json_if_exists(filename: str, default):
    path = ARTIFACTS_DIR / filename
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return default


SLANG_DICT: Dict[str, str] = _load_json_if_exists("slang_dict.json", {})
CUSTOM_STOPWORDS = set(_load_json_if_exists("custom_stopwords.json", DEFAULT_CUSTOM_STOPWORDS))
PROTECTED_WORDS = set(_load_json_if_exists("protected_words.json", DEFAULT_PROTECTED_WORDS))
SPECIAL_CASES = _load_json_if_exists("special_cases.json", DEFAULT_SPECIAL_CASES)
BASIC_STOPWORDS = set(_load_json_if_exists("basic_stopwords.json", list(DEFAULT_BASIC_STOPWORDS)))

_factory = StemmerFactory()
_stemmer = _factory.create_stemmer()


def cleaning_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", " ", text)
    text = re.sub(r"@\w+", " ", text)
    text = re.sub(r"#\w+", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    return text


def case_folding(text: str) -> str:
    return text.lower().strip() if isinstance(text, str) else ""


def normalize_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    words = text.split()
    normalized = [SLANG_DICT.get(word.strip().lower(), word.strip().lower()) for word in words]
    return " ".join(normalized)


def tokenize_text(text: str) -> List[str]:
    return text.split() if isinstance(text, str) else []


def remove_stopwords(tokens: List[str]) -> List[str]:
    return [
        token for token in tokens
        if token not in BASIC_STOPWORDS and token not in CUSTOM_STOPWORDS
    ]


def stem_tokens(tokens: List[str]) -> List[str]:
    stemmed = []
    for token in tokens:
        clean = token.strip().lower()
        if not clean:
            continue
        if clean in PROTECTED_WORDS:
            stemmed.append(clean)
        else:
            stemmed.append(_stemmer.stem(clean))
    return stemmed


def preprocess_text(text: str) -> Dict[str, object]:
    cleaned = cleaning_text(text)
    folded = case_folding(cleaned)
    normalized = normalize_text(folded)
    tokens = tokenize_text(normalized)
    filtered_tokens = remove_stopwords(tokens)
    stemmed_tokens = stem_tokens(filtered_tokens)
    stemmed_text = " ".join(stemmed_tokens)

    return {
        "original": text if isinstance(text, str) else "",
        "cleaned": cleaned,
        "case_folded": folded,
        "normalized": normalized,
        "tokens": tokens,
        "filtered_tokens": filtered_tokens,
        "stemmed_tokens": stemmed_tokens,
        "stemmed_text": stemmed_text,
    }
