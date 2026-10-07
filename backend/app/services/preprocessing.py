import re
import unicodedata
from dataclasses import dataclass
from typing import Optional

try:
    from nltk.stem import WordNetLemmatizer
    _lemmatizer = WordNetLemmatizer()
    # Test lemmatizer availability
    _lemmatizer.lemmatize("testing")
except Exception:
    _lemmatizer = None


@dataclass
class PreprocessedInput:
    """Container holding multiple stages of the preprocessed input text."""

    original_text: str
    normalized_text: str
    tokens: list[str]
    lemmatized_tokens: list[str]
    processed_text: str


def normalize_whitespace(text: str) -> str:
    """Replace multiple whitespace characters with a single space and strip edges."""
    return re.sub(r"\s+", " ", text).strip()


def remove_accent_marks(text: str) -> str:
    """Normalize unicode characters and remove accent marks."""
    normalized = unicodedata.normalize("NFKD", text)
    return "".join(c for c in normalized if not unicodedata.combining(c))


def normalize_text(text: str) -> str:
    """
    Perform low-level string normalization:
    - Unicode normalization
    - Lowercasing
    - Stripping surrounding and extra whitespace
    """
    if not text:
        return ""
    text = remove_accent_marks(text)
    text = text.lower()
    text = normalize_whitespace(text)
    return text


def tokenize(text: str) -> list[str]:
    """
    Split normalized text into word tokens using an alphanumeric-aware regex.
    Preserves hyphens and apostrophes within words where applicable.
    """
    if not text:
        return []
    # Extract alphanumeric words or contractions
    return re.findall(r"\b[\w'-]+\b", text)


def lemmatize_tokens(tokens: list[str]) -> list[str]:
    """Lemmatize tokens using NLTK WordNetLemmatizer if available; otherwise returns unchanged tokens."""
    if not tokens:
        return []
    if _lemmatizer is not None:
        try:
            return [_lemmatizer.lemmatize(token) for token in tokens]
        except Exception:
            return tokens
    return tokens


def clean_punctuation(text: str) -> str:
    """Remove unwanted punctuation while preserving alphanumeric characters and spaces."""
    return re.sub(r"[^\w\s]", " ", text)


def preprocess(text: str) -> PreprocessedInput:
    """
    Execute full text preprocessing pipeline:
    1. Preserves original text for NER extraction.
    2. Lowercases and cleans whitespace.
    3. Tokenizes and lemmatizes words.
    4. Builds a clean, space-separated string suitable for vectorizers and ML classifiers.
    """
    original = text or ""
    normalized = normalize_text(original)
    cleaned_punct = clean_punctuation(normalized)
    cleaned_norm = normalize_whitespace(cleaned_punct)

    tokens = tokenize(cleaned_norm)
    lemmatized = lemmatize_tokens(tokens)
    processed_text = " ".join(lemmatized) if lemmatized else cleaned_norm

    return PreprocessedInput(
        original_text=original,
        normalized_text=normalized,
        tokens=tokens,
        lemmatized_tokens=lemmatized,
        processed_text=processed_text,
    )
