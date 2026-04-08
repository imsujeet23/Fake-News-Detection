from __future__ import annotations

import re
import string
from pathlib import Path

import pandas as pd

DEFAULT_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "has",
    "he", "in", "is", "it", "its", "of", "on", "that", "the", "to", "was",
    "were", "will", "with", "this", "their", "they", "you", "your", "but",
    "or", "not", "have", "had", "after", "before", "into", "than", "then",
}


def load_raw_data(path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    required_columns = {"title", "text", "label"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing required columns: {sorted(missing)}")
    return df.copy()


def clean_text(text: str, stopwords: set[str] | None = None) -> str:
    if pd.isna(text):
        return ""

    stopwords = stopwords or DEFAULT_STOPWORDS
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [token for token in text.split() if token not in stopwords]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    processed = df.copy()
    processed["title"] = processed["title"].fillna("")
    processed["text"] = processed["text"].fillna("")
    processed["content"] = (processed["title"] + " " + processed["text"]).str.strip()
    processed["clean_text"] = processed["content"].apply(clean_text)
    processed = processed[processed["clean_text"].str.len() > 0].reset_index(drop=True)
    return processed


def save_processed_data(df: pd.DataFrame, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output, index=False)
