from __future__ import annotations

from sklearn.feature_extraction.text import TfidfVectorizer


def build_tfidf_vectorizer(max_features: int = 5000) -> TfidfVectorizer:
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=(1, 2),
        min_df=1,
        max_df=0.95,
        sublinear_tf=True,
    )


def fit_transform_features(text_series, vectorizer: TfidfVectorizer | None = None):
    vectorizer = vectorizer or build_tfidf_vectorizer()
    features = vectorizer.fit_transform(text_series)
    return features, vectorizer


def transform_features(text_series, vectorizer: TfidfVectorizer):
    return vectorizer.transform(text_series)
