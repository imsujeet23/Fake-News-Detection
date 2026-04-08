from __future__ import annotations

from pathlib import Path

import joblib

from src.preprocessing import clean_text


def load_model_bundle(model_path: str | Path):
    return joblib.load(model_path)


def predict_news(text: str, model_path: str | Path) -> dict:
    bundle = load_model_bundle(model_path)
    cleaned = clean_text(text)
    features = bundle["vectorizer"].transform([cleaned]).toarray()
    prediction = bundle["model"].predict(features)[0]
    probabilities = bundle["model"].predict_proba(features)[0]
    labels = bundle["model"].classes_
    confidence = {label: float(score) for label, score in zip(labels, probabilities)}

    return {
        "input_text": text,
        "cleaned_text": cleaned,
        "prediction": prediction,
        "confidence": confidence,
        "model_name": bundle["model_name"],
    }
