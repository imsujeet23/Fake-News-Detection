from __future__ import annotations

from pathlib import Path

import streamlit as st

from main import MODEL_PATH, train_pipeline
from src.predict import predict_news

PROJECT_ROOT = Path(__file__).resolve().parent

st.set_page_config(
    page_title="Fake News Detection",
    page_icon="📰",
    layout="centered",
)


@st.cache_resource
def ensure_model() -> Path:
    if not MODEL_PATH.exists():
        train_pipeline()
    return MODEL_PATH


def main() -> None:
    st.title("Fake News Detection Using NLP")
    st.write(
        "Paste a news headline or article snippet below to classify it as "
        "`REAL` or `FAKE` using the trained NLP model."
    )

    ensure_model()

    sample_text = (
        "Officials released a verified public report after independent review "
        "and confirmation from multiple trusted institutions."
    )
    text = st.text_area(
        "Enter news text",
        value=sample_text,
        height=220,
        help="Use a headline, paragraph, or full short article for prediction.",
    )

    if st.button("Predict", type="primary"):
        if not text.strip():
            st.warning("Please enter some news text before predicting.")
            return

        result = predict_news(text, MODEL_PATH)
        prediction = result["prediction"]
        confidence = result["confidence"]

        if prediction == "FAKE":
            st.error(f"Prediction: {prediction}")
        else:
            st.success(f"Prediction: {prediction}")

        st.subheader("Confidence Scores")
        st.write(confidence)

        st.subheader("Cleaned Text Used By The Model")
        st.code(result["cleaned_text"], language="text")

        st.caption(f"Model selected during training: {result['model_name']}")


if __name__ == "__main__":
    main()
