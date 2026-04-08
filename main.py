from __future__ import annotations

from pathlib import Path

import joblib

from src.evaluation import evaluate_predictions, save_confusion_matrix, save_results
from src.feature_extraction import fit_transform_features
from src.model_training import split_dataset, train_and_select_model
from src.predict import predict_news
from src.preprocessing import load_raw_data, preprocess_dataframe, save_processed_data

PROJECT_ROOT = Path(__file__).resolve().parent
RAW_DATA_PATH = PROJECT_ROOT / "data" / "raw" / "fake_news.csv"
PROCESSED_DATA_PATH = PROJECT_ROOT / "data" / "processed" / "cleaned_data.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "trained_model.pkl"
RESULTS_PATH = PROJECT_ROOT / "outputs" / "results.txt"
CONFUSION_MATRIX_PATH = PROJECT_ROOT / "outputs" / "confusion_matrix.png"


def train_pipeline() -> dict:
    raw_df = load_raw_data(RAW_DATA_PATH)
    processed_df = preprocess_dataframe(raw_df)
    save_processed_data(processed_df, PROCESSED_DATA_PATH)

    X, vectorizer = fit_transform_features(processed_df["clean_text"])
    y = processed_df["label"].astype(str).str.upper().to_numpy()

    X_train, X_test, y_train, y_test = split_dataset(X, y)
    bundle = train_and_select_model(X_train, X_test, y_train, y_test)

    metrics = evaluate_predictions(bundle.y_test, bundle.predictions)
    save_results(metrics, bundle.model_name, RESULTS_PATH)
    save_confusion_matrix(bundle.y_test, bundle.predictions, CONFUSION_MATRIX_PATH)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_name": bundle.model_name,
            "model": bundle.model,
            "vectorizer": vectorizer,
            "metrics": metrics,
        },
        MODEL_PATH,
    )

    return metrics


def demo_prediction() -> None:
    sample_text = (
        "Breaking news: scientists confirm a verified public health report "
        "after reviewing evidence from multiple trusted institutions."
    )
    result = predict_news(sample_text, MODEL_PATH)
    print("Prediction:", result["prediction"])
    print("Confidence:", result["confidence"])


if __name__ == "__main__":
    metrics = train_pipeline()
    print("Training complete.")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"F1-score: {metrics['f1_score']:.4f}")
    demo_prediction()
