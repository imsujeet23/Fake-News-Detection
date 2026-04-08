from __future__ import annotations

import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path.cwd() / ".matplotlib"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score, precision_score, recall_score


def evaluate_predictions(y_true, y_pred) -> dict:
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, pos_label="FAKE"),
        "recall": recall_score(y_true, y_pred, pos_label="FAKE"),
        "f1_score": f1_score(y_true, y_pred, pos_label="FAKE"),
        "classification_report": classification_report(y_true, y_pred),
    }


def save_confusion_matrix(y_true, y_pred, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred, labels=["REAL", "FAKE"])

    plt.figure(figsize=(6, 4))
    sns.heatmap(
        matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
    )
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(output)
    plt.close()


def save_results(metrics: dict, model_name: str, output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(
            [
                f"Best model: {model_name}",
                f"Accuracy: {metrics['accuracy']:.4f}",
                f"Precision: {metrics['precision']:.4f}",
                f"Recall: {metrics['recall']:.4f}",
                f"F1-score: {metrics['f1_score']:.4f}",
                "",
                "Classification Report:",
                metrics["classification_report"],
            ]
        ),
        encoding="utf-8",
    )
