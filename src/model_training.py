from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB


@dataclass
class TrainingBundle:
    model_name: str
    model: object
    X_test: object
    y_test: np.ndarray
    predictions: np.ndarray


def split_dataset(X, y, test_size: float = 0.2, random_state: int = 42):
    return train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y,
    )


def apply_smote(X_train, y_train, random_state: int = 42):
    dense_matrix = X_train.toarray()
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(dense_matrix, y_train)


def train_and_select_model(X_train, X_test, y_train, y_test) -> TrainingBundle:
    candidate_models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "naive_bayes": MultinomialNB(),
    }

    best_bundle = None
    best_score = -1.0

    X_train_smote, y_train_smote = apply_smote(X_train, y_train)
    X_test_dense = X_test.toarray()

    for model_name, model in candidate_models.items():
        model.fit(X_train_smote, y_train_smote)
        predictions = model.predict(X_test_dense)
        score = f1_score(y_test, predictions, pos_label="FAKE")

        if score > best_score:
            best_score = score
            best_bundle = TrainingBundle(
                model_name=model_name,
                model=model,
                X_test=X_test_dense,
                y_test=y_test,
                predictions=predictions,
            )

    if best_bundle is None:
        raise RuntimeError("No model was trained successfully.")

    return best_bundle
