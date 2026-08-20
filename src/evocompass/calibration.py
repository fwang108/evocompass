from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from .models import LogisticCalibration


def fit_logistic_calibration(logits: np.ndarray, labels: np.ndarray) -> LogisticCalibration:
    scores = np.asarray(logits, dtype=float).reshape(-1, 1)
    targets = np.asarray(labels, dtype=int)
    if len(np.unique(targets)) != 2:
        raise ValueError("Calibration requires positive and negative examples")
    classifier = LogisticRegression(max_iter=1000, random_state=0)
    classifier.fit(scores, targets)
    return LogisticCalibration(
        coefficient=float(classifier.coef_[0, 0]),
        intercept=float(classifier.intercept_[0]),
    )


def probability_to_hazard(probability: np.ndarray, interval: float = 1.0) -> np.ndarray:
    if interval <= 0:
        raise ValueError("Interval must be positive")
    values = np.asarray(probability, dtype=float)
    values = np.clip(values, 1e-8, 1 - 1e-8)
    return -np.log1p(-values) / interval

