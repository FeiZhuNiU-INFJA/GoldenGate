"""Classification metrics helpers."""
from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support


CLASS_NAMES = ["neutral", "buy", "sell"]


def summarize_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1, 2], zero_division=0
    )
    report = classification_report(
        y_true, y_pred, labels=[0, 1, 2], target_names=CLASS_NAMES, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1, 2])
    return {
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "support": support.tolist(),
        "macro_f1": float(np.mean(f1)),
        "report": report,
        "confusion_matrix": cm.tolist(),
    }
