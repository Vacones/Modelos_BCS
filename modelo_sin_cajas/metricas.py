from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .datos import CLASS_VALUES


@dataclass(frozen=True)
class CowPrediction:
    vaca_id: str
    true_bcs: float
    predicted_bcs: float
    probabilities: np.ndarray
    photo_count: int


def aggregate_by_cow(
    probabilities: np.ndarray,
    cow_ids: list[str],
    true_bcs: list[float],
) -> dict[str, CowPrediction]:
    result: dict[str, CowPrediction] = {}

    for cow_id in dict.fromkeys(cow_ids):
        indices = [index for index, value in enumerate(cow_ids) if value == cow_id]
        cow_labels = {true_bcs[index] for index in indices}
        if len(cow_labels) > 1:
            raise ValueError(f"La vaca {cow_id} tiene etiquetas BCS distintas: {sorted(cow_labels)}")
        average = probabilities[indices].mean(axis=0)
        predicted_bcs = CLASS_VALUES[int(np.argmax(average))]
        result[cow_id] = CowPrediction(
            vaca_id=cow_id,
            true_bcs=next(iter(cow_labels)),
            predicted_bcs=predicted_bcs,
            probabilities=average,
            photo_count=len(indices),
        )

    return result


def classification_metrics(
    predictions: dict[str, CowPrediction],
) -> dict[str, float | int]:
    true_values = np.array(
        [prediction.true_bcs for prediction in predictions.values()],
        dtype=np.float64,
    )
    predicted_values = np.array(
        [prediction.predicted_bcs for prediction in predictions.values()],
        dtype=np.float64,
    )
    absolute_errors = np.abs(predicted_values - true_values)

    return {
        "cow_count": len(predictions),
        "exact_accuracy": float(np.mean(absolute_errors == 0)),
        "within_0_25_accuracy": float(np.mean(absolute_errors <= 0.25)),
        "mean_absolute_error": float(np.mean(absolute_errors)),
    }


def confusion_matrix(
    predictions: dict[str, CowPrediction],
) -> np.ndarray:
    matrix = np.zeros((len(CLASS_VALUES), len(CLASS_VALUES)), dtype=np.int64)
    class_indices = {value: index for index, value in enumerate(CLASS_VALUES)}

    for prediction in predictions.values():
        true_index = class_indices[prediction.true_bcs]
        predicted_index = class_indices[prediction.predicted_bcs]
        matrix[true_index, predicted_index] += 1

    return matrix
