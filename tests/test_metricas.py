import numpy as np
import pytest


def test_aggregate_by_cow_averages_photo_probabilities():
    try:
        from modelo_sin_cajas.metricas import aggregate_by_cow
    except ModuleNotFoundError as error:
        pytest.fail(f"Falta implementar la agregación por vaca: {error}")

    probabilities = np.array(
        [
            [0.02, 0.03, 0.05, 0.80, 0.05, 0.03, 0.02],
            [0.02, 0.03, 0.05, 0.50, 0.30, 0.07, 0.03],
            [0.03, 0.75, 0.10, 0.05, 0.03, 0.02, 0.02],
        ],
        dtype=np.float64,
    )

    result = aggregate_by_cow(
        probabilities=probabilities,
        cow_ids=["100", "100", "200"],
        true_bcs=[3.0, 3.0, 2.5],
    )

    assert set(result) == {"100", "200"}
    np.testing.assert_allclose(
        result["100"].probabilities,
        probabilities[:2].mean(axis=0),
    )
    assert result["100"].predicted_bcs == 3.0
    assert result["100"].true_bcs == 3.0
    assert result["100"].photo_count == 2
    assert result["200"].predicted_bcs == 2.5
    assert result["200"].photo_count == 1


def test_aggregate_by_cow_rejects_conflicting_true_labels():
    from modelo_sin_cajas.metricas import aggregate_by_cow

    probabilities = np.full((2, 7), 1.0 / 7, dtype=np.float64)

    with pytest.raises(ValueError, match="etiquetas BCS distintas"):
        aggregate_by_cow(
            probabilities=probabilities,
            cow_ids=["100", "100"],
            true_bcs=[3.0, 3.25],
        )


def test_classification_metrics_are_calculated_per_cow():
    try:
        from modelo_sin_cajas.metricas import CowPrediction, classification_metrics
    except ImportError as error:
        pytest.fail(f"Falta implementar las métricas por vaca: {error}")

    probabilities = np.full(7, 1.0 / 7, dtype=np.float64)
    predictions = {
        "100": CowPrediction("100", 2.5, 2.5, probabilities, 2),
        "200": CowPrediction("200", 2.75, 3.0, probabilities, 3),
        "300": CowPrediction("300", 3.0, 3.5, probabilities, 1),
        "400": CowPrediction("400", 3.25, 3.0, probabilities, 4),
    }

    metrics = classification_metrics(predictions)

    assert metrics["cow_count"] == 4
    assert metrics["exact_accuracy"] == pytest.approx(0.25)
    assert metrics["within_0_25_accuracy"] == pytest.approx(0.75)
    assert metrics["mean_absolute_error"] == pytest.approx(0.25)


def test_confusion_matrix_uses_true_rows_and_predicted_columns():
    try:
        from modelo_sin_cajas.metricas import CowPrediction, confusion_matrix
    except ImportError as error:
        pytest.fail(f"Falta implementar la matriz de confusión: {error}")

    probabilities = np.full(7, 1.0 / 7, dtype=np.float64)
    predictions = {
        "100": CowPrediction("100", 2.5, 2.5, probabilities, 1),
        "200": CowPrediction("200", 2.5, 2.75, probabilities, 1),
        "300": CowPrediction("300", 3.0, 2.75, probabilities, 1),
    }

    matrix = confusion_matrix(predictions)

    assert matrix.shape == (7, 7)
    assert matrix.sum() == 3
    assert matrix[1, 1] == 1
    assert matrix[1, 2] == 1
    assert matrix[3, 2] == 1
