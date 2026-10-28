import csv
import json

import numpy as np
import pytest
from PIL import Image

from modelo_sin_cajas.metricas import CowPrediction


def test_write_evaluation_reports_creates_metrics_predictions_and_matrix(tmp_path):
    try:
        from modelo_sin_cajas.reportes import write_evaluation_reports
    except ModuleNotFoundError as error:
        pytest.fail(f"Falta implementar los reportes: {error}")

    probabilities = np.array([0.01, 0.04, 0.10, 0.70, 0.10, 0.04, 0.01])
    evaluation = {
        "cow_predictions": {
            "100": CowPrediction("100", 3.0, 3.0, probabilities, 3),
        },
        "metrics": {
            "cow_count": 1,
            "exact_accuracy": 1.0,
            "within_0_25_accuracy": 1.0,
            "mean_absolute_error": 0.0,
        },
        "confusion_matrix": np.diag([0, 0, 0, 1, 0, 0, 0]),
    }

    write_evaluation_reports(tmp_path, "validation", evaluation)

    metrics_path = tmp_path / "metrics_validation.json"
    predictions_path = tmp_path / "predicciones_por_vaca_validation.csv"
    matrix_path = tmp_path / "confusion_matrix_validation.csv"
    assert metrics_path.is_file()
    assert predictions_path.is_file()
    assert matrix_path.is_file()
    assert json.loads(metrics_path.read_text(encoding="utf-8"))["cow_count"] == 1
    with predictions_path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["vaca_id"] == "100"
    assert float(rows[0]["bcs_predicho"]) == 3.0


def test_write_evaluation_reports_creates_confusion_matrix_png(tmp_path):
    from modelo_sin_cajas.reportes import write_evaluation_reports

    probabilities = np.array([0.01, 0.04, 0.10, 0.70, 0.10, 0.04, 0.01])
    evaluation = {
        "cow_predictions": {
            "100": CowPrediction("100", 3.0, 3.0, probabilities, 3),
        },
        "metrics": {
            "cow_count": 1,
            "exact_accuracy": 1.0,
            "within_0_25_accuracy": 1.0,
            "mean_absolute_error": 0.0,
        },
        "confusion_matrix": np.diag([0, 0, 0, 1, 0, 0, 0]),
    }

    write_evaluation_reports(tmp_path, "validation", evaluation)

    png_path = tmp_path / "confusion_matrix_validation.png"
    assert png_path.is_file()
    with Image.open(png_path) as image:
        assert image.width >= 600
        assert image.height >= 600


def test_write_evaluation_reports_creates_photo_diagnostics(tmp_path):
    from modelo_sin_cajas.reportes import write_evaluation_reports

    probabilities = np.array(
        [[0.01, 0.04, 0.10, 0.70, 0.10, 0.04, 0.01]],
        dtype=np.float64,
    )
    evaluation = {
        "photo_predictions": {
            "probabilities": probabilities,
            "cow_ids": ["100"],
            "image_ids": ["img-1"],
            "true_bcs": [3.0],
        },
        "cow_predictions": {
            "100": CowPrediction("100", 3.0, 3.0, probabilities[0], 1),
        },
        "metrics": {
            "cow_count": 1,
            "exact_accuracy": 1.0,
            "within_0_25_accuracy": 1.0,
            "mean_absolute_error": 0.0,
        },
        "confusion_matrix": np.diag([0, 0, 0, 1, 0, 0, 0]),
    }

    write_evaluation_reports(tmp_path, "validation", evaluation)

    path = tmp_path / "predicciones_por_imagen_validation.csv"
    assert path.is_file()
    with path.open(newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file))
    assert rows[0]["imagen_id"] == "img-1"
    assert rows[0]["vaca_id"] == "100"
    assert float(rows[0]["bcs_predicho"]) == 3.0


def test_write_versions_records_reproducibility_environment(tmp_path):
    try:
        from modelo_sin_cajas.reportes import write_versions
    except ImportError as error:
        pytest.fail(f"Falta implementar el registro de versiones: {error}")

    write_versions(tmp_path)

    path = tmp_path / "versions.json"
    assert path.is_file()
    versions = json.loads(path.read_text(encoding="utf-8"))
    assert versions["python"]
    assert versions["torch"]
    assert versions["torchvision"]
    assert versions["numpy"]
    assert versions["pillow"]
