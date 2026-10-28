from collections import Counter
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset_sin_cajas_v2_aprobado"
MANIFEST = DATASET_ROOT / "_control" / "dataset_manifest.csv"
requires_private_dataset = pytest.mark.skipif(
    not MANIFEST.is_file(),
    reason="El dataset privado no está disponible en esta instalación.",
)


@requires_private_dataset
def test_loads_approved_manifest():
    try:
        from modelo_sin_cajas.datos import load_records
    except ModuleNotFoundError as error:
        pytest.fail(f"Falta implementar el lector del manifiesto: {error}")

    records = load_records(MANIFEST, DATASET_ROOT)

    assert len(records) == 335
    assert len({record.vaca_id for record in records}) == 66
    assert Counter(record.conjunto for record in records) == {
        "ENTRENAMIENTO": 243,
        "VALIDACION": 36,
        "PRUEBA": 56,
    }


def test_rejects_cow_assigned_to_multiple_splits(tmp_path):
    try:
        from modelo_sin_cajas.datos import ImageRecord, validate_records
    except ImportError as error:
        pytest.fail(f"Falta implementar la validación de fugas: {error}")

    image = tmp_path / "imagen.jpg"
    image.touch()
    records = [
        ImageRecord("img-1", "100", 3.0, "ENTRENAMIENTO", image),
        ImageRecord("img-2", "100", 3.0, "PRUEBA", image),
    ]

    with pytest.raises(ValueError, match="más de un conjunto"):
        validate_records(records)


def test_rejects_cow_with_multiple_bcs_values(tmp_path):
    from modelo_sin_cajas.datos import ImageRecord, validate_records

    image = tmp_path / "imagen.jpg"
    image.touch()
    records = [
        ImageRecord("img-1", "100", 3.0, "ENTRENAMIENTO", image),
        ImageRecord("img-2", "100", 3.25, "ENTRENAMIENTO", image),
    ]

    with pytest.raises(ValueError, match="más de un BCS"):
        validate_records(records)


def test_rejects_missing_image_file(tmp_path):
    from modelo_sin_cajas.datos import ImageRecord, validate_records

    missing_image = tmp_path / "no_existe.jpg"
    records = [
        ImageRecord("img-1", "100", 3.0, "ENTRENAMIENTO", missing_image),
    ]

    with pytest.raises(FileNotFoundError, match="no existe"):
        validate_records(records)


def test_bcs_class_catalog_has_fixed_order():
    try:
        from modelo_sin_cajas.datos import CLASS_VALUES
    except ImportError as error:
        pytest.fail(f"Falta definir el catálogo BCS: {error}")

    assert CLASS_VALUES == (2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75)


def test_rejects_unknown_split_name(tmp_path):
    from modelo_sin_cajas.datos import ImageRecord, validate_records

    image = tmp_path / "imagen.jpg"
    image.touch()
    records = [
        ImageRecord("img-1", "100", 3.0, "VALIDACIONN", image),
    ]

    with pytest.raises(ValueError, match="conjunto no válido"):
        validate_records(records)


def test_rejects_bcs_outside_class_catalog(tmp_path):
    from modelo_sin_cajas.datos import ImageRecord, validate_records

    image = tmp_path / "imagen.jpg"
    image.touch()
    records = [
        ImageRecord("img-1", "100", 4.0, "ENTRENAMIENTO", image),
    ]

    with pytest.raises(ValueError, match="BCS no válido"):
        validate_records(records)
