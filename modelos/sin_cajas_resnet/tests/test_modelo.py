import pytest
import torch


def test_resnet18_returns_seven_bcs_scores_per_image():
    try:
        from modelo_sin_cajas.modelo import build_model
    except ModuleNotFoundError as error:
        pytest.fail(f"Falta implementar el modelo BCS: {error}")

    model = build_model(num_classes=7, pretrained=False)
    images = torch.zeros((2, 3, 224, 224), dtype=torch.float32)

    with torch.no_grad():
        scores = model(images)

    assert tuple(scores.shape) == (2, 7)


@pytest.mark.parametrize(
    ("mps_available", "expected_type"),
    [(True, "mps"), (False, "cpu")],
)
def test_select_device_uses_mps_with_cpu_fallback(
    monkeypatch,
    mps_available,
    expected_type,
):
    try:
        from modelo_sin_cajas.modelo import select_device
    except ImportError as error:
        pytest.fail(f"Falta implementar la selección de dispositivo: {error}")

    monkeypatch.setattr(torch.backends.mps, "is_available", lambda: mps_available)

    assert select_device().type == expected_type
