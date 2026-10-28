import math
from pathlib import Path
import random

import pytest
import torch
from torch import nn
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset_sin_cajas_v2_aprobado"
MANIFEST = DATASET_ROOT / "_control" / "dataset_manifest.csv"
requires_private_dataset = pytest.mark.skipif(
    not MANIFEST.is_file(),
    reason="El dataset privado no está disponible en esta instalación.",
)


def test_train_one_epoch_updates_model_weights():
    try:
        from modelo_sin_cajas.entrenar import train_one_epoch
    except ModuleNotFoundError as error:
        pytest.fail(f"Falta implementar el ciclo de entrenamiento: {error}")

    torch.manual_seed(20261026)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 7))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    loader = [
        {
            "image": torch.randn(4, 3, 8, 8),
            "class_index": torch.tensor([0, 1, 2, 3]),
        }
    ]
    before = model[1].weight.detach().clone()

    loss = train_one_epoch(
        model=model,
        loader=loader,
        optimizer=optimizer,
        criterion=criterion,
        device=torch.device("cpu"),
    )

    assert math.isfinite(loss)
    assert loss > 0
    assert not torch.equal(before, model[1].weight.detach())


def test_predict_loader_returns_probabilities_and_identifiers():
    try:
        from modelo_sin_cajas.entrenar import predict_loader
    except ImportError as error:
        pytest.fail(f"Falta implementar la evaluación del modelo: {error}")

    torch.manual_seed(20261026)
    model = nn.Sequential(nn.Flatten(), nn.Linear(3 * 8 * 8, 7))
    loader = [
        {
            "image": torch.randn(2, 3, 8, 8),
            "class_index": torch.tensor([3, 1]),
            "bcs": torch.tensor([3.0, 2.5]),
            "vaca_id": ["100", "200"],
            "imagen_id": ["img-1", "img-2"],
        }
    ]

    result = predict_loader(model, loader, torch.device("cpu"))

    assert result["probabilities"].shape == (2, 7)
    np.testing.assert_allclose(
        result["probabilities"].sum(axis=1),
        np.ones(2),
        rtol=1e-6,
    )
    assert result["cow_ids"] == ["100", "200"]
    assert result["image_ids"] == ["img-1", "img-2"]
    assert result["true_bcs"] == [3.0, 2.5]


def test_evaluate_by_cow_connects_predictions_and_metrics():
    try:
        from modelo_sin_cajas.entrenar import evaluate_by_cow
    except ImportError as error:
        pytest.fail(f"Falta conectar la evaluación por vaca: {error}")

    class IdentityScores(nn.Module):
        def forward(self, images):
            return images[:, :, 0, 0]

    scores = torch.tensor(
        [
            [0.0, 0.0, 0.0, 5.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 4.0, 1.0, 0.0, 0.0],
            [0.0, 5.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    ).reshape(3, 7, 1, 1)
    loader = [
        {
            "image": scores,
            "class_index": torch.tensor([3, 3, 1]),
            "bcs": torch.tensor([3.0, 3.0, 2.5]),
            "vaca_id": ["100", "100", "200"],
            "imagen_id": ["img-1", "img-2", "img-3"],
        }
    ]

    result = evaluate_by_cow(IdentityScores(), loader, torch.device("cpu"))

    assert set(result["cow_predictions"]) == {"100", "200"}
    assert result["metrics"]["cow_count"] == 2
    assert result["metrics"]["exact_accuracy"] == pytest.approx(1.0)
    assert result["confusion_matrix"].sum() == 2


@requires_private_dataset
def test_build_data_loaders_uses_approved_partition():
    try:
        from modelo_sin_cajas.entrenar import build_data_loaders
    except ImportError as error:
        pytest.fail(f"Falta implementar los cargadores del dataset: {error}")

    loaders = build_data_loaders(
        manifest_path=MANIFEST,
        dataset_root=DATASET_ROOT,
        batch_size=16,
        image_size=224,
        seed=20261026,
        num_workers=0,
    )

    assert len(loaders["train"].dataset) == 243
    assert len(loaders["validation"].dataset) == 36
    assert len(loaders["test"].dataset) == 56


def test_fit_model_saves_best_checkpoint_from_validation(tmp_path):
    try:
        from modelo_sin_cajas.entrenar import fit_model
    except ImportError as error:
        pytest.fail(f"Falta implementar el ajuste y checkpoint: {error}")

    torch.manual_seed(20261026)
    model = nn.Sequential(nn.Flatten(), nn.Linear(7, 7))
    train_loader = [
        {
            "image": torch.eye(7).reshape(7, 7, 1, 1),
            "class_index": torch.arange(7),
        }
    ]
    validation_loader = [
        {
            "image": torch.eye(7).reshape(7, 7, 1, 1),
            "class_index": torch.arange(7),
            "bcs": torch.tensor([2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75]),
            "vaca_id": [str(index) for index in range(7)],
            "imagen_id": [f"img-{index}" for index in range(7)],
        }
    ]
    checkpoint_path = tmp_path / "best_model.pt"

    history = fit_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-3,
        patience=1,
        checkpoint_path=checkpoint_path,
    )

    assert len(history) == 1
    assert checkpoint_path.is_file()
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert checkpoint["epoch"] == 1
    assert checkpoint["selection_source"] == "validation_by_cow"
    assert "validation_metrics" in checkpoint


def test_set_seed_reproduces_python_numpy_and_torch_values():
    try:
        from modelo_sin_cajas.entrenar import set_seed
    except ImportError as error:
        pytest.fail(f"Falta implementar la semilla reproducible: {error}")

    set_seed(20261026)
    first = (random.random(), np.random.random(), torch.rand(1).item())
    set_seed(20261026)
    second = (random.random(), np.random.random(), torch.rand(1).item())

    assert second == pytest.approx(first)


def test_fit_model_reports_epoch_progress(tmp_path, capsys):
    from modelo_sin_cajas.entrenar import fit_model

    torch.manual_seed(20261026)
    model = nn.Sequential(nn.Flatten(), nn.Linear(7, 7))
    train_loader = [
        {
            "image": torch.eye(7).reshape(7, 7, 1, 1),
            "class_index": torch.arange(7),
        }
    ]
    validation_loader = [
        {
            "image": torch.eye(7).reshape(7, 7, 1, 1),
            "class_index": torch.arange(7),
            "bcs": torch.tensor([2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75]),
            "vaca_id": [str(index) for index in range(7)],
            "imagen_id": [f"img-{index}" for index in range(7)],
        }
    ]

    fit_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        device=torch.device("cpu"),
        epochs=1,
        learning_rate=1e-3,
        patience=1,
        checkpoint_path=tmp_path / "best_model.pt",
    )

    progress = capsys.readouterr().err
    assert "Época 1/1" in progress
    assert "MAE validación" in progress
