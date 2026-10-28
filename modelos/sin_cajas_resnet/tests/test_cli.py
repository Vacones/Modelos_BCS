import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_ROOT = PROJECT_ROOT / "dataset_sin_cajas_v2_aprobado"
MANIFEST = DATASET_ROOT / "_control" / "dataset_manifest.csv"
requires_private_dataset = pytest.mark.skipif(
    not MANIFEST.is_file(),
    reason="El dataset privado no está disponible en esta instalación.",
)


@requires_private_dataset
def test_validation_only_command_reports_dataset_without_training(capsys):
    try:
        from entrenar_sin_cajas import main
    except ModuleNotFoundError as error:
        pytest.fail(f"Falta implementar el comando principal: {error}")

    exit_code = main(
        [
            "--solo-validar",
            "--dataset-root",
            str(DATASET_ROOT),
            "--batch-size",
            "16",
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["mode"] == "validation_only"
    assert output["device"] in {"mps", "cpu"}
    assert output["images"] == {
        "train": 243,
        "validation": 36,
        "test": 56,
    }
    assert output["batches"] == {
        "train": 16,
        "validation": 3,
        "test": 4,
    }


def test_quick_mode_dispatches_one_epoch_without_test_evaluation(
    monkeypatch,
    tmp_path,
    capsys,
):
    import entrenar_sin_cajas as cli

    def fake_quick_experiment(args):
        return {
            "mode": "quick_test",
            "epochs": args.epochs,
            "test_evaluated": False,
            "output_dir": str(args.output_dir),
        }

    monkeypatch.setattr(
        cli,
        "run_quick_experiment",
        fake_quick_experiment,
        raising=False,
    )

    exit_code = cli.main(
        [
            "--prueba-rapida",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["mode"] == "quick_test"
    assert output["epochs"] == 1
    assert output["test_evaluated"] is False


def test_quick_experiment_writes_config_history_and_checkpoint(
    monkeypatch,
    tmp_path,
):
    import entrenar_sin_cajas as cli

    try:
        run_quick_experiment = cli.run_quick_experiment
    except AttributeError as error:
        pytest.fail(f"Falta implementar la prueba rápida real: {error}")

    loaders = {
        "train": object(),
        "validation": object(),
        "test": object(),
    }
    monkeypatch.setattr(cli, "build_data_loaders", lambda **kwargs: loaders)
    monkeypatch.setattr(cli, "build_model", lambda **kwargs: object(), raising=False)
    monkeypatch.setattr(cli, "select_device", lambda: "mps")

    def fake_fit_model(**kwargs):
        assert kwargs["train_loader"] is loaders["train"]
        assert kwargs["validation_loader"] is loaders["validation"]
        kwargs["checkpoint_path"].write_bytes(b"checkpoint")
        return [
            {
                "epoch": 1,
                "train_loss": 1.5,
                "validation_cow_count": 10,
                "validation_exact_accuracy": 0.2,
                "validation_within_0_25_accuracy": 0.7,
                "validation_mean_absolute_error": 0.25,
            }
        ]

    monkeypatch.setattr(cli, "fit_model", fake_fit_model, raising=False)
    monkeypatch.setattr(
        cli,
        "evaluate_by_cow",
        lambda model, loader, device: {
            "cow_predictions": {},
            "metrics": {"cow_count": 10},
            "confusion_matrix": np.zeros((7, 7), dtype=int),
        },
    )
    monkeypatch.setattr(
        cli,
        "write_evaluation_reports",
        lambda output_dir, split_name, evaluation: None,
    )
    args = SimpleNamespace(
        dataset_root=DATASET_ROOT,
        output_dir=tmp_path,
        run_name="smoke-test",
        batch_size=16,
        image_size=224,
        seed=20261026,
        num_workers=0,
        epochs=1,
        learning_rate=1e-4,
        patience=1,
    )

    summary = run_quick_experiment(args)
    run_dir = tmp_path / "smoke-test"

    assert summary["mode"] == "quick_test"
    assert summary["test_evaluated"] is False
    assert (run_dir / "config.json").is_file()
    assert (run_dir / "history.csv").is_file()
    assert (run_dir / "best_model.pt").is_file()


def test_quick_experiment_reports_validation_without_using_test(
    monkeypatch,
    tmp_path,
):
    import entrenar_sin_cajas as cli

    train_loader = object()
    validation_loader = object()
    test_loader = object()
    loaders = {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
    }
    model = object()
    monkeypatch.setattr(cli, "build_data_loaders", lambda **kwargs: loaders)
    monkeypatch.setattr(cli, "build_model", lambda **kwargs: model)
    monkeypatch.setattr(cli, "select_device", lambda: "mps")
    monkeypatch.setattr(cli, "set_seed", lambda seed: None)

    def fake_fit_model(**kwargs):
        kwargs["checkpoint_path"].write_bytes(b"checkpoint")
        return [{"epoch": 1, "train_loss": 1.0}]

    monkeypatch.setattr(cli, "fit_model", fake_fit_model)
    evaluation = {
        "cow_predictions": {},
        "metrics": {"cow_count": 10},
        "confusion_matrix": np.zeros((7, 7), dtype=int),
    }

    def fake_evaluate_by_cow(received_model, received_loader, device):
        assert received_model is model
        assert received_loader is validation_loader
        assert received_loader is not test_loader
        return evaluation

    report_calls = []
    monkeypatch.setattr(cli, "evaluate_by_cow", fake_evaluate_by_cow, raising=False)
    monkeypatch.setattr(
        cli,
        "write_evaluation_reports",
        lambda output_dir, split_name, result: report_calls.append(
            (output_dir, split_name, result)
        ),
        raising=False,
    )
    args = SimpleNamespace(
        dataset_root=DATASET_ROOT,
        output_dir=tmp_path,
        run_name="reports-test",
        batch_size=16,
        image_size=224,
        seed=20261026,
        num_workers=0,
        epochs=1,
        learning_rate=1e-4,
        patience=1,
    )

    summary = cli.run_quick_experiment(args)

    assert report_calls == [(tmp_path / "reports-test", "validation", evaluation)]
    assert summary["validation_metrics"] == {"cow_count": 10}
    assert summary["test_evaluated"] is False


def test_full_mode_dispatches_requested_epochs_and_final_test(
    monkeypatch,
    tmp_path,
    capsys,
):
    import entrenar_sin_cajas as cli

    def fake_full_experiment(args):
        return {
            "mode": "full_training",
            "epochs_requested": args.epochs,
            "patience": args.patience,
            "test_evaluated": True,
        }

    monkeypatch.setattr(
        cli,
        "run_full_experiment",
        fake_full_experiment,
        raising=False,
    )

    exit_code = cli.main(
        [
            "--entrenar",
            "--epochs",
            "30",
            "--patience",
            "7",
            "--output-dir",
            str(tmp_path),
        ]
    )
    output = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert output["mode"] == "full_training"
    assert output["epochs_requested"] == 30
    assert output["patience"] == 7
    assert output["test_evaluated"] is True


def test_full_experiment_evaluates_test_only_after_model_selection(
    monkeypatch,
    tmp_path,
):
    import entrenar_sin_cajas as cli

    try:
        run_full_experiment = cli.run_full_experiment
    except AttributeError as error:
        pytest.fail(f"Falta implementar el entrenamiento completo: {error}")

    train_loader = object()
    validation_loader = object()
    test_loader = object()
    loaders = {
        "train": train_loader,
        "validation": validation_loader,
        "test": test_loader,
    }
    model = object()
    call_order = []
    monkeypatch.setattr(cli, "build_data_loaders", lambda **kwargs: loaders)
    monkeypatch.setattr(cli, "build_model", lambda **kwargs: model)
    monkeypatch.setattr(cli, "select_device", lambda: "mps")
    monkeypatch.setattr(cli, "set_seed", lambda seed: None)

    def fake_fit_model(**kwargs):
        call_order.append("fit")
        assert kwargs["train_loader"] is train_loader
        assert kwargs["validation_loader"] is validation_loader
        kwargs["checkpoint_path"].write_bytes(b"checkpoint")
        return [{"epoch": 1, "train_loss": 1.0}]

    monkeypatch.setattr(cli, "fit_model", fake_fit_model)

    def fake_evaluate_by_cow(received_model, loader, device):
        split = "validation" if loader is validation_loader else "test"
        call_order.append(split)
        return {
            "cow_predictions": {},
            "metrics": {"cow_count": 10, "split": split},
            "confusion_matrix": np.zeros((7, 7), dtype=int),
        }

    report_calls = []
    monkeypatch.setattr(cli, "evaluate_by_cow", fake_evaluate_by_cow)
    monkeypatch.setattr(
        cli,
        "write_evaluation_reports",
        lambda output_dir, split_name, evaluation: report_calls.append(split_name),
    )
    args = SimpleNamespace(
        dataset_root=DATASET_ROOT,
        output_dir=tmp_path,
        run_name="full-test",
        batch_size=16,
        image_size=224,
        seed=20261026,
        num_workers=0,
        epochs=30,
        learning_rate=1e-4,
        patience=7,
    )

    summary = run_full_experiment(args)

    assert call_order == ["fit", "validation", "test"]
    assert report_calls == ["validation", "test"]
    assert summary["test_evaluated"] is True
    assert summary["validation_metrics"]["split"] == "validation"
    assert summary["test_metrics"]["split"] == "test"
