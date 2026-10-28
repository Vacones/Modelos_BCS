from __future__ import annotations

import argparse
import csv
from datetime import datetime
import json
from pathlib import Path

from modelo_sin_cajas.datos import CLASS_VALUES
from modelo_sin_cajas.entrenar import (
    build_data_loaders,
    evaluate_by_cow,
    fit_model,
    set_seed,
)
from modelo_sin_cajas.modelo import build_model, select_device
from modelo_sin_cajas.reportes import write_evaluation_reports, write_versions


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET_ROOT = PROJECT_ROOT / "dataset_sin_cajas_v2_aprobado"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Entrenamiento del clasificador BCS sin bounding boxes."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--solo-validar", action="store_true")
    mode.add_argument("--prueba-rapida", action="store_true")
    mode.add_argument("--entrenar", action="store_true")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "resultados_sin_cajas")
    parser.add_argument("--run-name")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--seed", type=int, default=20261026)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=1)
    return parser


def run_quick_experiment(args) -> dict:
    run_name = args.run_name or datetime.now().strftime("prueba-rapida-%Y%m%d-%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    set_seed(args.seed)
    manifest_path = args.dataset_root / "_control" / "dataset_manifest.csv"
    loaders = build_data_loaders(
        manifest_path=manifest_path,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    device = select_device()
    model = build_model(num_classes=len(CLASS_VALUES), pretrained=True)
    checkpoint_path = run_dir / "best_model.pt"
    config = {
        "mode": "quick_test",
        "dataset_root": str(args.dataset_root),
        "manifest_path": str(manifest_path),
        "device": device.type if hasattr(device, "type") else str(device),
        "classes": list(CLASS_VALUES),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "epochs": args.epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1",
        "test_evaluated": False,
    }
    (run_dir / "config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_versions(run_dir)
    history = fit_model(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
    )
    history_path = run_dir / "history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    validation = evaluate_by_cow(model, loaders["validation"], device)
    write_evaluation_reports(run_dir, "validation", validation)

    return {
        "mode": "quick_test",
        "device": config["device"],
        "epochs": len(history),
        "test_evaluated": False,
        "output_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "validation_metrics": validation["metrics"],
    }


def run_full_experiment(args) -> dict:
    run_name = args.run_name or datetime.now().strftime("entrenamiento-%Y%m%d-%H%M%S")
    run_dir = args.output_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=False)

    set_seed(args.seed)
    manifest_path = args.dataset_root / "_control" / "dataset_manifest.csv"
    loaders = build_data_loaders(
        manifest_path=manifest_path,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    device = select_device()
    model = build_model(num_classes=len(CLASS_VALUES), pretrained=True)
    checkpoint_path = run_dir / "best_model.pt"
    config_path = run_dir / "config.json"
    config = {
        "mode": "full_training",
        "dataset_root": str(args.dataset_root),
        "manifest_path": str(manifest_path),
        "device": device.type if hasattr(device, "type") else str(device),
        "classes": list(CLASS_VALUES),
        "batch_size": args.batch_size,
        "image_size": args.image_size,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "epochs_requested": args.epochs,
        "learning_rate": args.learning_rate,
        "patience": args.patience,
        "pretrained_weights": "ResNet18_Weights.IMAGENET1K_V1",
        "selection_source": "validation_by_cow",
        "test_evaluated": False,
    }
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_versions(run_dir)

    history = fit_model(
        model=model,
        train_loader=loaders["train"],
        validation_loader=loaders["validation"],
        device=device,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        patience=args.patience,
        checkpoint_path=checkpoint_path,
    )
    with (run_dir / "history.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)

    validation = evaluate_by_cow(model, loaders["validation"], device)
    write_evaluation_reports(run_dir, "validation", validation)
    test = evaluate_by_cow(model, loaders["test"], device)
    write_evaluation_reports(run_dir, "test", test)

    config["epochs_completed"] = len(history)
    config["test_evaluated"] = True
    config_path.write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return {
        "mode": "full_training",
        "device": config["device"],
        "epochs_requested": args.epochs,
        "epochs_completed": len(history),
        "patience": args.patience,
        "test_evaluated": True,
        "output_dir": str(run_dir),
        "checkpoint": str(checkpoint_path),
        "validation_metrics": validation["metrics"],
        "test_metrics": test["metrics"],
    }


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.entrenar:
        summary = run_full_experiment(args)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    if args.prueba_rapida:
        args.epochs = 1
        summary = run_quick_experiment(args)
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return 0

    manifest_path = args.dataset_root / "_control" / "dataset_manifest.csv"
    loaders = build_data_loaders(
        manifest_path=manifest_path,
        dataset_root=args.dataset_root,
        batch_size=args.batch_size,
        image_size=args.image_size,
        seed=args.seed,
        num_workers=args.num_workers,
    )
    summary = {
        "mode": "validation_only",
        "device": select_device().type,
        "images": {
            "train": len(loaders["train"].dataset),
            "validation": len(loaders["validation"].dataset),
            "test": len(loaders["test"].dataset),
        },
        "batches": {
            "train": len(loaders["train"]),
            "validation": len(loaders["validation"]),
            "test": len(loaders["test"]),
        },
        "training_started": False,
    }
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
