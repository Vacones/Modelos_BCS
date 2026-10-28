from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .datos import load_records, validate_records
from .imagenes import CowImageDataset, build_train_sampler, build_transforms
from .metricas import aggregate_by_cow, classification_metrics, confusion_matrix


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_one_epoch(
    model: nn.Module,
    loader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    loss_sum = 0.0
    sample_count = 0

    for batch in loader:
        images = batch["image"].to(device)
        targets = batch["class_index"].to(device)

        optimizer.zero_grad(set_to_none=True)
        scores = model(images)
        loss = criterion(scores, targets)
        loss.backward()
        optimizer.step()

        batch_size = images.shape[0]
        loss_sum += loss.item() * batch_size
        sample_count += batch_size

    return loss_sum / sample_count


def predict_loader(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict:
    model.eval()
    probability_batches = []
    cow_ids: list[str] = []
    image_ids: list[str] = []
    true_bcs: list[float] = []

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device)
            scores = model(images)
            probabilities = torch.softmax(scores, dim=1)
            probability_batches.append(probabilities.cpu().numpy())
            cow_ids.extend(batch["vaca_id"])
            image_ids.extend(batch["imagen_id"])
            values = batch["bcs"].tolist() if hasattr(batch["bcs"], "tolist") else batch["bcs"]
            true_bcs.extend(float(value) for value in values)

    return {
        "probabilities": np.concatenate(probability_batches, axis=0),
        "cow_ids": cow_ids,
        "image_ids": image_ids,
        "true_bcs": true_bcs,
    }


def evaluate_by_cow(
    model: nn.Module,
    loader,
    device: torch.device,
) -> dict:
    photo_predictions = predict_loader(model, loader, device)
    cow_predictions = aggregate_by_cow(
        probabilities=photo_predictions["probabilities"],
        cow_ids=photo_predictions["cow_ids"],
        true_bcs=photo_predictions["true_bcs"],
    )
    return {
        "photo_predictions": photo_predictions,
        "cow_predictions": cow_predictions,
        "metrics": classification_metrics(cow_predictions),
        "confusion_matrix": confusion_matrix(cow_predictions),
    }


def build_data_loaders(
    manifest_path,
    dataset_root,
    batch_size: int,
    image_size: int,
    seed: int,
    num_workers: int = 0,
) -> dict[str, DataLoader]:
    records = load_records(manifest_path, dataset_root)
    validate_records(records)

    train_records = [record for record in records if record.conjunto == "ENTRENAMIENTO"]
    validation_records = [record for record in records if record.conjunto == "VALIDACION"]
    test_records = [record for record in records if record.conjunto == "PRUEBA"]

    train_dataset = CowImageDataset(
        train_records,
        transform=build_transforms(train=True, image_size=image_size),
    )
    validation_dataset = CowImageDataset(
        validation_records,
        transform=build_transforms(train=False, image_size=image_size),
    )
    test_dataset = CowImageDataset(
        test_records,
        transform=build_transforms(train=False, image_size=image_size),
    )

    common = {
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    return {
        "train": DataLoader(
            train_dataset,
            sampler=build_train_sampler(train_records, seed=seed),
            **common,
        ),
        "validation": DataLoader(validation_dataset, shuffle=False, **common),
        "test": DataLoader(test_dataset, shuffle=False, **common),
    }


def fit_model(
    model: nn.Module,
    train_loader,
    validation_loader,
    device: torch.device,
    epochs: int,
    learning_rate: float,
    patience: int,
    checkpoint_path: Path,
) -> list[dict]:
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss()
    history: list[dict] = []
    best_selection: tuple[float, float] | None = None
    epochs_without_improvement = 0
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            criterion=criterion,
            device=device,
        )
        validation = evaluate_by_cow(model, validation_loader, device)
        metrics = validation["metrics"]
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                **{f"validation_{key}": value for key, value in metrics.items()},
            }
        )

        selection = (
            float(metrics["mean_absolute_error"]),
            -float(metrics["exact_accuracy"]),
        )
        checkpoint_status = "sin mejora"
        should_stop = False
        if best_selection is None or selection < best_selection:
            best_selection = selection
            epochs_without_improvement = 0
            checkpoint_status = "nuevo mejor guardado"
            cpu_state = {
                name: value.detach().cpu()
                for name, value in model.state_dict().items()
            }
            torch.save(
                {
                    "epoch": epoch,
                    "selection_source": "validation_by_cow",
                    "validation_metrics": metrics,
                    "model_state_dict": cpu_state,
                },
                checkpoint_path,
            )
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                should_stop = True

        print(
            f"Época {epoch}/{epochs} | "
            f"pérdida {train_loss:.4f} | "
            f"MAE validación {metrics['mean_absolute_error']:.4f} | "
            f"exactitud {metrics['exact_accuracy']:.1%} | "
            f"{checkpoint_status}",
            file=sys.stderr,
            flush=True,
        )
        if should_stop:
            print(
                f"Parada temprana: {patience} épocas consecutivas sin mejora.",
                file=sys.stderr,
                flush=True,
            )
            break

    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    return history
