from __future__ import annotations

import csv
from importlib.metadata import version
import json
import platform
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .datos import CLASS_VALUES


def write_versions(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    versions = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": version("torch"),
        "torchvision": version("torchvision"),
        "numpy": version("numpy"),
        "pillow": version("pillow"),
        "matplotlib": version("matplotlib"),
    }
    (output_dir / "versions.json").write_text(
        json.dumps(versions, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_evaluation_reports(
    output_dir: Path,
    split_name: str,
    evaluation: dict,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_path = output_dir / f"metrics_{split_name}.json"
    metrics_path.write_text(
        json.dumps(evaluation["metrics"], indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    probability_columns = [f"prob_bcs_{value:g}" for value in CLASS_VALUES]
    prediction_columns = [
        "vaca_id",
        "bcs_real",
        "bcs_predicho",
        "fotos",
        *probability_columns,
    ]
    predictions_path = output_dir / f"predicciones_por_vaca_{split_name}.csv"
    with predictions_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=prediction_columns)
        writer.writeheader()
        for cow_id, prediction in sorted(evaluation["cow_predictions"].items()):
            row = {
                "vaca_id": cow_id,
                "bcs_real": prediction.true_bcs,
                "bcs_predicho": prediction.predicted_bcs,
                "fotos": prediction.photo_count,
            }
            row.update(
                {
                    column: float(probability)
                    for column, probability in zip(
                        probability_columns,
                        prediction.probabilities,
                    )
                }
            )
            writer.writerow(row)

    if "photo_predictions" in evaluation:
        photo_columns = [
            "imagen_id",
            "vaca_id",
            "bcs_real",
            "bcs_predicho",
            *probability_columns,
        ]
        photo_path = output_dir / f"predicciones_por_imagen_{split_name}.csv"
        photo_predictions = evaluation["photo_predictions"]
        with photo_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=photo_columns)
            writer.writeheader()
            for image_id, cow_id, true_bcs, probabilities in zip(
                photo_predictions["image_ids"],
                photo_predictions["cow_ids"],
                photo_predictions["true_bcs"],
                photo_predictions["probabilities"],
            ):
                predicted_index = max(
                    range(len(probabilities)),
                    key=lambda index: probabilities[index],
                )
                row = {
                    "imagen_id": image_id,
                    "vaca_id": cow_id,
                    "bcs_real": true_bcs,
                    "bcs_predicho": CLASS_VALUES[predicted_index],
                }
                row.update(
                    {
                        column: float(probability)
                        for column, probability in zip(
                            probability_columns,
                            probabilities,
                        )
                    }
                )
                writer.writerow(row)

    matrix_path = output_dir / f"confusion_matrix_{split_name}.csv"
    with matrix_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        writer.writerow(["bcs_real", *CLASS_VALUES])
        for bcs, row in zip(CLASS_VALUES, evaluation["confusion_matrix"]):
            writer.writerow([bcs, *(int(value) for value in row)])

    matrix = evaluation["confusion_matrix"]
    figure, axis = plt.subplots(figsize=(7, 7), dpi=120)
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_title(f"Matriz de confusión por vaca: {split_name}")
    axis.set_xlabel("BCS predicho")
    axis.set_ylabel("BCS real")
    axis.set_xticks(range(len(CLASS_VALUES)), labels=[f"{value:g}" for value in CLASS_VALUES])
    axis.set_yticks(range(len(CLASS_VALUES)), labels=[f"{value:g}" for value in CLASS_VALUES])
    threshold = matrix.max() / 2 if matrix.max() else 0
    for row_index in range(len(CLASS_VALUES)):
        for column_index in range(len(CLASS_VALUES)):
            value = int(matrix[row_index, column_index])
            axis.text(
                column_index,
                row_index,
                str(value),
                ha="center",
                va="center",
                color="white" if value > threshold else "black",
            )
    figure.colorbar(image, ax=axis, fraction=0.046, pad=0.04)
    figure.tight_layout()
    figure.savefig(output_dir / f"confusion_matrix_{split_name}.png")
    plt.close(figure)
