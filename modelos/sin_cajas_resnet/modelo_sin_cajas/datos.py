from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv


CLASS_VALUES = (2.25, 2.5, 2.75, 3.0, 3.25, 3.5, 3.75)
SPLIT_VALUES = ("ENTRENAMIENTO", "VALIDACION", "PRUEBA")


@dataclass(frozen=True)
class ImageRecord:
    imagen_id: str
    vaca_id: str
    bcs: float
    conjunto: str
    ruta_imagen: Path


def validate_records(records: list[ImageRecord]) -> None:
    splits_by_cow: dict[str, set[str]] = {}
    bcs_by_cow: dict[str, set[float]] = {}

    for record in records:
        if record.bcs not in CLASS_VALUES:
            raise ValueError(f"El BCS no válido '{record.bcs}' aparece en {record.imagen_id}")
        if record.conjunto not in SPLIT_VALUES:
            raise ValueError(f"El conjunto no válido '{record.conjunto}' aparece en {record.imagen_id}")
        if not record.ruta_imagen.is_file():
            raise FileNotFoundError(f"La imagen no existe: {record.ruta_imagen}")
        splits_by_cow.setdefault(record.vaca_id, set()).add(record.conjunto)
        bcs_by_cow.setdefault(record.vaca_id, set()).add(record.bcs)

    leaking_cows = [
        cow_id
        for cow_id, splits in splits_by_cow.items()
        if len(splits) > 1
    ]
    if leaking_cows:
        cows = ", ".join(sorted(leaking_cows))
        raise ValueError(f"Cada vaca debe pertenecer a un solo conjunto; vacas en más de un conjunto: {cows}")

    conflicting_cows = [
        cow_id
        for cow_id, values in bcs_by_cow.items()
        if len(values) > 1
    ]
    if conflicting_cows:
        cows = ", ".join(sorted(conflicting_cows))
        raise ValueError(f"Cada vaca debe tener un solo valor BCS; vacas con más de un BCS: {cows}")


def load_records(manifest_path: Path, dataset_root: Path) -> list[ImageRecord]:
    records: list[ImageRecord] = []

    with manifest_path.open(newline="", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            records.append(
                ImageRecord(
                    imagen_id=row["imagen_id"],
                    vaca_id=row["vaca_id"],
                    bcs=float(row["bcs"]),
                    conjunto=row["conjunto"],
                    ruta_imagen=dataset_root / row["ruta_dataset"],
                )
            )

    return records
