from __future__ import annotations

from collections import Counter

from PIL import Image, ImageOps
import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torchvision import transforms

from .datos import CLASS_VALUES, ImageRecord


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def letterbox(
    image: Image.Image,
    size: int,
    fill: tuple[int, int, int] = (0, 0, 0),
) -> Image.Image:
    contained = ImageOps.contain(
        image.convert("RGB"),
        (size, size),
        method=Image.Resampling.LANCZOS,
    )
    canvas = Image.new("RGB", (size, size), color=fill)
    left = (size - contained.width) // 2
    top = (size - contained.height) // 2
    canvas.paste(contained, (left, top))
    return canvas


def build_transforms(train: bool, image_size: int):
    steps = [transforms.Lambda(lambda image: letterbox(image, image_size))]
    if train:
        steps.extend(
            [
                transforms.RandomHorizontalFlip(p=0.5),
                transforms.ColorJitter(
                    brightness=0.15,
                    contrast=0.15,
                    saturation=0.10,
                    hue=0.02,
                ),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return transforms.Compose(steps)


class CowImageDataset(Dataset):
    def __init__(self, records: list[ImageRecord], transform) -> None:
        self.records = records
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> dict:
        record = self.records[index]
        with Image.open(record.ruta_imagen) as source:
            image = source.convert("RGB")

        return {
            "image": self.transform(image),
            "class_index": CLASS_VALUES.index(record.bcs),
            "bcs": record.bcs,
            "vaca_id": record.vaca_id,
            "imagen_id": record.imagen_id,
        }


def build_train_sampler(
    records: list[ImageRecord],
    seed: int,
) -> WeightedRandomSampler:
    images_per_cow = Counter(record.vaca_id for record in records)
    weights = [1.0 / images_per_cow[record.vaca_id] for record in records]
    generator = torch.Generator().manual_seed(seed)
    return WeightedRandomSampler(
        weights=weights,
        num_samples=len(records),
        replacement=True,
        generator=generator,
    )
