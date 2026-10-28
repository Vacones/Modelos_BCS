from PIL import Image
import pytest
import torch

from modelo_sin_cajas.datos import ImageRecord


def test_letterbox_preserves_full_wide_image_with_padding():
    try:
        from modelo_sin_cajas.imagenes import letterbox
    except ModuleNotFoundError as error:
        pytest.fail(f"Falta implementar el preprocesamiento sin recorte: {error}")

    original = Image.new("RGB", (400, 200), color=(255, 0, 0))

    result = letterbox(original, size=224)

    assert result.size == (224, 224)
    assert result.getpixel((112, 112)) == (255, 0, 0)
    assert result.getpixel((112, 0)) == (0, 0, 0)
    assert result.getpixel((112, 223)) == (0, 0, 0)


def test_evaluation_transform_returns_resnet_tensor_shape():
    try:
        from modelo_sin_cajas.imagenes import build_transforms
    except ImportError as error:
        pytest.fail(f"Falta implementar la transformación del modelo: {error}")

    original = Image.new("RGB", (400, 200), color=(127, 127, 127))

    tensor = build_transforms(train=False, image_size=224)(original)

    assert tuple(tensor.shape) == (3, 224, 224)
    assert tensor.dtype.is_floating_point


def test_dataset_returns_tensor_class_and_cow_identity(tmp_path):
    try:
        from modelo_sin_cajas.imagenes import CowImageDataset, build_transforms
    except ImportError as error:
        pytest.fail(f"Falta implementar el dataset de imágenes: {error}")

    image_path = tmp_path / "vaca.jpg"
    Image.new("RGB", (400, 200), color=(127, 127, 127)).save(image_path)
    record = ImageRecord("img-1", "100", 3.0, "ENTRENAMIENTO", image_path)
    dataset = CowImageDataset(
        [record],
        transform=build_transforms(train=False, image_size=224),
    )

    sample = dataset[0]

    assert tuple(sample["image"].shape) == (3, 224, 224)
    assert sample["class_index"] == 3
    assert sample["bcs"] == 3.0
    assert sample["vaca_id"] == "100"
    assert sample["imagen_id"] == "img-1"


def test_train_sampler_gives_equal_total_weight_to_each_cow(tmp_path):
    try:
        from modelo_sin_cajas.imagenes import build_train_sampler
    except ImportError as error:
        pytest.fail(f"Falta implementar el muestreo por vaca: {error}")

    image = tmp_path / "imagen.jpg"
    records = [
        ImageRecord("img-1", "100", 3.0, "ENTRENAMIENTO", image),
        ImageRecord("img-2", "200", 3.0, "ENTRENAMIENTO", image),
        ImageRecord("img-3", "200", 3.0, "ENTRENAMIENTO", image),
    ]

    sampler = build_train_sampler(records, seed=20261026)
    weights = sampler.weights.to(dtype=torch.float64)

    assert weights[0].item() == pytest.approx(1.0)
    assert weights[1].item() == pytest.approx(0.5)
    assert weights[2].item() == pytest.approx(0.5)
    assert weights[0].item() == pytest.approx(weights[1:].sum().item())


def test_training_transform_adds_variation_without_changing_shape():
    from modelo_sin_cajas.imagenes import build_transforms

    original = Image.new("RGB", (400, 200), color=(255, 0, 0))
    for x in range(200, 400):
        for y in range(200):
            original.putpixel((x, y), (0, 0, 255))
    transform = build_transforms(train=True, image_size=224)

    torch.manual_seed(1)
    first = transform(original)
    torch.manual_seed(2)
    second = transform(original)

    assert tuple(first.shape) == (3, 224, 224)
    assert tuple(second.shape) == (3, 224, 224)
    assert not torch.equal(first, second)
