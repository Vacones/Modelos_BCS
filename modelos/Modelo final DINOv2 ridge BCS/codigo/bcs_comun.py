"""Piezas comunes del modelo final de BCS: elección de la grupa, recorte y características de DINOv2-S."""
import os
import numpy as np, torch, timm
from PIL import Image

MARGEN = 0.10
SIZE = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
ID_ASS, ID_COW = 0, 1  # índices de clase en el export de Roboflow "Cajas v2": ['ass', 'cow']


def elegir_grupa(cajas):
    """Elige la caja de grupa que se usa en una foto.

    cajas: lista de (clase, confianza, (cx, cy, w, h)) en coordenadas normalizadas.
    Devuelve (confianza, caja) de la grupa o None.
    Regla: se toma la vaca de mayor confianza y, dentro de ella, la grupa de mayor confianza cuyo centro
    queda dentro de la caja de la vaca. Si no hay caja de vaca, se toma la grupa de mayor confianza.
    """
    cows = [(c, b) for k, c, b in cajas if k == ID_COW]
    asses = [(c, b) for k, c, b in cajas if k == ID_ASS]
    if not asses:
        return None
    if not cows:
        return max(asses, key=lambda t: (t[0], t[1][2] * t[1][3]))
    _, (vx, vy, vw, vh) = max(cows, key=lambda t: (t[0], t[1][2] * t[1][3]))
    dentro = [a for a in asses if abs(a[1][0] - vx) <= vw / 2 and abs(a[1][1] - vy) <= vh / 2]
    return max(dentro, key=lambda t: (t[0], t[1][2] * t[1][3])) if dentro else None


def leer_txt_yolo(ruta):
    """Lee un txt en formato YOLO. La sexta columna, si existe, es la confianza."""
    out = []
    for ln in open(ruta):
        p = ln.split()
        if len(p) >= 5:
            out.append((int(float(p[0])), float(p[5]) if len(p) > 5 else 1.0, tuple(map(float, p[1:5]))))
    return out


def recortar(img, b):
    """Recorta la caja b con un margen de 10 % por lado, sin salirse de la foto."""
    W, H = img.size; cx, cy, w, h = b
    w2, h2 = w * (1 + 2 * MARGEN) / 2, h * (1 + 2 * MARGEN) / 2
    box = (max(0, (cx - w2) * W), max(0, (cy - h2) * H), min(W, (cx + w2) * W), min(H, (cy + h2) * H))
    return img.crop(tuple(int(round(v)) for v in box))


def a_tensor(img):
    im = img.resize((SIZE, SIZE), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1)
    return (x - torch.tensor(MEAN)[:, None, None]) / torch.tensor(STD)[:, None, None]


def cargar_dinov2(ruta_pesos):
    """Carga DINOv2 ViT-S/14 de timm con los pesos oficiales en formato safetensors, sin la capa de salida."""
    from safetensors.torch import load_file
    from timm.models.vision_transformer import checkpoint_filter_fn
    m = timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=0, img_size=SIZE)
    sd = {k: v for k, v in load_file(ruta_pesos).items() if not k.startswith("head.")}
    missing, _ = m.load_state_dict(checkpoint_filter_fn(sd, m), strict=False)
    assert not missing, f"missing keys: {missing[:5]}"
    return m.eval()


@torch.no_grad()
def features(model, crops, bs=16):
    """Token CLS más la media de los tokens de parche, promediado con la imagen espejada. Devuelve (n, 768)."""
    def f(x):
        t = model.forward_features(x)
        return torch.cat([t[:, 0], t[:, model.num_prefix_tokens:].mean(1)], 1)
    F = []
    for j in range(0, len(crops), bs):
        x = torch.stack([a_tensor(c) for c in crops[j:j + bs]])
        F.append(((f(x) + f(torch.flip(x, dims=[3]))) / 2).numpy())
    return np.concatenate(F)
