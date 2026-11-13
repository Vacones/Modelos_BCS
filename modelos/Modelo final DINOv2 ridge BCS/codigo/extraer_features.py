"""Pilot: crop the annotated rump ('ass') box with 10% margin and extract frozen
features with horizontal-flip test-time averaging.

Usage:
  python extraer_features.py --data DIR_YOLO --meta metadata_bcs_validos.csv \
      --out FEATS_DIR --b0-weights efficientnet_b0_ra-3dd342df.pth \
      [--dino-s PATH_vits14.(pth|safetensors)] [--dino-b PATH_vitb14...]
Features are written to FEATS_DIR (keep it OUTSIDE shared folders: derived from private photos).
"""
import argparse, os, time, json
import numpy as np, pandas as pd, torch, timm
from PIL import Image

MARGEN = 0.10
SPLIT_DIR = {"train": "train", "val": "valid", "test": "test"}


def leer_cajas(ruta_txt, id_ass=0, id_cow=1):
    """Same rule as modeling/con_cajas_ordinal/bcs_ordinal.py: rump box inside the cow box."""
    cows, asses = [], []
    for ln in open(ruta_txt):
        p = ln.split()
        if len(p) < 5:
            continue
        c = int(float(p[0])); b = tuple(map(float, p[1:5])); conf = float(p[5]) if len(p) > 5 else 1.0
        area = b[2] * b[3]
        (cows if c == id_cow else asses if c == id_ass else []).append((conf, area, b))
    if not asses:
        return None
    if not cows:
        return max(asses, key=lambda t: (t[0], t[1]))[2]
    vx, vy, vw, vh = max(cows, key=lambda t: (t[0], t[1]))[2]
    dentro = [a for a in asses if abs(a[2][0] - vx) <= vw / 2 and abs(a[2][1] - vy) <= vh / 2]
    return max(dentro, key=lambda t: (t[0], t[1]))[2] if dentro else None


def recortar(img, b):
    W, H = img.size; cx, cy, w, h = b
    w2, h2 = w * (1 + 2 * MARGEN) / 2, h * (1 + 2 * MARGEN) / 2
    box = (max(0, (cx - w2) * W), max(0, (cy - h2) * H), min(W, (cx + w2) * W), min(H, (cy + h2) * H))
    return img.crop(tuple(int(round(v)) for v in box))


def a_tensor(img, size, mean, std):
    im = img.resize((size, size), Image.BICUBIC)
    x = torch.from_numpy(np.asarray(im, dtype=np.float32) / 255.).permute(2, 0, 1)
    return (x - torch.tensor(mean)[:, None, None]) / torch.tensor(std)[:, None, None]


def cargar_pesos(model, path):
    if path.endswith(".safetensors"):
        from safetensors.torch import load_file; sd = load_file(path)
    else:
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("model", sd) if isinstance(sd, dict) else sd
    sd = {k: v for k, v in sd.items() if not k.startswith(("classifier.", "head.", "fc."))}
    try:
        missing, unexpected = model.load_state_dict(sd, strict=False)
    except RuntimeError:  # e.g. pos_embed size mismatch from original FB checkpoints
        from timm.models.vision_transformer import checkpoint_filter_fn
        missing, unexpected = model.load_state_dict(checkpoint_filter_fn(sd, model), strict=False)
    assert len(missing) == 0, f"missing keys: {missing[:5]}"
    return model


def modelos(args):
    IMN = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    out = {}
    if args.b0_weights:
        m = cargar_pesos(timm.create_model("efficientnet_b0", pretrained=False, num_classes=0), args.b0_weights).eval()
        out["b0_224"] = (m, 224, IMN, "pool")
        out["b0_320"] = (m, 320, IMN, "pool")
    for key, name, path in (("dinoS", "vit_small_patch14_dinov2", args.dino_s), ("dinoB", "vit_base_patch14_dinov2", args.dino_b)):
        if path and os.path.exists(path):
            m = cargar_pesos(timm.create_model(name, pretrained=False, num_classes=0, img_size=224), path).eval()
            out[f"{key}_224"] = (m, 224, IMN, "cls+mean")
    return out


@torch.no_grad()
def feats(m, x, modo):
    if modo == "pool":
        return m(x)
    t = m.forward_features(x)  # (B, 1+N, D) for timm dinov2 (no reg tokens)
    return torch.cat([t[:, 0], t[:, m.num_prefix_tokens:].mean(1)], 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True); ap.add_argument("--meta", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--b0-weights"); ap.add_argument("--dino-s"); ap.add_argument("--dino-b")
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    meta = pd.read_csv(a.meta); meta = meta[meta.metadata_status == "ok"].reset_index(drop=True)
    crops, keep, sin_caja = [], [], 0
    for i, r in meta.iterrows():
        d = os.path.join(a.data, SPLIT_DIR[r.split])
        lab = os.path.join(d, "labels", os.path.splitext(r.file_name)[0] + ".txt")
        b = leer_cajas(lab) if os.path.exists(lab) else None
        if b is None:
            sin_caja += 1; continue
        crops.append(recortar(Image.open(os.path.join(d, "images", r.file_name)).convert("RGB"), b)); keep.append(i)
    meta = meta.loc[keep].reset_index(drop=True)
    meta[["image_id", "cow_id", "bcs"]].to_csv(os.path.join(a.out, "index.csv"), index=False)
    info = {"n_images": len(meta), "sin_caja": sin_caja, "timing_s_per_img": {}}
    torch.set_num_threads(os.cpu_count())
    for k, (m, size, (mu, sd), modo) in modelos(a).items():
        t0 = time.time(); F = []
        for j in range(0, len(crops), 16):
            x = torch.stack([a_tensor(c, size, mu, sd) for c in crops[j:j + 16]])
            F.append(((feats(m, x, modo) + feats(m, torch.flip(x, dims=[3]), modo)) / 2).numpy())
        F = np.concatenate(F); np.save(os.path.join(a.out, f"{k}.npy"), F)
        info["timing_s_per_img"][k] = (time.time() - t0) / len(crops)
        print(k, F.shape, f"{info['timing_s_per_img'][k]:.3f}s/img (with flip TTA)", flush=True)
    json.dump(info, open(os.path.join(a.out, "info.json"), "w"), indent=1); print(info)


if __name__ == "__main__":
    main()
