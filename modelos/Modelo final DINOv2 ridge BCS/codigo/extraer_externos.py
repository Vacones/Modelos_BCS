"""Sample Huang (ScienceDB) images, crop the annotated box (+10% margin) and extract
DINOv2-S features exactly as extraer_features.py; also re-extract CAETEC DINOv2-S.

Usage: python extraer_externos.py HUANG_DIR CAETEC_YOLO_DIR META_CSV WEIGHTS OUT_DIR
Outputs (OUT_DIR, delete after use: derived from private / unlicensed photos):
  huang_index.csv, huang_dinoS.npy, caetec_index.csv, caetec_dinoS.npy, info.json
"""
import sys, os, re, glob, time, json, hashlib, random
import xml.etree.ElementTree as ET
from collections import defaultdict
import numpy as np, pandas as pd, torch, timm
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from extraer_features import leer_cajas, recortar, a_tensor, cargar_pesos, feats, SPLIT_DIR  # noqa: E402

IMN = ((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
PER_VIEW_CLASS = 150   # target images per (view, BCS level)
PER_ID = 2             # max images per identifiable cow / L-R index
LEVELS = ["3.25", "3.5", "3.75", "4.0", "4.25"]


def view_and_group(stem, lev):
    m = re.match(r"(GS|YM)_(\d+)_(\d+)$", stem)
    if m:  # id can recur in another class (re-scored cow or reused id): keep (id, class) as group,
        # but tie groups across classes by id so CV never splits the same id
        return m.group(1), f"{m.group(1)}_{m.group(2)}"
    m = re.match(r"([LR])-i(\d+)$", stem)
    if m:  # L-i<n> and R-i<n> share the index -> probably same cow/pass
        return m.group(1) + "-i", f"LR_{m.group(2)}"
    return None, None


def caja_huang(xml_path):
    r = ET.parse(xml_path).getroot()
    W, H = int(r.find("size/width").text), int(r.find("size/height").text)
    best = None
    for o in r.findall("object"):
        b = o.find("bndbox"); x0, y0, x1, y1 = (float(b.find(k).text) for k in ("xmin", "ymin", "xmax", "ymax"))
        a = (x1 - x0) * (y1 - y0)
        if best is None or a > best[0]:
            best = (a, ((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H))
    return best[1] if best else None


def muestrear_huang(root, seed=0):
    rng = random.Random(seed)
    pool = defaultdict(lambda: defaultdict(list))  # (view, lev) -> group -> [paths]
    for lev in LEVELS:
        for p in glob.glob(os.path.join(root, lev, "*.jpg")):
            v, g = view_and_group(os.path.splitext(os.path.basename(p))[0], lev)
            if v:
                pool[(v, lev)][g].append(p)
    rows = []
    for (v, lev), groups in sorted(pool.items()):
        gs = sorted(groups); rng.shuffle(gs); take = []
        for g in gs:  # round 1: one per group; round 2: second image per group
            take.append((g, rng.choice(groups[g])))
            if len(take) >= PER_VIEW_CLASS:
                break
        if len(take) < PER_VIEW_CLASS:
            used = {p for _, p in take}
            for g in gs:
                extra = [p for p in groups[g] if p not in used][: PER_ID - 1]
                take += [(g, p) for p in extra]
                if len(take) >= PER_VIEW_CLASS:
                    break
        rows += [{"path": p, "view": v, "group": g, "bcs": float(lev)} for g, p in take[:PER_VIEW_CLASS]]
    df = pd.DataFrame(rows)
    df["md5"] = [hashlib.md5(open(p, "rb").read()).hexdigest() for p in df.path]
    return df.drop_duplicates("md5").reset_index(drop=True)


def extraer(model, crops, bs=16):
    F = []
    for j in range(0, len(crops), bs):
        x = torch.stack([a_tensor(c, 224, *IMN) for c in crops[j:j + bs]])
        F.append(((feats(model, x, "cls+mean") + feats(model, torch.flip(x, dims=[3]), "cls+mean")) / 2).numpy())
    return np.concatenate(F)


def main():
    huang, caetec, meta_csv, w, out = sys.argv[1:6]
    os.makedirs(out, exist_ok=True); torch.set_num_threads(os.cpu_count())
    model = cargar_pesos(timm.create_model("vit_small_patch14_dinov2", pretrained=False, num_classes=0, img_size=224), w).eval()
    info = {}
    # --- CAETEC (same rule as extraer_features.py) ---
    meta = pd.read_csv(meta_csv); meta = meta[meta.metadata_status == "ok"].reset_index(drop=True)
    crops, keep = [], []
    for i, r in meta.iterrows():
        d = os.path.join(caetec, SPLIT_DIR[r.split])
        lab = os.path.join(d, "labels", os.path.splitext(r.file_name)[0] + ".txt")
        b = leer_cajas(lab) if os.path.exists(lab) else None
        if b is not None:
            crops.append(recortar(Image.open(os.path.join(d, "images", r.file_name)).convert("RGB"), b)); keep.append(i)
    meta = meta.loc[keep].reset_index(drop=True)
    t0 = time.time(); F = extraer(model, crops); info["caetec_s_per_img"] = (time.time() - t0) / len(crops)
    np.save(os.path.join(out, "caetec_dinoS.npy"), F); meta[["image_id", "cow_id", "bcs"]].to_csv(os.path.join(out, "caetec_index.csv"), index=False)
    print("caetec", F.shape, f"{info['caetec_s_per_img']:.3f}s/img", flush=True)
    # --- Huang ---
    df = muestrear_huang(huang)
    crops, ok, rel = [], [], []
    for i, r in df.iterrows():
        b = caja_huang(os.path.splitext(r.path)[0] + ".xml")
        if b is None:
            continue
        crops.append(recortar(Image.open(r.path).convert("RGB"), b)); ok.append(i); rel.append(b[2] * b[3])
    df = df.loc[ok].reset_index(drop=True); df["box_area_frac"] = rel
    t0 = time.time(); F = extraer(model, crops); info["huang_s_per_img"] = (time.time() - t0) / len(crops)
    np.save(os.path.join(out, "huang_dinoS.npy"), F)
    df.drop(columns=["path", "md5"]).to_csv(os.path.join(out, "huang_index.csv"), index=False)
    info["huang_counts"] = df.groupby(["view", "bcs"]).size().unstack().to_dict()
    info["huang_groups_per_view"] = df.groupby("view").group.nunique().to_dict()
    info["huang_box_area_frac_median"] = df.groupby("view").box_area_frac.median().to_dict()
    print("huang", F.shape, f"{info['huang_s_per_img']:.3f}s/img", flush=True)
    json.dump(info, open(os.path.join(out, "info.json"), "w"), indent=1); print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
