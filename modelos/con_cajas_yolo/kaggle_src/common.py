"""Utilidades sin importar frameworks GPU."""
import csv
import hashlib
import json
import math
import os
import shutil
import time
from pathlib import Path

SCHEMA = 1


def require(value, message):
    if not value:
        raise RuntimeError(message)


def load(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + '.tmp')
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')
    os.replace(temp, path)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def write_csv(path, records, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = fields or list(records[0])
    temp = path.with_name(path.name + '.tmp')
    with temp.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    temp.replace(path)


def copy_atomic(source, target):
    target = Path(target)
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + '.tmp')
    shutil.copy2(source, temp)
    temp.replace(target)


def artifacts(cfg):
    return Path(cfg['WORK_ROOT']) / 'artifacts'


def stage(cfg, name, fingerprint, outputs, details=None):
    root = artifacts(cfg)
    save(root / 'state' / (name + '.json'), {
        'schema': SCHEMA, 'status': 'complete', 'fingerprint': fingerprint,
        'outputs': {str(Path(p).relative_to(root)): sha(p) for p in outputs},
        'details': details or {},
    })


def reusable(cfg, name, fingerprint):
    root = artifacts(cfg)
    path = root / 'state' / (name + '.json')
    if not path.exists():
        return False
    state = load(path)
    require(state['fingerprint'] == fingerprint,
            f'Etapa {name} incompatible: cambió código, datos o configuración. Usa otro WORK_ROOT/EXPERIMENT_ID.')
    require(all((root / p).is_file() and sha(root / p) == h for p, h in state['outputs'].items()),
            f'Archivos incompletos o alterados en {name}. Recupera un export íntegro.')
    print('Reutilizar:', name)
    return True


def contract(cfg):
    root = artifacts(cfg)
    manifest = root / 'manifests' / 'manifest.json'
    split = root / 'manifests' / 'splits.json'
    require(manifest.is_file() and split.is_file(), 'Ejecuta auditoría y preparación primero.')
    m = load(manifest)
    s = load(split)
    require(s['dataset_hash'] == m['dataset_hash'], 'Particiones de otra versión de datos.')
    return {'schema': SCHEMA, 'dataset_hash': m['dataset_hash'], 'split_hash': s['split_hash'],
            'code_hash': cfg['CODE_HASH']}


def rows(cfg, split=None, supervised=False):
    data = load(artifacts(cfg) / 'manifests' / 'manifest.json')['rows']
    return [r for r in data if (split is None or r['split'] == split)
            and (not supervised or r['bcs_status'] == 'ok')]


def image_path(cfg, row):
    return Path(cfg['DATA_ROOT']) / row['image']


def crop(image, xyxy, size):
    from PIL import Image, ImageOps
    x1, y1, x2, y2 = xyxy
    box = (max(0, math.floor(x1)), max(0, math.floor(y1)),
           min(image.width, math.ceil(x2)), min(image.height, math.ceil(y2)))
    require(box[2] > box[0] and box[3] > box[1], 'Recorte vacío')
    return ImageOps.pad(image.convert('RGB').crop(box), (size, size),
                        method=Image.Resampling.BILINEAR, color=(0, 0, 0))


def pair_boxes(boxes, coverage=.8):
    pairs = []
    for cow in [b for b in boxes if b['class'] == 0]:
        a, b, c, d = cow['xyxy']
        for rump in [b for b in boxes if b['class'] == 1]:
            x, y, z, w = rump['xyxy']
            area = (z-x)*(w-y)
            inside = max(0, min(c, z)-max(a, x))*max(0, min(d, w)-max(b, y))
            if area > 0 and a <= (x+z)/2 <= c and b <= (y+w)/2 <= d and inside/area >= coverage:
                pairs.append((cow, rump))
    return ('ok', pairs[0]) if len(pairs) == 1 else (
        'sin_par_valido' if not pairs else 'asociacion_ambigua', None)


def remaining(cfg):
    deadline = cfg.get('DEADLINE_UNIX')
    return None if deadline is None else deadline - time.time()


def can_start(cfg, seconds):
    left = remaining(cfg)
    return left is None or left > max(0, seconds) * 1.25


def restore(cfg):
    source = cfg.get('RESUME_ROOT')
    if not source:
        return
    source = Path(source)
    candidates = [source] if (source / 'state').is_dir() else list(source.rglob('artifacts'))
    candidates = [p for p in candidates if (p / 'state').is_dir() and (p / 'configs').is_dir()]
    require(len(candidates) == 1, 'RESUME_ROOT debe contener exactamente una carpeta artifacts.')
    source = candidates[0]
    inventory = load(source / 'inventory.json')
    require(bool(inventory), 'Export sin inventario de integridad.')
    require(all(not Path(p).is_absolute() and '..' not in Path(p).parts for p in inventory),
            'Ruta insegura en inventario.')
    require(all((source / p).is_file() and sha(source / p) == h for p, h in inventory.items()),
            'Export previo incompleto o con hashes incorrectos.')
    marker = Path(cfg['WORK_ROOT']) / 'resume_import.json'
    if marker.exists():
        require(load(marker)['inventory_hash'] == digest(inventory),
                'Ya se importó otro export en este WORK_ROOT; usa una carpeta nueva.')
        print('Este export ya se importó; conservar el progreso actual de working.')
        return
    target = artifacts(cfg)
    for p, h in inventory.items():
        dest = target / p
        if dest.exists():
            require(sha(dest) == h, f'Conflicto al importar {p}; usa un WORK_ROOT vacío.')
        else:
            copy_atomic(source / p, dest)
    save(marker, {'inventory_hash': digest(inventory)})
    print('Artefactos verificados y copiados a working.')
