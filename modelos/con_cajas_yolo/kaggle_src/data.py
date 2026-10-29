"""COCO directo: auditoría, particiones originales y exportaciones reproducibles."""
import collections
import json
import math
import re
from pathlib import Path
from PIL import Image
from common import (artifacts, contract, crop, digest, image_path, load, require,
                    reusable, rows, save, sha, stage, write_csv)


def discover(input_root):
    root = Path(input_root)
    return sorted({p.parent.parent for p in root.rglob('_annotations.coco.json')
                   if p.parent.name == 'train'
                   and (p.parent.parent / 'valid/_annotations.coco.json').is_file()
                   and (p.parent.parent / 'test/_annotations.coco.json').is_file()})


def inspect(cfg):
    source = Path(cfg['DATA_ROOT'])
    for split in ('train', 'valid', 'test'):
        d = load(source / split / '_annotations.coco.json')
        print(split, 'imágenes:', len(d['images']), 'cajas:', len(d['annotations']),
              'categorías:', [(c['id'], c['name']) for c in d['categories']])


def audit(cfg):
    root = artifacts(cfg)
    source = Path(cfg['DATA_ROOT'])
    records, corrections, excluded = [], [], []
    cow_splits, hash_splits = collections.defaultdict(set), collections.defaultdict(set)
    for original, split in [('train', 'train'), ('valid', 'val'), ('test', 'test')]:
        d = load(source / original / '_annotations.coco.json')
        cats = {c['id']: c['name'] for c in d['categories']}
        require(len(cats) == len(d['categories']), 'IDs de categoría duplicados.')
        require({'cow', 'ass'} <= set(cats.values()), 'Faltan categorías cow/ass.')
        images = {im['id']: im for im in d['images']}
        require(len(images) == len(d['images']), 'IDs de imagen duplicados.')
        annotations = collections.defaultdict(list)
        seen_annotations = set()
        for ann in d['annotations']:
            require(ann['id'] not in seen_annotations, 'ID de anotación duplicado.')
            seen_annotations.add(ann['id'])
            require(ann['image_id'] in images and ann['category_id'] in cats, 'Referencia COCO inválida.')
            require(cats[ann['category_id']] in ('cow', 'ass'), 'Categoría anotada desconocida.')
            require(ann.get('iscrowd', 0) == 0, 'iscrowd requiere revisión manual.')
            annotations[ann['image_id']].append(ann)
        seen_names = set()
        for im in d['images']:
            name = im['file_name']
            require(Path(name).name == name and '\\' not in name, 'Nombre de archivo inseguro.')
            require(name not in seen_names, 'Archivo repetido en COCO.')
            seen_names.add(name)
            path = source / original / name
            require(path.is_file(), f'Falta foto: {path}')
            with Image.open(path) as picture:
                require(picture.size == (im['width'], im['height']), 'Dimensiones COCO incorrectas.')
                picture.verify()
            with Image.open(path) as picture:
                picture.convert('RGB').load()
            tags = im.get('extra', {}).get('user_tags', [])
            ids = sorted({str(t) for t in tags if re.fullmatch(r'\d+', str(t))})
            btags = sorted({str(t) for t in tags if str(t).startswith('BCS_')})
            require(ids, 'Falta identidad de vaca: no se puede verificar la partición.')
            for cow in ids:
                cow_splits[cow].add(split)
            bcs, status = None, 'ambiguo_o_ausente'
            if len(ids) == len(btags) == 1:
                match = re.fullmatch(r'BCS_(\d+)(?:_(\d+))?', btags[0])
                if match:
                    bcs = float(match[1] + ('.' + match[2] if match[2] else ''))
                    require(1 <= bcs <= 5, 'BCS fuera de 1..5.')
                    status = 'ok'
            boxes = []
            for ann in annotations[im['id']]:
                x, y, w, h = ann['bbox']
                require(all(math.isfinite(v) for v in (x, y, w, h)) and w > 0 and h > 0,
                        'Caja COCO no finita o vacía.')
                excess = max(0, -x, -y, x+w-im['width'], y+h-im['height'])
                require(excess <= cfg['BOX_ROUNDING_TOLERANCE_PX'] + 1e-9,
                        f'Caja fuera de imagen: {original}/{name}, exceso={excess}')
                xyxy = [max(0, x), max(0, y), min(im['width'], x+w), min(im['height'], y+h)]
                require(xyxy[2] > xyxy[0] and xyxy[3] > xyxy[1], 'Caja vacía al limitar.')
                if excess > 1e-9:
                    corrections.append({'key': f'{original}/{name}', 'annotation_id': ann['id'],
                                        'original_xywh': ann['bbox'], 'corrected_xyxy': xyxy})
                boxes.append({'class': {'cow': 0, 'ass': 1}[cats[ann['category_id']]], 'xyxy': xyxy})
            require(collections.Counter(b['class'] for b in boxes) == {0: 1, 1: 1},
                    'Se espera una caja cow y una ass por foto; revisa el COCO.')
            h = sha(path)
            hash_splits[h].add(split)
            row = {'key': f'{original}/{name}', 'image': f'{original}/{name}', 'split': split,
                   'image_id': im['id'], 'cow_ids': ids, 'cow_id': ids[0] if len(ids) == 1 else None,
                   'bcs': bcs, 'bcs_status': status, 'tags': tags, 'sha256': h,
                   'width': im['width'], 'height': im['height'], 'boxes': boxes}
            records.append(row)
            if status != 'ok':
                excluded.append({'key': row['key'], 'split': split, 'reason': status})
    overlap = {k: sorted(v) for k, v in cow_splits.items() if len(v) > 1}
    duplicates = [h for h, splits in hash_splits.items() if len(splits) > 1]
    issues = {'cross_split_cows': overlap, 'cross_split_hashes': duplicates,
              'box_corrections': corrections, 'excluded_bcs': excluded}
    # No sobrescribir la evidencia de una auditoría válida anterior si falla un intento nuevo.
    if overlap or duplicates:
        save(root / 'reports/audit_failed_attempt.json', issues)
    require(not overlap and not duplicates, 'Fuga entre particiones: revisa reports/audit_failed_attempt.json.')
    records.sort(key=lambda r: r['key'])
    require(all(any(r['split'] == s and r['bcs_status'] == 'ok' for r in records)
                for s in ('train', 'val', 'test')), 'Partición BCS vacía.')
    source_hashes = {s: sha(source / s / '_annotations.coco.json') for s in ('train', 'valid', 'test')}
    dataset_hash = digest({'rows': records, 'coco': source_hashes})
    fp = digest({'dataset': dataset_hash, 'code': cfg['CODE_HASH'],
                 'tolerance': cfg['BOX_ROUNDING_TOLERANCE_PX']})
    if reusable(cfg, 'audit', fp):
        return
    save(root / 'reports/audit_issues.json', issues)
    target = root / 'manifests/manifest.json'
    save(target, {'dataset_hash': dataset_hash, 'coco_hashes': source_hashes, 'rows': records})
    write_csv(root / 'manifests/excluded_bcs.csv', excluded, ['key', 'split', 'reason'])
    totals = [len(records), sum(r['bcs_status'] == 'ok' for r in records)]
    summary = []
    for split in ('train', 'val', 'test'):
        selected = [r for r in records if r['split'] == split]
        eligible = [r for r in selected if r['bcs_status'] == 'ok']
        summary.append({'split': split, 'detector_images': len(selected), 'regressor_images': len(eligible),
                        'excluded_bcs': len(selected)-len(eligible),
                        'detector_pct': 100*len(selected)/totals[0], 'regressor_pct': 100*len(eligible)/totals[1],
                        'detector_cows': len({c for r in selected for c in r['cow_ids']}),
                        'regressor_cows': len({r['cow_id'] for r in eligible}),
                        'bcs_distribution': dict(collections.Counter(str(r['bcs']) for r in eligible))})
    save(root / 'reports/audit_summary.json', summary)
    stage(cfg, 'audit', fp, [target, root / 'reports/audit_summary.json', root / 'reports/audit_issues.json',
                            root / 'manifests/excluded_bcs.csv'])
    print(json.dumps(summary, indent=2))


def partition(cfg):
    root = artifacts(cfg)
    m = load(root / 'manifests/manifest.json')
    assignments = {r['key']: r['split'] for r in m['rows']}
    fp = digest({'dataset': m['dataset_hash'], 'assignments': assignments, 'code': cfg['CODE_HASH']})
    if reusable(cfg, 'partitions', fp):
        return
    path = root / 'manifests/splits.json'
    save(path, {'dataset_hash': m['dataset_hash'], 'split_hash': digest(assignments),
                'policy': 'original_by_cow', 'assignments': assignments})
    stage(cfg, 'partitions', fp, [path])


def export_yolo(cfg):
    root = artifacts(cfg)
    fp = digest(contract(cfg))
    target = root / 'prepared/yolo'
    # Las rutas del YAML se regeneran para la sesión actual.
    yaml = root / 'configs/detector.yaml'
    if not reusable(cfg, 'yolo_export', fp):
        outputs = []
        for r in rows(cfg):
            dest = target / 'images' / r['split'] / (r['sha256'][:16] + '_' + Path(r['image']).name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            # Copias portables: no guardar symlinks hacia /kaggle/input.
            from common import copy_atomic
            copy_atomic(image_path(cfg, r), dest)
            label = target / 'labels' / r['split'] / (dest.stem + '.txt')
            label.parent.mkdir(parents=True, exist_ok=True)
            lines = []
            for b in r['boxes']:
                x, y, z, w = b['xyxy']
                lines.append(f"{b['class']} {(x+z)/2/r['width']:.10f} {(y+w)/2/r['height']:.10f} "
                             f"{(z-x)/r['width']:.10f} {(w-y)/r['height']:.10f}")
            label.write_text('\n'.join(lines)+'\n')
            outputs.extend([dest, label])
        stage(cfg, 'yolo_export', fp, outputs)
    save(yaml, {'path': str(target), 'train': 'images/train', 'val': 'images/val',
                'test': 'images/test', 'names': {0: 'cow', 1: 'ass'}})


def samples(cfg):
    root = artifacts(cfg)
    fp = digest({**contract(cfg), 'size': cfg['REGRESSOR']['size']})
    if reusable(cfg, 'regression_samples', fp):
        return
    output, index = [], []
    for r in rows(cfg, supervised=True):
        path = root / 'prepared/crops_gt' / r['split'] / (digest(r['key']) + '.png')
        path.parent.mkdir(parents=True, exist_ok=True)
        box = next(b['xyxy'] for b in r['boxes'] if b['class'] == 1)
        with Image.open(image_path(cfg, r)) as im:
            crop(im, box, cfg['REGRESSOR']['size']).save(path)
        output.append(path)
        index.append({'key': r['key'], 'split': r['split'], 'cow_id': r['cow_id'], 'bcs': r['bcs'],
                      'crop': str(path.relative_to(root))})
    path = root / 'manifests/regression_samples.json'
    save(path, index)
    stage(cfg, 'regression_samples', fp, output+[path])
