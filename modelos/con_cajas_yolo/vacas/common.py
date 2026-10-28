import csv
import hashlib
import json
import math
from pathlib import Path
from PIL import Image, ImageOps


def require(condition, message):
    if not condition:
        raise ValueError(message)


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')
    temp.replace(path)


def load_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def rows(path):
    with Path(path).open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def save_csv(path, records, fields=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    require(bool(records) or fields, f'No hay columnas para {path}')
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields or list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def crop_image(image, xyxy, size):
    """Mismo recorte y letterbox para ETL, evaluación e inferencia."""
    width, height = image.size
    x1,y1,x2,y2 = xyxy
    require(all(math.isfinite(x) for x in xyxy), 'Caja no finita')
    box = (max(0, math.floor(x1)), max(0, math.floor(y1)), min(width, math.ceil(x2)), min(height, math.ceil(y2)))
    require(box[2]>box[0] and box[3]>box[1], 'Caja vacía')
    crop = image.convert('RGB').crop(box)
    return ImageOps.pad(crop, (size,size), method=Image.Resampling.BILINEAR, color=(0,0,0), centering=(.5,.5))


def read_yolo(path, width, height):
    result=[]
    for line in Path(path).read_text().splitlines():
        fields=line.split()
        require(len(fields)==5, f'Etiqueta YOLO inválida: {path}')
        cls=int(fields[0]); xc,yc,w,h=map(float,fields[1:])
        require(cls in (0,1) and all(math.isfinite(v) and 0<=v<=1 for v in (xc,yc,w,h)) and w>0 and h>0, f'Caja inválida: {path}')
        box=[(xc-w/2)*width,(yc-h/2)*height,(xc+w/2)*width,(yc+h/2)*height]
        require(box[0]>=-1e-4 and box[1]>=-1e-4 and box[2]<=width+1e-4 and box[3]<=height+1e-4, 'Caja fuera de imagen')
        result.append({'class':cls,'xyxy':box})
    return result
