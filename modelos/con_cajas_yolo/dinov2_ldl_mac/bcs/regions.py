"""ROI geométrica revisable; NO detecta huesos ni recibe etiquetas BCS."""
import math
from PIL import Image, ImageOps, ImageDraw
from .common import require, rows

DEFAULT_ROI = [0., 0., 1., .65]


def validate_roi(box):
    require(len(box) == 4 and all(math.isfinite(float(v)) for v in box), 'ROI no finita')
    x1,y1,x2,y2 = map(float, box)
    require(0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1, 'ROI: 0 <= x1 < x2 <= 1, 0 <= y1 < y2 <= 1')
    return [x1,y1,x2,y2]


def load_rois(path):
    result = {}
    if not path:
        return result
    for r in rows(path):
        key = r['key']
        require(key not in result, 'ROI duplicada: '+key)
        result[key] = validate_roi([r[c] for c in ['x1','y1','x2','y2']])
    return result


def pixel_roi(image, roi):
    x1,y1,x2,y2 = validate_roi(roi)
    w,h = image.size
    return (math.floor(x1*w), math.floor(y1*h), min(w, math.ceil(x2*w)), min(h, math.ceil(y2*h)))


def region_image(image, roi):
    return ImageOps.pad(image.convert('RGB').crop(pixel_roi(image,roi)), (224,224),
                        method=Image.Resampling.BILINEAR, color=(0,0,0))


def draw_roi(image, roi):
    result=image.convert('RGB').copy()
    ImageDraw.Draw(result).rectangle(pixel_roi(image,roi), outline='cyan', width=3)
    return result
