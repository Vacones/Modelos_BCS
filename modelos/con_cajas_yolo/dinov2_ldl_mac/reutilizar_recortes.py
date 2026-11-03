#!/usr/bin/env python3
"""Reconstruye los recortes predichos exactos guardados en el CSV del notebook."""
import argparse,ast,csv,json
from pathlib import Path
from PIL import Image
from bcs.data import manifest
from bcs.common import require,save_json,crop_image,sha256
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--work',required=True);p.add_argument('--csv',required=True);p.add_argument('--out',required=True)
a=p.parse_args();_,cfg,rows=manifest(a.work)
with Path(a.csv).open() as f: original=list(csv.DictReader(f))
require(bool(original),'CSV vacío')
reference={}
for r in original:
    split,name=r['clave'].split('/',1);split='val' if split=='valid' else split
    require((split,name) not in reference,'Foto duplicada');reference[(split,name)]=r
splits={k[0] for k in reference};expected={(r['split'],r['file_name']):r for r in rows if r['status']=='ok' and r['split'] in splits}
require(set(reference)==set(expected),'Las fotos del CSV no coinciden con el manifest')
output=Path(a.out);require(not output.exists(),'Salida existente')
prepared=[]
for k,r in reference.items():
    require(float(r['real'])==float(expected[k]['bcs']),'Cambió etiqueta BCS')
    status=r['estado'];require(status in ('ok','sin_par_valido','asociacion_ambigua'),'Estado desconocido')
    box=ast.literal_eval(r['caja_predicha']) if status=='ok' else None
    if box is not None: require(len(box)==4,'Caja inválida')
    prepared.append((k,status,box))
output.mkdir(parents=True)
save_json(output/'settings.json',{'manifest_sha256':cfg['manifest_sha256'],'size':224,'source_csv_sha256':sha256(a.csv),'kind':'exact_notebook_boxes'})
with (output/'predictions.jsonl').open('w') as f:
    for k,status,box in prepared:
        path=Path('crops')/k[0]/(Path(k[1]).stem+'.png')
        if box is not None:
            dest=output/path;dest.parent.mkdir(parents=True,exist_ok=True)
            with Image.open(expected[k]['source_image']) as im:crop_image(im,box,224).save(dest)
        f.write(json.dumps({'split':k[0],'file_name':k[1],'status':status,'crop_path':str(path) if box is not None else ''})+'\n')
print('Recortes reconstruidos con las coordenadas originales:',output)
