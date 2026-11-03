"""Adaptar COCO original al esquema de entrada del ETL sin modificar el original."""
import json
import math
import re
import shutil
from pathlib import Path
from .common import require, load_json, save_csv


def normalize(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    require(not destination.exists(), f'Entrada normalizada ya existe: {destination}; usar --work nuevo')
    metadata = []
    for original, split in [('train','train'),('valid','val'),('test','test')]:
        folder=source/original
        coco=load_json(folder/'_annotations.coco.json')
        cats={c['id']:c['name'] for c in coco['categories']}
        require(len(cats)==len(coco['categories']), 'Categorías duplicadas')
        images={im['id']:im for im in coco['images']}
        require(len(images)==len(coco['images']), 'IDs COCO duplicados')
        anns={};ann_ids=set()
        for a in coco['annotations']:
            require(a['id'] not in ann_ids and a['image_id'] in images, 'Referencia COCO inválida')
            require(not a.get('iscrowd',0), 'Anotación crowd no admitida')
            require(cats[a['category_id']] in ('cow','ass'), 'Objeto COCO no admitido')
            ann_ids.add(a['id']);anns.setdefault(a['image_id'],[]).append(a)
        copied=destination/'coco_original'/original
        copied.mkdir(parents=True)
        shutil.copy2(folder/'_annotations.coco.json',copied/'_annotations.coco.json')
        for im in coco['images']:
            name=im['file_name']
            require(Path(name).name==name and '/' not in name and '\\' not in name,'Nombre de archivo no admitido')
            img=destination/'images'/split/name;img.parent.mkdir(parents=True,exist_ok=True)
            require(not img.exists(),'Nombre de imagen duplicado')
            require((folder/name).is_file(),f'Falta fotografía {name}')
            try:img.symlink_to(folder/name)
            except OSError:shutil.copy2(folder/name,img)
            tags=im.get('extra',{}).get('user_tags',[])
            ids=sorted({str(t) for t in tags if re.fullmatch(r'\d+',str(t))})
            btags=sorted({str(t) for t in tags if str(t).startswith('BCS_')})
            require(bool(ids),'Falta identidad para comprobar particiones')
            status,cow,bcs='etiquetas_ambiguas','',''
            if len(ids)==len(btags)==1:
                match=re.fullmatch(r'BCS_(\d+)(?:_(\d+))?',btags[0])
                require(match is not None,'Etiqueta BCS no válida')
                bcs=float(match[1]+('.'+match[2] if match[2] else ''))
                status,cow='ok',ids[0]
            metadata.append({'split':split,'image_id':str(im['id']),'file_name':name,
                'cow_id':cow,'bcs':bcs,'metadata_status':status,'raw_tags':json.dumps(tags)})
            encoded=[]
            for a in anns.get(im['id'],[]):
                x,y,w,h=map(float,a['bbox']);iw,ih=im['width'],im['height']
                require(all(math.isfinite(v) for v in (x,y,w,h)) and w>0 and h>0,'Caja no válida')
                require(max(0,-x,-y,x+w-iw,y+h-ih)<=.0110001,'Caja fuera de imagen; revisar COCO')
                x1,y1,x2,y2=max(0,x),max(0,y),min(iw,x+w),min(ih,y+h)
                require(x2>x1 and y2>y1,'Caja vacía')
                cls={'cow':0,'ass':1}[cats[a['category_id']]]
                encoded.append(f'{cls} {(x1+x2)/2/iw:.12f} {(y1+y2)/2/ih:.12f} {(x2-x1)/iw:.12f} {(y2-y1)/ih:.12f}')
            label=destination/'labels'/split/(Path(name).stem+'.txt');label.parent.mkdir(parents=True,exist_ok=True)
            require(not label.exists(),'Colisión de nombres de etiquetas')
            label.write_text('\n'.join(encoded)+'\n')
    save_csv(destination/'metadata.csv',metadata)
    (destination/'data.yaml').write_text('names:\n  0: cow\n  1: ass\n')
    return destination
