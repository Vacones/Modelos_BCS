import json
import re
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from PIL import Image
from .common import require, rows, save_csv, save_json, load_json, sha256, crop_image, read_yolo


def etl(args):
    source=Path(args.data).resolve(); work=Path(args.work).resolve()
    require(source.is_dir(), 'No existe --data')
    require(not work.exists(), 'La salida ETL ya existe. Usa --work con una carpeta nueva.')
    require(args.size>=32, '--size debe ser >=32')
    # Leer el esquema names del YAML generado por nuestro convertidor, sin ejecutarlo.
    yaml_text=(source/'data.yaml').read_text()
    require('names:' in yaml_text, 'data.yaml sin names')
    class_text=yaml_text.split('names:',1)[1]
    pairs=re.findall(r'^\s+(\d+):\s*[\"\']?(\w+)[\"\']?\s*$',class_text,re.MULTILINE)
    names={int(k):v for k,v in pairs}
    require(names=={0:'cow',1:'ass'}, 'Se esperaba 0=cow y 1=ass. No se entrenará con clases intercambiadas.')
    metadata=rows(source/'metadata.csv')
    require(bool(metadata), 'metadata.csv está vacío')
    keyset=set(); cow_splits=defaultdict(set); hashes=defaultdict(set)
    records=[]; originals={}; original_boxes={}; expected_files=set()
    for original,split in [('train','train'),('valid','val'),('test','test')]:
        coco=load_json(source/'coco_original'/original/'_annotations.coco.json')
        for im in coco['images']:
            originals[(split,str(im['id']))]=im
        cats={c['id']:c['name'] for c in coco['categories']}
        for ann in coco['annotations']:
            key=(split,str(ann['image_id']))
            original_boxes.setdefault(key,[]).append((cats[ann['category_id']],ann['bbox']))
    require(len(originals)==len(metadata), 'COCO y metadata tienen diferente cantidad de imágenes')
    for row in metadata:
        split=row['split']; name=row['file_name']; key=(split,row['image_id'])
        require(split in ('train','val','test') and key not in keyset, 'Partición o registro duplicado')
        keyset.add(key)
        require(Path(name).name==name and '/' not in name and '\\' not in name, 'Nombre de archivo no admitido')
        im=originals[key]
        require(name==im['file_name'] and json.loads(row['raw_tags'])==im.get('extra',{}).get('user_tags',[]), 'Los tags o la imagen no coinciden con COCO')
        tags=json.loads(row['raw_tags'])
        ids=list(dict.fromkeys(t for t in tags if re.fullmatch(r'\d+',t)))
        btags=list(dict.fromkeys(t for t in tags if t.startswith('BCS_')))
        for cow in ids:cow_splits[cow].add(split)
        path=source/'images'/split/name
        label=source/'labels'/split/(Path(name).stem+'.txt')
        require(path.is_file() and label.is_file(), f'Falta imagen/etiqueta {split}/{name}. No uses el ZIP sin fotos.')
        with Image.open(path) as image:
            require(image.size==(im['width'],im['height']), 'Las dimensiones no coinciden con COCO')
            image.verify()
        with Image.open(path) as image:
            image.convert('RGB').load()
        boxes=read_yolo(label,im['width'],im['height'])
        require(Counter(b['class'] for b in boxes)==Counter({0:1,1:1}), f'Se necesita un par único cow/ass: {name}')
        for class_name, bbox in original_boxes[key]:
            x,y,w,h=bbox
            actual=next(b['xyxy'] for b in boxes if b['class']=={'cow':0,'ass':1}[class_name])
            expected=[x,y,x+w,y+h]
            require(max(abs(a-b) for a,b in zip(actual,expected))<=.012,'La etiqueta YOLO no coincide con el COCO original')
        image_hash=sha256(path);hashes[image_hash].add(split)
        expected_files.add((split,name))
        status=row['metadata_status']; bcs='';cow_id='';crop_path=''
        if status=='ok':
            require(len(ids)==1 and len(btags)==1, 'Metadata ok con etiquetas ambiguas')
            match=re.fullmatch(r'BCS_(\d+)(?:_(\d+))?',btags[0])
            require(match is not None,'BCS inválido')
            bcs=float(match[1]+('.'+match[2] if match[2] else ''))
            cow_id=ids[0]
            require(1<=bcs<=5 and row['cow_id']==cow_id and abs(float(row['bcs'])-bcs)<1e-8,'BCS/identidad no coincide con tags')
        records.append({'split':split,'image_id':row['image_id'],'file_name':name,'source_image':str(path),
            'cow_id':cow_id,'bcs':bcs,'status':status,'crop_path':crop_path,'sha256':image_hash,
            'width':im['width'],'height':im['height'],'box_count':len(boxes),
            'ass_xyxy':json.dumps(next(b['xyxy'] for b in boxes if b['class']==1)),
            'raw_tags':row['raw_tags']})
    require(keyset==set(originals), 'Metadatos incompletos o sobrantes')
    overlap={c:sorted(s) for c,s in cow_splits.items() if len(s)>1}
    duplicates={h:sorted(s) for h,s in hashes.items() if len(s)>1}
    require(not overlap and not duplicates, f'Fuga entre particiones: cows={overlap}, hashes={duplicates}')
    all_images={(p.parent.name,p.name) for p in (source/'images').glob('*/*') if p.is_file()}
    require(all_images==expected_files, 'Hay fotos faltantes o sin metadata en images/')
    train_values=sorted({float(r['bcs']) for r in records if r['split']=='train' and r['status']=='ok'})
    require(len(train_values)>=2, 'Se necesitan al menos dos clases BCS de entrenamiento')
    require(all(float(r['bcs']) in train_values for r in records if r['status']=='ok'), 'Hay categorías fuera del catálogo de entrenamiento')
    work.mkdir(parents=True)
    for r in records:
        if r['status']!='ok':continue
        destination=work/'crops_gt'/r['split']/(Path(r['file_name']).stem+'.png')
        destination.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(r['source_image']) as image:
            crop_image(image,json.loads(r['ass_xyxy']),args.size).save(destination)
        r['crop_path']=str(destination)
    save_csv(work/'manifest.csv',records)
    save_csv(work/'excluidos_bcs.csv',[r for r in records if r['status']!='ok'],fields=list(records[0]))
    save_json(work/'classes.json',train_values)
    settings={'source':str(source),'size':args.size,'classes':train_values,
              'manifest_sha256':sha256(work/'manifest.csv'),'images':len(records),
              'metadata_ok':sum(r['status']=='ok' for r in records),'overlap':overlap,
              'source_metadata_sha256':sha256(source/'metadata.csv')}
    save_json(work/'etl.json',settings)
    detector={'train':str(source/'images/train'),'val':str(source/'images/val'),'test':str(source/'images/test'),'names':{0:'cow',1:'ass'}}
    # JSON es un subconjunto válido de YAML y Ultralytics lo lee en este archivo.
    (work/'detector.yaml').write_text(json.dumps(detector,indent=2))
    shutil.copy2(source/'metadata.csv',work/'metadata_original.csv')
    print(json.dumps(settings,indent=2))


def manifest(work):
    work=Path(work).resolve(); cfg=load_json(work/'etl.json')
    require(sha256(work/'manifest.csv')==cfg['manifest_sha256'],'manifest.csv fue modificado; vuelve a ejecutar ETL en otra carpeta')
    return work,cfg,rows(work/'manifest.csv')


def eda(args):
    import numpy as np
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    work,cfg,data=manifest(args.work)
    output=work/'eda';output.mkdir(exist_ok=True)
    classes=cfg['classes'];splits=['train','val','test']
    fig,ax=plt.subplots(figsize=(10,5)); x=np.arange(len(classes));width=.25
    table=[]
    for i,split in enumerate(splits):
        count=Counter(float(r['bcs']) for r in data if r['split']==split and r['status']=='ok')
        ax.bar(x+(i-1)*width,[count[c] for c in classes],width,label=split)
        for c in classes:table.append({'split':split,'bcs':c,'images':count[c]})
    ax.set_xticks(x,[str(c) for c in classes]);ax.set_xlabel('BCS');ax.set_ylabel('Imágenes inequívocas');ax.legend()
    fig.tight_layout();fig.savefig(output/'distribucion_bcs.png',dpi=160);plt.close(fig)
    save_csv(output/'distribucion_bcs.csv',table)
    cow_counts=Counter(r['cow_id'] for r in data if r['status']=='ok')
    fig,ax=plt.subplots(figsize=(8,4));ax.hist(list(cow_counts.values()),bins=range(1,max(cow_counts.values())+2));ax.set_xlabel('Fotos por vaca');ax.set_ylabel('Vacas');fig.tight_layout();fig.savefig(output/'fotos_por_vaca.png',dpi=160);plt.close(fig)
    valid=[r for r in data if r['status']=='ok']; sample=valid[:9]
    fig,axes=plt.subplots(3,3,figsize=(9,9))
    for ax,r in zip(axes.flat,sample):
        with Image.open(r['crop_path']) as im:ax.imshow(im)
        ax.set_title(f"{r['split']} | vaca {r['cow_id']} | BCS {r['bcs']}");ax.axis('off')
    for ax in list(axes.flat)[len(sample):]:ax.axis('off')
    fig.tight_layout();fig.savefig(output/'ejemplos_recortes.png',dpi=150);plt.close(fig)
    warnings=[]
    for c in classes:
        supports={s:sum(r['split']==s and r['status']=='ok' and float(r['bcs'])==c for r in data) for s in splits}
        if min(supports.values())==0 or supports['train']<5:warnings.append({'bcs':c,'support':supports})
    summary={'images_by_split':dict(Counter(r['split'] for r in data)), 'cows_by_split':{s:len({r['cow_id'] for r in valid if r['split']==s}) for s in splits},
             'excluded':sum(r['status']!='ok' for r in data),'rare_or_unrepresented_classes':warnings,
             'note':'Sin redistribución ni aumento de datos antes de separar particiones.'}
    save_json(output/'eda.json',summary);print(json.dumps(summary,indent=2))
