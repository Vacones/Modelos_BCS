import json
from pathlib import Path
from PIL import Image,ImageDraw
from .common import require,save_json,load_json,crop_image,sha256
from .data import manifest


def pair_boxes(boxes, coverage=.8):
    """Solo un par único: sin usar las cajas reales ni el BCS para escoger."""
    cows=[b for b in boxes if b['class']==0];rumps=[b for b in boxes if b['class']==1];pairs=[]
    for cow in cows:
        a,b,c,d=cow['xyxy']
        for rump in rumps:
            x,y,z,w=rump['xyxy']; area=(z-x)*(w-y)
            if area<=0:continue
            intersect=max(0,min(c,z)-max(a,x))*max(0,min(d,w)-max(b,y))
            if a<=(x+z)/2<=c and b<=(y+w)/2<=d and intersect/area>=coverage:
                pairs.append((cow,rump))
    if not pairs:return 'sin_par_valido',None
    if len(pairs)>1:return 'asociacion_ambigua',None
    return 'ok',pairs[0]


def get_detector(weights,device):
    import torch
    from ultralytics import YOLO
    require(torch.cuda.is_available() or device=='cpu','PyTorch no detecta CUDA. Revisa el entorno YOLO/nvidia-smi.')
    model=YOLO(weights)
    require(model.names=={0:'cow',1:'ass'},'Pesos no adaptados a tus dos clases. Usa runs/detector/weights/best.pt, no yolo26n.pt general.')
    return model


def infer(model,path,destination,size,imgsz,conf,device):
    results=model.predict(source=str(path),imgsz=imgsz,conf=conf,device=device,verbose=False)
    detection=results[0].boxes
    boxes=[]
    if detection is not None:
        for xyxy,cls,score in zip(detection.xyxy.cpu().numpy(),detection.cls.cpu().numpy(),detection.conf.cpu().numpy()):
            boxes.append({'class':int(cls),'score':float(score),'xyxy':[float(x) for x in xyxy]})
    status,pair=pair_boxes(boxes)
    info={'status':status,'boxes':boxes,'size':size,'crop_path':''}
    if status=='ok':
        destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
        with Image.open(path) as im:
            crop_image(im,pair[1]['xyxy'],size).save(destination)
        info['crop_path']=str(destination)
    return info


def train_detector(args):
    from ultralytics import YOLO
    import torch
    require(torch.cuda.is_available() or args.device=='cpu','No hay GPU CUDA para YOLO')
    work,cfg,data=manifest(args.work)
    if args.resume:
        model=YOLO(args.resume)
        model.train(resume=True,device=args.device)
        return
    run=Path(args.run).resolve();require(not run.exists(),'La carpeta detector ya existe. Usa --resume ruta/last.pt u otro --run')
    model=YOLO(args.weights)
    model.train(data=str(work/'detector.yaml'),epochs=args.epochs,imgsz=args.imgsz,batch=args.batch,
        device=args.device,workers=2,seed=42,deterministic=True,project=str(run.parent),name=run.name,
        exist_ok=False,patience=args.patience,save=True,save_period=5,plots=True,cache=False)
    best=run/'weights/best.pt'
    require(best.is_file(),'No se encontró el best.pt del entrenamiento')
    YOLO(str(best)).val(data=str(work/'detector.yaml'),split='val',imgsz=args.imgsz,device=args.device,
                       project=str(run),name='validation',plots=True)
    save_json(run/'pipeline_info.json',{'manifest_sha256':cfg['manifest_sha256'],'best':str(best),'weights_initial':args.weights})
    print('Detector guardado:',best)


def detector_test(args):
    work,cfg,data=manifest(args.work)
    model=get_detector(args.weights,args.device)
    out=Path(args.out).resolve();require(not out.exists(),'Usa --out nuevo para evaluación YOLO')
    model.val(data=str(work/'detector.yaml'),split=args.split,imgsz=args.imgsz,device=args.device,
              project=str(out.parent),name=out.name,plots=True)


def predicted_crops(args):
    work,cfg,data=manifest(args.work)
    output=Path(args.out).resolve();require(not output.exists(),'La salida de detección ya existe')
    model=get_detector(args.weights,args.device);output.mkdir(parents=True)
    save_json(output/'settings.json',{'manifest_sha256':cfg['manifest_sha256'],'size':cfg['size'],
        'detector':str(Path(args.weights).resolve()),'weights_sha256':sha256(args.weights),
        'imgsz':args.imgsz,'confidence':args.conf,'association_min_coverage':.8,'splits':args.splits})
    with (output/'predictions.jsonl').open('w') as f:
        for row in data:
            if row['split'] not in args.splits:continue
            relative=Path('crops')/row['split']/(Path(row['file_name']).stem+'.png')
            info=infer(model,row['source_image'],output/relative,cfg['size'],args.imgsz,args.conf,args.device)
            info.update(split=row['split'],file_name=row['file_name'],crop_path=relative.as_posix() if info['status']=='ok' else '')
            f.write(json.dumps(info)+'\n');f.flush()
            print(row['split'],row['file_name'],info['status'])


def detect_image(args):
    output=Path(args.out).resolve();require(not output.exists(),'Usa --out nuevo para cada imagen')
    model=get_detector(args.weights,args.device);output.mkdir(parents=True)
    info=infer(model,args.image,output/'crop.png',args.size,args.imgsz,args.conf,args.device)
    info['crop_path']='crop.png' if info['status']=='ok' else ''
    info['source_image']=str(Path(args.image).resolve())
    save_json(output/'prediction.json',info)
    with Image.open(args.image) as raw:
        preview=raw.convert('RGB')
    draw=ImageDraw.Draw(preview)
    for b in info['boxes']:
        color='lime' if b['class']==0 else 'orange'
        draw.rectangle(b['xyxy'],outline=color,width=4)
    preview.save(output/'detecciones.jpg');print(info['status'],output)
