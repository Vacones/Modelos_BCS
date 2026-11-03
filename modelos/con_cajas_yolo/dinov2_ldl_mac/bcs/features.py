"""DINOv2 congelado: vista completa + ROI superior o manual, ambas con CLS y parches."""
import random
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from .common import require, save_json, load_json, sha256
from .data import manifest
from .regions import DEFAULT_ROI, validate_roi, load_rois, region_image, draw_roi

DINO_COMMIT = '7764ea0f912e53c92e82eb78a2a1631e92725fc8'
DINO_REPO = 'facebookresearch/dinov2:'+DINO_COMMIT
DINO_NAME = 'dinov2_vits14'
BASE_DIM = 768
DIM = 1536
MEAN = [.485,.456,.406]
STD = [.229,.224,.225]


def device_for(name):
    if name=='cpu':return torch.device('cpu')
    require(name in ('mps','auto'),'En Mac usa --device mps o cpu')
    require(torch.backends.mps.is_available(),
        'GPU MPS no disponible. Ejecuta comprobar_gpu_mac.py. No se cambiará silenciosamente a CPU.')
    dev=torch.device('mps');torch.empty(1,device=dev)
    return dev


def encoder(device):
    net=torch.hub.load(DINO_REPO,DINO_NAME,pretrained=True,trust_repo=True)
    return net.eval().requires_grad_(False).to(device)


def transform(augment=False):
    # Aumentos se aplican a la imagen y ROI conjuntamente en embed_images.
    return transforms.Compose([transforms.ToTensor(),transforms.Normalize(MEAN,STD)])


def visual_features(model, tensor):
    out=model.forward_features(tensor)
    return torch.cat([out['x_norm_clstoken'],out['x_norm_patchtokens'].mean(1)],dim=1)


def embed_images(model, images, rois, device, mode='upper', augment=False):
    require(mode in ('upper','manual','full'), 'Modo ROI inválido')
    tfm=transform();full=[];focused=[]
    for image,roi in zip(images,rois):
        require(image.size==(224,224),'El recorte preparado debe medir 224x224')
        im=image.convert('RGB');box=validate_roi(roi)
        if augment:
            # Jitter compartido y volteo con transformación geométrica de la ROI.
            im=transforms.ColorJitter(brightness=.1,contrast=.1)(im)
            if random.random()<.5:
                im=im.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
                box=[1-box[2],box[1],1-box[0],box[3]]
        full.append(tfm(im));focused.append(tfm(region_image(im,box)))
    with torch.inference_mode():
        a=visual_features(model,torch.stack(full).to(device))
        b=torch.zeros_like(a) if mode=='full' else visual_features(model,torch.stack(focused).to(device))
        z=torch.cat([a,b],dim=1).float().cpu().numpy()
    require(z.shape==(len(images),DIM) and np.isfinite(z).all(),'Características inválidas')
    return z


def embed_paths(model,paths,device,batch=2,rois=None,mode='upper'):
    rois=rois or [DEFAULT_ROI]*len(paths)
    chunks=[]
    for offset in range(0,len(paths),batch):
        images=[]
        for path in paths[offset:offset+batch]:
            with Image.open(path) as im: images.append(im.convert('RGB'))
        chunks.append(embed_images(model,images,rois[offset:offset+batch],device,mode))
    return np.concatenate(chunks) if chunks else np.empty((0,DIM),np.float32)


def extract(args,model_factory=encoder):
    work,cfg,records=manifest(args.work);out=Path(args.out).resolve()
    require(not out.exists(),'Caché existente: utilizar --out nuevo')
    require(args.views>=1 and args.batch>=1,'views y batch deben ser positivos')
    require(not(args.boxes and args.views!=1),'Cajas predichas: --views 1')
    mode=getattr(args,'roi_mode','upper');manual=load_rois(getattr(args,'roi_csv',None))
    kind='predicted' if args.boxes else 'annotated'
    predicted={};boxes_cfg=None
    if args.boxes:
        folder=Path(args.boxes).resolve();boxes_cfg=load_json(folder/'settings.json')
        require(boxes_cfg['manifest_sha256']==cfg['manifest_sha256'] and boxes_cfg['size']==224,'Cajas de otro ETL')
        for line in (folder/'predictions.jsonl').read_text().splitlines():
            row=__import__('json').loads(line);key=(row['split'],row['file_name'])
            require(key not in predicted,'Detección duplicada');predicted[key]=row
    random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
    jobs=[];roi_lookup={}
    for row in records:
        if row['status']!='ok' or row['split'] not in args.splits:continue
        key=row['split']+'/'+row['file_name'];path=row['crop_path']
        if args.boxes:
            pkey=(row['split'],row['file_name'])
            require(pkey in predicted,'Falta estado de detección')
            if predicted[pkey]['status']!='ok':continue
            path=str((folder/predicted[pkey]['crop_path']).resolve())
        mkey=kind+'/'+key
        require(mode!='manual' or mkey in manual,'Falta ROI manual: '+mkey)
        roi=manual.get(mkey,DEFAULT_ROI);roi_lookup[key]=roi
        views=args.views if row['split']=='train' and not args.boxes else 1
        for view in range(views): jobs.append((row,path,view,roi))
    require(bool(jobs) or args.boxes,'No hay recortes para extraer')
    out.mkdir(parents=True);dev=device_for(args.device);model=model_factory(dev);chunks=[]
    for offset in range(0,len(jobs),args.batch):
        batch=jobs[offset:offset+args.batch]
        zs=np.empty((len(batch),DIM),dtype=np.float32)
        for is_aug in (False,True):
            indices=[i for i,(_,_,v,_) in enumerate(batch) if (v>0)==is_aug]
            if not indices:continue
            images=[];boxes=[]
            for i in indices:
                row,path,view,roi=batch[i]
                with Image.open(path) as im:images.append(im.convert('RGB'))
                boxes.append(roi)
                if view==0 and offset<args.batch*3:
                    preview=out/'vista_previa';preview.mkdir(exist_ok=True)
                    draw_roi(images[-1],roi).save(preview/(row['split']+'_'+Path(path).name))
            zs[indices]=embed_images(model,images,boxes,dev,mode,augment=is_aug)
        chunks.append(zs);print(f'Características {min(offset+len(batch),len(jobs))}/{len(jobs)}',flush=True)
    np.savez_compressed(out/'features.npz',x=np.concatenate(chunks) if chunks else np.empty((0,DIM),np.float32),
        keys=np.asarray([r['split']+'/'+r['file_name'] for r,p,v,b in jobs]),
        splits=np.asarray([r['split'] for r,p,v,b in jobs]),y=np.asarray([float(r['bcs']) for r,p,v,b in jobs],np.float32),
        views=np.asarray([v for r,p,v,b in jobs],np.int64),
        cows=np.asarray([r['cow_id'] for r,p,v,b in jobs]))
    save_json(out/'settings.json',{'manifest_sha256':cfg['manifest_sha256'],'classes':cfg['classes'],
        'encoder':DINO_NAME,'repo':DINO_REPO,'dim':DIM,'size':224,'mean':MEAN,'std':STD,
        'seed':args.seed,'train_views':args.views,'kind':kind,'roi_mode':mode,'default_roi':DEFAULT_ROI,
        'roi_csv_sha256':sha256(args.roi_csv) if getattr(args,'roi_csv',None) else None,'roi_lookup':roi_lookup,
        'statuses':{s+'/'+n:r['status'] for (s,n),r in predicted.items()},'boxes_settings':boxes_cfg,
        'splits':args.splits,'features_sha256':sha256(out/'features.npz')})


def cache(path):
    path=Path(path).resolve();cfg=load_json(path/'settings.json')
    require(sha256(path/'features.npz')==cfg['features_sha256'],'Caché modificado')
    require(cfg['encoder']==DINO_NAME and cfg['repo']==DINO_REPO and cfg['dim']==DIM,'Extractor incompatible')
    with np.load(path/'features.npz',allow_pickle=False) as f:data={k:f[k] for k in f.files}
    require(data['x'].shape==(len(data['keys']),DIM) and np.isfinite(data['x']).all(),'Caché inválido')
    return cfg,data
