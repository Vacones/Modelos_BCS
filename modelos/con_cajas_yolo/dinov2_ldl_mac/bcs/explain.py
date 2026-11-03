"""Oclusión: sensibilidad local de la estimación, NO atención ni detección de huesos."""
import json
import hashlib
from pathlib import Path
import numpy as np
from PIL import Image
from .common import require, save_json, save_csv, sha256, load_json
from .regions import pixel_roi, draw_roi
from .features import embed_images, encoder, cache, device_for
from .training import load_model, predict_features, model_fingerprint
from .data import manifest


def occlusion(image,roi,mode,backbone,head,saved,device,grid=7,batch=2):
    require(2<=grid<=16 and batch>=1,'grid entre 2 y 16; batch positivo')
    def predict(images):
        z=embed_images(backbone,images,[roi]*len(images),device,mode)
        return predict_features(head,z,saved,device)
    _,base,probs=predict([image]);base=float(base[0]);w,h=image.size
    fill=tuple(np.asarray(image).mean(axis=(0,1)).round().astype(int).tolist())
    borders_x=np.linspace(0,w,grid+1,dtype=int);borders_y=np.linspace(0,h,grid+1,dtype=int)
    cells=[];images=[]
    for iy in range(grid):
        for ix in range(grid):
            box=[int(borders_x[ix]),int(borders_y[iy]),int(borders_x[ix+1]),int(borders_y[iy+1])]
            im=image.copy();im.paste(fill,tuple(box));images.append(im)
            cells.append({'row':iy,'column':ix,'box':box})
    values=[]
    for offset in range(0,len(images),batch):
        _,expected,_=predict(images[offset:offset+batch]);values.extend(expected.tolist())
    for cell,value in zip(cells,values):
        cell.update(masked_bcs=float(value),delta_bcs=float(value-base),sensitivity_bcs=abs(float(value-base)))
    heat=np.zeros((h,w),np.float32)
    for cell in cells:
        x1,y1,x2,y2=cell['box'];heat[y1:y2,x1:x2]=cell['sensitivity_bcs']
    box=pixel_roi(image,roi)
    roi_hidden=image.copy();roi_hidden.paste(fill,box)
    only_roi=Image.new('RGB',image.size,fill);only_roi.paste(image.crop(box),box)
    _,abl,_=predict([roi_hidden,only_roi])
    return {'base_bcs':base,'probabilities':probs[0].tolist(),'cells':cells,'fill_rgb':fill,
        'roi':roi,'roi_mode':mode,'grid':grid,'max_sensitivity_bcs':float(heat.max()),
        'roi_hidden_bcs':float(abl[0]),'outside_roi_hidden_bcs':float(abl[1]),
        'roi_hidden_abs_delta':abs(float(abl[0])-base),'outside_roi_hidden_abs_delta':abs(float(abl[1])-base)},heat


def figure(image,original,roi,info,heat,real,classes,path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(1,4,figsize=(16,4))
    axes[0].imshow(original);axes[0].set_title('Fotografía original')
    axes[1].imshow(image if info['roi_mode']=='full' else draw_roi(image,roi))
    axes[1].set_title('Grupa (rama ROI desactivada)' if info['roi_mode']=='full' else 'Grupa + ROI (cian)')
    axes[2].imshow(image)
    layer=axes[2].imshow(heat,cmap='magma',alpha=.60,vmin=0,vmax=max(float(heat.max()),1e-6),interpolation='nearest')
    axes[2].set_title('Cambio absoluto al ocultar')
    fig.colorbar(layer,ax=axes[2],fraction=.045,pad=.03,label='Cambio en BCS')
    for ax in axes[:3]:ax.axis('off')
    axes[3].bar([str(c) for c in classes],info['probabilities']);axes[3].tick_params(axis='x',rotation=60)
    axes[3].set(ylabel='Probabilidad estimada',xlabel='BCS',ylim=(0,1))
    fig.suptitle(f"Real: {real} | Estimación: {info['base_bcs']:.3f} | Sensibilidad por oclusión; no prueba anatómica")
    fig.tight_layout();fig.savefig(path,dpi=140);plt.close(fig)


def explain(args,model_factory=encoder):
    _,dcfg,records=manifest(args.work);fcfg,features=cache(args.features)
    require(args.split!='test' or args.ack_test,'Test exploratorio: usar --ack-test')
    require(args.limit>=0,'limit debe ser >=0; 0 significa todas')
    require(fcfg['manifest_sha256']==dcfg['manifest_sha256'],'Dataset incompatible')
    predicted={}
    if fcfg['kind']=='predicted':
        require(args.boxes is not None,'Debes indicar --boxes con los recortes evaluados')
        folder=Path(args.boxes).resolve()
        require(load_json(folder/'settings.json')==fcfg['boxes_settings'],'Cajas distintas a las evaluadas')
        for line in (folder/'predictions.jsonl').read_text().splitlines():
            r=json.loads(line);predicted[r['split']+'/'+r['file_name']]=r
    dev=device_for(args.device);saved,head=load_model(args.model,dev)
    require(saved['config']['manifest_sha256']==dcfg['manifest_sha256'] and saved['config']['roi_mode']==fcfg['roi_mode'],'Modelo incompatible')
    chosen=[r for r in records if r['split']==args.split and r['status']=='ok']
    paths={}
    for r in chosen:
        key=r['split']+'/'+r['file_name']
        if fcfg['kind']=='annotated':paths[key]=Path(r['crop_path'])
        elif fcfg['statuses'].get(key)=='ok':paths[key]=folder/predicted[key]['crop_path']
    lookup={str(k):i for i,k in enumerate(features['keys']) if features['views'][i]==0}
    require(set(paths)<=set(lookup),'Faltan características de fotos a explicar')
    # Orden estable por archivo, sin escoger casos favorables ni consultar sus errores.
    chosen=sorted([r for r in chosen if r['split']+'/'+r['file_name'] in paths],key=lambda r:r['file_name'])
    if args.limit:chosen=chosen[:args.limit]
    out=Path(args.out).resolve();require(not out.exists(),'Salida existente');out.mkdir(parents=True)
    backbone=model_factory(dev) if chosen else None
    summary=[]
    for index,r in enumerate(chosen):
        key=r['split']+'/'+r['file_name'];roi=fcfg['roi_lookup'][key]
        with Image.open(paths[key]) as im:image=im.convert('RGB')
        info,heat=occlusion(image,roi,fcfg['roi_mode'],backbone,head,saved,dev,args.grid,args.batch)
        _,cached,_=predict_features(head,features['x'][lookup[key]:lookup[key]+1],saved,dev)
        require(abs(float(cached[0])-info['base_bcs'])<1e-4,'La predicción explicada no coincide con el caché evaluado')
        stem=f"{index:04d}_"+hashlib.sha256(key.encode()).hexdigest()[:12]
        info.update(key=key,true_bcs=float(r['bcs']),checkpoint_sha256=model_fingerprint(args.model))
        save_json(out/(stem+'.json'),info);np.save(out/(stem+'_sensibilidad.npy'),heat)
        with Image.open(r['source_image']) as im:original=im.convert('RGB')
        figure(image,original,roi,info,heat,r['bcs'],dcfg['classes'],out/(stem+'.png'))
        summary.append({'key':key,'real':float(r['bcs']),'pred_continuous':info['base_bcs'],
            'max_change_bcs':info['max_sensitivity_bcs'],'hide_roi_change_bcs':info['roi_hidden_abs_delta'],
            'hide_outside_roi_change_bcs':info['outside_roi_hidden_abs_delta'],'figure':stem+'.png'})
        print(f"Explicación {index+1}/{len(chosen)}: {key}",flush=True)
    save_csv(out/'sensibilidad.csv',summary,fields=['key','real','pred_continuous','max_change_bcs','hide_roi_change_bcs','hide_outside_roi_change_bcs','figure'])
    skipped=[{'key':r['split']+'/'+r['file_name'],'status':fcfg['statuses'].get(r['split']+'/'+r['file_name'],'no_evaluable')}
        for r in records if r['split']==args.split and r['status']=='ok' and r['split']+'/'+r['file_name'] not in paths]
    save_csv(out/'abstenciones_sin_mapa.csv',skipped,fields=['key','status'])
    save_json(out/'metodo.json',{'method':'occlusion_absolute_change_of_expected_bcs','grid':args.grid,
        'fill':'media RGB del recorte','roi_mode':fcfg['roi_mode'],'recompute_both_views':True,'explained':len(summary),'limit':args.limit,
        'checkpoint_sha256':model_fingerprint(args.model),'features_sha256':fcfg['features_sha256'],
        'note':'Valores en unidades BCS, sin normalizar cada mapa a 1. No es Grad-CAM ni detección de huesos. Oclusión introduce imágenes artificiales; no demuestra causalidad anatómica ni confianza calibrada.'})
    lines=['# Diagnóstico visual por oclusión','','El color muestra cuánto cambia la estimación al ocultar una celda del recorte y recalcular ambas vistas.',
        'La barra está en unidades BCS. Una región brillante indica sensibilidad a esa perturbación, no que la red haya identificado un hueso.',
        'Las regiones cian son las entradas elegidas, no mapas aprendidos. Las abstenciones no tienen estimación que explicar.','']
    for item in summary:lines+=['### '+item['key'],'',f"![Diagnóstico]({item['figure']})",'']
    (out/'LEEME.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
