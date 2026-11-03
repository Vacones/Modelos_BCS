"""Métricas condicionales y globales; abstenciones nunca se ocultan."""
import json
from pathlib import Path
import numpy as np
from .common import require, save_json, save_csv
from .data import manifest


def metrics(records, classes):
    from sklearn.metrics import f1_score, cohen_kappa_score
    require(bool(records),'No hay casos evaluables')
    accepted=[r for r in records if r['pred_bcs'] is not None]
    total,n=len(records),len(accepted)
    m={'eligible_images':total,'classified_images':n,'abstentions':total-n,'coverage':n/total,
       'accuracy_classified':None,'mae_classified':None,'rmse_classified':None,'bias_classified':None,
       'r2_classified':None,'macro_f1_classified_all_catalog':None,'quadratic_kappa_classified':None}
    for tol in (0,.25,.5):
        hits=sum(abs(r['pred_bcs']-r['true_bcs'])<=tol+1e-8 for r in accepted)
        m[f'accuracy_all_tolerance_{tol}']=hits/total
        m[f'accuracy_classified_tolerance_{tol}']=hits/n if n else None
    if n:
        y=np.array([r['true_bcs'] for r in accepted]);p=np.array([r['pred_bcs'] for r in accepted]);err=p-y
        yi=[classes.index(float(v)) for v in y];pi=[classes.index(float(v)) for v in p]
        m.update(accuracy_classified=float(np.mean(abs(err)<1e-8)),mae_classified=float(abs(err).mean()),
            rmse_classified=float(np.sqrt(np.mean(err**2))),bias_classified=float(err.mean()),
            macro_f1_classified_all_catalog=float(f1_score(yi,pi,labels=list(range(len(classes))),average='macro',zero_division=0)))
        denom=np.sum((y-y.mean())**2)
        if denom>0:m['r2_classified']=float(1-np.sum(err**2)/denom)
        if len(set(yi))>1:
            k=cohen_kappa_score(yi,pi,labels=list(range(len(classes))),weights='quadratic')
            if np.isfinite(k):m['quadratic_kappa_classified']=float(k)
    cows=[]
    for cow in sorted({r['cow_id'] for r in records}):
        group=[r for r in records if r['cow_id']==cow]
        cows.append({'cow_id':cow,'images':len(group),
            'coverage':sum(r['pred_bcs'] is not None for r in group)/len(group),
            'accuracy_all_tolerance_0.5':sum(r['pred_bcs'] is not None and abs(r['pred_bcs']-r['true_bcs'])<=.50000001 for r in group)/len(group)})
    m['mean_cow_accuracy_tolerance_0.5']=float(np.mean([c['accuracy_all_tolerance_0.5'] for c in cows]))
    m['success_025']=m['accuracy_all_tolerance_0.25']>=.5
    m['success_050']=m['accuracy_all_tolerance_0.5']>=.75
    m['note']='Global: todas las fotos elegibles; abstenciones cuentan como no aciertos. MAE/F1/R2: solo clasificadas.'
    return m,cows


def plots(records, classes, output):
    from sklearn.metrics import confusion_matrix, classification_report
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    acc=[r for r in records if r['pred_bcs'] is not None]
    if not acc:return
    y=[classes.index(r['true_bcs']) for r in acc];p=[classes.index(r['pred_bcs']) for r in acc]
    cm=confusion_matrix(y,p,labels=list(range(len(classes))))
    report=classification_report(y,p,labels=list(range(len(classes))),target_names=[str(v) for v in classes],zero_division=0,output_dict=True)
    save_json(output/'classification_report.json',report)
    fig,ax=plt.subplots(figsize=(7,6));ax.imshow(cm,cmap='Blues')
    ax.set_xticks(range(len(classes)),classes,rotation=45);ax.set_yticks(range(len(classes)),classes)
    ax.set_xlabel('BCS predicho');ax.set_ylabel('BCS real')
    for i in range(len(classes)):
        for j in range(len(classes)):ax.text(j,i,str(cm[i,j]),ha='center',va='center')
    fig.tight_layout();fig.savefig(output/'matriz_confusion.png',dpi=150);plt.close(fig)
    real=np.array([r['true_bcs'] for r in acc]);pred=np.array([r['pred_bcs'] for r in acc])
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].scatter(real,pred)
    axes[0].plot([min(classes),max(classes)],[min(classes),max(classes)],'k--')
    axes[0].set(xlabel='BCS real',ylabel='BCS predicho')
    axes[1].hist(pred-real,bins=np.arange(-3,3.125,.25));axes[1].set(xlabel='Error predicho - real',ylabel='Fotos')
    fig.tight_layout();fig.savefig(output/'errores_bcs.png',dpi=150);plt.close(fig)


def evaluate(args):
    from .features import cache,device_for
    from .training import load_model,predict_features
    work,dcfg,data=manifest(args.work);fcfg,features=cache(args.features)
    require(args.split!='test' or args.ack_test,'Test ya fue observado antes: confirmar evaluación exploratoria con --ack-test')
    dev=device_for(args.device);saved,model=load_model(args.model,dev)
    require(saved['config']['manifest_sha256']==dcfg['manifest_sha256']==fcfg['manifest_sha256'],'Mezcla de datasets')
    require(saved['config']['classes']==dcfg['classes']==fcfg['classes'],'Catálogo incompatible')
    require(saved['config']['roi_mode']==fcfg['roi_mode'],'Modo ROI diferente al entrenamiento')
    output=Path(args.out).resolve();require(not output.exists(),'Salida existente: usa --out nuevo')
    chosen=[r for r in data if r['split']==args.split and r['status']=='ok']
    require(bool(chosen),'Sin imágenes elegibles')
    mask=features['splits']==args.split
    require((features['views'][mask]==0).all(),'No evaluar vistas aumentadas')
    keys=features['keys'][mask].tolist();require(len(keys)==len(set(keys)),'Características duplicadas')
    preds,expected,probs=predict_features(model,features['x'][mask],saved,dev)
    lookup={k:(float(v),float(e),p.tolist(),float(y)) for k,v,e,p,y in zip(keys,preds,expected,probs,features['y'][mask])}
    eligible={r['split']+'/'+r['file_name'] for r in chosen}
    require(set(lookup)<=eligible,'Caché contiene fotos no elegibles')
    records=[]
    for r in chosen:
        key=r['split']+'/'+r['file_name']
        status=fcfg['statuses'].get(key) if fcfg['kind']=='predicted' else 'ok'
        require(status is not None,'Falta estado de detección; no omitir fallos')
        require((key in lookup)==(status=='ok'),'Caché y estados de detección no coinciden')
        if key in lookup:require(lookup[key][3]==float(r['bcs']),'Etiqueta de caché alterada')
        pred,exp,pr=(lookup[key][:3] if key in lookup else (None,None,[]))
        records.append({'split':r['split'],'file_name':r['file_name'],'cow_id':r['cow_id'],
            'true_bcs':float(r['bcs']),'pred_bcs':pred,'expected_bcs':exp,'status':status,'probabilities':json.dumps(pr)})
    require(any(r['split']=='train' and r['status']=='ok' for r in data),'Falta train para baseline')
    baseline_cont=float(np.median([float(r['bcs']) for r in data if r['split']=='train' and r['status']=='ok']))
    baseline=min(dcfg['classes'],key=lambda c:abs(c-baseline_cont))
    output.mkdir(parents=True)
    result,cows=metrics(records,dcfg['classes'])
    result['continuous_bcs']=continuous_metrics(records)
    result['stage']='end_to_end_predicted_boxes' if fcfg['kind']=='predicted' else 'classifier_ground_truth_boxes'
    base=[{**r,'pred_bcs':baseline,'expected_bcs':baseline_cont,'status':'baseline'} for r in records]
    result['baseline_all']=metrics(base,dcfg['classes'])[0]
    result['continuous_baseline_all']=continuous_metrics(base)
    accepted=[{**r,'pred_bcs':baseline,'status':'baseline'} for r in records if r['pred_bcs'] is not None]
    result['baseline_same_classified_images']=metrics(accepted,dcfg['classes'])[0] if accepted else None
    result['baseline_value']=baseline
    save_json(output/'metricas.json',result);save_csv(output/'predicciones.csv',records);save_csv(output/'por_vaca.csv',cows)
    save_json(output/'evaluation_settings.json',{'split':args.split,'checkpoint':str(Path(args.model).resolve()),
        'features':str(Path(args.features).resolve()),'test_is_exploratory':args.split=='test',
        'decoding':'LDL: esperanza continua y categoría más cercana','note':'MAE continuo primario; métricas de categoría por separado. No comparar silenciosamente ambos.'})
    plots(records,dcfg['classes'],output)
    continuous_plot(records,output)
    write_summary(result,output)
    print(json.dumps(result,indent=2,ensure_ascii=False))


def predict(args):
    import torch
    from PIL import Image,ImageDraw
    from .features import encoder,embed_paths,device_for
    from .training import load_model,predict_features
    from .detection import get_detector,infer
    output=Path(args.out).resolve();require(not output.exists(),'Salida existente')
    dev=device_for(args.device);output.mkdir(parents=True)
    detector=get_detector(args.detector,args.device)
    info=infer(detector,args.image,output/'crop.png',224,args.imgsz,args.conf,args.device)
    with Image.open(args.image) as im:preview=im.convert('RGB')
    draw=ImageDraw.Draw(preview)
    for box in info['boxes']:draw.rectangle(box['xyxy'],outline='lime' if box['class']==0 else 'orange',width=4)
    preview.save(output/'detecciones.jpg')
    del detector
    if dev.type=='mps':torch.mps.empty_cache()
    result={'status':info['status'],'bcs':None,'expected_bcs':None,'note':'No identifica al animal; scores sin calibrar.'}
    if info['status']=='ok':
        saved,head=load_model(args.model,dev)
        roi=args.roi or saved['config']['default_roi']
        require(saved['config']['roi_mode']!='manual' or args.roi is not None,'Modelo manual: debes proporcionar --roi x1 y1 x2 y2 para esta foto')
        backbone=encoder(dev);z=embed_paths(backbone,[output/'crop.png'],dev,rois=[roi],mode=saved['config']['roi_mode'])
        del backbone
        if dev.type=='mps':torch.mps.empty_cache()
        p,e,probs=predict_features(head,z,saved,dev)
        from .regions import draw_roi
        with Image.open(output/'crop.png') as im:draw_roi(im,roi).save(output/'region_superior.png')
        result.update(roi=roi,bcs=float(p[0]),expected_bcs=float(e[0]),classes=saved['config']['classes'],probabilities=probs[0].tolist())
    save_json(output/'deteccion.json',info);save_json(output/'prediccion_bcs.json',result)
    print(json.dumps(result,indent=2,ensure_ascii=False))


def continuous_metrics(records):
    accepted=[r for r in records if r['expected_bcs'] is not None]
    errors=np.asarray([r['expected_bcs']-r['true_bcs'] for r in accepted])
    result={'eligible_images':len(records),'classified_images':len(accepted),
        'mae_classified':float(np.abs(errors).mean()) if len(errors) else None,
        'rmse_classified':float(np.sqrt((errors**2).mean())) if len(errors) else None}
    for tol in [.25,.5]:
        hits=int((np.abs(errors)<=tol+1e-8).sum())
        result[f'accuracy_all_tolerance_{tol}']=hits/len(records)
        result[f'accuracy_classified_tolerance_{tol}']=hits/len(accepted) if accepted else None
    result['success_025']=result['accuracy_all_tolerance_0.25']>=.5
    result['success_050']=result['accuracy_all_tolerance_0.5']>=.75
    return result


def write_summary(result,output):
    c=result['continuous_bcs']
    lines=['# Evaluación BCS LDL','',f"Etapa: {result['stage']}",
        f"Fotos elegibles: {result['eligible_images']}; clasificadas: {result['classified_images']}; abstenciones: {result['abstentions']}.",
        '', '| Métrica | BCS continuo (principal) | Categoría redondeada |','|---|---:|---:|']
    for k in ['mae_classified','rmse_classified','accuracy_all_tolerance_0.25','accuracy_all_tolerance_0.5']:
        lines.append(f"| {k} | {c[k]} | {result[k]} |")
    lines+=['','Tolerancias como proporción: 0.75 equivale a 75%. Abstenciones cuentan como no aciertos en métricas globales.',
        'Las comparaciones con ordinal usan categorías en ambos modelos. Las métricas continuas se presentan aparte.',
        'La validación se usa para seleccionar el modelo; no demuestra rendimiento independiente.']
    (output/'resumen.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')


def continuous_plot(records,output):
    accepted=[r for r in records if r['expected_bcs'] is not None]
    if not accepted:return
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    y=np.asarray([r['true_bcs'] for r in accepted]);p=np.asarray([r['expected_bcs'] for r in accepted])
    fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].scatter(y,p)
    low=min(y.min(),p.min());high=max(y.max(),p.max())
    axes[0].plot([low,high],[low,high],'k--');axes[0].set(xlabel='BCS real',ylabel='BCS continuo predicho')
    axes[1].hist(p-y,bins=12);axes[1].set(xlabel='Error continuo (BCS)',ylabel='Fotos')
    fig.tight_layout();fig.savefig(output/'errores_continuos.png',dpi=150);plt.close(fig)
