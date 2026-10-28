from pathlib import Path
import numpy as np
from .common import require, save_json, load_json, save_csv
from .data import manifest


def metrics_and_plots(records,classes,output,stage):
    from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, mean_absolute_error, cohen_kappa_score
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    output=Path(output);output.mkdir(parents=True,exist_ok=True)
    require(bool(records),'No hay registros evaluables')
    accepted=[r for r in records if r['status']=='ok']
    total=len(records)
    result={'stage':stage,'eligible_images':total,'classified_images':len(accepted),'coverage':len(accepted)/total,
            'abstentions':total-len(accepted),
            'note':'Fallos de detección/asociación cuentan como no aciertos en métricas sobre todas las imágenes.'}
    for tolerance in (0,.25,.5):
        hits=sum(abs(float(r['true_bcs'])-float(r['pred_bcs']))<=tolerance+1e-8 for r in accepted)
        result[f'accuracy_all_tolerance_{tolerance}']=hits/total
    if accepted:
        true=np.array([classes.index(float(r['true_bcs'])) for r in accepted]);pred=np.array([classes.index(float(r['pred_bcs'])) for r in accepted])
        t=np.array([float(r['true_bcs']) for r in accepted]);p=np.array([float(r['pred_bcs']) for r in accepted])
        report=classification_report(true,pred,labels=list(range(len(classes))),target_names=[str(c) for c in classes],zero_division=0,output_dict=True)
        result.update(accuracy_classified=float(accuracy_score(true,pred)),
            macro_f1_all_catalog=float(f1_score(true,pred,labels=list(range(len(classes))),average='macro',zero_division=0)),
            mae_classified=float(mean_absolute_error(t,p)),rmse_classified=float(np.sqrt(np.mean((t-p)**2))))
        kappa=cohen_kappa_score(true,pred,labels=list(range(len(classes))),weights='quadratic') if len(set(true))>1 else float('nan')
        result['quadratic_kappa_classified']=float(kappa) if np.isfinite(kappa) else None
        save_json(output/'classification_report.json',report)
        cm=confusion_matrix(true,pred,labels=list(range(len(classes))))
        fig,ax=plt.subplots(figsize=(8,7));im=ax.imshow(cm,cmap='Blues');fig.colorbar(im,ax=ax)
        ax.set_xticks(range(len(classes)),[str(c) for c in classes],rotation=45);ax.set_yticks(range(len(classes)),[str(c) for c in classes]);ax.set_xlabel('Predicción');ax.set_ylabel('Referencia')
        for i in range(len(classes)):
            for j in range(len(classes)):ax.text(j,i,str(cm[i,j]),ha='center',va='center',color='white' if cm[i,j]>cm.max()/2 else 'black')
        fig.tight_layout();fig.savefig(output/'matriz_confusion.png',dpi=160);plt.close(fig)
        fig,axes=plt.subplots(1,2,figsize=(10,4));axes[0].scatter(t,p,alpha=.5);axes[0].plot([min(classes),max(classes)],[min(classes),max(classes)],'k--');axes[0].set_xlabel('BCS referencia');axes[0].set_ylabel('BCS predicho');axes[1].hist(p-t,bins=np.arange(-3,3.125,.25));axes[1].set_xlabel('Error BCS');axes[1].set_ylabel('Imágenes');fig.tight_layout();fig.savefig(output/'errores_bcs.png',dpi=160);plt.close(fig)
    else:
        result.update(accuracy_classified=None,macro_f1_all_catalog=None,mae_classified=None,rmse_classified=None)
    cow_rows=[]
    for cow in sorted({r['cow_id'] for r in records}):
        group=[r for r in records if r['cow_id']==cow]
        cow_rows.append({'cow_id':cow,'images':len(group),'coverage':sum(r['status']=='ok' for r in group)/len(group),
            'accuracy_all_tolerance_0.5':sum(r['status']=='ok' and abs(float(r['true_bcs'])-float(r['pred_bcs']))<=.5+1e-8 for r in group)/len(group)})
    result['mean_cow_accuracy_tolerance_0.5']=float(np.mean([r['accuracy_all_tolerance_0.5'] for r in cow_rows]))
    save_csv(output/'por_vaca.csv',cow_rows);save_csv(output/'predicciones.csv',records);save_json(output/'metricas.json',result)
    return result


def evaluate(args):
    from .tf_model import tensorflow_ready,dataset
    tf=tensorflow_ready(args.allow_cpu)
    work,cfg,data=manifest(args.work);model_path=Path(args.model).resolve();settings=load_json(model_path.parent/'settings.json')
    require(settings['classes']==cfg['classes'] and settings['size']==cfg['size'] and settings['manifest_sha256']==cfg['manifest_sha256'],'Modelo no corresponde al ETL')
    output=Path(args.out).resolve();require(not output.exists(),'La carpeta de evaluación ya existe; usa --out nuevo')
    selected=[r for r in data if r['split']==args.split and r['status']=='ok']
    require(bool(selected),'No hay referencias válidas para esta partición')
    predicted={}
    if args.boxes:
        folder=Path(args.boxes).resolve(); pconfig=load_json(folder/'settings.json')
        require(pconfig['manifest_sha256']==cfg['manifest_sha256'] and pconfig['size']==cfg['size'],'Cajas predichas de otro conjunto o tamaño')
        for line in (folder/'predictions.jsonl').read_text().splitlines():
            r=__import__('json').loads(line);key=(r['split'],r['file_name']);require(key not in predicted,'Predicciones duplicadas');predicted[key]=r
    prepared=[];result=[]
    for r in selected:
        record={'split':r['split'],'file_name':r['file_name'],'cow_id':r['cow_id'],'true_bcs':r['bcs'],'pred_bcs':'','confidence':'','status':'ok'}
        if args.boxes:
            p=predicted.get((r['split'],r['file_name']))
            require(p is not None,'Falta el registro de una foto en predictions.jsonl; no se permite omitir fallos')
            record['status']=p['status']
            if p['status']=='ok':prepared.append({**r,'crop_path':str((folder/p['crop_path']).resolve())})
        else:prepared.append(r)
        result.append(record)
    model=tf.keras.models.load_model(model_path)
    if prepared:
        ds=dataset(tf,prepared,cfg['classes'],cfg['size'],args.batch)
        probabilities=model.predict(ds.map(lambda x,y:x),verbose=0)
        lookup={(r['split'],r['file_name']):p for r,p in zip(prepared,probabilities)}
        for row in result:
            if row['status']!='ok':continue
            p=lookup[(row['split'],row['file_name'])];row['pred_bcs']=cfg['classes'][int(np.argmax(p))];row['confidence']=float(np.max(p))
    import json
    metrics=metrics_and_plots(result,cfg['classes'],output,'end_to_end_predicted_boxes' if args.boxes else 'classifier_ground_truth_boxes')
    save_json(output/'evaluation_settings.json',{'model':str(model_path),'split':args.split,'boxes':str(Path(args.boxes).resolve()) if args.boxes else None,
        'classes':cfg['classes'],'excluded_ambiguous_metadata':sum(r['split']==args.split and r['status']!='ok' for r in data)})
    print(json.dumps(metrics,indent=2))


def predict(args):
    from .tf_model import tensorflow_ready
    from PIL import Image
    tf=tensorflow_ready(args.allow_cpu);model_path=Path(args.model).resolve()
    settings=load_json(model_path.parent/'settings.json');bundle=Path(args.bundle).resolve();info=load_json(bundle/'prediction.json')
    require(info['size']==settings['size'],'Tamaño de recorte incompatible con el modelo')
    if info['status']!='ok':print('Sin clasificación:',info['status']);return
    model=tf.keras.models.load_model(model_path)
    with Image.open(bundle/info['crop_path']) as im:x=np.asarray(im.convert('RGB'),dtype=np.float32)
    p=model.predict(x[None],verbose=0)[0]
    result={'bcs':settings['classes'][int(np.argmax(p))],'score':float(np.max(p)),
        'scores':{str(c):float(v) for c,v in zip(settings['classes'],p)},'note':'Score softmax sin calibrar; no identifica a la vaca.'}
    save_json(bundle/'bcs_prediction.json',result);print(__import__('json').dumps(result,indent=2))
