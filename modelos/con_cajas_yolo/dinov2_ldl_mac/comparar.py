#!/usr/bin/env python3
"""Comparación pareada en todas las fotos y en la intersección clasificada."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np
from bcs.common import require,save_json,save_csv
from bcs.evaluation import metrics


def load(path):
    with Path(path).open() as f: rows=list(csv.DictReader(f))
    found={}
    for r in rows:
        key=(r['split'],r['file_name'])
        require(key not in found,'Fotos duplicadas')
        r['true_bcs']=float(r['true_bcs'])
        r['pred_bcs']=float(r['pred_bcs']) if r['pred_bcs'] else None
        require(np.isfinite(r['true_bcs']) and (r['pred_bcs'] is None or np.isfinite(r['pred_bcs'])),'BCS no finito')
        require((r['pred_bcs'] is not None)==(r['status']=='ok'),'Predicción/estado incompatible')
        found[key]=r
    require(bool(found),'CSV vacío')
    return found


def compare(old,new,classes):
    require(set(old)==set(new),'No coinciden las fotos o particiones; no es una comparación válida')
    for key in old:
        require(old[key]['true_bcs']==new[key]['true_bcs'],'Cambió la etiqueta real')
    common=[k for k in old if old[k]['pred_bcs'] is not None and new[k]['pred_bcs'] is not None]
    # Identidad añadida desde el nuevo manifest, nunca inferida del nombre.
    for k in old:old[k]['cow_id']=new[k]['cow_id']
    om=metrics(list(old.values()),classes)[0];nm=metrics(list(new.values()),classes)[0]
    same_status=all(old[k]['status']==new[k]['status'] for k in old)
    paired={}
    if common:
        a=metrics([old[k] for k in common],classes)[0];b=metrics([new[k] for k in common],classes)[0]
        paired={'n':len(common),'ordinal':a,'dinov2':b,'delta_mae':b['mae_classified']-a['mae_classified']}
    rows=[]
    for key in ['coverage','mae_classified','rmse_classified','accuracy_all_tolerance_0','accuracy_all_tolerance_0.25','accuracy_all_tolerance_0.5']:
        a,b=om[key],nm[key];rows.append({'metric':key,'ordinal_v4':a,'dinov2':b,'delta_dino_menos_ordinal':b-a if a is not None and b is not None else None})
    # Bootstrap pareado por vaca, no por foto: conserva dependencia entre fotos.
    rng=np.random.default_rng(42);cows=sorted({r['cow_id'] for r in new.values()});ci={}
    for tol in [.25,.5]:
        diffs=[];counts=[]
        for cow in cows:
            keys=[k for k in new if new[k]['cow_id']==cow]
            hit=lambda r:r['pred_bcs'] is not None and abs(r['pred_bcs']-r['true_bcs'])<=tol+1e-8
            diffs.append(sum(int(hit(new[k]))-int(hit(old[k])) for k in keys));counts.append(len(keys))
        ix=rng.integers(0,len(cows),(2000,len(cows)))
        draws=np.array(diffs)[ix].sum(1)/np.array(counts)[ix].sum(1)
        ci[str(tol)]={'delta':float(sum(diffs)/sum(counts)),'ci95_cluster_cow':np.quantile(draws,[.025,.975]).tolist()}
    return {'ordinal':om,'dinov2':nm,'same_detection_status':same_status,'same_classified_images':paired,
        'bootstrap_global_tolerances':ci,'table':rows,'note':'Validación exploratoria. Mismo estado no demuestra mismos píxeles: reutiliza detector y recortes originales. Los intervalos no corrigen selección reiterada por validación.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--ordinal',required=True);p.add_argument('--dinov2',required=True)
    p.add_argument('--work',required=True);p.add_argument('--out',required=True);args=p.parse_args()
    from bcs.data import manifest
    _,cfg,rows=manifest(args.work);old,new=load(args.ordinal),load(args.dinov2)
    lookup={(r['split'],r['file_name']):r for r in rows if r['status']=='ok'}
    require(set(new)<=set(lookup),'Predicciones fuera del manifest')
    for k,r in new.items():
        require(r['true_bcs']==float(lookup[k]['bcs']),'BCS no coincide con manifest')
        r['cow_id']=lookup[k]['cow_id']
    result=compare(old,new,cfg['classes']);out=Path(args.out);require(not out.exists(),'Salida existente');out.mkdir(parents=True)
    save_json(out/'comparacion.json',result);save_csv(out/'comparacion.csv',result['table'])
    lines=['# Comparación con ordinal v4','', '| Métrica | Ordinal v4 | DINOv2 | Diferencia |','|---|---:|---:|---:|']
    for r in result['table']:
        fmt=lambda x:'—' if x is None else f'{x:.6f}'
        lines.append('| '+' | '.join([r['metric']]+[fmt(r[k]) for k in ['ordinal_v4','dinov2','delta_dino_menos_ordinal']])+' |')
    lines.extend(['',result['note'],'','MAE menor es mejor; exactitud y cobertura mayores son mejores.',
        'Cumple ±0.25: '+str(result['dinov2']['success_025'])+'; cumple ±0.50: '+str(result['dinov2']['success_050']),
        'Estados de detección idénticos: '+str(result['same_detection_status']),
        'Si cambió el detector/recorte, la diferencia global corresponde al sistema completo; consulta también la intersección clasificada.'])
    (out/'comparacion.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':main()
