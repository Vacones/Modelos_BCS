#!/usr/bin/env python3
"""Genera regiones superiores y previsualizaciones; opcionalmente dibuja ROIs manuales."""
import argparse,json,math
from pathlib import Path
from PIL import Image
from bcs.data import manifest
from bcs.common import require,save_csv
from bcs.regions import DEFAULT_ROI,draw_roi,load_rois,validate_roi


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--work',required=True);p.add_argument('--out',required=True);p.add_argument('--boxes')
    p.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train','val'])
    p.add_argument('--interactive',action='store_true');p.add_argument('--from-csv')
    a=p.parse_args();_,cfg,rows=manifest(a.work);out=Path(a.out).resolve()
    require(not out.exists(),'Usa un --out nuevo para conservar tus anotaciones')
    previous=load_rois(a.from_csv);kind='predicted' if a.boxes else 'annotated';pred={}
    if a.boxes:
        folder=Path(a.boxes)
        for line in (folder/'predictions.jsonl').read_text().splitlines():
            r=json.loads(line);pred[r['split']+'/'+r['file_name']]=r
    jobs=[]
    for r in rows:
        if r['status']!='ok' or r['split'] not in a.splits:continue
        key=r['split']+'/'+r['file_name'];path=r['crop_path']
        if a.boxes:
            require(key in pred,'Falta detección: '+key)
            if pred[key]['status']!='ok':continue
            path=folder/pred[key]['crop_path']
        jobs.append((kind+'/'+key,path))
    require(bool(jobs),'No hay recortes para revisar')
    import matplotlib
    if not a.interactive:matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.widgets import RectangleSelector
    accepted=[];preview=out.with_suffix('');preview.mkdir(parents=True,exist_ok=True)
    for key,path in jobs:
        roi=previous.get(key,DEFAULT_ROI)
        if a.interactive:
            with Image.open(path) as im:image=im.convert('RGB')
            selected=[roi];decision=[None]
            fig,ax=plt.subplots();ax.imshow(image);ax.set_title(key+'\nArrastra ROI; Enter guarda; Q termina')
            def choose(click,release):
                x1,x2=sorted([click.xdata,release.xdata]);y1,y2=sorted([click.ydata,release.ydata])
                selected[0]=validate_roi([max(0,x1/224),max(0,y1/224),min(1,x2/224),min(1,y2/224)])
            selector=RectangleSelector(ax,choose,useblit=True,button=[1],minspanx=4,minspany=4,spancoords='pixels',interactive=True)
            selector.extents=(roi[0]*224,roi[2]*224,roi[1]*224,roi[3]*224)
            def keypress(event):
                if event.key in ['enter','q']:
                    decision[0]=event.key;plt.close(fig)
            fig.canvas.mpl_connect('key_press_event',keypress);plt.show()
            if decision[0]!='enter':break
            roi=selected[0]
        accepted.append({'key':key,**dict(zip(['x1','y1','x2','y2'],roi))})
        # Guardado incremental para no perder anotaciones ante una interrupción.
        save_csv(out,accepted)
    for offset in range(0,len(accepted),12):
        fig,axes=plt.subplots(3,4,figsize=(12,9))
        for ax in axes.flat:ax.axis('off')
        paths=dict(jobs)
        for ax,item in zip(axes.flat,accepted[offset:offset+12]):
            with Image.open(paths[item['key']]) as im:
                ax.imshow(draw_roi(im,[item[c] for c in ['x1','y1','x2','y2']]))
            ax.set_title(item['key'].split('/')[1]+' | '+item['key'].split('/')[-1][:18],fontsize=8)
        fig.suptitle('Cian: ROI de entrada. Aproximación geométrica o marca manual; NO detección de huesos.')
        fig.tight_layout();fig.savefig(preview/f'pagina_{offset//12+1:03}.png',dpi=140);plt.close(fig)
    print(f'Regiones guardadas: {len(accepted)}/{len(jobs)}; archivo: {out}; vistas: {preview}')


if __name__=='__main__':main()
