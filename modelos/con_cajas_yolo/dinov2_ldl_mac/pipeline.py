#!/usr/bin/env python3
"""Pipeline Mac: DINOv2/MPS y cabeza LDL TensorFlow/Metal."""
import argparse
from pathlib import Path


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    q=sub.add_parser('etl');q.add_argument('--data',required=True);q.add_argument('--work',required=True);q.add_argument('--size',type=int,default=224)
    q=sub.add_parser('eda');q.add_argument('--work',required=True)
    q=sub.add_parser('features');q.add_argument('--work',required=True);q.add_argument('--out',required=True)
    q.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['train','val'])
    q.add_argument('--boxes');q.add_argument('--views',type=int,default=1);q.add_argument('--batch',type=int,default=4)
    q.add_argument('--seed',type=int,default=42);q.add_argument('--device',default='mps')
    q.add_argument('--roi-mode',choices=['upper','manual','full'],default='upper');q.add_argument('--roi-csv')
    q=sub.add_parser('train');q.add_argument('--features',required=True);q.add_argument('--run',required=True)
    q.add_argument('--epochs',type=int,default=100);q.add_argument('--batch',type=int,default=32)
    q.add_argument('--lr',type=float,default=1e-3);q.add_argument('--weight-decay',type=float,default=1e-4)
    q.add_argument('--dropout',type=float,default=.30);q.add_argument('--patience',type=int,default=15)
    q.add_argument('--seed',type=int,default=42);q.add_argument('--device',default='mps');q.add_argument('--resume',action='store_true')
    q.add_argument('--sigma',type=float,default=.125);q.add_argument('--emd-weight',type=float,default=.5)
    q=sub.add_parser('evaluate');q.add_argument('--work',required=True);q.add_argument('--features',required=True)
    q.add_argument('--model',required=True);q.add_argument('--out',required=True);q.add_argument('--split',choices=['val','test'],default='val')
    q.add_argument('--device',default='mps');q.add_argument('--ack-test',action='store_true')
    q=sub.add_parser('predict');q.add_argument('--image',required=True);q.add_argument('--detector',required=True)
    q.add_argument('--model',required=True);q.add_argument('--out',required=True);q.add_argument('--device',default='mps')
    q.add_argument('--imgsz',type=int,default=640);q.add_argument('--conf',type=float,default=.25)
    q.add_argument('--roi',nargs=4,type=float)
    q=sub.add_parser('explain');q.add_argument('--work',required=True);q.add_argument('--features',required=True)
    q.add_argument('--model',required=True);q.add_argument('--out',required=True);q.add_argument('--boxes')
    q.add_argument('--split',choices=['val','test'],default='val');q.add_argument('--limit',type=int,default=12)
    q.add_argument('--grid',type=int,default=7);q.add_argument('--batch',type=int,default=2);q.add_argument('--device',default='mps')
    q.add_argument('--ack-test',action='store_true')
    q=sub.add_parser('detector-train');q.add_argument('--work',required=True);q.add_argument('--run',required=True)
    q.add_argument('--weights',default='yolo26n.pt');q.add_argument('--epochs',type=int,default=80);q.add_argument('--batch',type=int,default=4)
    q.add_argument('--imgsz',type=int,default=640);q.add_argument('--patience',type=int,default=15);q.add_argument('--device',default='mps');q.add_argument('--resume')
    q=sub.add_parser('detector-test');q.add_argument('--work',required=True);q.add_argument('--weights',required=True)
    q.add_argument('--out',required=True);q.add_argument('--split',choices=['val','test'],default='val')
    q.add_argument('--imgsz',type=int,default=640);q.add_argument('--device',default='mps')
    q=sub.add_parser('detector-crops');q.add_argument('--work',required=True);q.add_argument('--weights',required=True)
    q.add_argument('--out',required=True);q.add_argument('--splits',nargs='+',choices=['val','test'],default=['val'])
    q.add_argument('--imgsz',type=int,default=640);q.add_argument('--conf',type=float,default=.25);q.add_argument('--device',default='mps')
    args=parser.parse_args()
    if args.command=='etl':
        from bcs.common import require
        from bcs import data
        require(args.size==224,'DINOv2 de esta entrega espera 224; conserva --size 224')
        require(not Path(args.work).exists(),'La salida ETL ya existe; usa otra carpeta')
        original=Path(args.data).resolve()
        if not (original/'metadata.csv').is_file():
            from bcs.raw_coco import normalize
            work=Path(args.work).resolve()
            args.data=str(normalize(original,work.with_name(work.name+'_entrada_coco')))
        data.etl(args)
    elif args.command=='eda':
        from bcs.data import eda
        eda(args)
    elif args.command=='features':
        from bcs.features import extract
        extract(args)
    elif args.command=='explain':
        from bcs.explain import explain
        explain(args)
    elif args.command=='train':
        from bcs.training import train
        train(args)
    elif args.command in ('evaluate','predict'):
        from bcs import evaluation
        getattr(evaluation,args.command)(args)
    else:
        from bcs import detection
        {'detector-train':detection.train_detector,'detector-crops':detection.predicted_crops,
         'detector-test':detection.detector_test}[args.command](args)


if __name__=='__main__':main()
