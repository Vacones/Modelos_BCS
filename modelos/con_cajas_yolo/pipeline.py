#!/usr/bin/env python3
"""Pipeline local manual. Usa --help o SUBCOMANDO --help."""
import argparse


def main():
    p=argparse.ArgumentParser(description=__doc__);s=p.add_subparsers(dest='command',required=True)
    q=s.add_parser('etl');q.add_argument('--data',required=True);q.add_argument('--work',required=True);q.add_argument('--size',type=int,default=224)
    q=s.add_parser('eda');q.add_argument('--work',required=True)
    q=s.add_parser('train');q.add_argument('--work',required=True);q.add_argument('--run',required=True)
    q.add_argument('--phase',choices=['head','finetune'],default='head');q.add_argument('--init')
    q.add_argument('--weights',choices=['imagenet','none'],default='imagenet',help='none solo para prueba técnica, no para entrenamiento final')
    q.add_argument('--epochs',type=int,default=30);q.add_argument('--batch',type=int,default=4)
    q.add_argument('--lr',type=float,default=1e-3);q.add_argument('--fine-layers',type=int,default=20);q.add_argument('--patience',type=int,default=8)
    q.add_argument('--resume',action='store_true');q.add_argument('--mixed',action='store_true');q.add_argument('--allow-cpu',action='store_true')
    q=s.add_parser('evaluate');q.add_argument('--work',required=True);q.add_argument('--model',required=True);q.add_argument('--out',required=True)
    q.add_argument('--split',choices=['val','test'],default='test');q.add_argument('--boxes');q.add_argument('--batch',type=int,default=4);q.add_argument('--allow-cpu',action='store_true')
    q=s.add_parser('predict');q.add_argument('--model',required=True);q.add_argument('--bundle',required=True);q.add_argument('--allow-cpu',action='store_true')
    q=s.add_parser('detector-train');q.add_argument('--work',required=True);q.add_argument('--run',default='runs/detector');q.add_argument('--weights',default='yolo26n.pt')
    q.add_argument('--epochs',type=int,default=80);q.add_argument('--batch',type=int,default=4);q.add_argument('--imgsz',type=int,default=640);q.add_argument('--patience',type=int,default=15);q.add_argument('--device',default='0');q.add_argument('--resume')
    q=s.add_parser('detector-test');q.add_argument('--work',required=True);q.add_argument('--weights',required=True);q.add_argument('--out',required=True);q.add_argument('--split',choices=['val','test'],default='test');q.add_argument('--device',default='0');q.add_argument('--imgsz',type=int,default=640)
    q=s.add_parser('detector-crops');q.add_argument('--work',required=True);q.add_argument('--weights',required=True);q.add_argument('--out',required=True);q.add_argument('--splits',nargs='+',choices=['train','val','test'],default=['val','test'])
    q.add_argument('--device',default='0');q.add_argument('--imgsz',type=int,default=640);q.add_argument('--conf',type=float,default=.25)
    q=s.add_parser('detect-image');q.add_argument('--image',required=True);q.add_argument('--weights',required=True);q.add_argument('--out',required=True);q.add_argument('--size',type=int,default=224);q.add_argument('--device',default='0');q.add_argument('--imgsz',type=int,default=640);q.add_argument('--conf',type=float,default=.25)
    args=p.parse_args()
    if args.command in ('etl','eda'):
        from vacas import data
        getattr(data,args.command)(args)
    elif args.command=='train':
        from vacas.tf_model import train
        train(args)
    elif args.command in ('evaluate','predict'):
        from vacas import evaluation
        getattr(evaluation,args.command)(args)
    else:
        from vacas import detection
        funcs={'detector-train':detection.train_detector,'detector-test':detection.detector_test,
               'detector-crops':detection.predicted_crops,'detect-image':detection.detect_image}
        funcs[args.command](args)


if __name__=='__main__':main()
