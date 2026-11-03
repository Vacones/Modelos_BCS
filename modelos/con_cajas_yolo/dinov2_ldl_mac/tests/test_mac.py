"""Pruebas CPU con TensorFlow real; no son métricas de vacas."""
import unittest,tempfile,contextlib,io,json
from pathlib import Path
from types import SimpleNamespace as NS
import numpy as np
import torch
from PIL import Image
from bcs.common import save_json,load_json
from bcs.raw_coco import normalize
from bcs.data import etl,manifest
from bcs.features import extract,DIM,cache
from bcs.tf_model import tf,configure_tf,build_head,soft_targets,point_losses,decode_numpy,LDLLoss
from bcs.training import train,load_model,predict_features,model_fingerprint
from bcs.evaluation import evaluate
from bcs.explain import explain


def fixture(root):
    counter=0
    for split,values in [('train',[2.5,2.75,3.]),('valid',[2.5,3.]),('test',[2.75])]:
        folder=root/split;folder.mkdir(parents=True);images=[];anns=[]
        for i,value in enumerate(values):
            name=f'foto_{counter}.png'
            Image.new('RGB',(80,100),(30+counter*20,50,150-counter*15)).save(folder/name)
            images.append({'id':i,'file_name':name,'width':80,'height':100,
                'extra':{'user_tags':[str(100+counter),'BCS_'+str(value).replace('.','_')]}})
            anns.extend([{'id':2*i,'image_id':i,'category_id':1,'bbox':[0,0,80,100]},
                         {'id':2*i+1,'image_id':i,'category_id':2,'bbox':[10,20,50,40]}]);counter+=1
        save_json(folder/'_annotations.coco.json',{'images':images,'annotations':anns,
            'categories':[{'id':1,'name':'cow'},{'id':2,'name':'ass'}]})


class FakeEncoder(torch.nn.Module):
    def forward_features(self,x):
        z=x.mean((2,3)).repeat(1,128)
        return {'x_norm_clstoken':z,'x_norm_patchtokens':z[:,None,:].repeat(1,4,1)}


class MacTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):configure_tf('cpu');torch.set_num_threads(2)

    def test_tf_ldl_math_and_serialization(self):
        classes=[2.5,2.75,3.];q=soft_targets([0,1,2],classes).numpy()
        np.testing.assert_allclose(q.sum(-1),1,atol=1e-6)
        self.assertEqual(int(q[1].argmax()),1)
        logits=np.asarray([[.3,-.2,.7],[1.,0.,-1.],[-.5,.5,.2]],np.float32)
        p=np.exp(logits-logits.max(-1,keepdims=True));p/=p.sum(-1,keepdims=True)
        expected=-(q*np.log(p)).sum(-1)+.5*((p.cumsum(-1)[:,:-1]-q.cumsum(-1)[:,:-1])**2).mean(-1)
        np.testing.assert_allclose(point_losses([0,1,2],logits,classes).numpy(),expected,rtol=1e-5)
        cat,mean,prob=decode_numpy(np.log([[.1,.6,.3]]),classes)
        self.assertAlmostEqual(float(mean[0]),2.8,places=5);self.assertEqual(float(cat[0]),2.75)
        model=build_head(4,classes,.3);model.compile(optimizer=tf.keras.optimizers.AdamW(.001),loss=LDLLoss(classes))
        x=tf.ones((3,4));y=tf.constant([0,1,2])
        with tf.GradientTape() as tape:loss=tf.reduce_mean(model.loss(y,model(x,training=True)))
        gradients=tape.gradient(loss,model.trainable_variables)
        self.assertTrue(all(g is not None for g in gradients))
        model.optimizer.apply_gradients(zip(gradients,model.trainable_variables))
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'head.keras';model.save(path)
            restored=tf.keras.models.load_model(path)
            np.testing.assert_allclose(model(x,training=False).numpy(),restored(x,training=False).numpy(),atol=1e-6)
            self.assertEqual(int(model.optimizer.iterations.numpy()),int(restored.optimizer.iterations.numpy()))

    def test_etl_features_tf_train_resume_evaluate_explain(self):
        with tempfile.TemporaryDirectory() as temp,contextlib.redirect_stdout(io.StringIO()):
            root=Path(temp);fixture(root/'raw');converted=normalize(root/'raw',root/'converted')
            etl(NS(data=converted,work=root/'work',size=224));work,cfg,rows=manifest(root/'work')
            f=NS(work=work,out=root/'features',splits=['train','val'],boxes=None,views=2,batch=2,
                seed=42,device='cpu',roi_mode='upper',roi_csv=None)
            extract(f,lambda d:FakeEncoder())
            _,features=cache(f.out);self.assertEqual(features['x'].shape,(8,DIM))
            t=NS(features=f.out,run=root/'run',epochs=2,batch=2,lr=.001,dropout=.3,weight_decay=.0001,
                 patience=0,seed=42,device='cpu',resume=False,sigma=.125,emd_weight=.5)
            train(t);t.epochs=3;t.resume=True;train(t)
            saved,model=load_model(t.run);self.assertEqual(saved['epoch'],3)
            self.assertTrue((t.run/saved['last_checkpoint']).is_file())
            self.assertLessEqual(len(list((t.run/'checkpoints').glob('*.keras'))),2)
            self.assertEqual(len(model_fingerprint(t.run)),64)
            p,e,pr=predict_features(model,features['x'],saved)
            self.assertTrue(np.isfinite(e).all());np.testing.assert_allclose(pr.sum(-1),1,atol=1e-6)
            a=NS(work=work,features=f.out,model=t.run,out=root/'evaluation',split='val',device='cpu',ack_test=False)
            evaluate(a);self.assertEqual(load_json(a.out/'metricas.json')['coverage'],1)
            ex=NS(**vars(a));ex.out=root/'explain';ex.boxes=None;ex.limit=1;ex.grid=2;ex.batch=2
            explain(ex,lambda d:FakeEncoder())
            self.assertEqual(load_json(ex.out/'metodo.json')['explained'],1)
            a.split='test';a.out=root/'test'
            with self.assertRaisesRegex(ValueError,'Test'):evaluate(a)
            boxes=root/'boxes';boxes.mkdir()
            save_json(boxes/'settings.json',{'manifest_sha256':cfg['manifest_sha256'],'size':224})
            with (boxes/'predictions.jsonl').open('w') as out:
                for r in rows:
                    if r['split']=='val':out.write(json.dumps({'split':'val','file_name':r['file_name'],
                        'status':'sin_par_valido','crop_path':''})+'\n')
            f.out=root/'empty';f.boxes=boxes;f.splits=['val'];f.views=1;extract(f,lambda d:FakeEncoder())
            a.features=f.out;a.split='val';a.out=root/'all_failed';evaluate(a)
            self.assertEqual(load_json(a.out/'metricas.json')['coverage'],0)


if __name__=='__main__':unittest.main()
