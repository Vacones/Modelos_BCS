"""DINOv2 oficial + paso LDL TensorFlow + oclusión en CPU con imagen sintética."""
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import torch
from PIL import Image,ImageDraw
from bcs.features import encoder,embed_images,DIM
from bcs.regions import DEFAULT_ROI
from bcs.tf_model import tf,configure_tf,build_head,LDLLoss
from bcs.explain import occlusion


def main():
    torch.set_num_threads(2);configure_tf('cpu');tf.keras.utils.set_random_seed(42)
    image=Image.new('RGB',(224,224),(60,100,140));ImageDraw.Draw(image).ellipse([20,10,190,145],fill=(170,130,90))
    net=encoder(torch.device('cpu'))
    z=embed_images(net,[image],[DEFAULT_ROI],torch.device('cpu'))
    assert z.shape==(1,DIM) and np.isfinite(z).all()
    classes=[2.25,2.5,2.75,3.,3.25,3.5,3.75];model=build_head(DIM,classes)
    optimizer=tf.keras.optimizers.AdamW(.001)
    with tf.GradientTape() as tape:loss=tf.reduce_mean(LDLLoss(classes)(tf.constant([3]),model(z,training=True)))
    optimizer.apply_gradients(zip(tape.gradient(loss,model.trainable_variables),model.trainable_variables))
    saved={'mean':np.zeros(DIM,np.float32),'std':np.ones(DIM,np.float32),'config':{'classes':classes}}
    info,heat=occlusion(image,DEFAULT_ROI,'upper',net,model,saved,torch.device('cpu'),grid=2,batch=2)
    assert heat.shape==(224,224) and np.isfinite(heat).all() and len(info['cells'])==4
    assert abs(sum(info['probabilities'])-1)<1e-5
    print('OK: DINOv2 real congelado, dos vistas, gradientes TensorFlow LDL y oclusión integrada (CPU).')
    print('No es entrenamiento ni evaluación científica BCS.')


if __name__=='__main__':main()
