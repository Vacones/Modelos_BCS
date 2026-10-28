"""Pruebas sin TensorFlow ni GPU: python -m unittest tests.test_core -v"""
import unittest
import tempfile
from pathlib import Path
from PIL import Image
from vacas.common import crop_image,read_yolo
from vacas.detection import pair_boxes
from vacas.evaluation import metrics_and_plots


class CoreTests(unittest.TestCase):
    def test_association_rejects_ambiguity(self):
        cow={'class':0,'xyxy':[0,0,100,100]};ass={'class':1,'xyxy':[10,10,90,50]}
        self.assertEqual(pair_boxes([cow,ass])[0],'ok')
        self.assertEqual(pair_boxes([cow,cow,ass])[0],'asociacion_ambigua')
        self.assertEqual(pair_boxes([cow])[0],'sin_par_valido')

    def test_yolo_and_letterbox(self):
        with tempfile.TemporaryDirectory() as temp:
            path=Path(temp)/'box.txt';path.write_text('1 0.5 0.5 0.5 0.5\n')
            self.assertEqual(read_yolo(path,100,200)[0]['xyxy'],[25.,50.,75.,150.])
            path.write_text('1 0.5 0.5 3 2\n')
            with self.assertRaises(ValueError):read_yolo(path,100,200)
        result=crop_image(Image.new('RGB',(100,200)),[0,0,100,200],224)
        self.assertEqual(result.size,(224,224))

    def test_missing_detection_counts_as_failure(self):
        records=[{'cow_id':'1','true_bcs':2.5,'pred_bcs':2.5,'status':'ok'},
                 {'cow_id':'2','true_bcs':3.,'pred_bcs':'','status':'sin_par_valido'}]
        with tempfile.TemporaryDirectory() as temp:
            m=metrics_and_plots(records,[2.5,3.],temp,'end_to_end')
            self.assertEqual(m['coverage'],.5)
            self.assertEqual(m['accuracy_all_tolerance_0.5'],.5)
            self.assertEqual(m['accuracy_classified'],1.)
        with tempfile.TemporaryDirectory() as temp:
            m=metrics_and_plots([{**r,'status':'sin_par_valido','pred_bcs':''} for r in records],[2.5,3.],temp,'end_to_end')
            self.assertIsNone(m['mae_classified'])
            self.assertEqual(m['coverage'],0.)


if __name__=='__main__':unittest.main()
