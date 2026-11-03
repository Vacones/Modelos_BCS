import unittest
from comparar import compare
class ComparisonTests(unittest.TestCase):
 def test_pairs_and_abstentions(self):
  old={('val','a'):{'true_bcs':2.5,'pred_bcs':2.5,'status':'ok'},('val','b'):{'true_bcs':3.,'pred_bcs':None,'status':'sin_par_valido'}}
  new={('val','a'):{'true_bcs':2.5,'pred_bcs':2.75,'status':'ok','cow_id':'1'},('val','b'):{'true_bcs':3.,'pred_bcs':3.,'status':'ok','cow_id':'2'}}
  r=compare(old,new,[2.5,2.75,3.]);self.assertFalse(r['same_detection_status']);self.assertEqual(r['same_classified_images']['n'],1);self.assertEqual(r['same_classified_images']['delta_mae'],.25);self.assertEqual(r['bootstrap_global_tolerances']['0.5']['delta'],.5)
 def test_mismatched_photos_rejected(self):
  with self.assertRaisesRegex(ValueError,'coinciden'):compare({('val','a'):{}},{('test','a'):{}},[2.5,2.75])
