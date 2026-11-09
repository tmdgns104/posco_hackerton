"""Decision boundary contract tests for clean-only, budgeted calibration."""
import unittest
from unittest.mock import patch

import numpy as np
from experiments.inspection_binary.study import choose_threshold
from src.inspection_model import BinaryAdapter, prepare


class ThresholdTests(unittest.TestCase):
    def test_budget_duplicate_scores_and_priority(self):
        # With 10 normals exactly one false alarm is allowed. At .8 all
        # defects are found but two tied normals would both be flagged.
        y = np.array([1, 1, 1] + [0] * 10)
        z = np.array([.98, .9, .8, .93, .8] + [.1]*8)
        self.assertEqual(choose_threshold(y, z), .9)
        # One defect at .9, with no FP. Higher cutoff loses it; lowering
        # cutoff either adds an FP or loses the highest cutoff tie-break.
        self.assertEqual(choose_threshold([1, 0, 0], [.9, .3, .1]), .9)

    def test_infeasible_all_defect_tie_uses_all_normal_cutoff(self):
        t = choose_threshold([1, 0, 0], [0., 0., 0.])
        self.assertGreater(t, 0.)
        self.assertFalse((np.array([0., 0., 0.]) >= t).any())

    def test_nonzero_cutoff_is_inclusive_not_score_argmax(self):
        class Head:
            def decision_function(self, data):
                return np.array([.1, .3])
        m = BinaryAdapter("example", Head(), {"threshold":.3})
        with patch("src.inspection_model._shared_features", return_value=None):
            pred = m.predict([object(), object()])
        self.assertEqual([p["predicted_label"] for p in pred], ["normal","defect"])
        self.assertEqual(pred[0]["scores"], {"defect":.1,"normal":-.1})

    def test_entire_tall_image_reaches_model_including_both_edges(self):
        a=np.zeros((640,224,3),dtype=np.float32)
        a[:10,:,0]=1; a[-10:,:,2]=1
        p=prepare(a)
        self.assertEqual(p.shape,(512,192,3))
        np.testing.assert_allclose(p[0,:,0],1,atol=1e-6)
        np.testing.assert_allclose(p[-1,:,2],1,atol=1e-6)


if __name__=="__main__":
    unittest.main()
