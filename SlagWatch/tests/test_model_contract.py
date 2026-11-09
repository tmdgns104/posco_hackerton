import unittest

import numpy as np

from src.inspection_projects import load_project
from src.model_contract import BaseModelAdapter


class ImageContractTests(unittest.TestCase):
    def test_real_image_adapters_preserve_predictions_and_common_shape(self):
        for project_id in ("neu_cls6", "ksdd2"):
            project = load_project(project_id)
            raw = project.load(project.rows("validation").iloc[0])
            for model_id, adapter in project.models.items():
                with self.subTest(project=project_id, model=model_id):
                    self.assertIsInstance(adapter, BaseModelAdapter)
                    original = adapter.adapter.predict([raw])[0]
                    actual = adapter.infer(["sample"], [raw])[0]
                    self.assertEqual(actual["sample_id"], "sample")
                    self.assertEqual(actual["model_id"], model_id)
                    self.assertEqual(actual["predicted_class"], original["predicted_label"])
                    self.assertEqual(actual["scores"], original["scores"])
                    self.assertEqual(actual["score_type"], original["score_type"])
                    self.assertEqual(actual["inference_status"], original["status"])
                    self.assertEqual(actual["error_code"], "")
                    self.assertGreaterEqual(actual["inference_duration_ms"], 0)
                    with self.assertRaises(ValueError):
                        adapter.infer([], [raw])
                    with self.assertRaises(ValueError):
                        adapter.predict([np.array([np.nan], dtype=np.float32)])


if __name__ == "__main__":
    unittest.main()
