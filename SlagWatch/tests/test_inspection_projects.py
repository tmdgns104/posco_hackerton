"""Cross-project boundaries that the generic runner cannot check on its own."""
import unittest
from unittest.mock import patch

from src.inspection_projects import load_project


class ProjectScopeTests(unittest.TestCase):
    def test_validation_row_cannot_replace_registered_pixels_or_truth(self):
        project=load_project("ksdd2")
        base, other=project.rows("validation").iloc[:2].to_dict("records")
        import pandas as pd
        for field,value in (("image_path",other["image_path"]),("file_sha256",other["file_sha256"]),
                            ("true_label","defect" if base["true_label"]=="normal" else "normal")):
            with self.subTest(field=field), patch("src.inspection_projects.Image.open") as decode:
                with self.assertRaises(ValueError):
                    project.load(pd.Series({**base,field:value}))
                decode.assert_not_called()

    def test_no_other_project_or_negative_selection_can_open_official_test(self):
        project=load_project("neu_cls6")
        with self.assertRaises(ValueError):
            project.rows("test")
        binary=load_project("ksdd2")
        with patch("src.inspection_projects.read_json",return_value={"allowed":False}):
            with self.assertRaises(ValueError):
                binary.rows("test")
