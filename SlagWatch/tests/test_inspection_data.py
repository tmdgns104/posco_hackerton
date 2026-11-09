import csv
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from src.inspection_data import (DATA, MANIFEST_SHA, ROOT, assign, file_sha, image_group,
                                 load_pair, verify_files, verify_archive)


class OfficialDataTests(unittest.TestCase):
    def test_mask_rule_on_faint_image_and_invalid_annotations(self):
        with tempfile.TemporaryDirectory() as d:
            image, mask = Path(d)/"part.png", Path(d)/"part_GT.png"
            Image.new("RGB",(11,19),"black").save(image)
            with self.assertRaises(FileNotFoundError):
                load_pair(image, mask)
            ann = Image.new("L",(11,19),0)
            ann.save(mask)
            normal = load_pair(image, mask)
            self.assertEqual(normal["true_label"], "normal")
            ann.putpixel((7,11),255)
            ann.save(mask)
            defect = load_pair(image, mask)
            self.assertEqual((defect["true_label"],defect["defect_pixels"]),("defect",1))
            self.assertEqual(normal["file_sha256"],defect["file_sha256"])
            self.assertNotEqual(normal["mask_sha256"],defect["mask_sha256"])
            ann.putpixel((7,11),33)
            ann.save(mask)
            with self.assertRaisesRegex(ValueError,"Nonbinary"):
                load_pair(image, mask)
            ann.resize((3,4)).save(mask)
            with self.assertRaisesRegex(ValueError,"size mismatch"):
                load_pair(image, mask)
            mask.write_bytes(b"truncated PNG")
            with self.assertRaises(OSError):
                load_pair(image, mask)

    def test_reencoded_rotation_and_official_test_precedence(self):
        im = Image.new("RGB",(3,5))
        im.putpixel((2,0),(11,124,27))
        p1,g1 = image_group(im)
        p2,g2 = image_group(im.transpose(Image.Transpose.ROTATE_90))
        self.assertNotEqual(p1,p2)
        self.assertEqual(g1,g2)
        def row(id, side, grp, label="defect"):
            return dict(sample_id=id,official_split=side,group_id=grp,true_label=label,split=side)
        rows=assign([row("a","train",g1),row("b","test",g2),row("t1","train","g3"),
                     row("t2","train","g3")]+[row(f"id{i}","train",f"unique{i}") for i in range(12)])
        self.assertEqual([x['split'] for x in rows[:2]], ["excluded","test"])
        self.assertEqual(rows[0]["exclusion_reason"],"duplicate_of_official_test")
        self.assertEqual(rows[2]['split'],rows[3]['split'])
        self.assertIn("validation",{r['split'] for r in rows})
        with self.assertRaisesRegex(ValueError,"Conflicting"):
            assign([row("a","train",g1),row("b","test",g2,"normal")])
        with self.assertRaisesRegex(ValueError,"Duplicate IDs"):
            assign([row("a","test",g1),row("a","train",g1)])

    def test_frozen_release_and_untrusted_archive(self):
        self.assertEqual(file_sha(ROOT/DATA/"manifest.csv"),MANIFEST_SHA)
        with (ROOT/DATA/"manifest.csv").open(encoding="utf-8", newline="") as f:
            rows=list(csv.DictReader(f))
        self.assertEqual(len(rows),3336)
        train={r['group_id'] for r in rows if r['split']=='train'}
        valid={r['group_id'] for r in rows if r['split']=='validation'}
        test={r['group_id'] for r in rows if r['split']=='test'}
        self.assertFalse(train & valid or train & test or valid & test)
        self.assertEqual(len(test),1004)
        self.assertEqual(sum(r['true_label']=='defect' and r['split']=='validation' for r in rows),49)
        self.assertEqual([(r['part_id'],r['split']) for r in rows if r['exclusion_reason']],
                         [('10301','excluded')])
        with tempfile.TemporaryDirectory() as d:
            wrong=Path(d)/"not-KSDD2.zip"
            wrong.write_bytes(b"also not the source archive")
            with self.assertRaisesRegex(ValueError,"SHA-256 mismatch"):
                verify_archive(wrong)
            (Path(d)/DATA).mkdir(parents=True)
            (Path(d)/DATA/"manifest.csv").write_text("tampered!\n")
            with self.assertRaisesRegex(ValueError,"manifest SHA-256 mismatch"):
                verify_files(d,"validation")


if __name__ == "__main__":
    unittest.main()
