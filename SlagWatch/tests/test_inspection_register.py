"""Registration is an import boundary: received bytes never become executable paths."""
import csv
import hashlib
import io
import json
from pathlib import Path
import stat
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from PIL import Image

from src import inspection_register as reg


def img(color, size=(4, 3), mode="RGB", fmt="PNG"):
    source = Image.new(mode, size, color)
    source.putpixel((0, 1), (240, 11, 7) if mode == "RGB" else 241)
    buf = io.BytesIO()
    source.save(buf, format=fmt)
    return buf.getvalue()


def rgb_digest(blob):
    with Image.open(io.BytesIO(blob)) as im:
        rgb = im.convert("RGB")
        return hashlib.sha256(f"{rgb.width}x{rgb.height}:RGB:".encode("ascii") + rgb.tobytes()).hexdigest()


def table(rows):
    out = io.StringIO(newline="")
    wr = csv.DictWriter(out, fieldnames=list(rows[0]))
    wr.writeheader()
    wr.writerows(rows)
    return out.getvalue().encode("utf-8-sig")


def archive(rows, files):
    out = io.BytesIO()
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        z.writestr("manifest.csv", table(rows))
        for name, data in files:
            z.writestr(name, data)
    return out.getvalue()


def row(sid="001", path="images/제품.png", label="defect", **extra):
    return dict(sample_id=sid, image_path=path, true_label=label, **extra)


class TestRegistration(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        p = patch.object(reg, "ROOT", self.root)
        p.start()
        self.addCleanup(p.stop)
        self.ks = self.root / "data/ksdd2/manifest.csv"
        self.neu = self.root / "experiments/surface_v1/split_manifest.csv"
        self.ks.parent.mkdir(parents=True)
        self.neu.parent.mkdir(parents=True)
        self.ks_files = [img((i, i + 4, i + 8)) for i in range(3)]
        self.mask = img(7, mode="L")
        self.ks.write_bytes(table([
            dict(sample_id="ks"+str(i), true_label="defect", group_id="group"+str(i),
                 official_split="test" if split == "test" else "train", split=split,
                 file_sha256=hashlib.sha256(data).hexdigest(), pixel_sha256=rgb_digest(data),
                 mask_sha256=hashlib.sha256(self.mask).hexdigest(), mask_pixel_sha256=rgb_digest(self.mask))
            for i, (data, split) in enumerate(zip(self.ks_files, ["train","test","validation"]))]))
        self.neu_files = [img(i+50,(200,200),"L","BMP") for i in range(3)]
        self.neu.write_bytes(table([
            dict(sample_id="neu"+str(i), label="scratches", parent_image_id="parent"+str(i), split=sp,
                 file_sha256=hashlib.sha256(data).hexdigest(),
                 pixels_sha256=hashlib.sha256(Image.open(io.BytesIO(data)).tobytes()).hexdigest(),
                 canonical_sha256="a"*64)
            for i, (data, sp) in enumerate(zip(self.neu_files, ["train","test","validation"]))]))
        p = patch.dict(reg.INDEX_SHA, {"ksdd2":hashlib.sha256(self.ks.read_bytes()).hexdigest(),
                                      "neu_cls6":hashlib.sha256(self.neu.read_bytes()).hexdigest()})
        p.start()
        self.addCleanup(p.stop)
        self.data = img((40, 20, 80))
        self.one = [row()]
        self.files = [(self.one[0]["image_path"], self.data)]

    def submit(self, rows=None, files=None, template="ksdd2"):
        return reg.register_zip(archive(self.one if rows is None else rows, self.files if files is None else files),
                                "검사 개발", template)

    def test_zip_and_folder_publish_same_verified_project_preserving_native_bytes(self):
        key = self.submit()
        p = self.root / "inspection_projects" / key
        cfg = json.loads((p / "project.json").read_text(encoding="utf8"))
        self.assertEqual((cfg["id"],cfg["template_id"],cfg["title"]), (key,"ksdd2","검사 개발"))
        self.assertEqual(cfg["manifest_sha256"],hashlib.sha256((p/"manifest.csv").read_bytes()).hexdigest())
        with (p/"manifest.csv").open(encoding="utf8") as stream:
            r, = csv.DictReader(stream)
        self.assertEqual((r["sample_id"],r["true_label"],r["split"]),("001","defect","validation"))
        self.assertEqual((self.root/r["image_path"]).read_bytes(), self.data)
        self.assertEqual(r["file_sha256"], hashlib.sha256(self.data).hexdigest())
        f = self.root/"로컬 이미지"
        (f/"images").mkdir(parents=True)
        (f/self.files[0][0]).write_bytes(self.data)
        gt = self.root/"정답.csv"
        gt.write_bytes(table(self.one))
        self.assertEqual(key, reg.register_folder(f,gt,"새 이름","ksdd2"))
        self.assertEqual([p],list(p.parent.iterdir()))
        (self.root/r["image_path"]).write_bytes(b"changed")
        with self.assertRaises(ValueError):
            self.submit()

    def test_archive_and_csv_paths_and_links_do_not_escape(self):
        badnames = ["../evil.png", "/absolute.png", "C:/evil.png", "images/a:b.png",
                    "images/CON.png", "images/evil.png.", "a\\slash.png", "a/../x.png"]
        for name in badnames:
            with self.subTest(name=name), self.assertRaises(ValueError):
                self.submit([row(path=name)],[(name,self.data)])
        with self.assertRaises(ValueError): self.submit([row(path="../x.png")])
        with self.assertRaises(ValueError): self.submit(files=self.files+[("images/EXTRA.png",self.data)])
        with self.assertRaises(ValueError): self.submit(files=self.files+[("images/제품.PNG",self.data)])
        link = zipfile.ZipInfo("images/제품.png")
        link.create_system = 3
        link.external_attr = (stat.S_IFLNK | 0o777) << 16
        with self.assertRaises(ValueError): self.submit(files=[(link,b"/etc/passwd")])
        with self.assertRaises(ValueError): self.submit(files=self.files+[("model.pkl",b"not a model")])
        with self.assertRaises(ValueError): self.submit(template="../untrusted")
        self.assertFalse((self.root/"evil.png").exists())

    def test_refuses_bad_truth_duplicate_ids_repeated_inputs_and_unsupported_images(self):
        for rows in ([row(label="")],[row(label="unknown")],[row(path="missing.jpg")],
                     [row(split="test")], [row(sid="../a")], [row(), row()],
                     [dict(sample_id="001",image_path="images/제품.png")]):
            with self.subTest(rows=rows),self.assertRaises(ValueError): self.submit(rows)
        with self.assertRaises(ValueError):
            self.submit([row(),row("002","images/same.png")],self.files+[("images/same.png",self.data)])
        for data in (b"not an image",self.data[:-12],img((1,1,1),(5000,2))):
            with self.assertRaises(ValueError): self.submit(files=[(self.files[0][0],data)])
        with self.assertRaises(ValueError): self.submit(template="neu_cls6",rows=[row(label="crazing")])
        with patch.object(reg,"MAX_ARCHIVE_BYTES",10), self.assertRaises(ValueError): self.submit()
        with patch.object(reg,"MAX_EXPANDED_BYTES",10), self.assertRaises(ValueError): self.submit()
        self.assertFalse(list((self.root/"inspection_projects").glob("user_*")))

    def test_registration_pixels_fit_the_serialized_case_including_header_margin(self):
        import numpy as np
        from src.inspection_cases import _check_array

        # RGB float32 contains 12 bytes per pixel even when the user uploads an L PNG.
        # 2000×1398 fits 32 MiB before reserving room for the NPY header.
        for mode, size, color in [("RGB", (2048, 1536), (1, 7, 9)), ("L", (2000, 1398), 71)]:
            with self.subTest(mode=mode, size=size):
                blob = img(color, size, mode)
                with self.assertRaisesRegex(ValueError, "2,795,520"):
                    self.submit(files=[(self.files[0][0], blob)])
                with self.assertRaises(ValueError):
                    _check_array((size[1], size[0], 3), np.dtype("float32"))

        # Registration must also accept the exact case payload boundary and native NEU.
        for template, label, mode, size, color in [
            ("ksdd2", "normal", "RGB", (2048, 1365), (7, 34, 52)),
            ("neu_cls6", "patches", "L", (200, 200), 11),
        ]:
            with self.subTest(template=template):
                blob = img(color, size, mode)
                key = self.submit([row(label=label)], [(self.files[0][0], blob)], template)
                self.assertTrue((self.root/"inspection_projects"/key/"manifest.csv").is_file())
                shape = (size[1], size[0], 3) if template == "ksdd2" else (size[1], size[0])
                _check_array(shape, np.dtype("float32"))

    def test_known_train_test_masks_refused_across_templates_after_lossless_transform(self):
        for data in [*self.ks_files[:2],*self.neu_files[:2],self.mask]:
            with Image.open(io.BytesIO(data)) as original:
                rgb = original.convert("RGB").transpose(Image.Transpose.ROTATE_90)
                b = io.BytesIO()
                rgb.save(b, format="PNG")
            with self.subTest(data=hashlib.sha256(data).hexdigest()), self.assertRaises(ValueError):
                self.submit(files=[(self.files[0][0],b.getvalue())])
        with self.assertRaises(ValueError):
            self.submit(rows=[row(label="normal")],files=[(self.files[0][0],self.ks_files[2])])
        key = self.submit(files=[(self.files[0][0],self.ks_files[2])])
        with (self.root/"inspection_projects"/key/"manifest.csv").open() as f:
            r, = csv.DictReader(f)
        self.assertIn("ks2",r["source_id"])
        self.assertEqual(r["group_id"],"group2")
        self.ks.write_bytes(self.ks.read_bytes().replace(b"test,test,", b"train,validation,"))
        with self.assertRaises(ValueError): self.submit()
        self.ks.unlink()
        with self.assertRaises(ValueError): self.submit()

    def test_local_symlink_rejected_no_partial_import(self):
        good = self.root/"source"
        good.mkdir()
        (good/"images").symlink_to(self.root/"outside",target_is_directory=True)
        (self.root/"outside").mkdir()
        (self.root/"outside"/"제품.png").write_bytes(self.data)
        gt = good/"manifest.csv"
        gt.write_bytes(table(self.one))
        with self.assertRaises(ValueError):
            reg.register_folder(good,gt,"검사","ksdd2")
        self.assertFalse(list((self.root/"inspection_projects").glob("user_*")))
