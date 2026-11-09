"""Restoration must work offline when the checkout already has pinned inputs."""
import csv
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from zipfile import ZipFile


SCRIPT = Path(__file__).resolve().parents[1] / "public_inputs.py"
SPEC = importlib.util.spec_from_file_location("public_inputs", SCRIPT)
inputs = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(inputs)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def archive(files):
    output = io.BytesIO()
    with ZipFile(output, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return output.getvalue()


class RestoreTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "SlagWatch"
        self.cache = Path(self.temp.name) / "cache"
        self.root.mkdir()
        self.addCleanup(patch.stopall)
        patch.object(inputs, "ROOT", self.root).start()
        self.files = {}
        self.ksdd = {}
        rows = []
        # Contents are synthetic bytes. No model, image decoding or labels are used.
        for number, split in enumerate(("train", "validation", "test", "excluded")):
            side = "test" if split == "test" else "train"
            image, mask = f"{side}/{number}.png", f"{side}/{number}_GT.png"
            self.ksdd[image] = f"image-{number}".encode()
            self.ksdd[mask] = f"mask-{number}".encode()
            rows.append({"split": split, "image_path": "data/ksdd2/raw/" + image,
                         "mask_path": "data/ksdd2/raw/" + mask,
                         "file_sha256": digest(self.ksdd[image]),
                         "mask_sha256": digest(self.ksdd[mask])})
        metadata = "split_weakly_246.pyb"
        self.ksdd[metadata] = b"metadata-only-never-unpickle"
        for name, data in self.ksdd.items():
            self.write("data/ksdd2/raw/" + name, data)
        table = io.StringIO(newline="")
        writer = csv.DictWriter(table, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
        manifest = table.getvalue().encode()
        self.write("data/ksdd2/manifest.csv", manifest)
        self.ksdd_zip = archive(self.ksdd)
        self.write_json("data/ksdd2/audit.json", {
            "archive_sha256": digest(self.ksdd_zip), "archive_png_pairs": len(rows),
            "manifest_sha256": digest(manifest),
            "official_split_metadata_sha256": {metadata: digest(self.ksdd[metadata])}})
        self.ssfd = {"SSFD/E-1.csv": b"time,x,y,z\n0,1,2,3\n"}
        self.ssfd_zip = archive(self.ssfd)
        for name, data in self.ssfd.items():
            self.write("data/raw/" + name, data)
        self.write_json("data/raw/manifest.json", {
            "original_zip_sha256": digest(self.ssfd_zip),
            "files": [{"path": name, "sha256": digest(data)}
                      for name, data in self.ssfd.items()]})
        self.encoder = b"synthetic-pinned-weights"
        self.encoder_path = "experiments/surface_v1/models/resnet18-f37072fd.pth"
        self.write(self.encoder_path, self.encoder)
        self.write_json("experiments/surface_v1/protocol.json", {
            "weight_sha256": digest(self.encoder)})

    def write(self, name, data):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
        self.files[name] = data

    def write_json(self, name, data):
        self.write(name, json.dumps(data).encode())

    def run_repair(self, url, payload):
        with patch.object(inputs, "urlopen", return_value=io.BytesIO(payload)) as network:
            inputs.main(self.cache)
        network.assert_called_once_with(url, timeout=120)
        for name, expected in self.files.items():
            self.assertEqual((self.root / name).read_bytes(), expected, name)

    def test_complete_checkout_does_not_access_network_or_cache_archives(self):
        with patch.object(inputs, "urlopen", side_effect=AssertionError("offline")) as network:
            inputs.main(self.cache)
        network.assert_not_called()
        self.assertEqual(list(self.cache.iterdir()), [])

    def test_missing_training_image_restores_full_dataset(self):
        (self.root / "data/ksdd2/raw/train/0.png").unlink()
        self.run_repair(inputs.KSDD2, self.ksdd_zip)

    def test_corrupt_official_test_mask_is_repaired_without_evaluation(self):
        (self.root / "data/ksdd2/raw/test/2_GT.png").write_bytes(b"corrupt")
        self.run_repair(inputs.KSDD2, self.ksdd_zip)

    def test_missing_official_split_metadata_is_repaired_as_opaque_bytes(self):
        (self.root / "data/ksdd2/raw/split_weakly_246.pyb").unlink()
        self.run_repair(inputs.KSDD2, self.ksdd_zip)

    def test_corrupt_official_split_metadata_is_repaired_as_opaque_bytes(self):
        (self.root / "data/ksdd2/raw/split_weakly_246.pyb").write_bytes(b"corrupt")
        self.run_repair(inputs.KSDD2, self.ksdd_zip)

    def test_missing_excluded_duplicate_is_still_restored(self):
        (self.root / "data/ksdd2/raw/train/3.png").unlink()
        self.run_repair(inputs.KSDD2, self.ksdd_zip)

    def test_corrupt_registered_manifest_fails_before_network_access(self):
        (self.root / "data/ksdd2/manifest.csv").write_bytes(b"corrupt")
        with patch.object(inputs, "urlopen", side_effect=AssertionError("offline")) as network:
            with self.assertRaisesRegex(ValueError, "manifest hash mismatch"):
                inputs.main(self.cache)
        network.assert_not_called()

    def test_corrupt_ssfd_file_only_downloads_ssfd(self):
        (self.root / "data/raw/SSFD/E-1.csv").write_bytes(b"corrupt")
        self.run_repair(inputs.SSFD, self.ssfd_zip)

    def test_missing_encoder_only_downloads_encoder(self):
        (self.root / self.encoder_path).unlink()
        self.run_repair(inputs.RESNET, self.encoder)

    def test_bad_download_is_rejected_before_replacing_local_data(self):
        target = self.root / "data/raw/SSFD/E-1.csv"
        target.write_bytes(b"damaged-local-file")
        with patch.object(inputs, "urlopen", return_value=io.BytesIO(b"wrong archive")):
            with self.assertRaisesRegex(ValueError, "Download hash mismatch"):
                inputs.main(self.cache)
        self.assertEqual(target.read_bytes(), b"damaged-local-file")
        self.assertFalse((self.cache / "ssfd.part").exists())


if __name__ == "__main__":
    unittest.main()
