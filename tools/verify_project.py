"""Verify published project bytes without loading models or evaluating held-out data."""
import argparse
import hashlib
from pathlib import Path, PurePosixPath


def verify(root):
    entries = (root / "PROJECT_FILES.sha256").read_text(encoding="utf-8").splitlines()
    failures = []
    seen = set()
    for entry in entries:
        expected, name = entry.split("  ", 1)
        relative = PurePosixPath(name)
        if relative.is_absolute() or ".." in relative.parts or ":" in name or "\\" in name:
            raise ValueError(f"Unsafe manifest path: {name}")
        if name in seen or len(expected) != 64:
            raise ValueError(f"Invalid manifest entry: {name}")
        seen.add(name)
        path = root.joinpath(*relative.parts)
        if not path.is_file():
            failures.append(f"MISSING {name}")
            continue
        with path.open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != expected:
            failures.append(f"CHANGED {name}")
    if failures:
        raise ValueError("\n".join(failures))
    print(f"PASS: {len(seen)} project files match the published SHA-256 inventory.")
    print("Byte integrity only; no model deserialization, training or test evaluation.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    verify(parser.parse_args().root.resolve())
