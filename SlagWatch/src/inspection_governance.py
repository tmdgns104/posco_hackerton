"""Local evaluation-design snapshots; these do not unlock or authorize official tests."""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import uuid

import pandas as pd


def split_conflicts(index: pd.DataFrame) -> pd.DataFrame:
    if "split" not in index.columns:
        raise ValueError("Dataset split is required for overlap checks.")
    active = index[index["split"].isin(["train", "development", "validation", "test"])].copy()
    if active.empty:
        raise ValueError("No recognized dataset splits to inspect.")
    results = []
    for column in ("sample_id", "file_sha256", "group_id", "parent_original_id", "equipment_id", "capture_group"):
        if column not in active:
            continue
        values = active[["split", column]].dropna()
        values = values[values[column].astype(str).str.strip().ne("")]
        for value, group in values.groupby(column, sort=True):
            splits = sorted(group["split"].astype(str).unique())
            if len(splits) > 1:
                results.append({"field": column, "value": str(value), "splits": ", ".join(splits),
                                "rows": len(group)})
    return pd.DataFrame(results, columns=["field", "value", "splits", "rows"])


def _digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def design_state(project, split: str) -> dict:
    project.check_scope(split)
    if split != "validation":
        raise ValueError("Local design snapshots currently support validation only.")
    src = Path(__file__).resolve().parent
    files = ("inspection_runner.py", "inspection_inputs.py", "inspection_reliability.py", "inspection_governance.py")
    return {
        "project_id": project.id, "split": split, "dataset_sha256": project.index_hash,
        "protocol_sha256": project.protocol_file_sha256,
        "conditions": [list(c) for c in project.conditions(split)],
        "models": {name: adapter.identity for name, adapter in project.models.items()},
        "code_sha256": {name: _digest(src / name) for name in files},
    }


def _canonical(value: dict) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False, separators=(",", ":")).encode("utf-8")


def freeze_design(project, split: str, root: Path, *, actor: str, hypothesis: str, criteria: str) -> Path:
    for name, value in (("actor", actor), ("hypothesis", hypothesis), ("criteria", criteria)):
        if not isinstance(value, str) or not value.strip() or len(value) > 4000:
            raise ValueError(f"{name} must be present and at most 4000 characters.")
    conflicts = split_conflicts(project.index)
    if not conflicts.empty:
        raise ValueError("Cross-split overlaps must be resolved before freezing a design.")
    state = design_state(project, split)
    identifier = uuid.uuid4().hex
    snapshot = {"schema": 1, "id": identifier, "created_utc": datetime.now(timezone.utc).isoformat(),
                "actor_self_reported": actor.strip(), "hypothesis": hypothesis.strip(),
                "success_criteria": criteria.strip(), "state": state,
                "purpose": "local validation design record; not final-test authorization"}
    snapshot["sha256"] = hashlib.sha256(_canonical(snapshot)).hexdigest()
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / (identifier + ".json")
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(snapshot, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    return path


def verify_design(path: Path, project, split: str) -> dict:
    snapshot = json.loads(Path(path).read_text(encoding="utf-8"))
    digest = snapshot.get("sha256")
    body = {key: value for key, value in snapshot.items() if key != "sha256"}
    if digest != hashlib.sha256(_canonical(body)).hexdigest():
        raise ValueError("Frozen design has been modified.")
    if snapshot.get("schema") != 1 or snapshot.get("state") != design_state(project, split):
        raise ValueError("Current model, dataset, conditions, protocol or code differs from the frozen design.")
    return snapshot
