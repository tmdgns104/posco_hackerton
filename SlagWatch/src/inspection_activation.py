"""Local workspace selection; never deploys a model or authorizes an official test."""
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile

from filelock import FileLock
import pandas as pd

from .inspection_runner import DATA_COLS, load_run, recompute_decision
from . import security


def _state_path(root, project_id):
    if not isinstance(project_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]+", project_id):
        raise ValueError("Invalid project ID for local model selection.")
    return Path(root) / (project_id + ".json")


def active_model(root, project):
    path = _state_path(root, project.id)
    if not path.exists():
        return next(iter(project.models))
    state = json.loads(path.read_text(encoding="utf-8"))
    if state.get("schema") != 1 or state.get("project_id") != project.id or not isinstance(state.get("model_id"), str):
        raise ValueError("Invalid local active model record.")
    return state["model_id"]


def check_compatibility(project, before, after, adapters):
    if before == after or before not in adapters or after not in adapters:
        raise ValueError("Choose two different available models.")
    if project.kind != "binary" or project.template_id != "ksdd2":
        raise ValueError("Local model selection currently supports KSDD2 binary adapters only.")
    if list(project.classes) != ["normal", "defect"]:
        raise ValueError("Unsupported project class mapping.")
    for model in (before, after):
        identity = adapters[model].identity
        if (not isinstance(identity, dict) or identity.get("model_id") != model
                or identity.get("class_mapping") != {"0": "normal", "1": "defect"}
                or not callable(getattr(adapters[model], "predict", None))):
            raise ValueError(f"Model {model} does not match the binary image adapter contract.")
    from .inspection_model import ENCODER_SHA, PREP
    for model in (before, after):
        identity = adapters[model].identity
        if identity.get("encoder_sha256") != ENCODER_SHA or identity.get("preprocessing") != PREP:
            raise ValueError(f"Model {model} has incompatible encoder or preprocessing.")
    return {"before": before, "after": after, "classes": list(project.classes), "preprocessing": PREP}


def activation_evidence(project, run_dir, before, after, adapters):
    project.check_scope("validation")
    check_compatibility(project, before, after, adapters)
    run_dir = Path(run_dir)
    manifest = load_run(run_dir)
    if (manifest["run_id"] != run_dir.name or manifest["project_id"] != project.id
            or manifest["kind"] != project.kind or manifest["split"] != "validation"
            or manifest["scope"] != "full" or manifest["failed_pairs"] != 0
            or manifest["before_model_id"] != before or manifest["after_model_id"] != after
            or manifest["classes"] != list(project.classes)
            or manifest.get("source_manifest_sha256") != project.index_hash
            or manifest.get("protocol_file_sha256") != project.protocol_file_sha256
            or manifest["protocol"] != project.protocol
            or manifest["conditions"] != [list(item) for item in project.conditions("validation")]
            or manifest["models"] != {mid: adapters[mid].identity for mid in (before, after)}):
        raise ValueError("The complete validation run does not match the current project, model identities or conditions.")
    current = project.rows("validation")[list(DATA_COLS)].astype(str).sort_values("sample_id").reset_index(drop=True)
    recorded = pd.read_csv(run_dir / "data_manifest.csv", dtype=str, keep_default_na=False)
    if not set(DATA_COLS).issubset(recorded.columns):
        raise ValueError("The validation run is missing dataset lineage.")
    recorded = recorded[list(DATA_COLS)].astype(str).sort_values("sample_id").reset_index(drop=True)
    if not current.equals(recorded) or len(current) != manifest["n_originals"]:
        raise ValueError("The validation run does not cover the current entire validation dataset.")
    decision = recompute_decision(run_dir)
    if decision["status"] != "pass":
        raise ValueError("The full development-validation gate did not pass.")
    return {"run_id": manifest["run_id"], "manifest_sha256": hashlib.sha256((run_dir / "manifest.json").read_bytes()).hexdigest(),
            "before": before, "after": after, "status": decision["status"]}


def activate_model(root, project, run_dir, before, after, adapters, *, actor, confirmed, auth_token=None):
    if confirmed is not True or not isinstance(actor, str) or not actor.strip() or len(actor.strip()) > 120:
        raise ValueError("A self-reported operator and explicit confirmation are required.")
    principal = security.authorize(auth_token)
    if actor.strip() != principal["username"]:
        raise security.SecurityError("확인자는 로그인한 계정과 일치해야 합니다.")
    audit = security.begin_action(auth_token, "image_model_activation",
                                  {"project_id": project.id, "before": before, "after": after,
                                   "run_id": Path(run_dir).name})
    path = _state_path(root, project.id)
    path.parent.mkdir(parents=True, exist_ok=True)
    with FileLock(str(path.parent / (project.id + ".lock")), timeout=10):
        if active_model(root, project) != before:
            raise ValueError("The local active model changed; select a new matching evaluation.")
        evidence = activation_evidence(project, run_dir, before, after, adapters)
        state = {"schema": 1, "project_id": project.id, "model_id": after,
                 "model_identity": adapters[after].identity, "previous_model_id": before,
                 "run_id": evidence["run_id"], "run_manifest_sha256": evidence["manifest_sha256"],
                 "actor_authenticated": principal["username"], "created_utc": datetime.now(timezone.utc).isoformat(),
                 "scope": "local workspace selection only; no deployment or official approval"}
        temporary = None
        try:
            with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", newline="\n",
                                             dir=path.parent, prefix=".active-", delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(state, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
                stream.write("\n")
            os.replace(temporary, path)
        finally:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
    security.finish_action(audit, "completed")
    return state
