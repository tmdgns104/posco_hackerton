"""Local operational events, not authenticated or tamper-proof audit records."""
from datetime import datetime, timezone
import getpass
import json
from pathlib import Path
import uuid

from filelock import FileLock


def record_event(root, event, *, project_id, outcome, details=None):
    if not all(isinstance(v, str) and v.strip() for v in (event, project_id, outcome)):
        raise ValueError("Event type, project and outcome are required.")
    entry = {
        "id": uuid.uuid4().hex,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "actor_os_account_unverified": getpass.getuser(),
        "event": event,
        "project_id": project_id,
        "outcome": outcome,
        "details": details or {},
    }
    payload = json.dumps(entry, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n"
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    with FileLock(str(root / "events.lock"), timeout=10):
        with (root / "events.jsonl").open("a", encoding="utf-8", newline="\n") as stream:
            stream.write(payload)
            stream.flush()
    return entry


def read_events(root, *, project_id, limit=100):
    if not isinstance(limit, int) or not 1 <= limit <= 1000:
        raise ValueError("Event limit must be between 1 and 1000.")
    path = Path(root) / "events.jsonl"
    if not path.exists():
        return []
    with FileLock(str(path.parent / "events.lock"), timeout=10):
        with path.open(encoding="utf-8") as stream:
            entries = []
            for number, line in enumerate(stream, 1):
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as exc:
                    raise ValueError(f"Invalid event record at line {number}.") from exc
                if not isinstance(entry, dict):
                    raise ValueError(f"Invalid event record at line {number}.")
                if entry.get("project_id") == project_id:
                    entries.append(entry)
    return entries[-limit:][::-1]
