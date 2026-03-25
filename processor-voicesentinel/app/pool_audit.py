import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Dict

_pool_audit_logger = logging.getLogger("voicesentinel.pool_audit")


def setup_pool_audit(enabled: bool, log_path: str = "logs/pooled_server_audit.jsonl") -> None:
    _pool_audit_logger.handlers.clear()
    if not enabled:
        _pool_audit_logger.addHandler(logging.NullHandler())
        _pool_audit_logger.setLevel(logging.CRITICAL)
        _pool_audit_logger.propagate = False
        return
    parent = Path(log_path).parent
    if str(parent) and str(parent) != ".":
        parent.mkdir(parents=True, exist_ok=True)
    fh = logging.FileHandler(log_path, encoding="utf-8")
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter("%(message)s"))
    _pool_audit_logger.addHandler(fh)
    _pool_audit_logger.setLevel(logging.INFO)
    _pool_audit_logger.propagate = False


def pool_audit_emit(
    cfg: dict,
    event: str,
    client_id: str,
    license_fingerprint: str,
    license_key: str = "",
    **fields: Any,
) -> None:
    if not cfg or not cfg.get("pool_server"):
        return
    record = {
        "event": event,
        "client_id": client_id,
        "license_fingerprint": license_fingerprint or "",
        "license_key": license_key or "",
        **fields,
    }
    _pool_audit_logger.info(json.dumps(record, separators=(",", ":"), ensure_ascii=False))


def pool_transcript_append(
    cfg: dict,
    license_fingerprint: str,
    license_key: str,
    record: Dict[str, Any],
) -> None:
    if not cfg or not cfg.get("pool_server"):
        return
    root = Path(cfg.get("pool_server_transcripts_dir", "logs/pool_transcripts_by_license"))
    root.mkdir(parents=True, exist_ok=True)
    fp = (license_fingerprint or "").strip()
    if not fp:
        fp = hashlib.sha256((license_key or "unknown").encode("utf-8")).hexdigest()
    path = root / f"{fp}.jsonl"
    out = {
        **record,
        "license_key": license_key or "",
        "license_fingerprint": license_fingerprint or "",
    }
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(out, separators=(",", ":"), ensure_ascii=False) + "\n")
