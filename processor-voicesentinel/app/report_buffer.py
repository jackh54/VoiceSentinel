from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

from app.storage import get_storage, storage_key

_last_cleanup: Dict[str, float] = {}
_CLEANUP_INTERVAL = 3600.0

logger = logging.getLogger(__name__)

_FINGERPRINT_RE = re.compile(r"^[a-f0-9]{64}$")

_evidence_rate: Dict[str, List[float]] = {}
_evidence_lock = asyncio.Lock()
MAX_EVIDENCE_PER_MINUTE_PER_LICENSE = 60

_evidence_ip_rate: Dict[str, List[float]] = {}
MAX_EVIDENCE_PER_MINUTE_PER_IP = 120


def _sanitize_fingerprint(fp: str) -> Optional[str]:
    if not fp or not isinstance(fp, str):
        return None
    n = fp.strip().lower()
    if not _FINGERPRINT_RE.match(n):
        return None
    return n


def server_key_partition(server_key: str) -> str:
    raw = (server_key or "").strip()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _tenant_transcript_key(root: str, license_fp: str, server_key: str) -> str:
    skp = server_key_partition(server_key)
    return storage_key(root, license_fp, skp, "transcripts.jsonl")


def _legacy_transcript_key(root: str, license_fp: str) -> str:
    return storage_key(root, license_fp, "transcripts.jsonl")


def sanitize_player_query(name: str) -> Optional[str]:
    if not name or not isinstance(name, str):
        return None
    n = name.strip()
    if not n or len(n) > 64:
        return None
    if any(c in n for c in "./\\"):
        return None
    if any(ord(c) < 32 for c in n):
        return None
    return n


def sanitize_seconds_seconds(raw: Any, default: int = 300) -> int:
    try:
        s = int(raw)
    except (TypeError, ValueError):
        s = default
    return max(30, min(86400, s))


def report_buffer_append(
    cfg: dict,
    license_fingerprint: str,
    server_key: str,
    record: Dict[str, Any],
    audio_wav: Optional[bytes] = None,
) -> None:
    rb = cfg.get("report_buffer") or {}
    if not rb.get("enabled"):
        return
    fp = _sanitize_fingerprint(license_fingerprint or "")
    if not fp:
        return
    root = rb.get("path", "report_buffer/")
    store = get_storage()
    retention_seconds = float(rb.get("retention_seconds", 604800))
    _maybe_cleanup(store, root, fp, server_key, retention_seconds)
    key = _tenant_transcript_key(root, fp, server_key)
    store.ensure_prefix(storage_key(root, fp, server_key_partition(server_key)))
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
    store.append_bytes(key, line.encode("utf-8"))
    if rb.get("save_audio") and audio_wav:
        try:
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", str(record.get("player", "unknown")))[:32]
            safe_session = re.sub(r"[^a-zA-Z0-9_-]", "_", str(record.get("session_id", "")))[:48]
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            suffix = f"{ts}_{safe}_{safe_session}.wav" if safe_session else f"{ts}_{safe}.wav"
            audio_key = storage_key(root, fp, server_key_partition(server_key), "audio", suffix)
            store.write_bytes(audio_key, audio_wav)
        except OSError as e:
            logger.warning("report_buffer audio save failed: %s", e)


def _cleanup_tenant(
    store,
    root: str,
    tenant_parts: Tuple[str, ...],
    retention_seconds: float,
) -> None:
    cutoff = time.time() - retention_seconds
    transcript_key = storage_key(root, *tenant_parts, "transcripts.jsonl")
    lines = store.read_text_lines(transcript_key)
    if lines:
        kept = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            ts = obj.get("ts") or ""
            try:
                ts_sec = datetime.fromisoformat(ts.replace("Z", "+00:00")).timestamp()
            except (ValueError, TypeError):
                ts_sec = cutoff + 1
            if ts_sec >= cutoff:
                kept.append(line)
        store.write_text(transcript_key, "\n".join(kept) + ("\n" if kept else ""))

    audio_prefix = storage_key(root, *tenant_parts, "audio")
    for audio_key in store.list_keys(audio_prefix, suffix=".wav"):
        mtime = store.get_mtime(audio_key)
        if mtime is not None and mtime < cutoff:
            store.delete(audio_key)


def _maybe_cleanup(store, root: str, fp: str, server_key: str, retention_seconds: float) -> None:
    skp = server_key_partition(server_key)
    cleanup_key = storage_key(root, fp, skp)
    now = time.time()
    if now - _last_cleanup.get(cleanup_key, 0.0) < _CLEANUP_INTERVAL:
        return
    _last_cleanup[cleanup_key] = now
    _cleanup_tenant(store, root, (fp, skp), retention_seconds)


async def _rate_ok(bucket: Dict[str, List[float]], key: str, limit: int) -> bool:
    now = time.time()
    async with _evidence_lock:
        lst = bucket.get(key, [])
        lst = [t for t in lst if now - t < 60.0]
        if not lst:
            bucket.pop(key, None)
        if len(lst) >= limit:
            if lst:
                bucket[key] = lst
            return False
        lst.append(now)
        bucket[key] = lst
        return True


async def check_evidence_rate_limits(fingerprint: str, client_ip: str) -> bool:
    if not await _rate_ok(_evidence_rate, fingerprint, MAX_EVIDENCE_PER_MINUTE_PER_LICENSE):
        return False
    if client_ip:
        if not await _rate_ok(_evidence_ip_rate, client_ip, MAX_EVIDENCE_PER_MINUTE_PER_IP):
            return False
    return True


def _parse_ts_iso(s: str) -> Optional[float]:
    if not s:
        return None
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00")).timestamp()
    except (ValueError, TypeError):
        return None


def _read_evidence_file(store, key: str, pl: str, cutoff: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    try:
        for line in store.read_text_lines(key):
            line = line.strip()
            if not line:
                continue
            try:
                o = json.loads(line)
            except json.JSONDecodeError:
                continue
            if o.get("player", "").lower() != pl:
                continue
            ts = o.get("ts") or ""
            tsec = _parse_ts_iso(ts) if isinstance(ts, str) else None
            if tsec is not None and tsec < cutoff:
                continue
            out.append(
                {
                    "ts": ts,
                    "transcript": o.get("transcript", "") or "",
                    "session_id": o.get("session_id", ""),
                    "detected_language": o.get("detected_language", ""),
                }
            )
    except OSError as e:
        logger.warning("report evidence read failed: %s", e)
    return out


def query_report_evidence(
    cfg: dict,
    license_fingerprint: str,
    server_key_for_partition: str,
    player_query: str,
    seconds: int,
) -> List[Dict[str, Any]]:
    rb = cfg.get("report_buffer") or {}
    if not rb.get("enabled"):
        return []
    fp = _sanitize_fingerprint(license_fingerprint or "")
    if not fp:
        return []
    root = rb.get("path", "report_buffer/")
    store = get_storage()
    cutoff = time.time() - float(seconds)
    pq = sanitize_player_query(player_query)
    if not pq:
        return []
    pl = pq.lower()

    primary = _tenant_transcript_key(root, fp, server_key_for_partition)
    legacy = _legacy_transcript_key(root, fp)
    seen: Set[Tuple[str, str, str]] = set()
    merged: List[Dict[str, Any]] = []
    for key in (primary, legacy):
        for row in _read_evidence_file(store, key, pl, cutoff):
            dedupe = (
                row.get("ts") or "",
                row.get("transcript") or "",
                row.get("session_id") or "",
                row.get("detected_language") or "",
            )
            if dedupe in seen:
                continue
            seen.add(dedupe)
            merged.append(row)
    merged.sort(key=lambda r: r.get("ts") or "")
    return merged


def query_report_audio(
    cfg: dict,
    license_fingerprint: str,
    server_key_for_partition: str,
    player_query: str,
    session_id: str,
) -> Optional[bytes]:
    rb = cfg.get("report_buffer") or {}
    if not rb.get("enabled") or not rb.get("save_audio"):
        return None
    fp = _sanitize_fingerprint(license_fingerprint or "")
    if not fp:
        return None
    pq = sanitize_player_query(player_query)
    if not pq:
        return None
    sid = (session_id or "").strip()
    if not sid:
        return None

    root = rb.get("path", "report_buffer/")
    store = get_storage()
    skp = server_key_partition(server_key_for_partition)
    audio_prefix = storage_key(root, fp, skp, "audio")
    if not store.is_dir(audio_prefix):
        return None

    safe_player = re.sub(r"[^a-zA-Z0-9_]", "_", pq)[:32]
    safe_session = re.sub(r"[^a-zA-Z0-9_-]", "_", sid)[:48]
    candidates = sorted(
        store.list_keys(audio_prefix, suffix=f"_{safe_player}_{safe_session}.wav"),
        reverse=True,
    )
    if not candidates:
        candidates = sorted(
            store.list_keys(audio_prefix, suffix=f"_{safe_player}.wav"),
            reverse=True,
        )
    if not candidates:
        return None
    return store.read_bytes(candidates[0])


async def report_buffer_append_async(
    cfg: dict,
    license_fingerprint: str,
    server_key: str,
    record: Dict[str, Any],
    audio_wav: Optional[bytes] = None,
) -> None:
    await asyncio.to_thread(report_buffer_append, cfg, license_fingerprint, server_key, record, audio_wav)
