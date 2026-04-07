from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

_last_cleanup: Dict[str, float] = {}
_CLEANUP_INTERVAL = 3600.0  # run at most once per hour per tenant

logger = logging.getLogger(__name__)

_FINGERPRINT_RE = re.compile(r"^[a-f0-9]{64}$")

_evidence_rate: Dict[str, List[float]] = {}
_evidence_lock = asyncio.Lock()
MAX_EVIDENCE_PER_MINUTE_PER_LICENSE = 60

# Optional per-IP rate limit (abuse on shared pools)
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
    """Stable 64-hex directory name from the same server_key string used in WebSocket auth."""
    raw = (server_key or "").strip()
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _tenant_transcript_path(root: Path, license_fp: str, server_key: str) -> Path:
    skp = server_key_partition(server_key)
    return root / license_fp / skp / "transcripts.jsonl"


def _legacy_transcript_path(root: Path, license_fp: str) -> Path:
    return root / license_fp / "transcripts.jsonl"


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
    root = Path(rb.get("path", "report_buffer/"))
    root.mkdir(parents=True, exist_ok=True)
    retention_seconds = float(rb.get("retention_seconds", 604800))
    _maybe_cleanup(root, fp, server_key, retention_seconds)
    path = _tenant_transcript_path(root, fp, server_key)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, separators=(",", ":"), ensure_ascii=False) + "\n"
    with open(path, "a", encoding="utf-8") as f:
        f.write(line)
    if rb.get("save_audio") and audio_wav:
        try:
            safe = re.sub(r"[^a-zA-Z0-9_]", "_", str(record.get("player", "unknown")))[:32]
            ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f")
            audio_dir = path.parent / "audio"
            audio_dir.mkdir(parents=True, exist_ok=True)
            (audio_dir / f"{ts}_{safe}.wav").write_bytes(audio_wav)
        except OSError as e:
            logger.warning("report_buffer audio save failed: %s", e)


def _cleanup_tenant(tenant_dir: Path, retention_seconds: float) -> None:
    """Delete transcript lines and audio files older than retention_seconds."""
    cutoff = time.time() - retention_seconds

    # Prune old lines from transcripts.jsonl
    transcript_path = tenant_dir / "transcripts.jsonl"
    if transcript_path.is_file():
        try:
            kept = []
            with open(transcript_path, encoding="utf-8") as f:
                for line in f:
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
                        ts_sec = cutoff + 1  # keep lines with unparseable timestamps
                    if ts_sec >= cutoff:
                        kept.append(line)
            with open(transcript_path, "w", encoding="utf-8") as f:
                for line in kept:
                    f.write(line + "\n")
        except OSError as e:
            logger.warning("report_buffer retention cleanup (transcripts) failed: %s", e)

    # Delete old audio files
    audio_dir = tenant_dir / "audio"
    if audio_dir.is_dir():
        try:
            for wav in audio_dir.glob("*.wav"):
                if wav.stat().st_mtime < cutoff:
                    wav.unlink(missing_ok=True)
        except OSError as e:
            logger.warning("report_buffer retention cleanup (audio) failed: %s", e)


def _maybe_cleanup(root: Path, fp: str, server_key: str, retention_seconds: float) -> None:
    tenant_dir = _tenant_transcript_path(root, fp, server_key).parent
    key = str(tenant_dir)
    now = time.time()
    if now - _last_cleanup.get(key, 0.0) < _CLEANUP_INTERVAL:
        return
    _last_cleanup[key] = now
    _cleanup_tenant(tenant_dir, retention_seconds)


async def _rate_ok(bucket: Dict[str, List[float]], key: str, limit: int) -> bool:
    now = time.time()
    async with _evidence_lock:
        lst = bucket.setdefault(key, [])
        lst[:] = [t for t in lst if now - t < 60.0]
        if len(lst) >= limit:
            return False
        lst.append(now)
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


def _read_evidence_file(path: Path, pl: str, cutoff: float) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    if not path.is_file():
        return out
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
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
    root = Path(rb.get("path", "report_buffer/"))
    cutoff = time.time() - float(seconds)
    pq = sanitize_player_query(player_query)
    if not pq:
        return []
    pl = pq.lower()

    primary = _tenant_transcript_path(root, fp, server_key_for_partition)
    legacy = _legacy_transcript_path(root, fp)
    seen: Set[Tuple[str, str, str]] = set()
    merged: List[Dict[str, Any]] = []
    for path in (primary, legacy):
        for row in _read_evidence_file(path, pl, cutoff):
            key = (row.get("ts") or "", row.get("transcript") or "", row.get("session_id") or "", row.get("detected_language") or "")
            if key in seen:
                continue
            seen.add(key)
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
    """Return the most recent audio WAV matching the given player+session, or None."""
    rb = cfg.get("report_buffer") or {}
    if not rb.get("enabled") or not rb.get("save_audio"):
        return None
    fp = _sanitize_fingerprint(license_fingerprint or "")
    if not fp:
        return None
    pq = sanitize_player_query(player_query)
    if not pq:
        return None

    root = Path(rb.get("path", "report_buffer/"))
    skp = server_key_partition(server_key_for_partition)
    audio_dir = root / fp / skp / "audio"
    if not audio_dir.is_dir():
        return None

    safe_player = re.sub(r"[^a-zA-Z0-9_]", "_", pq)[:32]
    candidates = sorted(audio_dir.glob(f"*_{safe_player}.wav"), reverse=True)
    if not candidates:
        return None
    return candidates[0].read_bytes()


async def report_buffer_append_async(
    cfg: dict,
    license_fingerprint: str,
    server_key: str,
    record: Dict[str, Any],
    audio_wav: Optional[bytes] = None,
) -> None:
    await asyncio.to_thread(report_buffer_append, cfg, license_fingerprint, server_key, record, audio_wav)
