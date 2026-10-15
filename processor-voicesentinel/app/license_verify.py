"""Server-side license verification for the VoiceSentinel processor.

This is the enforcement point that a cracked/patched plugin jar *cannot* bypass:
the check runs on the processor (which PandaScript operates), not in the client
jar. A crack can patch out the plugin's own LicenseManager, but it still has to
connect to a processor to get any transcription done. On the public pool that
processor is ours, so we independently confirm the presented license key is real,
active, and entitled to the product before doing any work for the connection.

Design (matches the posture chosen for VoiceSentinel):
  * Every license is verified against the PandaScript key API (`/api/client/verify`,
    Bearer = the license key) with primary -> fallback endpoint failover.
  * Good results are cached for `cache_ttl_seconds` so we do not hammer the API.
  * Grace window: if the key API is unreachable, a license that verified OK within
    the last `grace_seconds` keeps working; a never-verified license is rejected.
  * `mode: "log_only"` verifies and reports but never rejects — for a safe rollout
    where you want to collect offender IPs for a while before flipping enforcement.

No third-party HTTP deps: stdlib urllib run off-thread, same pattern as cdn_load.py.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import logging
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger("voicesentinel.license_verify")

# Product id for VoiceSentinel in the PandaScript key system (same value the plugin sends).
DEFAULT_PRODUCT_ID = "72862"

# key.pandascript.dev proxies to the Cloudflare worker and can serve from its backup DB
# when the worker/D1 is down, so it is the most resilient primary. The worker is the fallback.
DEFAULT_VERIFY_URLS = [
    "https://key.pandascript.dev/api/client/verify",
    "https://pandascript-license-worker.pandascript.workers.dev/api/client/verify",
]

# Statuses returned to the caller.
VALID = "valid"
INVALID = "invalid"
UNAVAILABLE = "unavailable"


@dataclass
class VerifyDecision:
    """Outcome of a verification for one connection."""
    allow: bool
    status: str          # valid | invalid | grace | unavailable | disabled | log_only_flagged
    reason: str          # short machine-ish reason for the audit log
    message: str = ""    # human message from the key API, if any
    max_servers: Optional[int] = None
    current_servers: Optional[int] = None


@dataclass
class _CacheEntry:
    status: str          # valid | invalid
    checked_ts: float    # when we last got a definitive answer
    last_valid_ts: float # when we last saw VALID (drives the grace window)
    message: str = ""
    max_servers: Optional[int] = None
    current_servers: Optional[int] = None


def _coerce_int(value) -> Optional[int]:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def client_handshake_signature(server_key: str, license_key: str, product_id: str, version: str) -> str:
    """Recompute the plugin's auth-handshake HMAC (see WebSocketHostBridgeImpl)."""
    canonical = f"{license_key or ''}\n{product_id or ''}\n{version or ''}".encode("utf-8")
    return hmac.new(
        (server_key or "").encode("utf-8"), canonical, hashlib.sha256
    ).hexdigest()


def classify_client_integrity(expected_sig: str, provided_sig: str) -> str:
    """'ok' | 'mismatch' | 'absent' — a tamper signal, never a hard gate on its own."""
    if not provided_sig:
        return "absent"
    return "ok" if hmac.compare_digest(expected_sig, provided_sig) else "mismatch"


class LicenseVerifier:
    def __init__(self, config: dict):
        cfg = (config or {}).get("license_enforcement") or {}
        pool_server = bool((config or {}).get("pool_server", False))

        # Default ON for the public pool (where cracked jars actually connect), OFF for
        # self-hosted installs unless explicitly enabled — a self-hosted operator controls
        # their own processor, so a server-side check there buys nothing and could break
        # an airgapped box. Operators can still opt in with license_enforcement.enabled.
        self.enabled = bool(cfg.get("enabled", pool_server))

        mode = str(cfg.get("mode", "enforce")).strip().lower()
        self.log_only = mode == "log_only"

        self.product_id = str(cfg.get("product_id") or DEFAULT_PRODUCT_ID).strip()

        urls = cfg.get("verify_urls")
        if isinstance(urls, list) and urls:
            self.verify_urls = [str(u).strip() for u in urls if str(u).strip()]
        else:
            self.verify_urls = list(DEFAULT_VERIFY_URLS)

        self.cache_ttl = max(30, int(cfg.get("cache_ttl_seconds", 900)))
        self.negative_ttl = max(30, int(cfg.get("negative_cache_seconds", 120)))
        self.grace_seconds = max(0, int(cfg.get("grace_seconds", 3600)))
        self.timeout = max(2, int(cfg.get("timeout_seconds", 8)))

        self._cache: dict[str, _CacheEntry] = {}
        self._lock = asyncio.Lock()

    def describe(self) -> str:
        if not self.enabled:
            return "disabled"
        return (
            f"mode={'log_only' if self.log_only else 'enforce'} "
            f"product={self.product_id} ttl={self.cache_ttl}s grace={self.grace_seconds}s"
        )

    async def verify(self, license_key: str, fingerprint: str) -> VerifyDecision:
        if not self.enabled:
            return VerifyDecision(allow=True, status="disabled", reason="enforcement_disabled")

        license_key = (license_key or "").strip()
        if not license_key or not fingerprint:
            # No usable license at all. Honour log_only so a rollout never hard-breaks.
            return VerifyDecision(
                allow=self.log_only,
                status=("log_only_flagged" if self.log_only else INVALID),
                reason="missing_license",
                message="No license key presented",
            )

        now = time.time()

        async with self._lock:
            cached = self._cache.get(fingerprint)
            if cached is not None:
                if cached.status == VALID and (now - cached.checked_ts) < self.cache_ttl:
                    return VerifyDecision(True, VALID, "cache_valid", cached.message,
                                          cached.max_servers, cached.current_servers)
                if cached.status == INVALID and (now - cached.checked_ts) < self.negative_ttl:
                    return self._reject(INVALID, "cache_invalid", cached.message)

        status, message, max_servers, current_servers = await self._remote_verify(license_key)

        async with self._lock:
            prev = self._cache.get(fingerprint)
            if status == VALID:
                self._cache[fingerprint] = _CacheEntry(
                    VALID, now, now, message, max_servers, current_servers)
                return VerifyDecision(True, VALID, "api_valid", message, max_servers, current_servers)

            if status == INVALID:
                last_valid = prev.last_valid_ts if prev else 0.0
                self._cache[fingerprint] = _CacheEntry(
                    INVALID, now, last_valid, message, max_servers, current_servers)
                return self._reject(INVALID, "api_invalid", message)

            # UNAVAILABLE — apply the grace window off the last successful verify.
            last_valid = prev.last_valid_ts if prev else 0.0
            if last_valid and (now - last_valid) < self.grace_seconds:
                return VerifyDecision(True, "grace", "api_unavailable_grace", message,
                                      prev.max_servers if prev else None,
                                      prev.current_servers if prev else None)
            return self._reject(UNAVAILABLE, "api_unavailable_no_grace", message)

    def _reject(self, status: str, reason: str, message: str) -> VerifyDecision:
        """In log_only mode we always allow but surface the flag for the audit log."""
        if self.log_only:
            return VerifyDecision(True, "log_only_flagged", reason, message)
        return VerifyDecision(False, status, reason, message)

    async def _remote_verify(self, license_key: str):
        """Try each endpoint; a definitive valid/invalid wins immediately, only
        transient failures fall through to the next endpoint."""
        result = (UNAVAILABLE, "", None, None)
        for url in self.verify_urls:
            result = await asyncio.to_thread(self._remote_verify_one, url, license_key)
            if result[0] != UNAVAILABLE:
                return result
        return result

    def _remote_verify_one(self, url: str, license_key: str):
        sep = "&" if "?" in url else "?"
        full = f"{url}{sep}product={urllib.parse.quote(self.product_id)}"
        headers = {
            "Accept": "application/json",
            "Authorization": f"Bearer {license_key}",
            "User-Agent": "VoiceSentinel-Processor-LicenseVerify",
        }
        req = urllib.request.Request(full, headers=headers, method="GET")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                code = resp.status
                raw = resp.read().decode("utf-8")
                body = json.loads(raw) if raw.strip() else None
        except urllib.error.HTTPError as e:
            code = e.code
            try:
                err = e.read().decode("utf-8")
                body = json.loads(err) if err.strip() else None
            except Exception:
                body = None
        except Exception as e:  # DNS, timeout, connection reset, TLS...
            logger.debug("license verify network error for %s: %s", url, e)
            return (UNAVAILABLE, "network error", None, None)

        return self._classify(code, body)

    @staticmethod
    def _classify(code: int, body):
        """Map an HTTP response to VALID / INVALID / UNAVAILABLE.

        The verify endpoint returns {valid: bool, message, maxServers, currentServers};
        the worker's client route uses {status_id: SUCCESS|FAILED, retryable}. Handle both.
        """
        max_servers = current_servers = None
        message = ""
        if isinstance(body, dict):
            message = str(body.get("message") or "")
            max_servers = _coerce_int(body.get("maxServers"))
            current_servers = _coerce_int(body.get("currentServers"))

        # Rate-limited / server errors are always transient.
        if code in (408, 429) or code >= 500:
            return (UNAVAILABLE, message or f"HTTP {code}", None, None)

        if isinstance(body, dict) and body.get("retryable") is True:
            return (UNAVAILABLE, message or "retryable", None, None)

        if not isinstance(body, dict):
            return (UNAVAILABLE, f"HTTP {code} (no body)", None, None)

        # Explicit valid flag (verify route).
        if "valid" in body:
            if body.get("valid") is True and 200 <= code < 300:
                return (VALID, message, max_servers, current_servers)
            if body.get("valid") is False:
                return (INVALID, message or "License not valid", max_servers, current_servers)

        # Worker status_id shape.
        status_id = str(body.get("status_id") or "").upper()
        if status_id == "SUCCESS" and 200 <= code < 300:
            return (VALID, message, max_servers, current_servers)
        if status_id == "FAILED":
            # Maintenance / "try again later" should not nuke paying users.
            low = (message or "").lower()
            if "maintenance" in low or "temporarily unavailable" in low or "try again later" in low:
                return (UNAVAILABLE, message, None, None)
            return (INVALID, message or "License validation failed", max_servers, current_servers)

        # 401/403 without a clear shape = reject; anything else = transient.
        if code in (400, 401, 403):
            return (INVALID, message or f"HTTP {code}", max_servers, current_servers)
        return (UNAVAILABLE, message or f"HTTP {code}", None, None)
