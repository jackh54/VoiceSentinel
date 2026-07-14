from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.error
import urllib.request
from typing import Any, Optional

logger = logging.getLogger(__name__)

LOAD_REPORT_INTERVAL_SECONDS = 30
REBALANCE_LOAD_THRESHOLD = 70
REBALANCE_GAP_PERCENT = 25
REBALANCE_COOLDOWN_SECONDS = 600
REBALANCE_INACTIVITY_SECONDS = 90


def _pool_load_block(config: dict) -> dict:
    block = config.get("pool_server_load")
    return block if isinstance(block, dict) else {}


def cdn_load_settings(config: dict) -> Optional[dict[str, Any]]:
    if not config.get("pool_server"):
        return None
    block = _pool_load_block(config)
    load_url = str(block.get("directory_load_url") or "").strip()
    report_key = str(block.get("report_key") or "").strip()
    server_ip = str(block.get("server_ip") or "").strip()
    server_id_raw = block.get("server_id")
    try:
        server_id = int(server_id_raw)
    except (TypeError, ValueError):
        return None
    if not (load_url and report_key and server_ip and server_id > 0):
        return None
    return {
        "directory_load_url": load_url,
        "report_key": report_key,
        "server_id": server_id,
        "server_ip": server_ip,
    }


def cdn_load_enabled(config: dict) -> bool:
    return cdn_load_settings(config) is not None


def directory_servers_url(load_url: str) -> str:
    url = load_url.rstrip("/")
    if url.endswith("/load"):
        return url[: -len("/load")] + "/servers"
    return url.replace("/load", "/servers")


def _cpu_percent() -> float:
    try:
        import psutil  # type: ignore

        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return 0.0


def compute_load_metrics(ws_manager: Any, config: dict) -> dict[str, Any]:
    processing = config.get("processing", {}) or {}
    try:
        queue_max = max(1, int(processing.get("queue_max_size", 500)))
    except (TypeError, ValueError):
        queue_max = 500

    qsize = ws_manager._processing_queue.qsize() if ws_manager._processing_queue else 0
    queue_util = 100.0 * min(qsize, queue_max) / queue_max

    active_sessions = len(ws_manager.active_connections)
    max_sessions = max(10, queue_max // 5)
    session_util = 100.0 * min(active_sessions, max_sessions) / max_sessions

    cpu_percent = _cpu_percent()
    raw = max(queue_util, session_util, cpu_percent)
    server_load = max(1, min(100, round(raw)))

    return {
        "serverLoad": server_load,
        "activeSessions": active_sessions,
        "cpuPercent": round(cpu_percent, 2),
        "queueUtilizationPercent": round(queue_util, 2),
    }


def _http_json(
    method: str,
    url: str,
    bearer: str,
    body: Optional[dict] = None,
    timeout: float = 10.0,
) -> tuple[int, Optional[dict]]:
    data = None
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {bearer}",
        "User-Agent": "VoiceSentinel-Processor",
    }
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read().decode("utf-8")
            if not raw.strip():
                return resp.status, None
            return resp.status, json.loads(raw)
    except urllib.error.HTTPError as e:
        try:
            err_body = e.read().decode("utf-8")
            parsed = json.loads(err_body) if err_body.strip() else None
        except Exception:
            parsed = None
        return e.code, parsed
    except Exception:
        return 0, None


async def post_load_report(ws_manager: Any, config: dict) -> bool:
    settings = cdn_load_settings(config)
    if not settings:
        return False

    metrics = compute_load_metrics(ws_manager, config)
    payload = {
        "serverId": settings["server_id"],
        "serverIp": settings["server_ip"],
        "serverLoad": metrics["serverLoad"],
        "activeSessions": metrics["activeSessions"],
        "cpuPercent": metrics["cpuPercent"],
    }
    status, _ = await asyncio.to_thread(
        _http_json,
        "POST",
        settings["directory_load_url"],
        settings["report_key"],
        payload,
    )
    if status and 200 <= status < 300:
        logger.debug(
            "CDN load report ok serverLoad=%s activeSessions=%s",
            metrics["serverLoad"],
            metrics["activeSessions"],
        )
        return True
    logger.warning("CDN load report failed HTTP %s", status or "error")
    return False


def _parse_directory_peers(data: Optional[dict], self_server_ip: str) -> list[dict]:
    if not data:
        return []
    servers = data
    if isinstance(data, dict):
        servers = data.get("servers")
    if not isinstance(servers, list):
        return []
    self_norm = self_server_ip.strip().rstrip("/").lower()
    peers: list[dict] = []
    for row in servers:
        if not isinstance(row, dict):
            continue
        ip = row.get("serverIp")
        if not isinstance(ip, str) or not ip.strip():
            continue
        ip_trim = ip.strip().rstrip("/")
        if ip_trim.lower() == self_norm:
            continue
        try:
            load = int(row.get("serverLoad", 100))
        except (TypeError, ValueError):
            load = 100
        load = max(1, min(100, load))
        peers.append({"serverIp": ip_trim, "serverLoad": load})
    return peers


async def fetch_directory_peers(config: dict) -> list[dict]:
    settings = cdn_load_settings(config)
    if not settings:
        return []
    servers_url = directory_servers_url(settings["directory_load_url"])
    status, data = await asyncio.to_thread(
        _http_json, "GET", servers_url, settings["report_key"], None
    )
    if not status or status < 200 or status >= 300:
        logger.warning("CDN directory fetch for rebalance failed HTTP %s", status or "error")
        return []
    return _parse_directory_peers(data, settings["server_ip"])


class CdnLoadMonitor:
    def __init__(self, ws_manager: Any, config: dict):
        self.ws_manager = ws_manager
        self.config = config
        self._last_rebalance_at = 0.0
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if not cdn_load_enabled(self.config):
            return
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def _run(self) -> None:
        logger.info(
            "CDN load monitor started (report every %ss, rebalance threshold=%s gap=%s cooldown=%ss)",
            LOAD_REPORT_INTERVAL_SECONDS,
            REBALANCE_LOAD_THRESHOLD,
            REBALANCE_GAP_PERCENT,
            REBALANCE_COOLDOWN_SECONDS,
        )
        while True:
            try:
                await self._tick()
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("CDN load monitor tick error: %s", e)
            await asyncio.sleep(LOAD_REPORT_INTERVAL_SECONDS)

    async def _tick(self) -> None:
        await post_load_report(self.ws_manager, self.config)
        await self._maybe_rebalance()

    async def _maybe_rebalance(self) -> None:
        metrics = compute_load_metrics(self.ws_manager, self.config)
        server_load = metrics["serverLoad"]
        if server_load < REBALANCE_LOAD_THRESHOLD:
            return

        last_activity = getattr(self.ws_manager, "_last_activity_at", 0.0)
        if last_activity and (time.time() - last_activity) < REBALANCE_INACTIVITY_SECONDS:
            return

        now = time.time()
        if now - self._last_rebalance_at < REBALANCE_COOLDOWN_SECONDS:
            return

        if not self.ws_manager.active_connections:
            return

        peers = await fetch_directory_peers(self.config)
        if not peers:
            return

        best_other_load = min(p["serverLoad"] for p in peers)
        gap = server_load - best_other_load
        if gap < REBALANCE_GAP_PERCENT:
            return

        payload = {
            "reason": "load",
            "serverLoad": server_load,
            "bestOtherLoad": best_other_load,
        }
        sent = await self.ws_manager.broadcast_rebalance(payload)
        if sent > 0:
            self._last_rebalance_at = now
            logger.info(
                "CDN rebalance signal sent to %s client(s) (load=%s bestOther=%s gap=%s)",
                sent,
                server_load,
                best_other_load,
                gap,
            )
