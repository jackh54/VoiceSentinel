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

# Move a single Minecraft server to the high_volume pool when it dominates the queue.
HEAVY_CLIENT_LOAD_THRESHOLD = 50
HEAVY_CLIENT_SHARE_THRESHOLD = 0.35
HEAVY_CLIENT_MIN_ENQUEUES = 20
HEAVY_CLIENT_WINDOW_SECONDS = 120
HEAVY_CLIENT_COOLDOWN_SECONDS = 300


def _pool_load_block(config: dict) -> dict:
    block = config.get("pool_server_load")
    return block if isinstance(block, dict) else {}


def _parse_pool_name(value: Any) -> str:
    if not isinstance(value, str):
        return "default"
    normalized = value.strip().lower()
    if normalized in ("high_volume", "overflow", "big"):
        return "high_volume"
    return "default"


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
        "pool": _parse_pool_name(block.get("pool")),
    }


def cdn_load_enabled(config: dict) -> bool:
    return cdn_load_settings(config) is not None


def directory_servers_url(load_url: str) -> str:
    url = load_url.rstrip("/")
    if url.endswith("/load"):
        return url[: -len("/load")] + "/servers"
    return url.replace("/load", "/servers")


_cpu_primed = False
_cpu_unavailable_logged = False


def _prime_cpu_sampler() -> None:
    """Non-blocking prime so the first real sample is not always 0.0."""
    global _cpu_primed, _cpu_unavailable_logged
    try:
        import psutil  # type: ignore

        psutil.cpu_percent(interval=None)
        _cpu_primed = True
    except Exception as e:
        if not _cpu_unavailable_logged:
            logger.warning("CPU monitoring unavailable (install psutil): %s", e)
            _cpu_unavailable_logged = True


def _cpu_percent() -> float:
    global _cpu_primed, _cpu_unavailable_logged
    try:
        import psutil  # type: ignore

        if not _cpu_primed:
            psutil.cpu_percent(interval=None)
            _cpu_primed = True
            return 0.0
        return float(psutil.cpu_percent(interval=None))
    except Exception as e:
        if not _cpu_unavailable_logged:
            logger.warning("CPU monitoring unavailable (install psutil): %s", e)
            _cpu_unavailable_logged = True
        return 0.0


def compute_load_metrics(ws_manager: Any, config: dict) -> dict[str, Any]:
    processing = config.get("processing", {}) or {}
    try:
        queue_max = max(1, int(processing.get("queue_max_size", 500)))
    except (TypeError, ValueError):
        queue_max = 500

    qsize = ws_manager._processing_queue.qsize() if ws_manager._processing_queue else 0
    # Bound depth so utilization never exceeds 100% if the queue is over capacity.
    bounded_qsize = min(qsize, queue_max)
    queue_util = 100.0 * bounded_qsize / queue_max

    active_sessions = len(ws_manager.active_connections)
    cpu_percent = _cpu_percent()
    # Directory load is queue depth / max only — CPU and sessions are informational.
    # Empty queue must report 0 (not 1) so balancers prefer idle processors.
    server_load = max(0, min(100, round(queue_util)))

    return {
        "serverLoad": server_load,
        "activeSessions": active_sessions,
        "cpuPercent": round(cpu_percent, 2),
        "processingQueueSize": qsize,
        "queueMaxSize": queue_max,
        "queueUtilizationPercent": round(queue_util, 2),
    }


def _http_json(
    method: str,
    url: str,
    bearer: str,
    body: Optional[dict] = None,
    timeout: float = 10.0,
) -> tuple[int, Any]:
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
        "processingQueueSize": metrics["processingQueueSize"],
        "queueMaxSize": metrics["queueMaxSize"],
        "queueUtilizationPercent": metrics["queueUtilizationPercent"],
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
            "CDN load report ok serverLoad=%s queue=%s/%s cpuPercent=%s",
            metrics["serverLoad"],
            metrics["processingQueueSize"],
            metrics["queueMaxSize"],
            metrics["cpuPercent"],
        )
        return True
    logger.warning("CDN load report failed HTTP %s", status or "error")
    return False


def _parse_server_rows(rows: Any, self_server_ip: str) -> list[dict]:
    if not isinstance(rows, list):
        return []
    self_norm = self_server_ip.strip().rstrip("/").lower()
    peers: list[dict] = []
    for row in rows:
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
        load = max(0, min(100, load))
        peers.append({"serverIp": ip_trim, "serverLoad": load})
    return peers


def _parse_directory_payload(data: Any, self_server_ip: str) -> dict[str, list[dict]]:
    """Split directory response into default peers and high_volume targets."""
    if isinstance(data, list):
        return {
            "servers": _parse_server_rows(data, self_server_ip),
            "highVolumeServers": [],
        }
    if not isinstance(data, dict):
        return {"servers": [], "highVolumeServers": []}
    return {
        "servers": _parse_server_rows(data.get("servers"), self_server_ip),
        "highVolumeServers": _parse_server_rows(data.get("highVolumeServers"), self_server_ip),
    }


async def fetch_directory(config: dict) -> dict[str, list[dict]]:
    settings = cdn_load_settings(config)
    if not settings:
        return {"servers": [], "highVolumeServers": []}
    servers_url = directory_servers_url(settings["directory_load_url"])
    status, data = await asyncio.to_thread(
        _http_json, "GET", servers_url, settings["report_key"], None
    )
    if not status or status < 200 or status >= 300:
        logger.warning("CDN directory fetch for rebalance failed HTTP %s", status or "error")
        return {"servers": [], "highVolumeServers": []}
    return _parse_directory_payload(data, settings["server_ip"])


async def fetch_directory_peers(config: dict) -> list[dict]:
    settings = cdn_load_settings(config)
    directory = await fetch_directory(config)
    if not settings:
        return []
    if settings["pool"] == "high_volume":
        return directory["highVolumeServers"]
    return directory["servers"]


def pick_lowest_load(servers: list[dict]) -> Optional[dict]:
    if not servers:
        return None
    return min(servers, key=lambda p: p["serverLoad"])


def high_volume_mark_url(load_url: str) -> str:
    url = load_url.rstrip("/")
    if url.endswith("/load"):
        return url[: -len("/load")] + "/high-volume"
    return url.replace("/load", "/high-volume")


async def mark_license_high_volume(config: dict, license_key: str) -> bool:
    settings = cdn_load_settings(config)
    if not settings or not license_key:
        return False
    status, body = await asyncio.to_thread(
        _http_json,
        "POST",
        high_volume_mark_url(settings["directory_load_url"]),
        settings["report_key"],
        {"licenseKey": license_key},
    )
    if status and 200 <= status < 300:
        return True
    logger.warning(
        "Failed to mark license for high_volume pool HTTP %s body=%s",
        status or "error",
        body,
    )
    return False


class CdnLoadMonitor:
    def __init__(self, ws_manager: Any, config: dict):
        self.ws_manager = ws_manager
        self.config = config
        self._last_rebalance_at = 0.0
        self._last_heavy_move_at: dict[str, float] = {}
        self._task: Optional[asyncio.Task] = None

    def start(self) -> None:
        if not cdn_load_enabled(self.config):
            return
        _prime_cpu_sampler()
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
        settings = cdn_load_settings(self.config) or {}
        logger.info(
            "CDN load monitor started (pool=%s report every %ss, rebalance threshold=%s gap=%s cooldown=%ss)",
            settings.get("pool", "default"),
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
        await self._maybe_move_heavy_clients()
        await self._maybe_rebalance()

    async def _maybe_move_heavy_clients(self) -> None:
        settings = cdn_load_settings(self.config)
        if not settings or settings["pool"] == "high_volume":
            return

        metrics = compute_load_metrics(self.ws_manager, self.config)
        if metrics["serverLoad"] < HEAVY_CLIENT_LOAD_THRESHOLD:
            return

        counts = self.ws_manager.recent_enqueue_counts(HEAVY_CLIENT_WINDOW_SECONDS)
        total = sum(counts.values())
        if total < HEAVY_CLIENT_MIN_ENQUEUES:
            return

        heavy_clients = [
            (client_id, count)
            for client_id, count in counts.items()
            if count / total >= HEAVY_CLIENT_SHARE_THRESHOLD
            and client_id in self.ws_manager.active_connections
        ]
        if not heavy_clients:
            return

        directory = await fetch_directory(self.config)
        if not directory["highVolumeServers"]:
            logger.warning(
                "Heavy client detected but no high_volume processors are available in the directory"
            )
            return

        best_hv = pick_lowest_load(directory["highVolumeServers"])
        if not best_hv:
            return
        # Don't shove more servers onto HV when it's already worse than this processor.
        if best_hv["serverLoad"] + 10 >= metrics["serverLoad"]:
            logger.info(
                "Skipping heavy-client move; high_volume load=%s is not better than local load=%s",
                best_hv["serverLoad"],
                metrics["serverLoad"],
            )
            return

        now = time.time()
        for client_id, count in sorted(heavy_clients, key=lambda item: item[1], reverse=True):
            last = self._last_heavy_move_at.get(client_id, 0.0)
            if now - last < HEAVY_CLIENT_COOLDOWN_SECONDS:
                continue

            license_key = self.ws_manager.get_license_plain(client_id)
            if not license_key:
                logger.warning(
                    "Heavy client %s has no license key; cannot mark for high_volume pool",
                    client_id,
                )
                continue

            marked = await mark_license_high_volume(self.config, license_key)
            if not marked:
                continue

            share = count / total
            # Existing plugin rebalance: refetch directory and connect to lowest-load entry.
            # After the mark, that license's directory only contains high_volume processors.
            payload = {
                "reason": "load",
                "serverLoad": metrics["serverLoad"],
                "bestOtherLoad": best_hv["serverLoad"],
            }
            sent = await self.ws_manager.send_rebalance(client_id, payload)
            if sent:
                self._last_heavy_move_at[client_id] = now
                logger.info(
                    "Marked heavy client %s for high_volume pool and sent rebalance (share=%.0f%% count=%s/%s load=%s)",
                    client_id,
                    share * 100,
                    count,
                    total,
                    metrics["serverLoad"],
                )

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
