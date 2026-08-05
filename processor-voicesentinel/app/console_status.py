from __future__ import annotations

import os
import sys
import time
from collections import deque
from typing import Any, Callable, Optional

from rich.console import Console, Group
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

RuntimeProvider = Callable[[], dict[str, Any]]


def _is_tty() -> bool:
    try:
        return os.isatty(sys.stdout.fileno())
    except Exception:
        return False


def _is_limited_console() -> bool:
    """Pterodactyl / web consoles get a PTY but mishandle Rich Live (flash + clipped lines)."""
    if os.environ.get("P_SERVER_UUID") or os.environ.get("PTERODACTYL"):
        return True
    if os.environ.get("CONSOLE_STABLE", "").strip().lower() in ("1", "true", "yes"):
        return True
    home = os.environ.get("HOME", "")
    if home.rstrip("/") == "/home/container":
        return True
    term = (os.environ.get("TERM") or "").strip().lower()
    if term in ("dumb",):
        return True
    return False


def _format_ago(ts: Optional[float], now: float) -> str:
    if ts is None:
        return "—"
    return f"{max(0, int(now - ts))}s ago"


def _format_uptime(started_at: float, now: float) -> str:
    seconds = max(0, int(now - started_at))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h {minutes:02d}m {secs:02d}s"
    if minutes:
        return f"{minutes}m {secs:02d}s"
    return f"{secs}s"


def _cpu_percent() -> Optional[float]:
    try:
        import psutil  # type: ignore

        return float(psutil.cpu_percent(interval=None))
    except Exception:
        return None


def _prime_cpu() -> None:
    try:
        import psutil  # type: ignore

        psutil.cpu_percent(interval=None)
    except Exception:
        pass


def _pool_label(config: dict) -> str:
    if not config.get("pool_server"):
        return "standalone"
    block = config.get("pool_server_load")
    if not isinstance(block, dict):
        return "pool"
    raw = block.get("pool")
    if isinstance(raw, str):
        normalized = raw.strip().lower()
        if normalized in ("high_volume", "overflow", "big"):
            return "high_volume"
    return "default"


def _bar(pct: float, width: int = 16) -> Text:
    filled = max(0, min(width, int(round((pct / 100.0) * width))))
    empty = width - filled
    style = "green"
    if pct >= 85:
        style = "red"
    elif pct >= 60:
        style = "yellow"
    return Text(f"{'█' * filled}{'░' * empty}", style=style)


class ConsoleStatus:
    def __init__(self, config: dict, version: str = ""):
        self.config = config
        self.version = version
        self.stats = {
            "processed": 0,
            "currently_processing": 0,
            "flagged": 0,
            "muted": 0,
            "last_flagged_player": None,
            "last_flagged_time": None,
            "last_muted_player": None,
            "last_muted_time": None,
        }
        self.recent_transcripts = deque(maxlen=10)
        console_cfg = config.get("console", {}) or {}
        self.log_transcripts = bool(console_cfg.get("log_transcripts", False))
        self.live_display = bool(console_cfg.get("live_display", True))
        self.started_at = time.time()
        self._runtime_provider: Optional[RuntimeProvider] = None
        self._console = Console(
            file=sys.stdout,
            force_terminal=_is_tty() or None,
            soft_wrap=False,
            highlight=False,
        )
        self._live: Optional[Live] = None
        self._use_live = False
        self._use_stable = False
        self._last_compact_line = ""
        self._panel_line_count = 0
        self._last_stable_draw_at = 0.0

    def attach_runtime(self, provider: RuntimeProvider) -> None:
        self._runtime_provider = provider

    def increment_processed(self):
        self.stats["processed"] += 1

    def increment_processing(self):
        self.stats["currently_processing"] += 1

    def decrement_processing(self):
        self.stats["currently_processing"] = max(0, self.stats["currently_processing"] - 1)

    def increment_flagged(self, player_name: str):
        self.stats["flagged"] += 1
        self.stats["last_flagged_player"] = player_name
        self.stats["last_flagged_time"] = time.time()

    def increment_muted(self, player_name: str):
        self.stats["muted"] += 1
        self.stats["last_muted_player"] = player_name
        self.stats["last_muted_time"] = time.time()

    def add_transcript(
        self,
        player_name: str,
        transcript: str,
        language: str,
        flagged: bool,
        muted: bool,
    ):
        if self.log_transcripts:
            self.recent_transcripts.append(
                {
                    "player": player_name,
                    "transcript": transcript,
                    "language": language,
                    "flagged": flagged,
                    "muted": muted,
                    "time": time.time(),
                }
            )

    def _runtime(self) -> dict[str, Any]:
        if not self._runtime_provider:
            return {}
        try:
            data = self._runtime_provider()
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}

    def _build_renderable(self):
        now = time.time()
        runtime = self._runtime()
        connections = int(runtime.get("active_connections", 0) or 0)
        queue_size = int(runtime.get("processing_queue_size", 0) or 0)
        queue_max = max(1, int(runtime.get("queue_max_size", 500) or 500))
        workers = int(runtime.get("worker_count", 1) or 1)
        stt = int(runtime.get("stt_concurrency", 1) or 1)
        queue_pct = 100.0 * min(queue_size, queue_max) / queue_max
        cpu = _cpu_percent()
        pool = _pool_label(self.config)

        header = Text()
        header.append("VoiceSentinel Processor Status", style="bold cyan")
        if self.version:
            header.append(f"  v{self.version}", style="dim")
        header.append(f"  ·  {_format_uptime(self.started_at, now)}", style="dim")
        header.append(f"  ·  {pool}", style="magenta")

        metrics = Table.grid(expand=True, padding=(0, 2))
        metrics.add_column(ratio=1)
        metrics.add_column(ratio=1)
        metrics.add_column(ratio=1)
        metrics.add_column(ratio=1)

        queue_cell = Table.grid(padding=(0, 0))
        queue_cell.add_row(Text("Queue", style="dim"))
        queue_cell.add_row(Text(f"{queue_size}/{queue_max}", style="bold"))
        queue_cell.add_row(_bar(queue_pct))

        cpu_text = "—" if cpu is None else f"{cpu:.0f}%"
        cpu_style = "green"
        if cpu is not None:
            if cpu >= 85:
                cpu_style = "red"
            elif cpu >= 60:
                cpu_style = "yellow"

        metrics.add_row(
            Text.from_markup(f"[dim]Connections[/dim]\n[bold]{connections}[/bold]"),
            queue_cell,
            Text.from_markup(
                f"[dim]Processing[/dim]\n[bold]{self.stats['currently_processing']}[/bold]"
            ),
            Text.from_markup(f"[dim]CPU[/dim]\n[bold {cpu_style}]{cpu_text}[/]"),
        )

        counters = Table.grid(expand=True, padding=(0, 2))
        counters.add_column(ratio=1)
        counters.add_column(ratio=1)
        counters.add_column(ratio=1)
        counters.add_column(ratio=1)
        counters.add_row(
            Text.from_markup(f"[dim]Processed[/dim]\n[bold]{self.stats['processed']}[/bold]"),
            Text.from_markup(
                f"[dim]Flagged[/dim]\n[bold yellow]{self.stats['flagged']}[/bold yellow]"
            ),
            Text.from_markup(f"[dim]Muted[/dim]\n[bold red]{self.stats['muted']}[/bold red]"),
            Text.from_markup(f"[dim]Workers / STT[/dim]\n[bold]{workers} / {stt}[/bold]"),
        )

        activity = Table.grid(expand=True, padding=(0, 2))
        activity.add_column(ratio=1)
        activity.add_column(ratio=1)
        flagged_player = self.stats["last_flagged_player"] or "—"
        muted_player = self.stats["last_muted_player"] or "—"
        activity.add_row(
            Text.from_markup(
                f"[dim]Last flagged[/dim]\n[yellow]{flagged_player}[/yellow]  "
                f"[dim]{_format_ago(self.stats['last_flagged_time'], now)}[/dim]"
            ),
            Text.from_markup(
                f"[dim]Last muted[/dim]\n[red]{muted_player}[/red]  "
                f"[dim]{_format_ago(self.stats['last_muted_time'], now)}[/dim]"
            ),
        )

        sections: list[Any] = [
            Panel(header, border_style="cyan", padding=(0, 1)),
            Panel(metrics, title="Runtime", border_style="blue", padding=(0, 1)),
            Panel(counters, title="Totals", border_style="blue", padding=(0, 1)),
            Panel(activity, title="Activity", border_style="blue", padding=(0, 1)),
        ]

        if self.log_transcripts:
            transcript_table = Table(
                expand=True,
                show_header=True,
                header_style="bold dim",
                box=None,
                padding=(0, 1),
            )
            transcript_table.add_column("Status", width=9)
            transcript_table.add_column("Player", width=16, overflow="ellipsis")
            transcript_table.add_column("Transcript", overflow="ellipsis")
            transcript_table.add_column("Lang", width=5)
            transcript_table.add_column("Age", width=7, justify="right")

            rows = list(self.recent_transcripts)[-5:]
            if not rows:
                transcript_table.add_row("—", "—", "No transcripts yet", "—", "—")
            else:
                for item in rows:
                    if item["muted"]:
                        status = Text("MUTED", style="bold red")
                    elif item["flagged"]:
                        status = Text("FLAGGED", style="bold yellow")
                    else:
                        status = Text("CLEAN", style="green")
                    transcript_table.add_row(
                        status,
                        str(item["player"]),
                        str(item["transcript"]),
                        str(item.get("language") or "—"),
                        _format_ago(item.get("time"), now),
                    )
            sections.append(
                Panel(
                    transcript_table,
                    title="Recent Transcripts",
                    border_style="blue",
                    padding=(0, 0),
                )
            )

        # Trailing blank line prevents web consoles from clipping the bottom border.
        sections.append(Text(""))
        return Group(*sections)

    def _compact_line(self) -> str:
        now = time.time()
        runtime = self._runtime()
        connections = int(runtime.get("active_connections", 0) or 0)
        queue_size = int(runtime.get("processing_queue_size", 0) or 0)
        queue_max = max(1, int(runtime.get("queue_max_size", 500) or 500))
        cpu = _cpu_percent()
        cpu_part = f" cpu={cpu:.0f}%" if cpu is not None else ""
        return (
            "VoiceSentinel Processor Status | "
            f"conn={connections} queue={queue_size}/{queue_max} "
            f"proc={self.stats['currently_processing']} "
            f"done={self.stats['processed']} "
            f"flag={self.stats['flagged']} mute={self.stats['muted']}"
            f"{cpu_part} "
            f"up={_format_uptime(self.started_at, now)}"
        )

    def _capture_panel_text(self) -> str:
        with self._console.capture() as capture:
            self._console.print(self._build_renderable())
        text = capture.get()
        if not text.endswith("\n"):
            text += "\n"
        return text

    def _stable_redraw(self, force: bool = False) -> None:
        now = time.time()
        # Throttle redraws on limited consoles to cut flicker.
        if not force and (now - self._last_stable_draw_at) < 2.0:
            return
        text = self._capture_panel_text()
        lines = text.splitlines()
        n = len(lines)
        out = sys.stdout
        try:
            if self._panel_line_count > 0:
                out.write(f"\033[{self._panel_line_count}A")
            for i, line in enumerate(lines):
                out.write("\033[2K\r")
                out.write(line)
                out.write("\n")
            if self._panel_line_count > n:
                for _ in range(self._panel_line_count - n):
                    out.write("\033[2K\n")
                out.write(f"\033[{self._panel_line_count - n}A")
            out.flush()
            self._panel_line_count = n
            self._last_stable_draw_at = now
        except Exception:
            self._console.print(self._build_renderable())
            self._panel_line_count = 0

    def start(self) -> None:
        _prime_cpu()
        self.started_at = time.time()
        limited = _is_limited_console()
        tty = _is_tty()
        self._use_live = bool(self.live_display and tty and not limited)
        self._use_stable = bool(self.live_display and tty and limited)

        if self._use_stable:
            self._stable_redraw(force=True)
            return

        if not self._use_live:
            line = self._compact_line()
            self._console.print(line)
            self._last_compact_line = line
            return

        if self._live is not None:
            return
        # auto_refresh=False: we drive updates once per second (avoids double-redraw flash).
        self._live = Live(
            self._build_renderable(),
            console=self._console,
            auto_refresh=False,
            transient=False,
            vertical_overflow="visible",
        )
        self._live.start()
        self._live.update(self._build_renderable(), refresh=True)

    def stop(self) -> None:
        if self._live is not None:
            try:
                self._live.stop()
            except Exception:
                pass
            self._live = None

    def refresh(self) -> None:
        if not self.live_display and self._live is None and not self._use_stable:
            return
        if self._use_live and self._live is not None:
            self._live.update(self._build_renderable(), refresh=True)
            return
        if self._use_stable:
            self._stable_redraw(force=False)
            return
        line = self._compact_line()
        if line != self._last_compact_line:
            self._console.print(line)
            self._last_compact_line = line

    def print_status(self, force: bool = False):
        if not self.live_display and not force:
            return
        if self._use_live:
            if self._live is None:
                self.start()
            else:
                self.refresh()
            return
        if self._use_stable:
            self._stable_redraw(force=True)
            return
        line = self._compact_line()
        if force or line != self._last_compact_line:
            self._console.print(line)
            self._last_compact_line = line

    def has_changed(self) -> bool:
        return True
