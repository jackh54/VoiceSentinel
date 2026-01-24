import time
from typing import Optional
from collections import deque

class ConsoleStatus:
    def __init__(self, config: dict):
        self.config = config
        self.stats = {
            "processed": 0,
            "currently_processing": 0,
            "flagged": 0,
            "muted": 0,
            "last_flagged_player": None,
            "last_flagged_time": None,
            "last_muted_player": None,
            "last_muted_time": None
        }
        self.recent_transcripts = deque(maxlen=10)
        self.log_transcripts = config.get("console", {}).get("log_transcripts", False)
        self.live_display = config.get("console", {}).get("live_display", True)
        self.last_update = time.time()
        self.last_stats_hash = None
        
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
        
    def add_transcript(self, player_name: str, transcript: str, language: str, flagged: bool, muted: bool):
        if self.log_transcripts:
            self.recent_transcripts.append({
                "player": player_name,
                "transcript": transcript,
                "language": language,
                "flagged": flagged,
                "muted": muted,
                "time": time.time()
            })
            
    def get_status_display(self) -> str:
        lines = []
        lines.append("=" * 60)
        lines.append("VoiceSentinel Processor Status")
        lines.append("=" * 60)
        lines.append(f"Currently Processing: {self.stats['currently_processing']}")
        lines.append(f"Total Processed: {self.stats['processed']}")
        lines.append(f"Total Flagged: {self.stats['flagged']}")
        lines.append(f"Total Muted: {self.stats['muted']}")
        
        if self.stats["last_flagged_player"]:
            time_ago = int(time.time() - self.stats["last_flagged_time"])
            lines.append(f"Last Flagged: {self.stats['last_flagged_player']} ({time_ago}s ago)")
            
        if self.stats["last_muted_player"]:
            time_ago = int(time.time() - self.stats["last_muted_time"])
            lines.append(f"Last Muted: {self.stats['last_muted_player']} ({time_ago}s ago)")
            
        if self.log_transcripts and self.recent_transcripts:
            lines.append("")
            lines.append("Recent Transcripts:")
            lines.append("-" * 60)
            for item in list(self.recent_transcripts)[-5:]:
                status = "MUTED" if item["muted"] else ("FLAGGED" if item["flagged"] else "CLEAN")
                lines.append(f"[{status}] {item['player']}: '{item['transcript']}' ({item['language']})")
                
        lines.append("=" * 60)
        return "\n".join(lines)
        
    def has_changed(self) -> bool:
        current_hash = hash((
            self.stats["processed"],
            self.stats["currently_processing"],
            self.stats["flagged"],
            self.stats["muted"],
            self.stats["last_flagged_player"],
            self.stats["last_muted_player"]
        ))
        if current_hash != self.last_stats_hash:
            self.last_stats_hash = current_hash
            return True
        return False
    
    def print_status(self, force: bool = False):
        if not self.live_display and not force:
            return
            
        try:
            import sys
            import os
            if os.isatty(sys.stdout.fileno()):
                print("\033[2J\033[H", end="", flush=True)
            print(self.get_status_display(), flush=True)
        except:
            print(self.get_status_display(), flush=True)
