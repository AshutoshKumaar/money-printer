from __future__ import annotations

import logging
import time
import json
from collections.abc import Callable
from datetime import datetime, timezone, timedelta
import zoneinfo
from pathlib import Path

from config import Settings


def get_timezone_info(tz_name: str):
    try:
        return zoneinfo.ZoneInfo(tz_name)
    except Exception:
        # Fallback for Windows where standard tzdata package is not installed
        if tz_name == "Asia/Kolkata":
            return timezone(timedelta(hours=5, minutes=30))
        return timezone.utc


def get_next_run(now_dt: datetime) -> tuple[datetime, str]:
    candidates = []
    # 1. Shorts candidates for today, tomorrow, and day after tomorrow
    for offset in (0, 1, 2):
        d = now_dt + timedelta(days=offset)
        candidates.append((d.replace(hour=6, minute=0, second=0, microsecond=0), "short"))
        candidates.append((d.replace(hour=18, minute=0, second=0, microsecond=0), "short"))
    
    # 2. Long video candidates for the next two weeks
    for offset in range(14):
        d = now_dt + timedelta(days=offset)
        if d.weekday() == 4: # Friday
            candidates.append((d.replace(hour=12, minute=0, second=0, microsecond=0), "long"))
            
    future_candidates = [c for c in candidates if c[0] > now_dt]
    return min(future_candidates, key=lambda x: x[0])


def get_missed_slots(start_dt: datetime, end_dt: datetime) -> list[tuple[datetime, str]]:
    """Returns scheduled slots that fell in the range (start_dt, end_dt]."""
    slots = []
    current = start_dt.replace(minute=0, second=0, microsecond=0) + timedelta(hours=1)
    while current <= end_dt:
        if current.hour in (6, 18):
            slots.append((current, "short"))
        elif current.weekday() == 4 and current.hour == 12:
            slots.append((current, "long"))
        current += timedelta(hours=1)
    return slots


class DailyScheduler:
    """Runs the automation workflow:
    - Shorts: Daily at 06:00 AM and 06:00 PM IST.
    - Long Videos: Fridays at 12:00 PM IST.
    """

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger
        self.state_file = settings.storage_dir / "scheduler_state.json"

    def _load_state(self) -> dict:
        if self.state_file.exists():
            try:
                return json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {}

    def _save_state(self, state: dict) -> None:
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            self.logger.warning("Failed to save scheduler state: %s", e)

    def run_forever(self, job: Callable[[str], None]) -> None:
        self.logger.info("Scheduler initialized successfully")
        self.logger.info("Timezone: Asia/Kolkata")
        self.logger.info("Scheduled Runs: Shorts: 06:00 AM, 06:00 PM | Long Videos: Fridays 12:00 PM")
        
        tz = get_timezone_info("Asia/Kolkata")
        job_running = False
        
        # Load state and check for missed runs on startup
        state = self._load_state()
        now = datetime.now(tz)
        last_run_str = state.get("last_run_time")
        
        if last_run_str:
            try:
                last_run = datetime.fromisoformat(last_run_str)
                if last_run.tzinfo is None:
                    last_run = last_run.replace(tzinfo=tz)
                else:
                    last_run = last_run.astimezone(tz)
                
                # Cap catch-up search window to the last 24 hours to avoid spamming
                if now - last_run > timedelta(days=1):
                    self.logger.warning("Last run time is too old (%s). Capping catch-up search window to 24 hours.", last_run_str)
                    last_run = now - timedelta(days=1)
                
                missed = get_missed_slots(last_run, now)
                if missed:
                    self.logger.warning("Detected %d missed scheduled runs due to restart.", len(missed))
                    missed_shorts = [m for m in missed if m[1] == "short"]
                    missed_longs = [m for m in missed if m[1] == "long"]
                    
                    to_run = []
                    if missed_shorts:
                        to_run.append(missed_shorts[-1])
                    if missed_longs:
                        to_run.append(missed_longs[-1])
                    
                    to_run.sort(key=lambda x: x[0])
                    
                    for run_time, vtype in to_run:
                        self.logger.info("Executing missed %s job (scheduled for %s IST)...", vtype, run_time.strftime("%Y-%m-%d %H:%M:%S"))
                        job_running = True
                        try:
                            job(vtype)
                        except Exception as exc:
                            self.logger.error("Missed job catch-up execution failed: %s", exc)
                        finally:
                            job_running = False
            except Exception as exc:
                self.logger.error("Failed to check for missed scheduler runs: %s", exc)
        else:
            self.logger.info("First boot or no state found. Writing initial last_run_time: %s", now.isoformat())
            
        state["last_run_time"] = now.isoformat()
        self._save_state(state)
        
        while True:
            now = datetime.now(tz)
            server_now = datetime.now(timezone.utc)
            next_run_dt, next_run_type = get_next_run(now)
            remaining = next_run_dt - now
            
            rem_hours = remaining.seconds // 3600
            rem_minutes = (remaining.seconds % 3600) // 60
            remaining_str = f"{rem_hours}h {rem_minutes}m"
            
            self.logger.info("--- Railway Deploy Status ---")
            self.logger.info("Current Server Time (UTC): %s", server_now.strftime("%Y-%m-%d %H:%M:%S UTC"))
            self.logger.info("Current IST Time:          %s", now.strftime("%Y-%m-%d %H:%M:%S IST"))
            self.logger.info("Next Scheduled Run:        %s (%s IST)", next_run_dt.strftime("%Y-%m-%d %H:%M:%S"), next_run_type)
            self.logger.info("Time Remaining:            %s", remaining_str)
            self.logger.info("Job Running:               %s", "YES" if job_running else "NO")
            self.logger.info("Scheduler Waiting:         %s", "YES" if not job_running else "NO")
            self.logger.info("-----------------------------")
            
            is_short_slot = now.hour in (6, 18) and now.minute == 0
            is_long_slot = now.weekday() == 4 and now.hour == 12 and now.minute == 0
            
            triggered_type = None
            if is_short_slot:
                triggered_type = "short"
            elif is_long_slot:
                triggered_type = "long"
                
            if triggered_type:
                state = self._load_state()
                last_run_str = state.get("last_run_time")
                run_already_completed = False
                if last_run_str:
                    try:
                        last_run_dt = datetime.fromisoformat(last_run_str)
                        if last_run_dt.tzinfo is None:
                            last_run_dt = last_run_dt.replace(tzinfo=tz)
                        if last_run_dt.date() == now.date() and last_run_dt.hour == now.hour:
                            run_already_completed = True
                    except Exception:
                        pass
                
                if not run_already_completed:
                    self.logger.info("Scheduled %s run reached! Starting video generation...", triggered_type)
                    job_running = True
                    try:
                        job(triggered_type)
                    except Exception as e:
                        self.logger.error("Scheduled job execution failed: %s", e)
                    finally:
                        job_running = False
                        
                    state["last_run_time"] = datetime.now(tz).isoformat()
                    self._save_state(state)
                    self.logger.info("Scheduled job execution completed.")
                    
            time.sleep(30)
