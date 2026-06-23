from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime, timezone, timedelta
import zoneinfo

from config import Settings


def get_timezone_info(tz_name: str):
    try:
        return zoneinfo.ZoneInfo(tz_name)
    except Exception:
        # Fallback for Windows where standard tzdata package is not installed
        if tz_name == "Asia/Kolkata":
            return timezone(timedelta(hours=5, minutes=30))
        return timezone.utc


def get_next_run(now_dt: datetime) -> datetime:
    candidates = []
    # Today's 6 AM
    candidates.append(now_dt.replace(hour=6, minute=0, second=0, microsecond=0))
    # Today's 6 PM
    candidates.append(now_dt.replace(hour=18, minute=0, second=0, microsecond=0))
    # Tomorrow's 6 AM
    candidates.append((now_dt + timedelta(days=1)).replace(hour=6, minute=0, second=0, microsecond=0))
    # Tomorrow's 6 PM
    candidates.append((now_dt + timedelta(days=1)).replace(hour=18, minute=0, second=0, microsecond=0))
    
    future_candidates = [c for c in candidates if c > now_dt]
    return min(future_candidates)


class DailyScheduler:
    """Runs the automation workflow daily at exactly 06:00 AM and 06:00 PM Asia/Kolkata."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger

    def run_forever(self, job: Callable[[], None]) -> None:
        self.logger.info("Scheduler initialized successfully")
        self.logger.info("Timezone: Asia/Kolkata")
        self.logger.info("Scheduled Runs: 06:00 AM, 06:00 PM")
        
        tz = get_timezone_info("Asia/Kolkata")
        job_running = False
        last_run_time: datetime | None = None
        
        while True:
            now = datetime.now(tz)
            server_now = datetime.now(timezone.utc)
            next_run = get_next_run(now)
            remaining = next_run - now
            
            # Format time remaining (e.g. 11h 45m)
            rem_hours = remaining.seconds // 3600
            rem_minutes = (remaining.seconds % 3600) // 60
            remaining_str = f"{rem_hours}h {rem_minutes}m"
            
            self.logger.info("--- Railway Deploy Status ---")
            self.logger.info("Current Server Time (UTC): %s", server_now.strftime("%Y-%m-%d %H:%M:%S UTC"))
            self.logger.info("Current IST Time:          %s", now.strftime("%Y-%m-%d %H:%M:%S IST"))
            self.logger.info("Next Scheduled Run:        %s", next_run.strftime("%Y-%m-%d %H:%M:%S IST"))
            self.logger.info("Time Remaining:            %s", remaining_str)
            self.logger.info("Job Running:               %s", "YES" if job_running else "NO")
            self.logger.info("Scheduler Waiting:         %s", "YES" if not job_running else "NO")
            self.logger.info("-----------------------------")
            
            # Check if it matches exactly 6 AM or 6 PM slot
            is_schedule_hour = now.hour in {6, 18} and now.minute == 0
            
            if is_schedule_hour and (last_run_time is None or last_run_time.date() != now.date() or last_run_time.hour != now.hour):
                last_run_time = now
                self.logger.info("Scheduled run reached! Starting video generation job...")
                job_running = True
                try:
                    job()
                except Exception as e:
                    self.logger.error("Scheduled job execution failed: %s", e)
                finally:
                    job_running = False
                self.logger.info("Scheduled job execution completed.")
                
            time.sleep(30)
