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


class DailyScheduler:
    """Runs the automation workflow every day at the configured local time."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger

    def run_forever(self, job: Callable[[], None]) -> None:
        self.logger.info("Scheduler started. Daily run time: %s (%s)", self.settings.schedule_time, self.settings.timezone)
        tz = get_timezone_info(self.settings.timezone)
        last_run_date: str | None = None
        while True:
            now = datetime.now(tz)
            today = now.strftime("%Y-%m-%d")
            if now.strftime("%H:%M") == self.settings.schedule_time and last_run_date != today:
                last_run_date = today
                job()
            time.sleep(30)
