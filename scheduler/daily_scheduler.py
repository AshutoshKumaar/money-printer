from __future__ import annotations

import logging
import time
from collections.abc import Callable
from datetime import datetime

from config import Settings


class DailyScheduler:
    """Runs the automation workflow every day at the configured local time."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger

    def run_forever(self, job: Callable[[], None]) -> None:
        self.logger.info("Scheduler started. Daily run time: %s (%s)", self.settings.schedule_time, self.settings.timezone)
        last_run_date: str | None = None
        while True:
            now = datetime.now()
            today = now.strftime("%Y-%m-%d")
            if now.strftime("%H:%M") == self.settings.schedule_time and last_run_date != today:
                last_run_date = today
                job()
            time.sleep(30)
