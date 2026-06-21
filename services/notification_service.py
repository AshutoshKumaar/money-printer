from __future__ import annotations

import logging

import requests

from config import Settings


class NotificationService:
    """Sends optional webhook notifications for automation runs."""

    def __init__(self, settings: Settings, logger: logging.Logger) -> None:
        self.settings = settings
        self.logger = logger

    def send(self, title: str, message: str, success: bool = True) -> None:
        if not self.settings.notification_webhook_url:
            self.logger.info("%s: %s", title, message)
            return
        payload = {"title": title, "message": message, "success": success}
        try:
            requests.post(
                self.settings.notification_webhook_url,
                json=payload,
                timeout=self.settings.request_timeout_seconds,
            ).raise_for_status()
        except Exception as exc:
            self.logger.warning("Notification failed: %s", exc)
