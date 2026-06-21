from __future__ import annotations

import logging
import time
from collections.abc import Callable
from typing import TypeVar

T = TypeVar("T")


def retry_call(
    operation: Callable[[], T],
    *,
    attempts: int,
    backoff_seconds: float,
    logger: logging.Logger,
    label: str,
    retry_exceptions: tuple[type[BaseException], ...] = (Exception,),
) -> T:
    """Run an operation with exponential backoff and clear logging."""
    last_error: BaseException | None = None
    for attempt in range(1, attempts + 1):
        try:
            return operation()
        except retry_exceptions as exc:
            last_error = exc
            if attempt >= attempts:
                break
            sleep_for = backoff_seconds * attempt
            logger.warning(
                "%s failed on attempt %s/%s: %s. Retrying in %.1fs",
                label,
                attempt,
                attempts,
                exc,
                sleep_for,
            )
            time.sleep(sleep_for)
    assert last_error is not None
    raise last_error
