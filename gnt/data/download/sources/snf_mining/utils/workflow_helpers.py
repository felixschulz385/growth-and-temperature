"""Shared helpers for workflow step implementations."""

from __future__ import annotations

import logging
import time

logger = logging.getLogger(__name__)


def sleep_politely(seconds: float) -> None:
    if seconds > 0:
        logger.debug("Sleeping for %.2f second(s).", seconds)
        time.sleep(seconds)


def retry(fn, attempts: int, sleep_seconds: float, label: str):
    last_exc: Exception | None = None
    tries = max(1, attempts)
    for attempt in range(1, tries + 1):
        logger.debug("Attempt %d/%d for %s", attempt, tries, label)
        try:
            result = fn()
            logger.debug("Attempt %d/%d succeeded for %s", attempt, tries, label)
            return result
        except Exception as exc:
            last_exc = exc
            if attempt < tries:
                logger.info(
                    "Retrying (%d/%d) for %s after error: %s",
                    attempt,
                    tries,
                    label,
                    exception_brief(exc),
                )
                sleep_politely(sleep_seconds * attempt)
            else:
                logger.warning(
                    "Final retry attempt failed for %s: %s",
                    label,
                    exception_brief(exc),
                )
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Retry failed unexpectedly for: {label}")


def exception_brief(exc: Exception) -> str:
    message = str(exc).strip()
    first_line = message.splitlines()[0] if message else "<no message>"
    return f"{type(exc).__name__}: {first_line}"


def reset_profile_page_state(
    driver,
    profile_url: str,
    step_sleep_seconds: float,
    mine_id: str,
) -> None:
    logger.debug(
        "Resetting page state for mine_id=%s by navigating to the profile twice: %s",
        mine_id,
        profile_url,
    )
    driver.get(profile_url)
    sleep_politely(step_sleep_seconds)
    driver.get(profile_url)
    sleep_politely(step_sleep_seconds)
