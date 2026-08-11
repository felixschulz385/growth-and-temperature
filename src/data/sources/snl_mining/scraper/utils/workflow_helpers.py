"""Shared helpers for workflow step implementations."""

from __future__ import annotations

import logging
import time

from selenium.common.exceptions import TimeoutException
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait

logger = logging.getLogger(__name__)


class ImmediateRetryError(RuntimeError):
    """Signals a retry-worthy failure that should be retried without backoff."""


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
                if isinstance(exc, ImmediateRetryError):
                    logger.info(
                        "Retrying immediately (%d/%d) for %s after error: %s",
                        attempt,
                        tries,
                        label,
                        exception_brief(exc),
                    )
                else:
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
        "Resetting page state for mine_id=%s by navigating to the profile: %s",
        mine_id,
        profile_url,
    )
    driver.get(profile_url)
    try:
        _wait_for_profile_page_ready(driver, profile_url=profile_url)
        return
    except TimeoutException:
        logger.debug(
            "Initial profile navigation did not settle for mine_id=%s; retrying once.",
            mine_id,
        )
        sleep_politely(step_sleep_seconds)
        driver.get(profile_url)
        _wait_for_profile_page_ready(driver, profile_url=profile_url)


def _wait_for_profile_page_ready(driver, profile_url: str, timeout: int = 10) -> None:
    def _profile_ready(current_driver) -> bool:
        try:
            current_url = current_driver.current_url or ""
        except Exception:
            return False

        if current_url and not current_url.startswith(profile_url):
            return False

        ready_state = current_driver.execute_script("return document.readyState")
        if ready_state != "complete":
            return False

        wrappers = current_driver.find_elements(By.CSS_SELECTOR, "div.page-sidebar-wrapper")
        return any(wrapper.is_displayed() for wrapper in wrappers)

    WebDriverWait(driver, timeout, poll_frequency=0.2).until(_profile_ready)
