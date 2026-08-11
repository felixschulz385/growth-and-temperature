"""S&P Global / Capital IQ authentication helpers."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from selenium.common.exceptions import (
    ElementClickInterceptedException,
    ElementNotInteractableException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from ..config import (
    DEFAULT_WAIT_SECONDS,
    LOGIN_URL,
    SEL,
    SHORT_WAIT_SECONDS,
    SMALL_SLEEP_SECONDS,
)

logger = logging.getLogger(__name__)


def load_credentials(credentials_path: str | Path) -> dict:
    logger.info("Loading credentials from %s", credentials_path)
    with open(credentials_path, "r", encoding="utf-8") as fh:
        credentials = json.load(fh)
    logger.debug(
        "Credentials loaded successfully (username_present=%s, password_present=%s)",
        bool(credentials.get("username")),
        bool(credentials.get("password")),
    )
    return credentials


def login(driver: WebDriver, credentials: dict, wait: int = DEFAULT_WAIT_SECONDS) -> None:
    logger.info("Starting login flow (wait=%ss)", wait)
    try:
        logger.info("Navigating to login page: %s", LOGIN_URL)
        driver.get(LOGIN_URL)
        time.sleep(SMALL_SLEEP_SECONDS)

        _wait = WebDriverWait(driver, wait)
        username_field = _wait.until(
            EC.visibility_of_element_located((By.XPATH, SEL["username_input"]))
        )
        username_field.clear()
        username_field.send_keys(credentials["username"])
        _click_when_ready(driver, (By.XPATH, SEL["submit_btn"]), timeout=wait, label="username submit")
        time.sleep(SMALL_SLEEP_SECONDS)
        logger.info("Username submitted.")

        password_field = _wait.until(
            EC.visibility_of_element_located((By.XPATH, SEL["password_input"]))
        )
        password_field.clear()
        password_field.send_keys(credentials["password"])
        _click_when_ready(driver, (By.XPATH, SEL["submit_btn"]), timeout=wait, label="password submit")
        time.sleep(SMALL_SLEEP_SECONDS)
        logger.info("Password submitted; waiting for dashboard URL change.")

        _wait.until(EC.url_changes(LOGIN_URL))
        logger.info("URL changed successfully. Current URL: %s", driver.current_url)

        accepted = accept_cookies(driver, timeout=DEFAULT_WAIT_SECONDS)
        if accepted:
            logger.info("Cookie banner accepted.")
        else:
            logger.info("Cookie banner not present or not accepted within timeout.")

        logger.info("Login successful.")
    except Exception:
        logger.exception("Login flow failed before completion.")
        raise


def logout(driver: WebDriver, wait: int = DEFAULT_WAIT_SECONDS) -> None:
    logger.info("Starting logout flow (wait=%ss)", wait)
    try:
        opened_menu = click_hamburger_menu(driver, timeout=SHORT_WAIT_SECONDS)
        logger.debug("Hamburger menu open result: %s", opened_menu)
        time.sleep(SMALL_SLEEP_SECONDS)

        _click_when_ready(
            driver,
            (By.CSS_SELECTOR, SEL["profile_icon"]),
            timeout=wait,
            label="profile icon",
        )
        time.sleep(SMALL_SLEEP_SECONDS)
        _click_when_ready(
            driver,
            (By.XPATH, SEL["sign_out_link"]),
            timeout=wait,
            label="sign out",
        )
        logger.info("Logout successful.")
    except Exception as exc:  # pragma: no cover
        logger.warning("Logout failed (may have already been logged out): %s", exc)


def accept_cookies(driver: WebDriver, timeout: int = DEFAULT_WAIT_SECONDS) -> bool:
    try:
        WebDriverWait(driver, timeout).until(
            EC.presence_of_element_located((By.ID, SEL["cookie_banner"]))
        )
        _click_when_ready(
            driver,
            (By.ID, SEL["cookie_accept"]),
            timeout=timeout,
            label="cookie accept",
        )
        return True
    except TimeoutException:
        return False
    except Exception as exc:
        logger.warning("Cookie banner detected but acceptance failed non-fatally: %s", exc)
        return False


def click_hamburger_menu(driver: WebDriver, timeout: int = SHORT_WAIT_SECONDS) -> bool:
    try:
        _click_when_ready(
            driver,
            (By.CSS_SELECTOR, SEL["hamburger_menu"]),
            timeout=timeout,
            label="hamburger menu",
        )
        return True
    except TimeoutException:
        return False


def _click_when_ready(
    driver: WebDriver,
    locator: tuple[str, str],
    timeout: int,
    label: str,
) -> None:
    wait = WebDriverWait(driver, timeout)
    last_exc: Exception | None = None

    for attempt in range(1, 4):
        try:
            element = wait.until(EC.presence_of_element_located(locator))
            driver.execute_script(
                "arguments[0].scrollIntoView({block: 'center', inline: 'center'});",
                element,
            )
            time.sleep(SMALL_SLEEP_SECONDS)

            try:
                wait.until(EC.element_to_be_clickable(locator))
                element.click()
            except (ElementClickInterceptedException, ElementNotInteractableException) as exc:
                last_exc = exc
                logger.debug(
                    "Standard click failed for %s on attempt %d/3; trying JS click. Error: %s",
                    label,
                    attempt,
                    exc,
                )
                driver.execute_script("arguments[0].click();", element)
            return
        except (
            ElementClickInterceptedException,
            ElementNotInteractableException,
            StaleElementReferenceException,
            TimeoutException,
        ) as exc:
            last_exc = exc
            logger.debug("Click attempt %d/3 failed for %s: %s", attempt, label, exc)
            time.sleep(SMALL_SLEEP_SECONDS * attempt)

    if last_exc is not None:
        raise last_exc
    raise TimeoutException(f"Unable to click {label}")
