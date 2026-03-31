"""Chrome WebDriver lifecycle management."""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
from contextlib import contextmanager
from typing import Generator

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from ..config import DEFAULT_WINDOW_HEIGHT, DEFAULT_WINDOW_WIDTH

logger = logging.getLogger(__name__)
_GRACEFUL_QUIT_TIMEOUT_SECONDS = 10


class ManagedBrowser:
    """Context manager that owns a Chrome WebDriver instance."""

    def __init__(
        self,
        headless: bool = False,
        download_dir: str | None = None,
        extra_options: list[str] | None = None,
        keep_open_on_error: bool = False,
    ) -> None:
        self.headless = headless
        self.download_dir = download_dir
        self.extra_options = extra_options or []
        self.keep_open_on_error = keep_open_on_error
        self._driver: webdriver.Chrome | None = None

    def __enter__(self) -> webdriver.Chrome:
        self._driver = self._create_driver()
        return self._driver

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None and self.keep_open_on_error:
            logger.warning(
                "Preserving Chrome window for debugging because an exception occurred: %s",
                exc_type.__name__,
            )
            return False

        self.quit()
        return False

    def quit(self) -> None:
        if self._driver is not None:
            driver = self._driver
            service = getattr(driver, "service", None)
            try:
                if self._quit_driver_with_timeout(driver, timeout_seconds=_GRACEFUL_QUIT_TIMEOUT_SECONDS):
                    logger.debug("Chrome driver quit successfully.")
                else:
                    logger.warning(
                        "Chrome driver did not quit within %ss; forcing service shutdown.",
                        _GRACEFUL_QUIT_TIMEOUT_SECONDS,
                    )
            finally:
                self._stop_service(service)
                self._driver = None

    @property
    def current_driver(self) -> webdriver.Chrome | None:
        return self._driver

    def restart(self) -> webdriver.Chrome:
        logger.warning("Restarting Chrome driver.")
        self.quit()
        self._driver = self._create_driver()
        return self._driver

    def _create_driver(self) -> webdriver.Chrome:
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument(f"--window-size={DEFAULT_WINDOW_WIDTH},{DEFAULT_WINDOW_HEIGHT}")

        if self.download_dir:
            prefs = {
                "download.default_directory": str(self.download_dir),
                "download.prompt_for_download": False,
                "download.directory_upgrade": True,
                "safebrowsing.enabled": True,
            }
            options.add_experimental_option("prefs", prefs)

        for flag in self.extra_options:
            options.add_argument(flag)

        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=options)
        driver.set_window_size(DEFAULT_WINDOW_WIDTH, DEFAULT_WINDOW_HEIGHT)
        logger.debug("Chrome driver created.")
        return driver

    def _quit_driver_with_timeout(
        self,
        driver: webdriver.Chrome,
        timeout_seconds: int,
    ) -> bool:
        result: dict[str, BaseException | None] = {"error": None}

        def _graceful_quit() -> None:
            try:
                driver.quit()
            except Exception as exc:  # pragma: no cover
                result["error"] = exc

        worker = threading.Thread(target=_graceful_quit, daemon=True)
        worker.start()
        worker.join(timeout_seconds)
        if worker.is_alive():
            return False

        if result["error"] is not None:
            logger.warning("Error while quitting driver: %s", result["error"])
            return False
        return True

    def _close_all_windows(self, driver: webdriver.Chrome) -> None:
        try:
            handles = list(driver.window_handles)
        except Exception as exc:
            logger.debug("Unable to enumerate Chrome windows before quit: %s", exc)
            return

        if not handles:
            return

        logger.debug("Closing %d Chrome window(s) before quit.", len(handles))
        for handle in reversed(handles):
            try:
                driver.switch_to.window(handle)
                driver.close()
            except Exception as exc:
                logger.debug("Ignoring window close failure for handle=%s: %s", handle, exc)

    def _stop_service(self, service: Service | None) -> None:
        if service is None:
            return

        process = getattr(service, "process", None)
        pid = getattr(process, "pid", None)

        try:
            service.stop()
            logger.debug("ChromeDriver service stopped successfully.")
        except Exception as exc:
            logger.warning("Error while stopping ChromeDriver service: %s", exc)

        if pid is not None and self._process_is_running(pid):
            logger.warning(
                "ChromeDriver service process pid=%s survived normal shutdown; forcing termination.",
                pid,
            )
            self._terminate_process_tree(pid)

    def _process_is_running(self, pid: int) -> bool:
        if pid <= 0:
            return False

        try:
            if os.name == "nt":
                result = subprocess.run(
                    ["tasklist", "/FI", f"PID eq {pid}"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
                return str(pid) in result.stdout

            os.kill(pid, 0)
            return True
        except Exception:
            return False

    def _terminate_process_tree(self, pid: int) -> None:
        try:
            if os.name == "nt":
                subprocess.run(
                    ["taskkill", "/PID", str(pid), "/T", "/F"],
                    check=False,
                    capture_output=True,
                    text=True,
                )
            else:
                os.kill(pid, signal.SIGTERM)
        except Exception as exc:  # pragma: no cover
            logger.warning("Failed to force-terminate ChromeDriver process pid=%s: %s", pid, exc)


@contextmanager
def open_browser(
    headless: bool = False,
    download_dir: str | None = None,
    extra_options: list[str] | None = None,
    keep_open_on_error: bool = False,
) -> Generator[webdriver.Chrome, None, None]:
    with ManagedBrowser(
        headless=headless,
        download_dir=download_dir,
        extra_options=extra_options,
        keep_open_on_error=keep_open_on_error,
    ) as driver:
        yield driver
