"""Chrome WebDriver lifecycle management."""

from __future__ import annotations

import logging
import os
import signal
import shutil
import subprocess
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from ..config import DATA_DIR, DEFAULT_WINDOW_HEIGHT, DEFAULT_WINDOW_WIDTH, SCRATCH_DIR

logger = logging.getLogger(__name__)
_GRACEFUL_QUIT_TIMEOUT_SECONDS = 10
# Ephemeral Selenium user-data dirs -- scratch, not durable data (SCRATCH_DIR).
_PROFILE_ROOT = SCRATCH_DIR / "browser_profiles"
# Chromedriver logs are durable diagnostics, kept centrally under DATA_DIR.
_LOG_ROOT = DATA_DIR / "logs"
_CHROMEDRIVER_VERSION_ROOT = Path.home() / ".wdm" / "drivers" / "chromedriver"
_WINDOWS_CHROME_CANDIDATES = (
    Path(r"C:\Program Files\Google\Chrome\Application\chrome.exe"),
    Path(r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"),
)


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
        self._active_profile_dir: Path | None = None
        self._active_chromedriver_log_path: Path | None = None

    def __enter__(self) -> webdriver.Chrome:
        self.purge_stale_processes()
        self._driver = self._create_driver()
        return self._driver

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        if exc_type is not None and self.keep_open_on_error:
            logger.warning(
                "An exception occurred (%s); closing Chrome windows despite keep_open_on_error to prevent stale browser buildup.",
                exc_type.__name__,
            )

        self.quit()
        return False

    def quit(self) -> None:
        if self._driver is not None:
            driver = self._driver
            service = getattr(driver, "service", None)
            try:
                self._close_all_windows(driver)
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
                self.purge_stale_processes()
                self._remove_active_profile_dir()

    @property
    def current_driver(self) -> webdriver.Chrome | None:
        return self._driver

    def restart(self) -> webdriver.Chrome:
        logger.warning("Restarting Chrome driver.")
        self.quit()
        self.purge_stale_processes()
        self._driver = self._create_driver()
        return self._driver

    def _create_driver(self) -> webdriver.Chrome:
        options = webdriver.ChromeOptions()
        if self.headless:
            options.add_argument("--headless=new")
        options.add_argument(f"--window-size={DEFAULT_WINDOW_WIDTH},{DEFAULT_WINDOW_HEIGHT}")
        options.add_argument("--no-first-run")
        options.add_argument("--no-default-browser-check")

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

        if not self._has_user_data_dir_option():
            self._active_profile_dir = self._new_profile_dir()
            options.add_argument(f"--user-data-dir={self._active_profile_dir}")

        chrome_binary = _find_chrome_binary()
        if chrome_binary is not None:
            options.binary_location = str(chrome_binary)

        log_path = self._new_chromedriver_log_path()
        self._active_chromedriver_log_path = log_path
        service = Service(
            self._resolve_chromedriver_path(chrome_binary),
            log_output=str(log_path),
        )
        try:
            driver = webdriver.Chrome(service=service, options=options)
        except Exception:
            logger.exception(
                "Chrome driver startup failed. chromedriver_log=%s chrome_binary=%s profile_dir=%s headless=%s",
                log_path,
                chrome_binary,
                self._active_profile_dir,
                self.headless,
            )
            self._stop_service(service)
            self._remove_active_profile_dir()
            raise
        logger.debug("Chrome driver created.")
        return driver

    def _resolve_chromedriver_path(self, chrome_binary: Path | None) -> str:
        chrome_version = _get_chrome_version(chrome_binary)
        cached_driver = _find_cached_chromedriver(chrome_version)
        if cached_driver is not None:
            logger.info(
                "Using cached ChromeDriver %s for Chrome version %s",
                cached_driver,
                chrome_version or "unknown",
            )
            return str(cached_driver)

        manager_kwargs: dict[str, str] = {}
        if chrome_version:
            manager_kwargs["driver_version"] = chrome_version
        logger.info(
            "No cached ChromeDriver found for Chrome version %s; resolving via webdriver_manager.",
            chrome_version or "unknown",
        )
        return ChromeDriverManager(**manager_kwargs).install()

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

    def purge_stale_processes(self) -> None:
        """Terminate scraper-owned Chrome processes left behind by previous sessions."""
        _PROFILE_ROOT.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            self._purge_stale_processes_windows(_PROFILE_ROOT)
        else:
            logger.debug("Stale Chrome process purge is currently implemented for Windows only.")
        self._remove_stale_profile_dirs()

    def _purge_stale_processes_windows(self, profile_root: Path) -> None:
        script = r"""
$root = [System.IO.Path]::GetFullPath($args[0])
$escaped = [WildcardPattern]::Escape($root)
Get-CimInstance Win32_Process |
  Where-Object {
    ($_.Name -eq 'chrome.exe' -or $_.Name -eq 'chromedriver.exe') -and
    $_.CommandLine -like "*$escaped*"
  } |
  ForEach-Object {
    try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop } catch {}
  }
"""
        try:
            subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-Command",
                    script,
                    str(profile_root),
                ],
                check=False,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except Exception as exc:
            logger.debug("Unable to purge stale scraper-owned Chrome processes: %s", exc)

    def _new_profile_dir(self) -> Path:
        profile_dir = _PROFILE_ROOT / f"profile_{os.getpid()}_{int(time.time() * 1000)}"
        profile_dir.mkdir(parents=True, exist_ok=True)
        return profile_dir

    def _new_chromedriver_log_path(self) -> Path:
        _LOG_ROOT.mkdir(parents=True, exist_ok=True)
        return _LOG_ROOT / f"chromedriver_{os.getpid()}_{int(time.time() * 1000)}.log"

    def _has_user_data_dir_option(self) -> bool:
        return any(str(flag).startswith("--user-data-dir=") for flag in self.extra_options)

    def _remove_active_profile_dir(self) -> None:
        if self._active_profile_dir is None:
            return
        profile_dir = self._active_profile_dir
        self._active_profile_dir = None
        try:
            shutil.rmtree(profile_dir, ignore_errors=True)
        except Exception as exc:
            logger.debug("Unable to remove Chrome profile directory %s: %s", profile_dir, exc)

    def _remove_stale_profile_dirs(self) -> None:
        if not _PROFILE_ROOT.exists():
            return
        cutoff = time.time() - 60
        for profile_dir in _PROFILE_ROOT.iterdir():
            if not profile_dir.is_dir() or profile_dir == self._active_profile_dir:
                continue
            try:
                if profile_dir.stat().st_mtime > cutoff:
                    continue
                shutil.rmtree(profile_dir, ignore_errors=True)
            except Exception as exc:
                logger.debug("Unable to remove stale Chrome profile directory %s: %s", profile_dir, exc)

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
                )
                return str(pid).encode("ascii") in (result.stdout or b"")

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
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
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


def _find_chrome_binary() -> Path | None:
    candidates = [
        *_WINDOWS_CHROME_CANDIDATES,
        Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
    ]
    for candidate in candidates:
        if candidate and candidate.exists():
            return candidate
    return None


def _get_chrome_version(chrome_binary: Path | None) -> str | None:
    if chrome_binary is None or not chrome_binary.exists():
        return None

    if os.name != "nt":
        return None

    try:
        completed = subprocess.run(
            [
                "powershell",
                "-NoProfile",
                "-Command",
                "(Get-Item $args[0]).VersionInfo.ProductVersion",
                str(chrome_binary),
            ],
            check=False,
            capture_output=True,
            text=True,
        )
        version = (completed.stdout or "").strip()
        return version or None
    except Exception as exc:
        logger.debug("Unable to determine Chrome version from %s: %s", chrome_binary, exc)
        return None


def _find_cached_chromedriver(chrome_version: str | None) -> Path | None:
    if chrome_version is None or not _CHROMEDRIVER_VERSION_ROOT.exists():
        return None

    major_version = chrome_version.split(".", maxsplit=1)[0]
    candidates: list[tuple[tuple[int, ...], Path]] = []
    for version_dir in _CHROMEDRIVER_VERSION_ROOT.iterdir():
        if not version_dir.is_dir() or not version_dir.name.startswith(f"{major_version}."):
            continue
        for driver_path in version_dir.rglob("chromedriver.exe"):
            version_key = _parse_version_tuple(version_dir.name)
            if version_key is not None:
                candidates.append((version_key, driver_path))

    if not candidates:
        return None

    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _parse_version_tuple(version: str) -> tuple[int, ...] | None:
    try:
        return tuple(int(part) for part in version.split("."))
    except ValueError:
        return None
