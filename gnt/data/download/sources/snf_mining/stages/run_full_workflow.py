"""Top-level stage orchestration for the SNF mining workflow."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Collection, Iterable

from ..config import (
    DEFAULT_DB_PATH,
    DEFAULT_WAIT_SECONDS,
    DOWNLOAD_WAIT_SECONDS,
    EXPORT_DIR,
    PERIODIC_BROWSER_RESTART_MINE_INTERVAL,
    SMALL_SLEEP_SECONDS,
)
from ..scrapers.id_collection import collect_all_ids
from ..session.auth import load_credentials, login, logout
from ..session.browser import ManagedBrowser
from ..storage.database import get_connection
from .names import Stage, resolve_stages
from .parse_detail_exports import parse_detail_exports
from .scrape_detail_exports import scrape_detail_exports

logger = logging.getLogger(__name__)


def run_full_workflow(
    credentials_path: str | Path,
    db_path: str | Path = DEFAULT_DB_PATH,
    stages: Collection[Stage | str] | None = None,
    headless: bool = False,
    wait: int = DEFAULT_WAIT_SECONDS,
    download_wait: int = DOWNLOAD_WAIT_SECONDS,
    mine_ids: Iterable[str] | None = None,
    subsections: Iterable[str] | None = None,
    max_attempts: int = 3,
    sidebar_reload_attempts: int = 2,
    continue_on_error: bool = True,
    force_stages: Collection[Stage | str] | None = None,
    step_sleep_seconds: float = SMALL_SLEEP_SECONDS,
    restart_session_every_mines: int | None = PERIODIC_BROWSER_RESTART_MINE_INTERVAL,
) -> dict[str, object]:
    """Execute the requested SNF mining workflow stages."""
    logger.info(
        "Initializing workflow (credentials_path=%s, db_path=%s, headless=%s, wait=%s, download_wait=%s)",
        credentials_path,
        db_path,
        headless,
        wait,
        download_wait,
    )
    active_stages = resolve_stages(stages)
    forced_stages = resolve_stages(force_stages) if force_stages else frozenset()
    logger.info("Running stages: %s", [stage.value for stage in active_stages])
    if forced_stages:
        logger.info("Force rerun requested for stages: %s", [stage.value for stage in forced_stages])

    credentials = load_credentials(credentials_path)
    conn = get_connection(db_path)
    results: dict[str, object] = {}

    try:
        EXPORT_DIR.mkdir(parents=True, exist_ok=True)

        if Stage.DETAIL_EXPORTS in active_stages or Stage.IDS in active_stages:
            browser = ManagedBrowser(headless=headless, download_dir=str(EXPORT_DIR))
            with browser as driver:
                login(driver, credentials, wait=wait)
                try:
                    if Stage.IDS in active_stages:
                        ids = collect_all_ids(
                            driver,
                            conn,
                            wait=wait,
                            download_wait=download_wait,
                        )
                        results["ids"] = {"mine_count": len(ids)}

                    if Stage.DETAIL_EXPORTS in active_stages:
                        def _recover_driver():
                            restarted_driver = browser.restart()
                            login(restarted_driver, credentials, wait=wait)
                            return restarted_driver

                        results["detail_exports"] = scrape_detail_exports(
                            driver,
                            conn,
                            mine_ids=mine_ids,
                            wait=wait,
                            download_wait=download_wait,
                            step_sleep_seconds=step_sleep_seconds,
                            max_attempts=max_attempts,
                            sidebar_reload_attempts=sidebar_reload_attempts,
                            subsections=subsections,
                            force=Stage.DETAIL_EXPORTS in forced_stages,
                            recover_driver=_recover_driver,
                            restart_session_every_mines=restart_session_every_mines,
                        )
                finally:
                    active_driver = browser.current_driver
                    if active_driver is not None:
                        logout(active_driver, wait=wait)

        if Stage.DETAIL_PARSE in active_stages:
            results["detail_parse"] = parse_detail_exports(
                conn,
                mine_ids=mine_ids,
                subsections=subsections,
                continue_on_error=continue_on_error,
                force=Stage.DETAIL_PARSE in forced_stages,
            )
    finally:
        conn.close()

    return results
