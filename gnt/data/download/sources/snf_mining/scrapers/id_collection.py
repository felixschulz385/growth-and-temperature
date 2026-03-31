"""Collect mine IDs from the Capital IQ screener using the built-in export."""

from __future__ import annotations

import logging
import time
from pathlib import Path

import duckdb
import pandas as pd
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from ..config import (
    DEFAULT_WAIT_SECONDS,
    DOWNLOAD_WAIT_SECONDS,
    EXPORT_DIR,
    SCREENER_URL,
    SEL,
)
from ..storage.database import upsert_mine_ids

logger = logging.getLogger(__name__)

_EXCEL_SUFFIXES = (".xls", ".xlsx")


def collect_all_ids(
    driver: WebDriver,
    conn: duckdb.DuckDBPyConnection,
    wait: int = DEFAULT_WAIT_SECONDS,
    download_wait: int = DOWNLOAD_WAIT_SECONDS,
    download_dir: str | Path | None = None,
) -> list[str]:
    _wait = WebDriverWait(driver, wait)
    export_dir = Path(download_dir) if download_dir else EXPORT_DIR
    export_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Opening screener...")
    driver.get(SCREENER_URL)
    run_btn = _wait.until(EC.presence_of_element_located((By.XPATH, SEL["run_screen_btn"])))
    driver.execute_script("arguments[0].click();", run_btn)
    logger.info("Screen submitted; waiting for results...")

    _wait.until(EC.presence_of_element_located((By.CSS_SELECTOR, SEL["pager_container"])))
    logger.info("Screener results loaded; starting export.")

    before = {
        p.name
        for suffix in _EXCEL_SUFFIXES
        for p in export_dir.glob(f"*{suffix}")
    }

    export_results_as_values(driver, timeout=wait)
    wait_for_download_modal_and_click(driver, timeout=wait)
    exported_file = wait_for_new_download(
        download_dir=export_dir,
        timeout=download_wait,
        known_filenames=before,
    )

    extracted_ids = extract_ids_from_xls(exported_file)
    inserted = upsert_mine_ids(conn, extracted_ids)
    logger.info(
        "ID collection complete: %d IDs extracted from export (%d newly inserted).",
        len(extracted_ids),
        inserted,
    )
    return _all_ids_from_db(conn)


def _all_ids_from_db(conn: duckdb.DuckDBPyConnection) -> list[str]:
    rows = conn.execute("SELECT mine_id FROM mines ORDER BY mine_id").fetchall()
    return [r[0] for r in rows]


def export_results_as_values(driver: WebDriver, timeout: int = DEFAULT_WAIT_SECONDS) -> None:
    select_el = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located(
            (By.XPATH, "//select[option[text()='Results As Table Function']]")
        )
    )
    driver.execute_script(
        """
        var select = arguments[0];
        select.value = '2';
        select.dispatchEvent(new Event('change'));
        """,
        select_el,
    )
    export_btn = WebDriverWait(driver, timeout).until(
        EC.element_to_be_clickable((By.XPATH, SEL["export_button"]))
    )
    export_btn.click()


def wait_for_download_modal_and_click(driver: WebDriver, timeout: int = 15) -> None:
    modal = WebDriverWait(driver, timeout).until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, SEL["download_modal"]))
    )
    modal.find_element(By.CSS_SELECTOR, SEL["download_link"]).click()


def wait_for_new_download(
    download_dir: str | Path,
    timeout: int,
    known_filenames: set[str] | None = None,
) -> Path:
    download_dir = Path(download_dir)
    known_filenames = known_filenames or set()
    start = time.time()

    while time.time() - start < timeout:
        candidates = sorted(
            (
                p
                for suffix in _EXCEL_SUFFIXES
                for p in download_dir.glob(f"*{suffix}")
            ),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        for candidate in candidates:
            if candidate.name in known_filenames:
                continue
            if candidate.stat().st_size <= 0:
                continue
            crdownload = candidate.with_suffix(candidate.suffix + ".crdownload")
            if crdownload.exists():
                continue
            return candidate
        time.sleep(0.5)

    raise TimeoutError(
        f"Timed out waiting for exported Excel file in {download_dir} after {timeout}s"
    )


def extract_ids_from_xls(xls_path: str | Path) -> list[str]:
    xls_path = Path(xls_path)
    df = pd.read_excel(xls_path, header=None)
    if df.empty:
        return []

    best_col = None
    best_score = -1
    for col in df.columns:
        values = (
            df[col]
            .dropna()
            .astype(str)
            .str.strip()
            .str.replace(r"\.0$", "", regex=True)
        )
        if values.empty:
            continue
        score = values.str.fullmatch(r"\d+").sum()
        if score > best_score:
            best_score = int(score)
            best_col = col

    if best_col is None:
        return []

    ids = (
        df[best_col]
        .dropna()
        .astype(str)
        .str.strip()
        .str.replace(r"\.0$", "", regex=True)
    )
    ids = ids[ids.str.fullmatch(r"\d+")]

    seen = set()
    unique_ids: list[str] = []
    for value in ids.tolist():
        if value not in seen:
            unique_ids.append(value)
            seen.add(value)
    return unique_ids
