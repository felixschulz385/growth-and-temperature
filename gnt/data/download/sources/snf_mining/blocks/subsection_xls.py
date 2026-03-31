"""Detail block: discover profile subsections and export XLS per subsection."""

from __future__ import annotations

import logging
from pathlib import Path
import re

from selenium.common.exceptions import (
    NoSuchElementException,
    StaleElementReferenceException,
    TimeoutException,
)
from selenium.webdriver.chrome.webdriver import WebDriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import WebDriverWait

from ..config import DOWNLOAD_WAIT_SECONDS, EXPORT_DIR
from ..scrapers.geometry import extract_property_and_linked_geometries
from ..scrapers.id_collection import wait_for_download_modal_and_click, wait_for_new_download
from ..storage.subsections import (
    build_subsection_records,
    clear_detail_rows,
    ensure_detail_tables,
    insert_export_row,
    insert_geometry_rows,
    insert_parsed_workbook,
    insert_parsed_cells,
    insert_subsection_records,
    update_export_parse_metadata,
)
from ..parsing.xls import (
    PROPERTY_PROFILE_LABEL,
    normalize_subsection_label,
    parse_subsection_xls,
)
from .base import BaseBlock

logger = logging.getLogger(__name__)

_EXCEL_SUFFIXES = (".xls", ".xlsx")


class ToolbarAction:
    EXPORT_EXCEL = 32


_RETRY_EXC = (NoSuchElementException, StaleElementReferenceException)


class SubsectionXlsBlock(BaseBlock):
    """Enumerate subsection links, export XLS for each, and parse workbook cells."""

    name = "subsection_xls"

    def ensure_schema(self) -> None:
        logger.debug("[%s] Ensuring subsection export schema exists.", self.name)
        ensure_detail_tables(self.conn)
        logger.debug("[%s] Subsection export schema ensured.", self.name)

    def scrape(self, driver: WebDriver, mine_id: str) -> None:
        logger.info("[%s] Starting scrape for mine_id=%s", self.name, mine_id)
        mine_sort_id = _sortable_mine_id(mine_id)
        export_dir = EXPORT_DIR / "detail" / "subsections" / mine_sort_id
        export_dir.mkdir(parents=True, exist_ok=True)
        logger.debug("[%s] Export directory prepared: %s", self.name, export_dir)

        subsections = get_sidebar_sections_and_subsections(driver, timeout=self.wait)
        logger.info(
            "[%s] Sidebar discovery returned %d top-level section(s) for mine_id=%s",
            self.name,
            len(subsections),
            mine_id,
        )
        records = build_subsection_records(subsections)
        logger.info(
            "[%s] Prepared %d subsection record(s) for mine_id=%s",
            self.name,
            len(records),
            mine_id,
        )

        logger.debug("[%s] Clearing existing rows for mine_id=%s", self.name, mine_id)
        clear_detail_rows(self.conn, mine_id, include_cells=True)

        if records:
            insert_subsection_records(self.conn, mine_id, records)
            logger.debug("[%s] Inserted subsection index rows for mine_id=%s", self.name, mine_id)

        for subsection_idx, record in enumerate(records, start=1):
            if not record.subsection_href:
                logger.info(
                    "[%s] mine_id=%s subsection='%s' has no href; skipping export.",
                    self.name,
                    mine_id,
                    record.subsection_label,
                )
                continue

            logger.info(
                "[%s] Processing subsection %d/%d for mine_id=%s: %s / %s",
                self.name,
                subsection_idx,
                len(records),
                mine_id,
                record.section_label,
                record.subsection_label,
            )

            driver.get(record.subsection_href)
            logger.debug("[%s] Navigated to subsection href: %s", self.name, record.subsection_href)
            if normalize_subsection_label(record.subsection_label) == PROPERTY_PROFILE_LABEL:
                logger.debug("[%s] Extracting property geometries for mine_id=%s", self.name, mine_id)
                geometries = extract_property_and_linked_geometries(driver, components="1")
                inserted_geometries = insert_geometry_rows(self.conn, mine_id, geometries)
                logger.info(
                    "[%s] Inserted %d geometry rows for mine_id=%s",
                    self.name,
                    inserted_geometries,
                    mine_id,
                )

            known = {
                p.name
                for suffix in _EXCEL_SUFFIXES
                for p in EXPORT_DIR.glob(f"*{suffix}")
            }
            logger.debug(
                "[%s] Known XLS files before export for mine_id=%s subsection=%s: %d",
                self.name,
                mine_id,
                record.subsection_label,
                len(known),
            )
            xls_path = export_current_subsection_xls(
                driver,
                download_dir=EXPORT_DIR,
                known_filenames=known,
                timeout=self.wait,
                download_wait=DOWNLOAD_WAIT_SECONDS,
            )
            logger.debug("[%s] Downloaded XLS path: %s", self.name, xls_path)

            managed_xls_path = _rename_exported_xls(
                downloaded_xls=xls_path,
                mine_id=mine_id,
                subsection_idx=subsection_idx,
                section_label=record.section_label,
                subsection_label=record.subsection_label,
                target_dir=export_dir,
            )
            logger.debug("[%s] Managed XLS path: %s", self.name, managed_xls_path)

            insert_export_row(self.conn, mine_id, record, str(managed_xls_path))
            logger.info(
                "[%s] Recorded export for mine_id=%s subsection=%s",
                self.name,
                mine_id,
                record.subsection_label,
            )

            logger.info(
                "[%s] Parsing exported workbook for mine_id=%s subsection=%s: %s",
                self.name,
                mine_id,
                record.subsection_label,
                managed_xls_path,
            )
            parsed = parse_subsection_xls(
                managed_xls_path,
                subsection_label=record.subsection_label,
            )
            if parsed:
                update_export_parse_metadata(
                    self.conn,
                    mine_id,
                    record,
                    str(managed_xls_path),
                    xls_sha256=parsed.xls_sha256,
                    workbook_title=parsed.workbook_title,
                    workbook_subtitle=parsed.workbook_subtitle,
                    primary_sheet_name=parsed.primary_sheet_name,
                    content_subsection_label=parsed.content_subsection_label,
                )
                insert_parsed_workbook(self.conn, mine_id, record, parsed)
                inserted_cells = insert_parsed_cells(
                    self.conn,
                    mine_id,
                    record.subsection_label,
                    parsed.flat_cells,
                )
                logger.info(
                    "[%s] Parsed and inserted %d cell rows for mine_id=%s subsection=%s",
                    self.name,
                    inserted_cells,
                    mine_id,
                    record.subsection_label,
                )
            else:
                logger.debug(
                    "[%s] No parsed rows returned for mine_id=%s subsection=%s",
                    self.name,
                    mine_id,
                    record.subsection_label,
                )

        logger.info("[%s] Completed scrape for mine_id=%s", self.name, mine_id)


def _sortable_mine_id(mine_id: str) -> str:
    mine_id_str = str(mine_id).strip()
    if mine_id_str.isdigit():
        return f"id_{int(mine_id_str):015d}"
    return f"id_{_slug(mine_id_str)}"


def _slug(value: str, fallback: str = "unknown") -> str:
    slug = re.sub(r"[^a-zA-Z0-9]+", "_", str(value).strip().lower())
    slug = slug.strip("_")
    return slug or fallback


def _rename_exported_xls(
    downloaded_xls: Path,
    mine_id: str,
    subsection_idx: int,
    section_label: str,
    subsection_label: str,
    target_dir: Path | None = None,
) -> Path:
    directory = Path(target_dir) if target_dir is not None else downloaded_xls.parent
    directory.mkdir(parents=True, exist_ok=True)
    stem = (
        f"{_sortable_mine_id(mine_id)}"
        f"__{subsection_idx:03d}"
        f"__{_slug(section_label)}"
        f"__{_slug(subsection_label)}"
    )
    suffix = downloaded_xls.suffix or ".xlsx"
    candidate = directory / f"{stem}{suffix}"

    duplicate_idx = 1
    while candidate.exists() and candidate.resolve() != downloaded_xls.resolve():
        candidate = directory / f"{stem}__{duplicate_idx:02d}{suffix}"
        duplicate_idx += 1

    if downloaded_xls.resolve() == candidate.resolve():
        return downloaded_xls
    return downloaded_xls.rename(candidate)


def _get_sidebar_nav(driver: WebDriver, timeout: int = 8):
    wrapper = WebDriverWait(driver, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "div.page-sidebar-wrapper"))
    )
    nav = WebDriverWait(wrapper, timeout).until(
        EC.presence_of_element_located((By.CSS_SELECTOR, "nav[data-testid='side-navigation']"))
    )
    return wrapper, nav


def _is_sidebar_expanded(driver: WebDriver, timeout: int = 3) -> bool:
    try:
        _, nav = _get_sidebar_nav(driver, timeout=timeout)
    except TimeoutException:
        return False
    return (nav.get_attribute("data-open") or "").lower() == "true"


def ensure_sidebar_expanded(driver: WebDriver, timeout: int = 8) -> bool:
    if _is_sidebar_expanded(driver, timeout=min(timeout, 3)):
        logger.debug("Sidebar is already expanded.")
        return False

    logger.debug("Sidebar is collapsed; expanding it before continuing.")
    toggle = WebDriverWait(driver, timeout).until(
        EC.element_to_be_clickable(
            (
                By.CSS_SELECTOR,
                'button[aria-label="Expand Page Side Navigation Menu"], '
                'button[aria-label="Collapse Page Side Navigation Menu"]',
            )
        )
    )
    try:
        toggle.click()
    except Exception:
        driver.execute_script("arguments[0].click();", toggle)

    WebDriverWait(driver, timeout).until(lambda d: _is_sidebar_expanded(d, timeout=1))
    logger.debug("Sidebar expanded successfully.")
    return True


def ensure_sidebar_collapsed(driver: WebDriver, timeout: int = 8) -> bool:
    if not _is_sidebar_expanded(driver, timeout=min(timeout, 3)):
        logger.debug("Sidebar is already collapsed.")
        return False

    logger.debug("Sidebar is expanded; collapsing it before continuing.")
    buttons = driver.find_elements(
        By.CSS_SELECTOR,
        "button[data-type='open-menu'][aria-label='Collapse Page Side Navigation Menu'], "
        "button[aria-label='Collapse Page Side Navigation Menu']",
    )
    for btn in buttons:
        if not btn.is_displayed():
            continue
        try:
            btn.click()
        except Exception:
            driver.execute_script("arguments[0].click();", btn)
        WebDriverWait(driver, timeout).until(lambda d: not _is_sidebar_expanded(d, timeout=1))
        logger.debug("Sidebar collapsed successfully.")
        return True

    logger.debug("No visible collapse button found for expanded sidebar; continuing without collapsing.")
    return False


def get_sidebar_sections_and_subsections(driver: WebDriver, timeout: int = 8) -> dict[str, list[dict[str, str]]]:
    logger.debug("Starting sidebar section/subsection discovery (timeout=%ss).", timeout)
    ensure_sidebar_expanded(driver, timeout=timeout)
    _, nav = _get_sidebar_nav(driver, timeout=timeout)

    try:
        expand_all_btn = nav.find_element(By.CSS_SELECTOR, "button[data-type='expand-all']")
        if (expand_all_btn.get_attribute("aria-expanded") or "").lower() == "false":
            try:
                expand_all_btn.click()
            except Exception:
                driver.execute_script("arguments[0].click();", expand_all_btn)
            logger.debug("Expanded all sidebar sections before discovery.")
    except Exception:
        logger.debug("Expand-all button not available; continuing with per-section expansion.")

    results: dict[str, list[dict[str, str]]] = {}
    menu_items = nav.find_elements(By.XPATH, ".//li[@data-type='menu-item' and @data-testid]")
    logger.debug("Found %d sidebar menu item(s).", len(menu_items))

    for item in menu_items:
        try:
            btn = item.find_element(By.XPATH, "./button")
            label = btn.find_element(By.XPATH, "./span[@data-button-text='true']/span").text.strip()
        except Exception:
            continue

        if not label:
            continue

        if btn.get_attribute("aria-expanded") != "true":
            try:
                btn.click()
            except Exception:
                driver.execute_script("arguments[0].click();", btn)
            WebDriverWait(driver, timeout).until(
                lambda d: btn.get_attribute("aria-expanded") == "true"
            )

        links = item.find_elements(By.XPATH, "./ul//a[@data-type='link']")

        results[label] = [
            {"label": a.text.strip(), "href": a.get_attribute("href") or ""}
            for a in links
            if a.text.strip() and a.is_displayed()
        ]
        logger.debug("Section '%s' yielded %d visible link(s).", label, len(results[label]))

    logger.info("Sidebar discovery complete: %d section(s).", len(results))
    return results


def _safe_call(fn):
    try:
        return fn()
    except _RETRY_EXC:
        return None


def _find_toolbar(driver: WebDriver, action_data_id: str, timeout: int):
    logger.debug("Locating toolbar for action data-id=%s", action_data_id)
    toolbar_xpath = (
        f"//div[contains(@class,'snl-hui-toolbar')]"
        f"[.//a[contains(@class,'hui-toolbutton') and @data-id='{action_data_id}']]"
    )

    def _pick_toolbar(_):
        candidates = driver.find_elements(By.XPATH, toolbar_xpath)
        if not candidates:
            return None
        visible = [el for el in candidates if el.is_displayed()]
        logger.debug(
            "Found %d toolbar candidate(s) for action data-id=%s; visible=%d",
            len(candidates),
            action_data_id,
            len(visible),
        )
        return visible[0] if visible else candidates[0]

    toolbar = WebDriverWait(driver, timeout).until(_pick_toolbar)
    logger.debug("Selected toolbar for action data-id=%s", action_data_id)
    return toolbar


def _find_target(scope, action_data_id: str):
    candidates = scope.find_elements(
        By.XPATH,
        f".//a[@data-id='{action_data_id}' and contains(@class,'hui-toolbutton')]",
    )
    if not candidates:
        raise NoSuchElementException(f"No hui-toolbutton with data-id={action_data_id}")
    visible = [a for a in candidates if a.is_displayed()]
    logger.debug(
        "Found %d candidate toolbar element(s) for action data-id=%s; visible=%d",
        len(candidates),
        action_data_id,
        len(visible),
    )
    return visible[0] if visible else candidates[0]


def _open_parent_dropdown(driver: WebDriver, target, timeout: int) -> bool:
    try:
        parent_li = target.find_element(
            By.XPATH, "./ancestor::li[contains(@class,'dropdown')][1]"
        )
        if "open" not in (parent_li.get_attribute("class") or ""):
            toggle = parent_li.find_element(By.XPATH, "./a[@data-toggle='dropdown']")
            try:
                toggle.click()
            except Exception:
                driver.execute_script("arguments[0].click();", toggle)
            WebDriverWait(driver, timeout).until(
                lambda d: "open" in (parent_li.get_attribute("class") or "")
            )
            logger.debug("Opened parent dropdown for toolbar action.")
            return True
    except NoSuchElementException:
        return False
    return False


def click_toolbar_action(driver: WebDriver, action_data_id: int | str, toolbar_el=None, timeout: int = 8) -> None:
    action_data_id = str(action_data_id)
    logger.debug("Clicking toolbar action data-id=%s", action_data_id)
    scope = toolbar_el or _find_toolbar(driver, action_data_id, timeout)

    target = WebDriverWait(driver, timeout).until(
        lambda d: _safe_call(lambda: _find_target(scope, action_data_id))
    )

    if _open_parent_dropdown(driver, target, timeout):
        target = _find_target(scope, action_data_id)

    if target.is_displayed():
        try:
            target.click()
        except Exception:
            driver.execute_script("arguments[0].click();", target)
    else:
        logger.debug(
            "Toolbar action target data-id=%s is not displayed; attempting JS click fallback.",
            action_data_id,
        )
        driver.execute_script("arguments[0].click();", target)
    logger.debug("Toolbar action clicked for data-id=%s", action_data_id)


def export_current_subsection_xls(
    driver: WebDriver,
    download_dir: str | Path,
    known_filenames: set[str],
    timeout: int,
    download_wait: int,
) -> Path:
    logger.debug("Starting XLS export from current subsection page.")
    ensure_sidebar_collapsed(driver, timeout=timeout)
    click_toolbar_action(driver, ToolbarAction.EXPORT_EXCEL, timeout=timeout)

    try:
        wait_for_download_modal_and_click(driver, timeout=timeout)
    except TimeoutException:
        # Some pages may trigger direct download with no modal.
        logger.debug("Download modal not shown; assuming direct download path.")
        pass

    downloaded = wait_for_new_download(
        download_dir=download_dir,
        timeout=download_wait,
        known_filenames=known_filenames,
    )
    logger.debug("XLS export complete: %s", downloaded)
    return downloaded


