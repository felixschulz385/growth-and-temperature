"""Stage implementation for scraping detail subsection exports."""

from __future__ import annotations

from dataclasses import dataclass
import logging
import sys
from typing import Callable, Iterable

from selenium.common.exceptions import TimeoutException, WebDriverException
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.ui import Select, WebDriverWait

from ..blocks.subsection_xls import (
    _rename_exported_xls,
    _sortable_mine_id,
    ensure_sidebar_expanded,
    export_current_subsection_xls,
    get_sidebar_sections_and_subsections,
)
from ..parsing.xls import PROPERTY_PROFILE_LABEL, normalize_subsection_label
from ..config import (
    DEFAULT_WAIT_SECONDS,
    DOWNLOAD_WAIT_SECONDS,
    EXPORT_DIR,
    PERIODIC_BROWSER_RESTART_MINE_INTERVAL,
    PROFILE_URL_TEMPLATE,
    SMALL_SLEEP_SECONDS,
)
from ..scrapers.geometry import extract_property_and_linked_geometries
from ..storage.database import (
    get_all_mine_ids,
    get_stage_pending_mine_ids,
    mark_stage_complete,
    reset_stage_completion,
)
from ..storage.subsections import (
    build_subsection_records,
    clear_stage_outputs,
    clear_subsection_stage_output,
    ensure_detail_tables,
    get_completed_stage_keys,
    insert_export_row,
    insert_geometry_rows,
    insert_subsection_records,
    upsert_subsection_stage_status,
)
from ..utils.workflow_helpers import exception_brief, reset_profile_page_state, retry, sleep_politely

logger = logging.getLogger(__name__)

_EXCEL_SUFFIXES = (".xls", ".xlsx")
_STAGE_NAME = "detail_exports"



@dataclass(frozen=True, slots=True)
class _SelectFilterOverride:
    control_label: str
    mode: str = "select_value"
    target_text: str | None = None
    target_value: str | None = None
    trigger_text: str | None = None
    apply_after: bool = True


_SUBSECTION_FILTER_OVERRIDES: dict[str, tuple[_SelectFilterOverride, ...]] = {
    normalize_subsection_label("financings"): (
        _SelectFilterOverride(
            control_label="Date Range",
            mode="select_value",
            target_text="All",
            target_value="-1",
            apply_after=True,
        ),
    ),
    normalize_subsection_label("reserves / resources & production chart"): (
        _SelectFilterOverride(
            control_label="Period",
            mode="select_value",
            target_text="All Years",
            target_value="3",
        ),
    ),
    normalize_subsection_label("ownership"): (
        _SelectFilterOverride(
            control_label="Periods",
            mode="custom_period_max",
            trigger_text="Customize Periods...",
            apply_after=False,
        ),
    ),
}

_TEMPORARILY_SKIPPED_SUBSECTIONS = {
    normalize_subsection_label("cost curve"),
}


@dataclass(slots=True)
class _DetailScrapeStatus:
    total_mines: int
    processed_mines: int = 0
    exported_mines: int = 0
    exported_subsections: int = 0


def _print_status_window(status: _DetailScrapeStatus, current_mine_id: str | None = None) -> None:
    lines = [
        "SNF Detail Export Status",
        f"Mines processed    : {status.processed_mines}/{status.total_mines}",
        f"Mines exported     : {status.exported_mines}",
        f"Subsections export : {status.exported_subsections}",
    ]
    if current_mine_id is not None:
        lines.append(f"Current mine       : {current_mine_id}")

    width = max(len(line) for line in lines)
    border = "+" + ("-" * (width + 2)) + "+"
    window = [border, *[f"| {line.ljust(width)} |" for line in lines], border]
    print("\n".join(window), file=sys.stdout, flush=True)


def scrape_detail_exports(
    driver,
    conn,
    mine_ids: Iterable[str] | None = None,
    wait: int = DEFAULT_WAIT_SECONDS,
    download_wait: int = DOWNLOAD_WAIT_SECONDS,
    step_sleep_seconds: float = SMALL_SLEEP_SECONDS,
    max_attempts: int = 3,
    sidebar_reload_attempts: int = 2,
    subsections: Iterable[str] | None = None,
    force: bool = False,
    recover_driver: Callable[[], object] | None = None,
    restart_session_every_mines: int | None = PERIODIC_BROWSER_RESTART_MINE_INTERVAL,
) -> dict[str, int]:
    """Scrape subsection XLS exports and Property Profile geometries."""
    subsection_filter = _normalize_subsection_filter(subsections)
    logger.info(
        "Starting detail export scrape (wait=%s, download_wait=%s, max_attempts=%s, sidebar_reload_attempts=%s, subsection_filter_count=%s, force=%s)",
        wait,
        download_wait,
        max_attempts,
        sidebar_reload_attempts,
        len(subsection_filter) if subsection_filter is not None else "all",
        force,
    )
    ensure_detail_tables(conn)

    if mine_ids is None:
        pending = get_all_mine_ids(conn) if force else get_stage_pending_mine_ids(conn, _STAGE_NAME)
    else:
        pending = [str(mid) for mid in mine_ids]
    pending = sorted(pending, key=lambda mid: int(mid) if str(mid).isdigit() else str(mid))
    logger.info("Prepared %d mine IDs for detail export scraping.", len(pending))

    if force and pending:
        logger.info("Force rerun requested for %d mine(s) in stage %s.", len(pending), _STAGE_NAME)
        reset_stage_completion(conn, pending, _STAGE_NAME)
        for mine_id in pending:
            clear_stage_outputs(conn, mine_id, _STAGE_NAME)

    total_exports = 0
    total_geometry_rows = 0
    mine_failures = 0
    subsection_failures = 0
    subsection_skips = 0
    status = _DetailScrapeStatus(total_mines=len(pending))
    _print_status_window(status)

    for mine_id in pending:
        if (
            recover_driver is not None
            and restart_session_every_mines is not None
            and restart_session_every_mines > 0
            and status.processed_mines > 0
            and status.processed_mines % restart_session_every_mines == 0
        ):
            logger.info(
                "Periodically restarting browser session after %d processed mine(s).",
                status.processed_mines,
            )
            driver = recover_driver()

        logger.info("Starting detail export processing for mine_id=%s", mine_id)
        _print_status_window(status, current_mine_id=mine_id)
        mine_setup_attempt = 0
        setup_ok = False
        mine_export_completed = False
        while True:
            try:
                mine_sort_id = _sortable_mine_id(mine_id)
                export_dir = EXPORT_DIR / "detail" / "subsections" / mine_sort_id
                export_dir.mkdir(parents=True, exist_ok=True)
                logger.debug("Export directory prepared for mine_id=%s at %s", mine_id, export_dir)

                profile_url = PROFILE_URL_TEMPLATE.format(mine_id=mine_id)
                reset_profile_page_state(
                    driver=driver,
                    profile_url=profile_url,
                    step_sleep_seconds=step_sleep_seconds,
                    mine_id=mine_id,
                )
                subsections = _discover_subsections_with_reload(
                    driver=driver,
                    profile_url=profile_url,
                    wait=wait,
                    max_attempts=max_attempts,
                    reload_attempts=sidebar_reload_attempts,
                    step_sleep_seconds=step_sleep_seconds,
                    mine_id=mine_id,
                )
                logger.info(
                    "Discovered %d top-level section(s) for mine_id=%s",
                    len(subsections),
                    mine_id,
                )
                setup_ok = True
                break
            except Exception as exc:
                if (
                    recover_driver is not None
                    and mine_setup_attempt == 0
                    and _is_restartable_browser_error(exc)
                ):
                    mine_setup_attempt += 1
                    logger.warning(
                            "Browser/session error during setup/discovery for mine_id=%s; restarting browser and retrying once. Error: %s",
                            mine_id,
                            exception_brief(exc),
                        )
                    driver = recover_driver()
                    continue
                mine_failures += 1
                status.processed_mines += 1
                _print_status_window(status, current_mine_id=mine_id)
                logger.warning(
                    "Skipping mine_id=%s due to setup/discovery failure: %s",
                    mine_id,
                    exception_brief(exc),
                )
                break

        if not setup_ok:
            continue

        records = build_subsection_records(subsections)
        if subsection_filter is not None:
            records = [
                record
                for record in records
                if normalize_subsection_label(record.subsection_label) in subsection_filter
            ]
        logger.info(
            "Prepared %d subsection record(s) for mine_id=%s",
            len(records),
            mine_id,
        )

        if records:
            insert_subsection_records(conn, mine_id, records)
            logger.debug("Inserted subsection metadata for mine_id=%s", mine_id)

        subsection_debug = subsection_filter is not None and mine_ids is not None
        completed_keys = set() if (force or subsection_debug) else get_completed_stage_keys(conn, mine_id, _STAGE_NAME)
        records_to_process = [
            record
            for record in records
            if (record.section_label, record.subsection_label) not in completed_keys
        ]
        subsection_skips += len(records) - len(records_to_process)
        logger.info(
            "Mine_id=%s has %d completed subsection export(s) already; %d pending.",
            mine_id,
            len(records) - len(records_to_process),
            len(records_to_process),
        )

        mine_had_failure = False
        for subsection_idx, record in enumerate(records, start=1):
            if (record.section_label, record.subsection_label) in completed_keys:
                continue
            if _should_temporarily_skip_subsection(record.subsection_label):
                subsection_skips += 1
                upsert_subsection_stage_status(
                    conn,
                    mine_id,
                    record,
                    stage_name=_STAGE_NAME,
                    status="completed",
                    error_msg="Temporarily skipped: Cost Curve export disabled",
                )
                logger.info(
                    "Temporarily skipping subsection (mine_id=%s, subsection=%s) while Cost Curve export is disabled.",
                    mine_id,
                    record.subsection_label,
                )
                continue
            if not record.subsection_href:
                logger.debug(
                    "Skipping subsection with empty href (mine_id=%s, subsection=%s)",
                    mine_id,
                    record.subsection_label,
                )
                continue

            subsection_attempt = 0
            while True:
                try:
                    logger.info(
                        "Processing subsection %d/%d for mine_id=%s: %s / %s",
                        subsection_idx,
                        len(records),
                        mine_id,
                        record.section_label,
                        record.subsection_label,
                    )
                    clear_subsection_stage_output(conn, mine_id, record, _STAGE_NAME)
                    driver.get(record.subsection_href)
                    sleep_politely(step_sleep_seconds)

                    if normalize_subsection_label(record.subsection_label) == PROPERTY_PROFILE_LABEL:
                        logger.debug("Extracting property geometries for mine_id=%s", mine_id)
                        try:
                            geometries = retry(
                                lambda: extract_property_and_linked_geometries(driver, components="1"),
                                attempts=max_attempts,
                                sleep_seconds=step_sleep_seconds,
                                label=f"extract geometries mine_id={mine_id}",
                            )
                        except Exception as exc:
                            logger.warning(
                                "Geometry extraction failed for mine_id=%s subsection=%s, continuing with XLS export: %s",
                                mine_id,
                                record.subsection_label,
                                exception_brief(exc),
                            )
                        else:
                            inserted_geometries = insert_geometry_rows(conn, mine_id, geometries)
                            total_geometry_rows += inserted_geometries
                            logger.info(
                                "Inserted %d geometry row(s) for mine_id=%s",
                                inserted_geometries,
                                mine_id,
                            )

                    known = {
                        path.name
                        for suffix in _EXCEL_SUFFIXES
                        for path in EXPORT_DIR.glob(f"*{suffix}")
                    }
                    downloaded = retry(
                        lambda: _export_subsection_xls_once(
                            driver=driver,
                            subsection_href=record.subsection_href,
                            subsection_label=record.subsection_label,
                            download_dir=EXPORT_DIR,
                            known_filenames=known,
                            timeout=wait,
                            download_wait=download_wait,
                            step_sleep_seconds=step_sleep_seconds,
                        ),
                        attempts=max_attempts,
                        sleep_seconds=step_sleep_seconds,
                        label=f"export xls mine_id={mine_id}, subsection={record.subsection_label}",
                    )

                    managed_xls = _rename_exported_xls(
                        downloaded_xls=downloaded,
                        mine_id=mine_id,
                        subsection_idx=subsection_idx,
                        section_label=record.section_label,
                        subsection_label=record.subsection_label,
                        target_dir=export_dir,
                    )

                    insert_export_row(conn, mine_id, record, str(managed_xls))
                    upsert_subsection_stage_status(
                        conn,
                        mine_id,
                        record,
                        stage_name=_STAGE_NAME,
                        status="completed",
                    )
                    total_exports += 1
                    status.exported_subsections = total_exports
                    _print_status_window(status, current_mine_id=mine_id)
                    logger.info(
                        "Recorded export %d for mine_id=%s subsection=%s",
                        total_exports,
                        mine_id,
                        record.subsection_label,
                    )
                    sleep_politely(step_sleep_seconds)
                    break
                except Exception as exc:
                    if _is_restartable_browser_error(exc):
                        crash_error = exception_brief(exc)
                        if recover_driver is not None:
                            logger.warning(
                                "Browser/session error during subsection export for mine_id=%s subsection=%s; restarting browser and marking subsection as skipped so it is not retried. Error: %s",
                                mine_id,
                                record.subsection_label,
                                crash_error,
                            )
                            driver = recover_driver()
                        else:
                            logger.warning(
                                "Browser/session error during subsection export for mine_id=%s subsection=%s; marking subsection as skipped so it is not retried. Error: %s",
                                mine_id,
                                record.subsection_label,
                                crash_error,
                            )

                        subsection_skips += 1
                        upsert_subsection_stage_status(
                            conn,
                            mine_id,
                            record,
                            stage_name=_STAGE_NAME,
                            status="completed",
                            error_msg=f"Skipped after browser/session crash during export: {crash_error}",
                        )
                        break

                    mine_had_failure = True
                    subsection_failures += 1
                    upsert_subsection_stage_status(
                        conn,
                        mine_id,
                        record,
                        stage_name=_STAGE_NAME,
                        status="failed",
                        error_msg=exception_brief(exc),
                    )
                    logger.warning(
                        "Skipping subsection due to failure (mine_id=%s, subsection=%s): %s",
                        mine_id,
                        record.subsection_label,
                        exception_brief(exc),
                    )
                    break

        if not mine_had_failure:
            mark_stage_complete(conn, mine_id, _STAGE_NAME)
            mine_export_completed = True

        status.processed_mines += 1
        if mine_export_completed:
            status.exported_mines += 1
        _print_status_window(status, current_mine_id=mine_id)

        logger.info(
            "Completed detail export processing for mine_id=%s (%d subsection record(s)).",
            mine_id,
            len(records),
        )

    logger.info(
        "Detail export scrape complete: %d mine(s), %d XLS export(s), %d geometry row(s), %d skipped subsection(s), %d mine failure(s), %d subsection failure(s).",
        len(pending),
        total_exports,
        total_geometry_rows,
        subsection_skips,
        mine_failures,
        subsection_failures,
    )
    return {
        "mine_count": len(pending),
        "export_count": total_exports,
        "geometry_count": total_geometry_rows,
        "skipped_subsection_count": subsection_skips,
        "mine_failure_count": mine_failures,
        "subsection_failure_count": subsection_failures,
    }


def _discover_subsections_with_reload(
    driver,
    profile_url: str,
    wait: int,
    max_attempts: int,
    reload_attempts: int,
    step_sleep_seconds: float,
    mine_id: str,
):
    last_exc: Exception | None = None
    tries = max(1, reload_attempts)
    logger.debug(
        "Starting sidebar discovery for mine_id=%s with up to %d reload attempt(s)",
        mine_id,
        tries,
    )

    for reload_idx in range(1, tries + 1):
        try:
            return retry(
                lambda: _discover_subsections_once(driver=driver, timeout=wait),
                attempts=max_attempts,
                sleep_seconds=step_sleep_seconds,
                label=f"discover subsections for mine_id={mine_id} (reload attempt {reload_idx}/{tries})",
            )
        except Exception as exc:
            last_exc = exc
            if reload_idx < tries:
                logger.info(
                    "Sidebar discovery failed for mine_id=%s; reloading profile page (%d/%d). Error: %s",
                    mine_id,
                    reload_idx,
                    tries,
                    exc,
                )
                driver.get(profile_url)
                sleep_politely(step_sleep_seconds * (reload_idx + 1))
                driver.refresh()
                sleep_politely(step_sleep_seconds * (reload_idx + 1))

    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"Sidebar discovery failed unexpectedly for mine_id={mine_id}")


def _discover_subsections_once(driver, timeout: int):
    # Ensure left nav is open before we query sections/links.
    ensure_sidebar_expanded(driver, timeout=timeout)
    return get_sidebar_sections_and_subsections(driver, timeout=timeout)


def _export_subsection_xls_once(
    driver,
    subsection_href: str,
    subsection_label: str,
    download_dir,
    known_filenames: set[str],
    timeout: int,
    download_wait: int,
    step_sleep_seconds: float,
):
    driver.get(subsection_href)
    sleep_politely(step_sleep_seconds)
    _apply_subsection_filter_overrides_if_needed(
        driver=driver,
        subsection_label=subsection_label,
        timeout=timeout,
    )
    return export_current_subsection_xls(
        driver,
        download_dir=download_dir,
        known_filenames=known_filenames,
        timeout=timeout,
        download_wait=download_wait,
    )


def _normalize_subsection_filter(subsections: Iterable[str] | None) -> set[str] | None:
    if subsections is None:
        return None
    normalized = {
        normalize_subsection_label(str(subsection))
        for subsection in subsections
        if str(subsection).strip()
    }
    return normalized or None


def _should_temporarily_skip_subsection(subsection_label: str) -> bool:
    return normalize_subsection_label(subsection_label) in _TEMPORARILY_SKIPPED_SUBSECTIONS


def _apply_subsection_filter_overrides_if_needed(
    driver,
    subsection_label: str,
    timeout: int,
) -> None:
    overrides = _get_subsection_filter_overrides(subsection_label)
    if not overrides:
        logger.debug(
            "No subsection filter override configured for subsection=%s",
            subsection_label,
        )
        return

    panel_wait = _build_override_wait(driver, timeout, cap=8)
    control_wait = _build_override_wait(driver, timeout, cap=10)
    verify_wait = _build_override_wait(driver, timeout, cap=6)
    menu_wait = _build_override_wait(driver, timeout, cap=8)
    apply_wait = _build_override_wait(driver, timeout, cap=8)

    logger.debug(
        "Applying %d subsection filter override(s) for subsection=%s",
        len(overrides),
        subsection_label,
    )
    try:
        needs_apply = False
        for override in overrides:
            _open_filter_panel_if_needed(
                driver,
                control_label=override.control_label,
                wait=panel_wait,
            )
            logger.debug(
                "Applying subsection filter override for subsection=%s label=%s mode=%s text=%s value=%s trigger=%s",
                subsection_label,
                override.control_label,
                override.mode,
                override.target_text,
                override.target_value,
                override.trigger_text,
            )
            if override.mode == "select_value":
                _apply_select_value_override(
                    driver=driver,
                    subsection_label=subsection_label,
                    override=override,
                    control_wait=control_wait,
                    verify_wait=verify_wait,
                    menu_wait=menu_wait,
                )
                needs_apply = needs_apply or override.apply_after
                continue
            if override.mode == "custom_period_max":
                _apply_maximum_custom_periods(
                    driver=driver,
                    subsection_label=subsection_label,
                    override=override,
                    control_wait=control_wait,
                    verify_wait=verify_wait,
                    menu_wait=menu_wait,
                )
                needs_apply = needs_apply or override.apply_after
                continue
            raise RuntimeError(f"Unsupported filter override mode: {override.mode}")

        if needs_apply and _has_visible_apply_button(driver):
            apply_wait.until(
                EC.element_to_be_clickable((By.CSS_SELECTOR, "button[id^='Apply_']"))
            ).click()
            logger.debug(
                "Filter Apply clicked successfully for subsection=%s",
                subsection_label,
            )
        else:
            logger.debug(
                "No top-level Apply click needed for subsection=%s",
                subsection_label,
            )
    except Exception as exc:
        raise RuntimeError(
            f"Failed to apply subsection filter override for subsection={subsection_label}"
        ) from exc


def _apply_select_value_override(
    driver,
    subsection_label: str,
    override: _SelectFilterOverride,
    control_wait: WebDriverWait,
    verify_wait: WebDriverWait,
    menu_wait: WebDriverWait,
) -> None:
    if override.target_text is None or override.target_value is None:
        raise RuntimeError("select_value override requires target_text and target_value")

    select_el = _find_select_control(
        driver,
        control_label=override.control_label,
        wait=control_wait,
    )
    _set_select_value_via_js(
        driver,
        select_el=select_el,
        target_value=override.target_value,
    )
    logger.debug(
        "Native select JS path executed for subsection=%s label=%s",
        subsection_label,
        override.control_label,
    )

    selectpicker_button = _find_visible_selectpicker_button(
        driver,
        select_el=select_el,
        control_label=override.control_label,
    )
    if selectpicker_button is None:
        _ensure_native_select_value(
            select_el=select_el,
            target_text=override.target_text,
            target_value=override.target_value,
            wait=verify_wait,
        )
        logger.debug(
            "Native select verification succeeded for subsection=%s label=%s",
            subsection_label,
            override.control_label,
        )
        return

    if _wait_for_selectpicker_button_text(
        selectpicker_button,
        target_text=override.target_text,
        wait=verify_wait,
    ):
        logger.debug(
            "Selectpicker button text updated via JS path for subsection=%s label=%s",
            subsection_label,
            override.control_label,
        )
        return

    logger.debug(
        "Falling back to visible selectpicker interaction for subsection=%s label=%s",
        subsection_label,
        override.control_label,
    )
    _set_selectpicker_value_via_ui(
        driver,
        selectpicker_button=selectpicker_button,
        target_text=override.target_text,
        wait=menu_wait,
    )
    _wait_for_selectpicker_button_text(
        selectpicker_button,
        target_text=override.target_text,
        wait=verify_wait,
        require_match=True,
    )


def _apply_maximum_custom_periods(
    driver,
    subsection_label: str,
    override: _SelectFilterOverride,
    control_wait: WebDriverWait,
    verify_wait: WebDriverWait,
    menu_wait: WebDriverWait,
) -> None:
    if override.trigger_text is None:
        raise RuntimeError("custom_period_max override requires trigger_text")

    logger.debug(
        "Entering custom_period_max override path for subsection=%s label=%s",
        subsection_label,
        override.control_label,
    )
    if not _ownership_historical_tab_present(driver):
        logger.debug(
            "Skipping custom_period_max override for subsection=%s because Historical tab is not present",
            subsection_label,
        )
        return

    select_el = _find_select_control(
        driver,
        control_label=override.control_label,
        wait=control_wait,
    )
    selectpicker_button = _find_visible_selectpicker_button(
        driver,
        select_el=select_el,
        control_label=override.control_label,
    )
    if selectpicker_button is None:
        raise RuntimeError(
            f"No visible selectpicker button found for control label={override.control_label}"
        )

    logger.debug(
        "Opening Periods dropdown for subsection=%s label=%s",
        subsection_label,
        override.control_label,
    )
    _open_selectpicker_dropdown(selectpicker_button, wait=menu_wait)
    logger.debug(
        "Selecting trigger option %s for subsection=%s",
        override.trigger_text,
        subsection_label,
    )
    _click_selectpicker_option_by_text(
        selectpicker_button=selectpicker_button,
        target_text=override.trigger_text,
        wait=menu_wait,
    )

    modal_wait = menu_wait
    modal_root = modal_wait.until(
        EC.visibility_of_element_located((By.CSS_SELECTOR, ".hui-custom-period-picker-modal"))
    )
    logger.debug("Custom period modal opened for subsection=%s", subsection_label)

    period_row = modal_root.find_element(
        By.CSS_SELECTOR, ".hui-cpp-availableperiods-row"
    )

    period_row.find_element(By.CSS_SELECTOR, ".btn").click()

    select_all = modal_wait.until(
        EC.visibility_of_element_located(
            (
                By.CSS_SELECTOR,
                ".hui-custom-period-picker-modal .period-picker-available-periods .bs-select-all",
            )
        )
    )
    try:
        select_all.click()
    except Exception:
        driver.execute_script("arguments[0].click();", select_all)
    logger.debug("Clicked Select All in custom period modal for subsection=%s", subsection_label)

    _close_inner_period_picker_if_present(modal_root, wait=menu_wait)

    done_button = modal_wait.until(
        EC.visibility_of_element_located(
            (By.CSS_SELECTOR, ".hui-custom-period-picker-modal .period-picker-apply")
        )
    )
    driver.execute_script("arguments[0].scrollIntoView({block: 'center'});", done_button)
    try:
        done_button.click()
    except Exception:
        driver.execute_script("arguments[0].click();", done_button)
    logger.debug("Clicked modal Done for subsection=%s", subsection_label)

    modal_wait.until(
        EC.invisibility_of_element_located((By.CSS_SELECTOR, ".hui-custom-period-picker-modal"))
    )
    logger.debug("Custom period modal closed successfully for subsection=%s", subsection_label)

    try:
        verify_wait.until(
            lambda _driver: select_el.get_attribute("value") == "Custom"
            or _normalize_visible_text(selectpicker_button.text) != "Last Ten Years"
        )
    except TimeoutException:
        verify_wait.until(lambda _driver: _modal_count(_driver) == 0)


def _ownership_historical_tab_present(driver) -> bool:
    historical_tabs = driver.find_elements(
        By.XPATH,
        (
            "//div[contains(@class,'hui-tabs')]"
            "//a[normalize-space()='Historical']"
        ),
    )
    return any(tab.is_displayed() for tab in historical_tabs)


def _get_subsection_filter_overrides(
    subsection_label: str,
) -> tuple[_SelectFilterOverride, ...]:
    normalized_subsection_label = normalize_subsection_label(subsection_label)
    return _SUBSECTION_FILTER_OVERRIDES.get(normalized_subsection_label, ())


def _build_override_wait(driver, timeout: int, cap: int) -> WebDriverWait:
    return WebDriverWait(
        driver,
        max(2, min(timeout, cap)),
        poll_frequency=0.2,
    )


def _open_filter_panel_if_needed(
    driver,
    control_label: str,
    wait: WebDriverWait,
) -> None:
    if _is_filter_panel_open(driver, control_label=control_label):
        logger.debug("Filter panel already open for control_label=%s", control_label)
        return

    togglers = [
        el
        for el in driver.find_elements(By.CSS_SELECTOR, ".snl-hui-settings-toggler")
        if el.is_displayed()
    ]

    if not togglers:
        logger.debug(
            "No filter panel toggler present for control_label=%s; assuming always-visible filter layout",
            control_label,
        )
        return

    wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, ".snl-hui-settings-toggler"))
    ).click()
    wait.until(lambda _driver: _is_filter_panel_open(_driver, control_label=control_label))
    logger.debug("Filter panel opened successfully for control_label=%s", control_label)


def _is_filter_panel_open(driver, control_label: str | None = None) -> bool:
    if _has_visible_apply_button(driver):
        return True

    visible_no_apply_boxes = [
        box
        for box in driver.find_elements(By.CSS_SELECTOR, ".snl-hui-filter-box-no-apply")
        if box.is_displayed()
    ]
    visible_filter_boxes = [
        box
        for box in driver.find_elements(By.CSS_SELECTOR, ".snl-hui-filter-box")
        if box.is_displayed()
    ]

    for box in visible_no_apply_boxes + visible_filter_boxes:
        if control_label is not None:
            selectors = _build_select_control_selectors(control_label)
            for selector in selectors:
                for el in box.find_elements(By.CSS_SELECTOR, selector):
                    if el.is_displayed():
                        return True
            for button in box.find_elements(
                By.CSS_SELECTOR,
                f'button.selectpickerbtn[aria-label="{control_label}"]',
            ):
                if button.is_displayed():
                    return True
            continue

        for el in box.find_elements(
            By.CSS_SELECTOR,
            "select[data-settingtitle], select[aria-label]",
        ):
            if el.is_displayed():
                return True
        for button in box.find_elements(
            By.CSS_SELECTOR,
            "button.selectpickerbtn[aria-label]",
        ):
            if button.is_displayed():
                return True

    return False


def _has_visible_apply_button(driver) -> bool:
    for button in driver.find_elements(By.CSS_SELECTOR, "button[id^='Apply_']"):
        if button.is_displayed():
            return True
    return False


def _find_select_control(driver, control_label: str, wait: WebDriverWait):
    selectors = _build_select_control_selectors(control_label)

    def _locate(_driver):
        for selector in selectors:
            matches = _driver.find_elements(By.CSS_SELECTOR, selector)
            if matches:
                return matches[0]
        return False

    select_el = wait.until(_locate)
    logger.debug("Matched select control for label=%s", control_label)
    return select_el


def _build_select_control_selectors(control_label: str) -> tuple[str, ...]:
    escaped_label = control_label.replace('"', '\\"')
    escaped_label_with_colon = f"{control_label}:".replace('"', '\\"')
    return (
        f'select[aria-label="{escaped_label}"]',
        f'select[data-settingtitle="{escaped_label}"]',
        f'select[data-settingtitle="{escaped_label_with_colon}"]',
        f'select[data-settingtitle^="{escaped_label}"]',
    )


def _set_select_value_via_js(driver, select_el, target_value: str) -> None:
    driver.execute_script(
        """
        const sel = arguments[0];
        const targetValue = arguments[1];
        sel.value = targetValue;
        sel.dispatchEvent(new Event('change', { bubbles: true }));
        sel.dispatchEvent(new Event('input', { bubbles: true }));
        """,
        select_el,
        target_value,
    )


def _ensure_native_select_value(
    select_el,
    target_text: str,
    target_value: str,
    wait: WebDriverWait,
) -> None:
    def _has_target_value(_driver):
        return select_el.get_attribute("value") == target_value

    try:
        wait.until(_has_target_value)
        return
    except TimeoutException:
        logger.debug(
            "Native select value did not update via JS; falling back to Select(text=%s)",
            target_text,
        )

    Select(select_el).select_by_visible_text(target_text)
    wait.until(_has_target_value)


def _find_visible_selectpicker_button(driver, select_el, control_label: str):
    parent = select_el.find_element(By.XPATH, "./parent::*")
    candidates = parent.find_elements(
        By.CSS_SELECTOR,
        f'button.selectpickerbtn[aria-label="{control_label}"]',
    )
    if not candidates:
        candidates = parent.find_elements(By.CSS_SELECTOR, "button.selectpickerbtn")
    if not candidates:
        candidates = driver.find_elements(
            By.CSS_SELECTOR,
            f'button.selectpickerbtn[aria-label="{control_label}"]',
        )
    for candidate in candidates:
        if candidate.is_displayed():
            return candidate
    return None


def _wait_for_selectpicker_button_text(
    selectpicker_button,
    target_text: str,
    wait: WebDriverWait,
    require_match: bool = False,
) -> bool:
    def _text_matches(_driver):
        return _normalize_visible_text(selectpicker_button.text) == _normalize_visible_text(
            target_text
        )

    try:
        wait.until(_text_matches)
        return True
    except TimeoutException:
        if require_match:
            raise
        return False


def _set_selectpicker_value_via_ui(
    selectpicker_button,
    target_text: str,
    wait: WebDriverWait,
) -> None:
    _open_selectpicker_dropdown(selectpicker_button, wait=wait)

    def _find_option(_driver):
        menu = _find_visible_selectpicker_menu_for_button(selectpicker_button)
        if menu is None:
            return False
        option_labels = menu.find_elements(
            By.XPATH,
            (
                ".//li[not(contains(@class,'disabled'))]"
                "//span[contains(@class,'text') and normalize-space()="
                f"{_xpath_literal(target_text)}]"
            ),
        )
        for option_label in option_labels:
            if option_label.is_displayed():
                return option_label
        return False

    option_label = wait.until(_find_option)
    option_label.find_element(By.XPATH, "./ancestor::li[1]").click()


def _find_visible_selectpicker_menu_for_button(selectpicker_button):
    parent = selectpicker_button.find_element(By.XPATH, "./parent::*")

    selectors = (
        "./div[contains(@class, 'dropdown-menu') and contains(@class, 'open')]",
        "./div[contains(@class, 'dropdown-menu')]",
    )

    for xpath in selectors:
        matches = parent.find_elements(By.XPATH, xpath)
        for menu in matches:
            if menu.is_displayed():
                return menu
    return None

def _open_selectpicker_dropdown(selectpicker_button, wait: WebDriverWait) -> None:
    selectpicker_button.click()
    wait.until(
        lambda _driver: _find_visible_selectpicker_menu_for_button(selectpicker_button) is not None
    )

def _click_selectpicker_option_by_text(
    selectpicker_button,
    target_text: str,
    wait: WebDriverWait,
) -> None:
    def _find_option(_driver):
        menu = _find_visible_selectpicker_menu_for_button(selectpicker_button)
        if menu is None:
            return False
        option_labels = menu.find_elements(
            By.XPATH,
            (
                ".//li[not(contains(@class,'disabled'))]"
                "//span[contains(@class,'text') and normalize-space()="
                f"{_xpath_literal(target_text)}]"
            ),
        )
        for option_label in option_labels:
            if option_label.is_displayed():
                return option_label.find_element(By.XPATH, "./ancestor::li[1]")
        return False

    wait.until(_find_option).click()


def _close_inner_period_picker_if_present(modal_root, wait: WebDriverWait) -> None:
    buttons = modal_root.find_elements(
        By.CSS_SELECTOR,
        ".period-picker-available-periods .snl-multi-select-close",
    )
    for button in buttons:
        if not button.is_displayed():
            continue
        button.click()
        wait.until(lambda _driver: not button.is_displayed())
        return


def _modal_count(driver) -> int:
    return sum(
        1
        for modal in driver.find_elements(By.CSS_SELECTOR, ".hui-custom-period-picker-modal")
        if modal.is_displayed()
    )


def _normalize_visible_text(value: str) -> str:
    return " ".join(value.split())


def _xpath_literal(value: str) -> str:
    if "'" not in value:
        return f"'{value}'"
    if '"' not in value:
        return f'"{value}"'
    parts = value.split("'")
    return "concat(" + ", \"'\", ".join(f"'{part}'" for part in parts) + ")"


def _is_restartable_browser_error(exc: Exception) -> bool:
    seen: set[int] = set()
    pending: list[BaseException] = [exc]

    while pending:
        current = pending.pop()
        if id(current) in seen:
            continue
        seen.add(id(current))

        text = f"{type(current).__module__}.{type(current).__name__}: {current}".lower()
        if isinstance(current, TimeoutException | WebDriverException):
            return True

        if any(
            marker in text
            for marker in (
                "read timeout",
                "readtimedout",
                "connection reset",
                "connection refused",
                "connection aborted",
                "remote host closed",
                "invalid session id",
                "disconnected",
                "target window already closed",
                "chrome not reachable",
            )
        ):
            return True

        cause = getattr(current, "__cause__", None)
        if cause is not None:
            pending.append(cause)

        context = getattr(current, "__context__", None)
        if context is not None:
            pending.append(context)

    return False
