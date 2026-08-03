"""Mine-detail scraping dispatcher."""

from __future__ import annotations

import logging
from typing import Iterable, Type

import duckdb
from selenium.webdriver.chrome.webdriver import WebDriver

from ..blocks.base import BaseBlock
from ..blocks.subsection_xls import SubsectionXlsBlock
from ..config import DEFAULT_WAIT_SECONDS, PROFILE_URL_TEMPLATE
from ..storage.database import get_unscraped_ids, log_error, mark_detail_scraped
from ..utils.workflow_helpers import reset_profile_page_state

logger = logging.getLogger(__name__)

DEFAULT_BLOCKS: list[Type[BaseBlock]] = [SubsectionXlsBlock]


def register_block(block_cls: Type[BaseBlock]) -> Type[BaseBlock]:
    if block_cls not in DEFAULT_BLOCKS:
        DEFAULT_BLOCKS.append(block_cls)
    return block_cls


class DetailScraper:
    """Apply all registered blocks to every unscraped mine profile."""

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        blocks: Iterable[Type[BaseBlock]] | None = None,
        wait: int = DEFAULT_WAIT_SECONDS,
        skip_already_scraped: bool = True,
    ) -> None:
        self.conn = conn
        self.wait = wait
        self.skip_already_scraped = skip_already_scraped

        block_classes = list(blocks) if blocks is not None else DEFAULT_BLOCKS
        self._blocks: list[BaseBlock] = [cls(conn=conn, wait=wait) for cls in block_classes]
        logger.info(
            "DetailScraper initialised with %d block(s): %s",
            len(self._blocks),
            [b.name for b in self._blocks],
        )

    def run(self, driver: WebDriver, mine_ids: list[str] | None = None) -> None:
        ids = mine_ids if mine_ids is not None else self._pending_ids()
        if not ids:
            logger.info("No pending mine IDs to scrape; skipping detail pass.")
            return

        logger.info("Starting detail pass for %d mine ID(s)...", len(ids))
        for mine_id in ids:
            self._process_one(driver, mine_id)
        logger.info("Detail pass complete.")

    def _pending_ids(self) -> list[str]:
        if self.skip_already_scraped:
            return get_unscraped_ids(self.conn)
        rows = self.conn.execute("SELECT mine_id FROM mines ORDER BY id_scraped_at").fetchall()
        return [r[0] for r in rows]

    def _process_one(self, driver: WebDriver, mine_id: str) -> None:
        logger.debug("Processing mine_id=%s", mine_id)

        profile_url = PROFILE_URL_TEMPLATE.format(mine_id=mine_id)
        reset_profile_page_state(
            driver=driver,
            profile_url=profile_url,
            step_sleep_seconds=0.0,
            mine_id=mine_id,
        )

        all_ok = True
        for block in self._blocks:
            try:
                block.scrape(driver, mine_id)
            except Exception as exc:
                all_ok = False
                block.on_error(mine_id, exc)
                log_error(self.conn, mine_id, block.name, str(exc))

        if all_ok:
            mark_detail_scraped(self.conn, mine_id)
