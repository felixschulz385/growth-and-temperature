"""
Abstract base class for mine-detail information blocks.

Each block is responsible for:
1. Navigating to (or operating on) the mine-profile page.
2. Extracting one logical section of data (e.g. "Overview", "Production").
3. Writing that data to the DuckDB connection.

Implementation guide
--------------------
Subclass :class:`BaseBlock`, implement :meth:`scrape`, and register the
class via :func:`~snf_mining.scrapers.detail.register_block` or list it in
:data:`~snf_mining.scrapers.detail.DEFAULT_BLOCKS`.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import logging

import duckdb
from selenium.webdriver.chrome.webdriver import WebDriver

logger = logging.getLogger(__name__)


class BaseBlock(ABC):
    """Interface every information block must implement.

    Parameters
    ----------
    conn : duckdb.DuckDBPyConnection
        Shared database connection.  Blocks must create their own tables if
        they do not yet exist (typically in :meth:`ensure_schema`).
    wait : int
        Default implicit wait timeout in seconds.
    """

    #: Human-readable name used in log messages and error records.
    name: str = "unnamed_block"

    def __init__(
        self,
        conn: duckdb.DuckDBPyConnection,
        wait: int = 10,
    ) -> None:
        self.conn = conn
        self.wait = wait
        self.ensure_schema()

    # ------------------------------------------------------------------
    # Mandatory interface
    # ------------------------------------------------------------------

    @abstractmethod
    def ensure_schema(self) -> None:
        """Create block-specific tables / columns if not yet present.

        Called once from ``__init__``.  Must be idempotent.
        """

    @abstractmethod
    def scrape(self, driver: WebDriver, mine_id: str) -> None:
        """Extract data for *mine_id* from the currently loaded page and
        persist it via ``self.conn``.

        The caller guarantees that ``driver`` is authenticated and that the
        profile page for *mine_id* is already loaded when this method is
        called.

        Parameters
        ----------
        driver : WebDriver
            Active, authenticated Chrome driver with the profile page open.
        mine_id : str
            The Capital IQ mine ID being processed.
        """

    # ------------------------------------------------------------------
    # Optional hook
    # ------------------------------------------------------------------

    def on_error(self, mine_id: str, exc: Exception) -> None:
        """Called by the dispatcher when :meth:`scrape` raises.

        The default implementation just logs a warning; override to add
        custom recovery logic.
        """
        logger.warning(
            "[%s] block failed for mine_id=%s: %s", self.name, mine_id, exc
        )
