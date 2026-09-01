"""
Shared helpers for the GNT CLI: logging setup and common argument registration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional

_NOISY_GEO_LOGGERS = (
    "rasterio",
    "rasterio.env",
    "rasterio._env",
    "rasterio._warp",
    "rasterio._base",
)

#: Library loggers that emit DEBUG-level noise unrelated to this codebase's
#: own progress (e.g. numcodecs re-announcing its whole codec registry on
#: every import) -- kept quiet even under --debug so debug mode stays useful
#: for this codebase's own logs instead of being drowned out.
_NOISY_LIB_LOGGERS = (
    "numcodecs",
    "zarr",
    # Selenium/Chrome plumbing for the EOG fetch: per-request wire logs
    # ("selenium"), connection-pool churn ("urllib3"), and the legacy
    # webdriver-manager banner ("WDM", now unused but harmless to pin).
    "selenium",
    "urllib3",
    "WDM",
)


class _RasterioWarpNoiseFilter(logging.Filter):
    """Filter out known noisy Rasterio/GDAL debug messages."""

    _SUPPRESSED_SUBSTRINGS = (
        "CPLE_AppDefined",
        "GDAL",
        "Warp",
        "warp",
        "Nodata success",
    )

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(s in message for s in self._SUPPRESSED_SUBSTRINGS)

def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    debug: bool = False,
) -> None:
    """Configure root logger.

    Parameters
    ----------
    level:
        Standard logging level name (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    log_file:
        Unused — SLURM redirects stdout/stderr to files automatically.
        Accepted for API compatibility.
    debug:
        When ``True`` forces the level to DEBUG regardless of *level*.
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level!r}")

    if debug:
        numeric_level = logging.DEBUG

    root = logging.getLogger()
    root.setLevel(numeric_level)

    # Replace all existing handlers with a single stdout handler
    for handler in root.handlers[:]:
        root.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    rasterio_noise_filter = _RasterioWarpNoiseFilter()
    handler.addFilter(rasterio_noise_filter)
    root.addHandler(handler)

    # Keep rasterio/GDAL chatter quiet even in debug mode so package debug logs remain usable.
    for name in _NOISY_GEO_LOGGERS:
        logging.getLogger(name).setLevel(logging.ERROR)

    # Same idea for other noisy library loggers (see _NOISY_LIB_LOGGERS docstring).
    for name in _NOISY_LIB_LOGGERS:
        logging.getLogger(name).setLevel(logging.WARNING)

    rasterio_env_logger = logging.getLogger("rasterio._env")
    for log_filter in rasterio_env_logger.filters[:]:
        if isinstance(log_filter, _RasterioWarpNoiseFilter):
            rasterio_env_logger.removeFilter(log_filter)
    rasterio_env_logger.addFilter(rasterio_noise_filter)


def add_logging_args(parser: argparse.ArgumentParser) -> None:
    """Add ``--log-level`` and ``--debug`` to *parser* (in-place)."""
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        default="INFO",
        help="Set the logging level (default: INFO)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (overrides --log-level to DEBUG)",
    )


#: Repo-relative path to the pipeline's standard config file, resolved
#: against the current working directory (matches the `analysis run
#: --config` default convention of a relative `orchestration/configs/...`
#: path -- both assume invocation from the repo root, e.g. via `run.py`).
DEFAULT_PIPELINE_CONFIG = "orchestration/configs/data.yaml"


def add_config_arg(parser: argparse.ArgumentParser, *, default: Optional[str] = DEFAULT_PIPELINE_CONFIG) -> None:
    """Add the ``--config`` argument to *parser* (in-place).

    Defaults to `DEFAULT_PIPELINE_CONFIG` so `--config` can be omitted in the
    common case. Pass `default=None` to make it required instead.
    """
    parser.add_argument(
        "--config",
        default=default,
        required=default is None,
        help="Path to unified configuration file (YAML or JSON)"
        + (f" (default: {default})" if default else ""),
    )


def add_source_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--source`` argument to *parser* (in-place)."""
    parser.add_argument(
        "--source",
        required=True,
        help="Data source name as defined in the configuration file",
    )
