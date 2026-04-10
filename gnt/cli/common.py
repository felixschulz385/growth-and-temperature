"""
Shared helpers for the GNT CLI: logging setup and common argument registration.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Optional


class _RasterioWarpNoiseFilter(logging.Filter):
    """Suppress known benign GDAL warp-option chatter from rasterio."""

    _NOISY_OPTIONS = ("DTYPE", "XSCALE", "YSCALE")

    def filter(self, record: logging.LogRecord) -> bool:
        if record.name != "rasterio._env":
            return True

        message = record.getMessage()
        if "CPLE_NotSupported" not in message or "does not support option" not in message:
            return True

        return not any(f"option {option}" in message for option in self._NOISY_OPTIONS)


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

    # Keep rasterio internals quiet even when the application runs in DEBUG.
    for name in ("rasterio", "rasterio.env", "rasterio._env", "rasterio._base"):
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


def add_config_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--config`` argument to *parser* (in-place)."""
    parser.add_argument(
        "--config",
        required=True,
        help="Path to unified configuration file (YAML or JSON)",
    )


def add_source_arg(parser: argparse.ArgumentParser) -> None:
    """Add the ``--source`` argument to *parser* (in-place)."""
    parser.add_argument(
        "--source",
        required=True,
        help="Data source name as defined in the configuration file",
    )
