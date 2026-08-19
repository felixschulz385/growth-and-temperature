"""src.cli.common.setup_logging -- noisy third-party logger suppression."""

import logging

from src.cli.common import setup_logging


def test_debug_mode_still_quiets_numcodecs_and_zarr():
    setup_logging(debug=True)
    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("numcodecs").level == logging.WARNING
    assert logging.getLogger("zarr").level == logging.WARNING


def test_default_info_level_also_quiets_numcodecs_and_zarr():
    setup_logging(level="INFO")
    assert logging.getLogger().level == logging.INFO
    assert logging.getLogger("numcodecs").level == logging.WARNING
    assert logging.getLogger("zarr").level == logging.WARNING


def test_rasterio_loggers_still_suppressed_to_error():
    setup_logging(debug=True)
    assert logging.getLogger("rasterio").level == logging.ERROR
