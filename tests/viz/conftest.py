"""Synthetic PREPARE `cell_id` parquet fixture for src/viz tests.

Builds a tiny slice of a real-shaped PREPARE tree (`ix=<row>/iy=<col>/
part-<year>.parquet`, `cell_id`/`year`/one variable column -- matching
`SpatialProcessor.process_tile_region`'s actual writer) against the *real*
canonical EASE6933 grid, not a fake small one, since `src.viz.grid.load_matrix`
always builds its own `canonical_ease_geobox()` internally. A small lon/lat
bbox keeps the fixture data itself tiny while staying dimensionally honest.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.data.assemble.constants import DEFAULT_TILE_SIZE
from src.data.common.geobox.canonical import canonical_ease_geobox
from src.data.common.geobox.cell_id import cell_tile_indices
from src.viz.grid import bbox_to_row_col

TEST_BBOX = (10.0, 45.0, 10.5, 45.3)  # small region in northern Italy
TEST_YEARS = [2019, 2020]
VARIABLE = "test_var"


def _write_tile(root, ix: int, iy: int, rows, cols, years, rng) -> None:
    tile_dir = root / f"ix={ix}" / f"iy={iy}"
    tile_dir.mkdir(parents=True, exist_ok=True)
    geobox = canonical_ease_geobox()
    _, width = geobox.shape
    row_grid, col_grid = np.meshgrid(rows, cols, indexing="ij")
    cell_id = (row_grid.astype(np.int64) * width + col_grid.astype(np.int64)).ravel().astype(np.uint32)
    for year in years:
        df = pd.DataFrame(
            {
                "cell_id": cell_id,
                "year": np.full(cell_id.shape, year, dtype=np.int32),
                VARIABLE: rng.uniform(0, 100, size=cell_id.shape).astype(np.float32),
            }
        )
        df.to_parquet(tile_dir / f"part-{year}.parquet", index=False, engine="pyarrow")


@pytest.fixture(scope="session")
def canonical_geobox():
    return canonical_ease_geobox()


@pytest.fixture()
def prepare_tree(tmp_path, canonical_geobox):
    """A small PREPARE parquet tree: real pixels inside `TEST_BBOX`'s tile,
    plus one decoy tile on the opposite side of the grid whose values must
    never appear in a `TEST_BBOX`-scoped query (tile-pruning correctness)."""
    rng = np.random.default_rng(0)
    root = tmp_path / "prepare_output"

    row0, row1, col0, col1 = bbox_to_row_col(TEST_BBOX, geobox=canonical_geobox)
    rows = np.arange(row0, row1)
    cols = np.arange(col0, col1)
    ix, iy = cell_tile_indices(row0, col0, DEFAULT_TILE_SIZE)
    _write_tile(root, int(ix), int(iy), rows, cols, TEST_YEARS, rng)

    height, width = canonical_geobox.shape
    decoy_row, decoy_col = height - 1, width - 1
    decoy_ix, decoy_iy = cell_tile_indices(decoy_row, decoy_col, DEFAULT_TILE_SIZE)
    _write_tile(
        root,
        int(decoy_ix),
        int(decoy_iy),
        np.array([decoy_row]),
        np.array([decoy_col]),
        TEST_YEARS,
        rng,
    )

    return {
        "root": str(root),
        "bbox": TEST_BBOX,
        "years": TEST_YEARS,
        "variable": VARIABLE,
        "row_range": (row0, row1),
        "col_range": (col0, col1),
    }
