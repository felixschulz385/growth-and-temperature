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


# ---------------------------------------------------------------------------
# Synthetic regression panel (src/viz/stats, models, regplot)
# ---------------------------------------------------------------------------

PANEL_N_UNITS = 40
PANEL_YEARS = list(range(2000, 2020))
PANEL_SLOPE = 2.0


def _build_panel() -> pd.DataFrame:
    """A tiny unbalanced panel with a known slope: `y = 2*x + unit_fe + year_fe
    + 0.5*treat + noise`. `ntl` is skewed-positive with ~10% missing, for
    log-transform / missingness tests."""
    rng = np.random.default_rng(12345)
    unit_fe = rng.normal(0.0, 1.0, PANEL_N_UNITS)
    year_fe = {y: rng.normal(0.0, 0.5) for y in PANEL_YEARS}

    rows = []
    for u in range(PANEL_N_UNITS):
        treat_start = int(rng.integers(PANEL_YEARS[5], PANEL_YEARS[-4]))
        for y in PANEL_YEARS:
            x = float(rng.normal(0.0, 1.0))
            treat = int(y >= treat_start)
            yy = PANEL_SLOPE * x + unit_fe[u] + year_fe[y] + 0.5 * treat + rng.normal(0.0, 0.3)
            rows.append(
                {
                    "unit": f"U{u:03d}",
                    "year": y,
                    "country": f"C{u % 5}",
                    "x": x,
                    "treat": treat,
                    "y": yy,
                    "ntl": float(rng.lognormal(0.0, 1.0)),
                }
            )
    df = pd.DataFrame(rows)
    df.loc[df.sample(frac=0.1, random_state=7).index, "ntl"] = np.nan
    return df


@pytest.fixture(scope="session")
def panel_frame() -> pd.DataFrame:
    return _build_panel()


@pytest.fixture()
def panel_parquet(tmp_path, panel_frame) -> str:
    """The panel as a single flat parquet file."""
    path = tmp_path / "panel.parquet"
    panel_frame.to_parquet(path, index=False, engine="pyarrow")
    return str(path)


@pytest.fixture()
def panel_hive(tmp_path, panel_frame) -> str:
    """The panel as a hive tree (`ix=*/iy=*/data.parquet`) -- one part per
    country, mapped to a fake `(ix, iy)`."""
    root = tmp_path / "panel_hive"
    for i, (country, part) in enumerate(panel_frame.groupby("country")):
        tile = root / f"ix={i}" / "iy=0"
        tile.mkdir(parents=True, exist_ok=True)
        part.drop(columns="country").to_parquet(tile / "data.parquet", index=False, engine="pyarrow")
    return str(root)


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
