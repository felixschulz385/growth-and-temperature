"""DuckDB-only assembly engine.

Replaces the former Dask/xarray + ``odc.reproject`` tile loop. Every GRID-stage
pixel-grid source is already ``cell_id``-keyed wide parquet, co-registered on the
one canonical EASE 6933 grid (``src/data/common/raster/spatial.py::
process_tile_region`` writes ``ix=<row>/iy=<col>/part[-<year>].parquet`` with a
``cell_id = row * W + col`` global row-major index, an optional ``year`` column,
and one column per variable). So at assembly time:

* **merge** = an equi-join on ``(pixel_id[, year])``;
* **coarsen 1 km -> N km** = an exact integer block aggregation
  ``GROUP BY pixel_id`` where ``pixel_id`` is a pure integer function of
  ``cell_id`` (``_pixel_id_sql``), with one SQL aggregate per variable chosen
  from its resampling method (``constants.SQL_RESAMPLING_AGGREGATES``);
* **grid-shake** = an integer native-pixel origin shift folded into that
  ``pixel_id`` function (``shaken`` offset ``(DR, DC)``), matching
  ``src/data/common/geobox/cell_id.py::shaken_cell_id``'s ``(row + dr) // f``
  convention.

No neighbourhood / stencil operation lives in assembly (disc convolution / ring
means are a separate, still-unwired stage), so a pure ``GROUP BY`` is exact.

The output keeps the historical shape: tile-partitioned parquet under
``<output_path>/ix=<tile_row>/iy=<tile_col>/`` with the tile-packed ``pixel_id``
(``[ix:16 | iy:16 | local:32]``) ``src/analysis`` consumes, plus any configured
derived pixel-id columns and ``join_on`` sidecar columns.
"""

from __future__ import annotations

import glob
import logging
import os
import re
import shutil
import tempfile
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import duckdb
import pandas as pd
import pyarrow.parquet as pq

from src.data.assemble.constants import (
    DEFAULT_RESAMPLING_METHOD,
    DEFAULT_TILE_SIZE,
    EXCLUDED_VARIABLES,
    SQL_RESAMPLING_AGGREGATES,
)
from src.data.assemble.parquet_raster import (
    _detect_years,
    _partitioned_parquet_files,
    is_tiled_parquet_dataset,
)
from src.data.assemble.utils import (
    normalize_derived_pixel_id_specs,
    resolve_resampling,
)
from src.data.common.geobox.canonical import canonical_ease_geobox

logger = logging.getLogger(__name__)

# Columns that are never source variables.
_NON_VAR_COLUMNS = set(EXCLUDED_VARIABLES) | {"cell_id", "year"}


# ---------------------------------------------------------------------------
# Grid facts
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class GridFacts:
    """Everything the SQL builder needs about the run's grid, all integers."""

    W: int          # full canonical grid pixel width
    H: int          # full canonical grid pixel height
    TS: int         # output tile size, native cells (== PREPARE's tile grid)
    F: int          # coarsening factor (native cells per output cell); 1 == native
    DR: int         # grid-shake row origin shift, native cells, in [0, F)
    DC: int         # grid-shake col origin shift, native cells, in [0, F)

    @classmethod
    def build(
        cls,
        resolution_m: Optional[float],
        shake_offset: Tuple[float, float],
        tile_size: int = DEFAULT_TILE_SIZE,
    ) -> "GridFacts":
        gb = canonical_ease_geobox()
        H, W = int(gb.shape[0]), int(gb.shape[1])
        if resolution_m is None:
            F = 1
        else:
            if resolution_m % 1000 != 0:
                raise ValueError(
                    f"assemble resolution {resolution_m} m is not a whole number of "
                    f"native 1 km cells; DuckDB block aggregation needs an integer factor."
                )
            F = int(resolution_m // 1000)
        dx_frac, dy_frac = float(shake_offset[0]), float(shake_offset[1])
        DC = int(round(dx_frac * F)) % F if F > 1 else 0
        DR = int(round(dy_frac * F)) % F if F > 1 else 0
        return cls(W=W, H=H, TS=int(tile_size), F=F, DR=DR, DC=DC)


def _ceil_div_sql(num: str, den: int) -> str:
    return f"(({num} + {den - 1}) // {den})"


def _pixel_id_sql(cell_col: str, g: GridFacts, factor: int, dr: int, dc: int) -> str:
    """SQL expression: tile-packed ``pixel_id`` for a native ``cell_id`` on a
    grid coarsened by *factor* with origin shift ``(dr, dc)``.

    Mirrors ``utils.make_pixel_ids`` exactly for ``factor == 1`` (native), and
    ``make_pixel_ids(ix, iy, tile_geobox.zoom_to(resolution))`` for the coarse
    case (coarse indexing restarts at each native tile's origin, matching the
    per-tile ``zoom_to`` the old pipeline did). ``dr``/``dc`` shift where the
    coarse block boundaries fall, per ``cell_id.shaken_cell_id``'s
    ``(row + dr) // factor`` convention.
    """
    W, TS = g.W, g.TS
    nrow = f"({cell_col} // {W})"
    ncol = f"({cell_col} % {W})"
    ix = f"({nrow} // {TS})"
    iy = f"({ncol} // {TS})"
    nrl = f"({nrow} - {ix} * {TS})"      # native row within tile
    ncl = f"({ncol} - {iy} * {TS})"      # native col within tile
    tw = f"least({TS}, {W} - {iy} * {TS})"  # native tile width (ragged last col)
    if factor == 1:
        cw = tw
        crl = nrl
        ccl = ncl
    else:
        cw = _ceil_div_sql(f"({tw} + {dc})", factor)
        crl = f"(({nrl} + {dr}) // {factor})"
        ccl = f"(({ncl} + {dc}) // {factor})"
    local = f"({crl} * {cw} + {ccl})"
    return (
        f"((({ix})::UBIGINT << 48) | (({iy})::UBIGINT << 32) | ({local})::UBIGINT)"
    )


def _tile_ix_sql(pixel_id_col: str) -> str:
    return f"(({pixel_id_col} >> 48) & 65535)::INTEGER"


def _tile_iy_sql(pixel_id_col: str) -> str:
    return f"(({pixel_id_col} >> 32) & 65535)::INTEGER"


# ---------------------------------------------------------------------------
# Source model
# ---------------------------------------------------------------------------

@dataclass
class _Source:
    name: str
    path: str
    glob: str
    is_annual: bool
    variables: List[str]              # raw variable names, schema order
    out_columns: List[str]            # prefixed names, schema order
    prefix: str
    method_by_var: Dict[str, str]     # raw var -> resampling method
    fillna: Any                       # None | scalar | {var: scalar}
    index_cols: List[str]
    winsorize: Optional[float]        # symmetric quantile cutoff, or None


def _source_glob(path: str) -> str:
    return os.path.join(path, "ix=*", "iy=*", "part*.parquet")


def _detect_variables_ordered(part_file: str) -> List[str]:
    names = pq.ParquetFile(part_file).schema_arrow.names
    return [c for c in names if c not in _NON_VAR_COLUMNS]


def _fillna_for(name: str, cfg: Dict[str, Any]) -> Any:
    fillna = cfg.get("fillna")
    if fillna is None and name == "snl_mining":
        return 0
    return fillna


def _build_sources(
    datasets: Dict[str, Dict[str, Any]],
    *,
    datasource_filter: Optional[str],
) -> Tuple[List[_Source], Dict[str, Tuple[str, Dict[str, Any]]]]:
    """Split the config's datasets into reprojected pixel-grid *sources* and
    ``join_on`` sidecars. Returns ``(sources, join_specs)`` where ``join_specs``
    maps name -> ``(join_col, dataset_cfg)``."""
    raster: List[_Source] = []
    join_specs: Dict[str, Tuple[str, Dict[str, Any]]] = {}

    for name, cfg in datasets.items():
        if cfg.get("join_on"):
            join_specs[name] = (cfg["join_on"], cfg)
            continue
        if datasource_filter and name != datasource_filter:
            continue

        path = cfg["path"]
        if not is_tiled_parquet_dataset(path):
            raise ValueError(
                f"Source {name!r} at {path} is not a tiled-parquet GRID output; the "
                f"DuckDB assembly engine only reads `run_tiled_prepare` output on the "
                f"canonical EASE grid."
            )
        part_files = _partitioned_parquet_files(path)
        years = _detect_years(part_files)
        all_vars = _detect_variables_ordered(part_files[0])

        wanted = cfg.get("columns")
        variables = [v for v in all_vars if v in wanted] if wanted else all_vars
        if not variables:
            raise ValueError(f"Source {name!r}: no usable variables found at {path}")

        prefix = cfg.get("column_prefix") or ""
        method_by_var = resolve_resampling(cfg.get("resampling"), variables)

        raster.append(
            _Source(
                name=name,
                path=path,
                glob=_source_glob(path),
                is_annual=years is not None,
                variables=variables,
                out_columns=[f"{prefix}{v}" for v in variables],
                prefix=prefix,
                method_by_var=method_by_var,
                fillna=_fillna_for(name, cfg),
                index_cols=list(cfg.get("index_cols", ["pixel_id"])),
                winsorize=cfg.get("winsorize"),
            )
        )

    if datasource_filter:
        raster = [s for s in raster if s.name == datasource_filter]
        if not raster:
            raise ValueError(f"Datasource {datasource_filter!r} not found among reprojected sources")
    return raster, join_specs


# ---------------------------------------------------------------------------
# SQL fragments
# ---------------------------------------------------------------------------

def _nan_to_null(col: str) -> str:
    # parquet floats carry NaN as nodata; avg()/sum() would propagate it.
    return f'CASE WHEN "{col}" != "{col}" THEN NULL ELSE "{col}" END'


def _agg_expr(method: str, col: str) -> str:
    fn = SQL_RESAMPLING_AGGREGATES.get(method or DEFAULT_RESAMPLING_METHOD)
    if fn is None:
        raise ValueError(f"resampling method {method!r} has no SQL aggregate")
    return fn(f'"{col}"')


def _src_select(
    src: _Source,
    g: GridFacts,
    *,
    use_land_cells: bool,
    year_range: Optional[Sequence[int]],
    want_repr: bool,
) -> str:
    """The row-level scan for one source: ``pixel_id`` (+ ``_repr_cell`` only
    when a derived pixel id needs it), ``[year]``, and each variable with NaN
    nodata mapped to NULL. Land masking is a semi-join against the ``land_cells``
    temp table built once by the driver."""
    pid = _pixel_id_sql("cell_id::BIGINT", g, g.F, g.DR, g.DC)
    read = f"read_parquet('{src.glob}', union_by_name = true)"
    where = []
    if use_land_cells:
        where.append("cell_id IN (SELECT cell_id FROM land_cells)")
    if src.is_annual and year_range:
        where.append(f"year BETWEEN {int(year_range[0])} AND {int(year_range[1])}")
    where_sql = f"\n    WHERE {' AND '.join(where)}" if where else ""

    cols = [f"{pid} AS pixel_id"]
    if want_repr:
        cols.append("cell_id::BIGINT AS _repr_cell")
    if src.is_annual:
        cols.append("year")
    cols += [f'{_nan_to_null(v)} AS "{out}"' for v, out in zip(src.variables, src.out_columns)]
    return "SELECT\n        " + ",\n        ".join(cols) + f"\n    FROM {read}{where_sql}"


def _agg_select(
    src: _Source,
    g: GridFacts,
    *,
    use_land_cells: bool,
    year_range: Optional[Sequence[int]],
    want_repr: bool,
) -> str:
    """A standalone ``SELECT`` (possibly ``WITH``-prefixed) that block-aggregates
    one source to the coarse ``pixel_id`` grid -- fed to ``CREATE TEMP TABLE
    agg_<name> AS ...`` so each source's scan+group-by completes and frees its
    memory before the next, instead of ten aggregation pipelines living at once
    inside a single query.
    """
    group_keys = "pixel_id, year" if src.is_annual else "pixel_id"
    repr_sel = f"any_value(_repr_cell) AS _repr_cell_{src.name},\n        " if want_repr else ""
    agg_val_cols = ",\n        ".join(
        f'{_agg_expr(src.method_by_var[v], out)} AS "{out}"'
        for v, out in zip(src.variables, src.out_columns)
    )
    not_all_null = " OR ".join(f'"{out}" IS NOT NULL' for out in src.out_columns)
    src_sql = _src_select(src, g, use_land_cells=use_land_cells, year_range=year_range, want_repr=want_repr)

    if src.winsorize and src.winsorize > 0:
        lo, hi = src.winsorize, 1.0 - src.winsorize
        # Bounded per-(variable[, year]) cutoffs, joined back -- not an unbounded
        # window over the whole column.
        cut_cols = ", ".join(
            f'quantile_cont("{out}", {lo}) AS lo_{i}, quantile_cont("{out}", {hi}) AS hi_{i}'
            for i, out in enumerate(src.out_columns)
        )
        clamp = ",\n        ".join(
            f'least(greatest(w."{out}", c.lo_{i}), c.hi_{i}) AS "{out}"'
            for i, out in enumerate(src.out_columns)
        )
        keep = "w.pixel_id" + (", w._repr_cell" if want_repr else "") + (", w.year" if src.is_annual else "")
        if src.is_annual:
            cut_body = f"SELECT year, {cut_cols} FROM s GROUP BY year"
            cut_join = "c ON w.year = c.year"
        else:
            cut_body = f"SELECT {cut_cols} FROM s"
            cut_join = "c ON TRUE"
        from_rel = (
            f"(\n    WITH s AS (\n    {src_sql}\n    ), cutoffs AS ({cut_body})\n"
            f"    SELECT {keep},\n        {clamp}\n"
            f"    FROM s w JOIN cutoffs {cut_join}\n    )"
        )
    else:
        from_rel = f"(\n    {src_sql}\n    )"

    return (
        f"SELECT {group_keys},\n        {repr_sel}{agg_val_cols}\n"
        f"FROM {from_rel} agg_src\n"
        f"GROUP BY {group_keys}\n"
        f"HAVING {not_all_null}"
    )


def _merge_sql(sources: List[_Source]) -> str:
    """FULL OUTER JOIN every ``agg_<name>`` on the widest shared key: annual
    sources on ``(pixel_id, year)``, static sources on ``(pixel_id)``. Static
    values broadcast across years, matching the old index-col-intersection
    merge policy."""
    annual = [s for s in sources if s.is_annual]
    static = [s for s in sources if not s.is_annual]
    ordered = annual + static
    first = ordered[0]
    sql = f"agg_{first.name}"
    for s in ordered[1:]:
        key = "pixel_id, year" if (s.is_annual and annual) else "pixel_id"
        sql += f"\n    FULL JOIN agg_{s.name} USING ({key})"
    return sql


def _repr_cell_coalesce(sources: List[_Source]) -> str:
    return "coalesce(" + ", ".join(f"_repr_cell_{s.name}" for s in sources) + ")"


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

#: Project root (``src/data/assemble/sql_engine.py`` -> up 3). ``scratch_nobackup``
#: under it is the repo-wide gitignored scratch convention (also used by the
#: snl_mining scraper and the analysis runner), so the engine spills to a
#: *run-private* subdirectory of it and only ever removes that subdirectory.
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
DEFAULT_SPILL_ROOT = os.path.join(_PROJECT_ROOT, "scratch_nobackup")
DEFAULT_MAX_TEMP_SIZE = "1TB"

_SIZE_UNITS = {
    "": 1, "B": 1,
    "KB": 1000, "MB": 1000**2, "GB": 1000**3, "TB": 1000**4,
    "KIB": 1024, "MIB": 1024**2, "GIB": 1024**3, "TIB": 1024**4,
}


def _parse_size(text: str) -> int:
    m = re.fullmatch(r"\s*(\d+(?:\.\d+)?)\s*([A-Za-z]*)\s*", text)
    if not m:
        raise ValueError(f"unparseable size {text!r}")
    return int(float(m.group(1)) * _SIZE_UNITS[m.group(2).upper()])


@dataclass
class DuckDBConfig:
    threads: Optional[int] = None
    memory_limit: Optional[str] = None
    #: Explicit spill directory, used as-is and never removed. ``None`` -> a
    #: private ``assemble_*`` subdir of ``<project_root>/scratch_nobackup``,
    #: created before the run and removed after.
    temp_dir: Optional[str] = None
    #: Requested cap on total spill. Clamped down to 90% of the volume's free
    #: space so it is always a ceiling, never a raise of DuckDB's own default.
    max_temp_size: str = DEFAULT_MAX_TEMP_SIZE


def _connect(cfg: DuckDBConfig, spill_dir: str) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    # 119 native tiles -> keep every partition file open for a single-pass write.
    con.execute("SET partitioned_write_max_open_files = 512")
    if cfg.threads:
        con.execute(f"SET threads TO {int(cfg.threads)}")
    if cfg.memory_limit:
        con.execute(f"SET memory_limit = '{cfg.memory_limit}'")
    os.makedirs(spill_dir, exist_ok=True)
    con.execute(f"SET temp_directory = '{spill_dir}'")
    if cfg.max_temp_size:
        try:
            free = shutil.disk_usage(spill_dir).free
            cap = min(_parse_size(cfg.max_temp_size), int(free * 0.9))
            con.execute(f"SET max_temp_directory_size = '{cap}B'")
        except Exception as exc:  # noqa: BLE001 -- fall back to DuckDB's default cap
            logger.warning("could not set max_temp_directory_size (%s); using DuckDB default", exc)
    return con


def _land_scan_sql(land_mask_path: Optional[str]) -> Optional[str]:
    """The ``SELECT cell_id ... WHERE land_mask`` scan for the shared
    ``land_cells`` CTE, or ``None`` to skip land masking."""
    if not land_mask_path:
        return None
    if not is_tiled_parquet_dataset(land_mask_path):
        raise ValueError(
            f"land mask at {land_mask_path} is not tiled parquet; the DuckDB assembly "
            f"engine needs a `run_tiled_prepare` land mask on the canonical grid."
        )
    glob = _source_glob(land_mask_path)
    return f"SELECT cell_id FROM read_parquet('{glob}', union_by_name = true) WHERE land_mask"


def _derived_column_sql(
    specs: List[Tuple[str, float]],
    g: GridFacts,
    shake_offset: Tuple[float, float],
    repr_cell_expr: str,
) -> List[str]:
    out = []
    for col, res_m in specs:
        if res_m % 1000 != 0:
            raise ValueError(f"derived pixel-id resolution {res_m} m is not a whole km")
        df = int(res_m // 1000)
        if df < g.F or df % g.F != 0:
            raise ValueError(
                f"derived pixel-id resolution {res_m} m (factor {df}) must be a multiple "
                f"of the run's grid factor {g.F}"
            )
        ddr = int(round(float(shake_offset[1]) * df)) % df if df > 1 else 0
        ddc = int(round(float(shake_offset[0]) * df)) % df if df > 1 else 0
        expr = _pixel_id_sql(f"({repr_cell_expr})", g, df, ddr, ddc)
        out.append(f'{expr} AS "{col}"')
    return out


def _sql_literal(value: Any) -> str:
    if isinstance(value, bool):
        return "TRUE" if value else "FALSE"
    if isinstance(value, (int, float)):
        return repr(value)
    return "'" + str(value).replace("'", "''") + "'"


def _fill_col_sql(out: str, fillna: Any, raw: Optional[str] = None) -> str:
    """``"<out>"`` or ``coalesce("<out>", <fill>) AS "<out>"`` per *fillna*
    (``None`` | scalar | ``{name: value}`` keyed by prefixed or raw name)."""
    if isinstance(fillna, dict):
        fill = fillna.get(out)
        if fill is None and raw is not None:
            fill = fillna.get(raw)
    elif fillna is not None:
        fill = fillna
    else:
        fill = None
    if fill is None:
        return f'"{out}"'
    return f'coalesce("{out}", {_sql_literal(fill)}) AS "{out}"'


def _final_select_sql(
    sources: List[_Source],
    join_cols: Dict[str, Tuple[Any, List[str]]],
    derived_col_sql: List[str],
    *,
    any_annual: bool,
    from_rel: str,
) -> str:
    """*join_cols* maps join dataset name -> ``(fillna, [prefixed value cols])``."""
    cols: List[str] = ["pixel_id"]
    if any_annual:
        cols.append("year")
    cols.extend(derived_col_sql)
    for s in sources:
        for raw, out in zip(s.variables, s.out_columns):
            cols.append(_fill_col_sql(out, s.fillna, raw=raw))
    for _jname, (fillna, value_cols) in join_cols.items():
        for c in value_cols:
            cols.append(_fill_col_sql(c, fillna))
    cols.append(f'{_tile_ix_sql("pixel_id")} AS ix')
    cols.append(f'{_tile_iy_sql("pixel_id")} AS iy')
    return "SELECT\n    " + ",\n    ".join(cols) + f"\nFROM {from_rel}"


def _register_join_tables(
    con: duckdb.DuckDBPyConnection,
    join_specs: Dict[str, Tuple[str, Dict[str, Any]]],
) -> Dict[str, Tuple[Any, List[str]]]:
    """Register each ``join_on`` sidecar as a DuckDB view ``join_<name>`` and
    return ``{name: (fillna, [prefixed value columns])}``."""
    out: Dict[str, Tuple[Any, List[str]]] = {}
    for jname, (jcol, jcfg) in join_specs.items():
        df = pd.read_parquet(jcfg["path"], columns=jcfg.get("columns"))
        if jcol not in df.columns:
            raise ValueError(f"join_on dataset {jname!r}: no {jcol!r} column at {jcfg['path']}")
        prefix = jcfg.get("column_prefix") or ""
        if prefix:
            df = df.rename(columns={c: f"{prefix}{c}" for c in df.columns if c != jcol})
        if df[jcol].duplicated().any():
            logger.warning("join_on %r: duplicate %r values, keeping first", jname, jcol)
            df = df.drop_duplicates(subset=[jcol], keep="first")
        con.register(f"join_{jname}", df)
        out[jname] = (jcfg.get("fillna"), [c for c in df.columns if c != jcol])
    return out


def _apply_joins_sql(base_rel: str, join_specs: Dict[str, Tuple[str, Dict[str, Any]]]) -> str:
    sql = base_rel
    for jname, (jcol, jcfg) in join_specs.items():
        sql += f'\n    LEFT JOIN join_{jname} USING ("{jcol}")'
    return sql


# ---------------------------------------------------------------------------
# Public entrypoints
# ---------------------------------------------------------------------------

def run_sql_assembly(
    datasets: Dict[str, Dict[str, Any]],
    output_path: str,
    *,
    resolution_m: Optional[float],
    shake_offset: Tuple[float, float],
    land_mask_path: Optional[str],
    compression: str,
    tile_size: int,
    year_range: Optional[Sequence[int]],
    derived_pixel_ids: Optional[Dict[str, Any]],
    mode: str,
    datasource: Optional[str],
    duckdb_cfg: DuckDBConfig,
) -> None:
    """Build one ``grid=<label>/shake=<label>`` assembled table with DuckDB.

    ``land_mask_path`` is ``None`` to skip land masking. Per-source
    ``winsorize`` is read from each dataset config by :func:`_build_sources`.
    """
    g = GridFacts.build(resolution_m, shake_offset, tile_size)
    logger.info(
        "SQL assembly: grid factor F=%d, shake origin (DR=%d, DC=%d), tile_size=%d, W=%d H=%d",
        g.F, g.DR, g.DC, g.TS, g.W, g.H,
    )

    sources, join_specs = _build_sources(datasets, datasource_filter=(datasource if mode == "update" else None))
    if not sources:
        raise ValueError("No reprojected sources to assemble")

    land_scan = _land_scan_sql(land_mask_path)
    derived_specs = normalize_derived_pixel_id_specs(derived_pixel_ids)

    if duckdb_cfg.temp_dir:
        # Caller owns it: use as-is, leave it in place.
        spill_dir = duckdb_cfg.temp_dir
        owns_spill_dir = False
    else:
        # A private subdir of the shared scratch root -- unique per run so
        # concurrent assembly jobs never collide, and cleanup can only ever
        # touch this run's own directory (never the shared root or a sibling).
        os.makedirs(DEFAULT_SPILL_ROOT, exist_ok=True)
        spill_dir = tempfile.mkdtemp(prefix="assemble_", dir=DEFAULT_SPILL_ROOT)
        owns_spill_dir = True
    logger.info("DuckDB spill: %s (requested max %s)", spill_dir, duckdb_cfg.max_temp_size)

    con = _connect(duckdb_cfg, spill_dir)
    try:
        if mode == "update":
            _run_update(con, sources[0], g, output_path,
                        land_scan=land_scan, year_range=year_range,
                        compression=compression)
        else:
            _check_no_column_collisions(sources, derived_specs)
            _run_create(con, sources, join_specs, g, shake_offset, output_path,
                        land_scan=land_scan, year_range=year_range,
                        derived_specs=derived_specs, compression=compression)
    finally:
        con.close()
        if owns_spill_dir:
            shutil.rmtree(spill_dir, ignore_errors=True)
            # tidy the scratch root if we created it and nothing else uses it
            try:
                os.rmdir(DEFAULT_SPILL_ROOT)
            except OSError:
                pass


def _check_no_column_collisions(sources: List[_Source], derived_specs: List[Tuple[str, float]]) -> None:
    """Every emitted column name must be unique -- the FULL JOIN merge has no
    ``_x``/``_y`` suffixing and a duplicate name breaks the parquet COPY."""
    seen: Dict[str, str] = {"pixel_id": "index", "year": "index", "ix": "partition", "iy": "partition"}
    for col, _ in derived_specs:
        if col in seen:
            raise ValueError(f"assembly column {col!r} (derived pixel id) collides with {seen[col]}")
        seen[col] = "derived pixel id"
    for s in sources:
        for col in s.out_columns:
            if col in seen:
                raise ValueError(
                    f"assembly column {col!r} from source {s.name!r} collides with {seen[col]!r}; "
                    f"set a distinct column_prefix"
                )
            seen[col] = s.name


def _materialize_aggregates(
    con, sources, g, *, land_scan, year_range, want_repr,
) -> None:
    """``CREATE TEMP TABLE land_cells`` (once) + ``agg_<name>`` per source,
    one at a time. Each source's scan + group-by completes and releases its
    memory before the next, so peak RAM is one aggregation, not ten -- and the
    temp tables spill to the configured directory. This is what keeps a
    full-grid create inside its memory limit."""
    if land_scan:
        con.execute(f"CREATE TEMP TABLE land_cells AS {land_scan}")
        logger.info("land mask: %d land cells", con.execute("SELECT count(*) FROM land_cells").fetchone()[0])
    for s in sources:
        sel = _agg_select(
            s, g, use_land_cells=bool(land_scan), year_range=year_range, want_repr=want_repr
        )
        con.execute(f"CREATE TEMP TABLE agg_{s.name} AS\n{sel}")
        rows = con.execute(f"SELECT count(*) FROM agg_{s.name}").fetchone()[0]
        logger.info("aggregated %s -> %d coarse rows", s.name, rows)


def _run_create(
    con, sources, join_specs, g, shake_offset, output_path,
    *, land_scan, year_range, derived_specs, compression,
) -> None:
    any_annual = any(s.is_annual for s in sources)
    want_repr = bool(derived_specs)

    _materialize_aggregates(
        con, sources, g, land_scan=land_scan, year_range=year_range, want_repr=want_repr
    )

    # Materialize the merged panel too, so the FULL OUTER JOIN of the coarse
    # aggregates completes and spills before the partitioned COPY sink starts --
    # the two never contend for memory at once.
    merged = _merge_sql(sources)  # FULL OUTER JOIN over the agg_<name> temp tables
    repr_sel = f", {_repr_cell_coalesce(sources)} AS _repr_cell" if want_repr else ""
    con.execute(f"CREATE TEMP TABLE panel AS SELECT *{repr_sel} FROM {merged}")
    logger.info(
        "merged panel -> %d rows", con.execute("SELECT count(*) FROM panel").fetchone()[0]
    )
    derived_col_sql = (
        _derived_column_sql(derived_specs, g, shake_offset, "_repr_cell") if want_repr else []
    )

    join_cols = _register_join_tables(con, join_specs)
    from_rel = _apply_joins_sql("panel", join_specs)
    select_sql = _final_select_sql(
        sources, join_cols, derived_col_sql, any_annual=any_annual, from_rel=from_rel
    )

    # Clear any prior tile partitions so a tile that is empty this run does not
    # keep a stale file (DuckDB's OVERWRITE only truncates partitions it rewrites).
    for stale in glob.glob(os.path.join(output_path, "ix=*")):
        shutil.rmtree(stale, ignore_errors=True)
    os.makedirs(output_path, exist_ok=True)
    copy_sql = (
        f"COPY (\n{select_sql}\n) TO '{output_path}' "
        f"(FORMAT parquet, PARTITION_BY (ix, iy), OVERWRITE_OR_IGNORE true, "
        f"COMPRESSION '{compression}', FILENAME_PATTERN 'data_{{i}}')"
    )
    logger.info("Running assembly COPY -> %s", output_path)
    logger.debug("assembly SQL:\n%s", copy_sql)
    con.execute(copy_sql)
    logger.info("Assembly write complete: %s", output_path)


def _run_update(
    con, source, g, output_path,
    *, land_scan, year_range, compression,
) -> None:
    """Refresh one raster *source*'s columns in an existing panel. Only that
    source's columns change; ``join_on`` sidecar columns are carried through
    unchanged (a stale sidecar needs a full ``create``, not an update)."""
    if not os.path.isdir(output_path):
        raise ValueError(f"update mode: assembled table {output_path} does not exist")

    panel_parts = glob.glob(os.path.join(output_path, "ix=*", "iy=*", "*.parquet"))
    if not panel_parts:
        raise ValueError(f"update mode: {output_path} has no tile parquet files")
    # hive partition columns (ix/iy) are in the path, not the file schema
    existing_cols = [c for c in pq.ParquetFile(panel_parts[0]).schema_arrow.names if c not in ("ix", "iy")]

    key = "pixel_id, year" if source.is_annual else "pixel_id"
    t_cols = set(source.out_columns)
    missing = t_cols.difference(existing_cols)
    if missing:
        raise ValueError(
            f"update mode: source {source.name!r} columns {sorted(missing)} are not in the "
            f"existing panel at {output_path}; run `assemble create` to add a new source"
        )

    _materialize_aggregates(
        con, [source], g, land_scan=land_scan, year_range=year_range, want_repr=False
    )

    # Guard: an empty refreshed aggregate (broken/all-NaN re-prepare, or every
    # group dropped by HAVING) would NULL out this source's columns and the
    # atomic swap would commit the corrupted panel. Abort instead.
    n = con.execute(f"SELECT count(*) FROM agg_{source.name}").fetchone()[0]
    if n == 0:
        raise ValueError(
            f"update mode: refreshed aggregate for {source.name!r} is empty -- refusing to "
            f"overwrite the panel's {source.name!r} columns with NULL. Check the source's "
            f"GRID output."
        )

    # Preserve the panel's existing column order: take each column from `e`,
    # except this source's columns which come from the refreshed `t`.
    proj = ", ".join(
        (f't."{c}" AS "{c}"' if c in t_cols else f'e."{c}"') for c in existing_cols
    )
    existing_glob = os.path.join(output_path, "ix=*", "iy=*", "*.parquet")

    tmp_dir = tempfile.mkdtemp(prefix=".assemble_update_", dir=os.path.dirname(output_path.rstrip("/")))
    try:
        copy_sql = (
            f"COPY (\n"
            f"SELECT {proj}, e.ix, e.iy\n"
            f"FROM read_parquet('{existing_glob}', hive_partitioning = true) e\n"
            f"LEFT JOIN agg_{source.name} t USING ({key})\n"
            f") TO '{tmp_dir}' (FORMAT parquet, PARTITION_BY (ix, iy), OVERWRITE_OR_IGNORE true, "
            f"COMPRESSION '{compression}', FILENAME_PATTERN 'data_{{i}}')"
        )
        logger.info("Running update COPY for %r -> staged %s", source.name, tmp_dir)
        con.execute(copy_sql)
        # carry non-tile files (e.g. _metadata.yaml) across the swap
        for name in os.listdir(output_path):
            if not name.startswith("ix="):
                shutil.move(os.path.join(output_path, name), os.path.join(tmp_dir, name))
        shutil.rmtree(output_path)
        os.rename(tmp_dir, output_path)
        logger.info("Update complete: %s", output_path)
    except BaseException:
        shutil.rmtree(tmp_dir, ignore_errors=True)
        raise
