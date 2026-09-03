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


def _source_ctes(
    src: _Source,
    g: GridFacts,
    *,
    use_land_cells: bool,
    year_range: Optional[Sequence[int]],
) -> str:
    """Return the comma-joined CTE fragment for one source: ``src_<name>``,
    optionally ``cutoffs_<name>`` + ``win_<name>`` (winsorize), then
    ``agg_<name>`` (columns ``pixel_id``, ``_repr_cell_<name>``, ``[year]``,
    then the source's prefixed value columns).

    ``use_land_cells`` semi-joins against the shared ``land_cells`` CTE (built
    once by :func:`_cte_block`), not a per-source re-scan of the mask.
    """
    winsorize = src.winsorize
    pid = _pixel_id_sql("cell_id::BIGINT", g, g.F, g.DR, g.DC)

    read = f"read_parquet('{src.glob}', union_by_name = true)"
    where = []
    if use_land_cells:
        where.append("cell_id IN (SELECT cell_id FROM land_cells)")
    if src.is_annual and year_range:
        where.append(f"year BETWEEN {int(year_range[0])} AND {int(year_range[1])}")
    where_sql = f" WHERE {' AND '.join(where)}" if where else ""

    val_cols = ",\n        ".join(
        f'{_nan_to_null(v)} AS "{out}"'
        for v, out in zip(src.variables, src.out_columns)
    )
    year_sel = "year,\n        " if src.is_annual else ""
    src_cte = (
        f"src_{src.name} AS (\n"
        f"    SELECT\n"
        f"        {pid} AS pixel_id,\n"
        f"        cell_id::BIGINT AS _repr_cell,\n"
        f"        {year_sel}{val_cols}\n"
        f"    FROM {read}{where_sql}\n"
        f")"
    )

    from_rel = f"src_{src.name}"
    if winsorize and winsorize > 0:
        lo, hi = winsorize, 1.0 - winsorize
        # Bounded aggregate (per year for annual sources), not an unbounded
        # window over the whole column -- one small row of cutoffs, joined back.
        cut_grp = "year" if src.is_annual else None
        cut_cols = ",\n        ".join(
            f'quantile_cont("{out}", {lo}) AS lo_{i}, quantile_cont("{out}", {hi}) AS hi_{i}'
            for i, out in enumerate(src.out_columns)
        )
        cut_sel = (f"year,\n        {cut_cols}" if cut_grp else cut_cols)
        cut_grp_sql = "\n    GROUP BY year" if cut_grp else ""
        clamp_cols = ",\n        ".join(
            f'least(greatest("{out}", c.lo_{i}), c.hi_{i}) AS "{out}"'
            for i, out in enumerate(src.out_columns)
        )
        keep = ("s.pixel_id, s._repr_cell, s.year" if src.is_annual else "s.pixel_id, s._repr_cell")
        join_on = "ON s.year = c.year" if cut_grp else "ON TRUE"
        src_cte += (
            f",\ncutoffs_{src.name} AS (\n"
            f"    SELECT {cut_sel}\n    FROM src_{src.name}{cut_grp_sql}\n"
            f"),\nwin_{src.name} AS (\n"
            f"    SELECT {keep},\n        {clamp_cols}\n"
            f"    FROM src_{src.name} s JOIN cutoffs_{src.name} c {join_on}\n"
            f")"
        )
        from_rel = f"win_{src.name}"

    agg_val_cols = ",\n        ".join(
        f'{_agg_expr(src.method_by_var[v], out)} AS "{out}"'
        for v, out in zip(src.variables, src.out_columns)
    )
    group_keys = "pixel_id, year" if src.is_annual else "pixel_id"
    not_all_null = " OR ".join(f'"{out}" IS NOT NULL' for out in src.out_columns)
    agg_cte = (
        f"agg_{src.name} AS (\n"
        f"    SELECT {group_keys},\n"
        f"        any_value(_repr_cell) AS _repr_cell_{src.name},\n"
        f"        {agg_val_cols}\n"
        f"    FROM {from_rel}\n"
        f"    GROUP BY {group_keys}\n"
        f"    HAVING {not_all_null}\n"
        f")"
    )
    return f"{src_cte},\n{agg_cte}"


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

@dataclass
class DuckDBConfig:
    threads: Optional[int] = None
    memory_limit: Optional[str] = None
    temp_dir: Optional[str] = None


def _connect(cfg: DuckDBConfig) -> duckdb.DuckDBPyConnection:
    con = duckdb.connect()
    con.execute("SET preserve_insertion_order = false")
    if cfg.threads:
        con.execute(f"SET threads TO {int(cfg.threads)}")
    if cfg.memory_limit:
        con.execute(f"SET memory_limit = '{cfg.memory_limit}'")
    if cfg.temp_dir:
        os.makedirs(cfg.temp_dir, exist_ok=True)
        con.execute(f"SET temp_directory = '{cfg.temp_dir}'")
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

    con = _connect(duckdb_cfg)
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


def _cte_block(sources: List[_Source], g: GridFacts, *, land_scan: Optional[str], year_range) -> str:
    parts = []
    if land_scan:
        parts.append(f"land_cells AS (\n    {land_scan}\n)")
    parts += [
        _source_ctes(s, g, use_land_cells=bool(land_scan), year_range=year_range)
        for s in sources
    ]
    return "WITH " + ",\n".join(parts)


def _run_create(
    con, sources, join_specs, g, shake_offset, output_path,
    *, land_scan, year_range, derived_specs, compression,
) -> None:
    any_annual = any(s.is_annual for s in sources)
    cte_sql = _cte_block(sources, g, land_scan=land_scan, year_range=year_range)

    merged = _merge_sql(sources)
    repr_cell = _repr_cell_coalesce(sources)
    merged_cte = (
        f"{cte_sql},\n"
        f"panel AS (\n    SELECT *, {repr_cell} AS _repr_cell\n    FROM {merged}\n)"
    )
    derived_col_sql = _derived_column_sql(derived_specs, g, shake_offset, "_repr_cell")

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
        f"COPY (\n{merged_cte}\n{select_sql}\n) TO '{output_path}' "
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

    cte_sql = _cte_block([source], g, land_scan=land_scan, year_range=year_range)
    key = "pixel_id, year" if source.is_annual else "pixel_id"
    t_cols = set(source.out_columns)
    missing = t_cols.difference(existing_cols)
    if missing:
        raise ValueError(
            f"update mode: source {source.name!r} columns {sorted(missing)} are not in the "
            f"existing panel at {output_path}; run `assemble create` to add a new source"
        )

    # Guard: an empty refreshed aggregate (broken/all-NaN re-prepare, or every
    # group dropped by HAVING) would NULL out this source's columns and the
    # atomic swap would commit the corrupted panel. Abort instead.
    n = con.execute(f"{cte_sql}\nSELECT count(*) FROM agg_{source.name}").fetchone()[0]
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
            f"COPY (\n{cte_sql}\n"
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
