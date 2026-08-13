"""PLAD (Political Leaders and Development): fetch + grid, no separate prepare.

docs/design/09-integrated-pipeline.md §5. Merges
`src/data/download/sources/harvard.py::HarvardDataSource` (Harvard Dataverse
API fetch -- registered under aliases `harvard_plad`/`harvard`, never
actually wired to the old preprocessor, which built its own `create_source()`
`None` special-case instead) and
`src/data/preprocess/sources/plad.py::PLADPreprocessor` (`stage="spatial"`
-> GRID). **No PREPARE step**: PLAD's raw `.dta` table already carries GADM's
native `gid_1`/`gid_2` string codes directly, so there's no vector-boundary
pre-step to build.

**No longer rasterized.** Regional favoritism (`reg_fav`) is constant across
every pixel of the favored admin unit for a given year -- it varies only by
`GID_1`/`GID_2` and year, never by pixel location -- so GRID now writes a
tiny `(GID_N, year)`-keyed parquet table of favored units instead of a full
pixel-grid zarr, and no longer needs GADM's polygon geometries at all (only
`GID_N_code_mapping.json`, to translate PLAD's native `gid_1`/`gid_2` string
codes into the same integer ids gadm's own per-pixel `GID_N` grid uses).
Assembly merges it directly onto rows via
`src.data.assemble.processors.TileProcessor`'s `join_on` mechanism, keyed on
`GID_N` (fillna=False for admin units absent from the favored-unit table, via
the assemble config, not baked in here).

**Quirk preserved, not "fixed"**: `get_hpc_output_path` hardcodes the string
`"plad"` as the output path prefix, never `self.data_path` -- so even a
configured `data_path` override would not change where output lands. Modeled
here via an `output_root()` override, mirroring the identical pattern already
used for GLASS's `path_prefix`.

`REQUIRES` on gadm's **GRID** (not PREPARE) -- changed from PREPARE now that
rasterization (and the polygon geometries it needed) is gone; only the
integer-id mapping sidecar (`GID_N_code_mapping.json`, produced by gadm's
GRID step) is still needed. Scoped to this source's own GRID step (only
`_plan_grid()` touches gadm), so FETCH runs unblocked before gadm exists.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
from src.data.sources import verify

if TYPE_CHECKING:
    from src.data.common.ledger.store import ArtifactRow

logger = logging.getLogger(__name__)

DEFAULT_DOI = "doi:10.7910/DVN/YUS575"


class PlaDSource(DataSource):
    """Political Leaders and Development: regional-favoritism (GID_N, year) table."""

    ID = "plad"
    ALIASES = ("harvard_plad", "harvard")
    STEPS = (PipelineStep.FETCH, PipelineStep.GRID)
    REQUIRES = ((PipelineStep.GRID, "gadm", PipelineStep.GRID),)

    DATA_SOURCE_NAME = "harvard"
    has_entrypoints = False
    OUTPUT_PREFIX = "plad"  # hardcoded in the old code, see module docstring

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="plad")
        if cfg.year_range is None:
            cfg = dataclasses.replace(cfg, year_range=(1980, 2022))  # old PLADPreprocessor default
        super().__init__(ctx, cfg)

        self.doi = cfg.raw.get("doi") or cfg.raw.get("base_url") or DEFAULT_DOI
        self.base_url = cfg.raw.get("base_url") or f"https://dataverse.harvard.edu/dataset.xhtml?persistentId={self.doi}"
        self.file_extensions = cfg.raw.get("file_extensions") or [".csv", ".nc", ".tif", ".zip"]

        self.admin_level = cfg.raw.get("admin_level", 1)
        if self.admin_level not in (1, 2):
            raise ValueError("admin_level must be 1 or 2")

        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="plad_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

    @property
    def _gid_column(self) -> str:
        return f"GID_{self.admin_level}"

    def output_root(self, step: PipelineStep, *, namespace: str | None = None) -> str:
        if step is PipelineStep.GRID:
            return layout.output_root(
                self.ctx.data_root,
                self.OUTPUT_PREFIX,
                step,
                namespace=namespace,
                grid_id=self.ctx.grid_id,
                layout=self.ctx.layout,
            )
        return super().output_root(step, namespace=namespace)

    # ------------------------------------------------------------------
    # RemoteFileCatalog contract (ports HarvardDataSource verbatim)
    # ------------------------------------------------------------------

    def list_remote_files(self, entrypoint: Optional[dict] = None) -> List[tuple]:
        import requests

        api_url = f"https://dataverse.harvard.edu/api/datasets/:persistentId?persistentId={self.doi}"
        try:
            response = requests.get(api_url)
            response.raise_for_status()
            files = response.json()["data"]["latestVersion"]["files"]
            result = []
            for file in files:
                label = file["label"]
                if not self.file_extensions or any(label.endswith(ext) for ext in self.file_extensions):
                    relative_path = file["dataFile"].get("originalFileName", label)
                    file_id = file["dataFile"]["id"]
                    result.append((relative_path, f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"))
            return result
        except Exception:
            logger.exception("Error listing files from Harvard Dataverse")
            return []

    def local_path(self, relative_path: str) -> str:
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)

    # get_file_hash: inherited from DataSource (src/data/sources/base.py).

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        return None

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        return []

    def download(self, file_url: str, output_path: str, session: Any = None) -> None:
        import time

        import requests

        s = session or requests.Session()
        time.sleep(0.5)
        r = s.get(file_url, stream=True)
        r.raise_for_status()
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    async def download_async(self, file_url: str, output_path: str, session: Any = None) -> None:
        import asyncio

        await asyncio.sleep(0.5)
        await asyncio.get_event_loop().run_in_executor(None, self.download, file_url, output_path, None)

    # ------------------------------------------------------------------
    # plan()/execute() dispatch
    # ------------------------------------------------------------------

    def _plan_fetch(self) -> List[StepTarget]:
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.FETCH, key="all",
                output_path=self.output_root(PipelineStep.FETCH), completion=Completion.NEVER,
            )
        ]

    def _plan(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        if step is PipelineStep.FETCH:
            return self._plan_fetch()
        if step is PipelineStep.GRID:
            return self._plan_grid()
        raise AssertionError(f"unreachable: {step}")

    def _discover(self, step: PipelineStep, selection: TargetSelection) -> List[StepTarget]:
        """Ground truth for `data reconcile` -- see gadm.py's identical
        `_discover()` for the full rationale. PLAD's targets are singletons
        with no year/key selection to apply; `selection` is accepted for
        interface symmetry only."""
        if step is PipelineStep.FETCH:
            return self._plan_fetch()
        if step is PipelineStep.GRID:
            return self._discover_grid()
        raise AssertionError(f"unreachable: {step}")

    def _plan_grid(self) -> List[StepTarget]:
        """Ledger-backed fast path. `inputs` (a single deterministic mapping-
        file path) is persisted directly in `meta` at discovery time, same
        pattern as gadm's PREPARE `raw_file` -- see gadm.py's `_plan_prepare()`."""

        def build_target(row: "ArtifactRow", _ledger: Any) -> Optional[StepTarget]:
            mapping_file = row.meta.get("mapping_file")
            if mapping_file is None or row.local_path is None:
                return None
            return StepTarget(
                source_id=self.ID, step=PipelineStep.GRID, key=row.unit_id,
                output_path=row.local_path, inputs=(mapping_file,),
                completion=Completion.PATH_EXISTS, meta=row.meta,
            )

        targets = self._plan_from_ledger(PipelineStep.GRID, TargetSelection(), build_target)
        if targets is not None:
            return targets
        logger.warning(
            "No ledger for source='%s' step='grid' -- falling back to live discovery; "
            "run `data reconcile --source %s --step grid` for faster planning.",
            self.ID, self.ID,
        )
        return self._discover_grid()

    def _discover_grid(self) -> List[StepTarget]:
        mapping_file = self._gid_mapping_file()
        if not os.path.exists(mapping_file):
            return []
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.GRID, key=f"adm{self.admin_level}",
                output_path=os.path.join(
                    self.output_root(PipelineStep.GRID),
                    f"plad_adm{self.admin_level}_reg_fav.parquet",
                ),
                inputs=(mapping_file,),
                completion=Completion.PATH_EXISTS,
                meta={
                    "admin_level": self.admin_level,
                    "year_range": self.cfg.year_range,
                    "mapping_file": mapping_file,
                    **verify.verification_meta(
                        self.cfg.raw, expected_vars=(self._gid_column, "year", "reg_fav")
                    ),
                },
            )
        ]

    def _execute(self, target: StepTarget) -> bool:
        if target.step is PipelineStep.FETCH:
            return self._execute_fetch(target)
        if target.step is PipelineStep.GRID:
            return self._execute_grid(target)
        raise AssertionError(f"unreachable: {target.step}")

    def _execute_fetch(self, target: StepTarget) -> bool:
        # FETCH is local-disk only now -- no HPC target required. `data
        # transfer` (separate, manual or auto per source config) is the only
        # thing that pushes to HPC.
        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

    # -- GRID: build the (GID_N, year) favored-unit table --------------------

    def _gid_mapping_file(self) -> str:
        from src.data.sources.misc.gadm import gid_mapping_path

        return gid_mapping_path(self.ctx.data_root, self.ctx.grid_id, self.ctx.layout, self._gid_column)

    def _resolve_plad_data_file(self) -> Optional[str]:
        from src.data.common.ledger.paths import ledger_path
        from src.data.common.ledger.store import SourceLedger

        local_ledger_path = ledger_path(self.ctx.local_index_dir, self.data_path)
        if not local_ledger_path or not os.path.exists(local_ledger_path):
            return None
        with SourceLedger.open(local_ledger_path, data_path=self.data_path, read_only=True) as ledger:
            relative_paths = ledger.completed_fetch_files()
        for rel_path in relative_paths:
            filename = os.path.basename(rel_path).lower()
            if "plad" in filename and filename.endswith(".dta"):
                raw = rel_path if os.path.isabs(rel_path) else os.path.join(self.output_root(PipelineStep.FETCH), rel_path)
                return raw
        return None

    def _build_reg_fav_table(self, mapping_file: str):
        """One row per (favored GID_N, year): PLAD's raw table already carries
        native `gid_1`/`gid_2` GADM string codes directly, so building this is
        just an expand-by-year-range + translate-to-gadm's-integer-id, no
        GADM polygon geometry needed."""
        import pandas as pd

        plad_data_path = self._resolve_plad_data_file()
        if not plad_data_path:
            raise ValueError("PLAD data file not found")

        plad = pd.read_table(plad_data_path)
        raw_gid_col = f"gid_{self.admin_level}"
        if raw_gid_col not in plad.columns:
            raise ValueError(f"PLAD data file missing expected column '{raw_gid_col}'")

        with open(mapping_file) as f:
            code_to_id: Dict[str, int] = json.load(f)

        start_year, end_year = self.cfg.year_range
        rows = []
        for _, row in plad.iterrows():
            code = row[raw_gid_col]
            if pd.isna(code):
                continue
            row_start = max(int(row["startyear"]), start_year)
            row_end = min(int(row["endyear"]), end_year)
            for year in range(row_start, row_end + 1):
                rows.append((code, year))

        if not rows:
            return pd.DataFrame(columns=[self._gid_column, "year", "reg_fav"])

        panel = pd.DataFrame(rows, columns=["gid_code", "year"]).drop_duplicates()
        panel[self._gid_column] = panel["gid_code"].map(lambda c: code_to_id.get(c, 0))
        panel = panel[panel[self._gid_column] != 0]
        panel["reg_fav"] = True
        return panel[[self._gid_column, "year", "reg_fav"]].drop_duplicates(
            subset=[self._gid_column, "year"]
        )

    def _execute_grid(self, target: StepTarget) -> bool:
        from src.data.sources.steps import is_complete

        if not self.cfg.override and is_complete(target):
            logger.info("Skipping PLAD reg_fav table, output already exists: %s", target.output_path)
            return True

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        try:
            mapping_file = target.inputs[0]
            reg_fav_table = self._build_reg_fav_table(mapping_file)
            reg_fav_table.to_parquet(target.output_path, index=False)
            logger.info(
                "PLAD reg_fav table complete: %d favored (%s, year) rows",
                len(reg_fav_table), self._gid_column,
            )
            return True
        except Exception:
            logger.exception("Error building PLAD reg_fav table")
            return False


registry.register(PlaDSource.ID, __name__, PlaDSource.__name__, PlaDSource.STEPS, aliases=PlaDSource.ALIASES, requires=PlaDSource.REQUIRES)
