"""PLAD (Political Leaders and Development): fetch + prepare, no grid step.

docs/design/09-integrated-pipeline.md §5. Merges
`src/data/download/sources/harvard.py::HarvardDataSource` (Harvard Dataverse
API fetch -- registered under aliases `harvard_plad`/`harvard`) and
`src/data/preprocess/sources/plad.py::PLADPreprocessor` (`stage="spatial"`).
PLAD's raw `.dta` table already carries GADM's native `gid_1`/`gid_2` string
codes directly, so there's no vector-boundary pre-step to build.

**Not rasterized.** Regional favoritism (`reg_fav`) is constant across
every pixel of the favored admin unit for a given year -- it varies only by
`GID_1`/`GID_2` and year, never by pixel location -- so PREPARE writes a
tiny `(GID_N, year)`-keyed parquet table of favored units instead of a full
pixel-grid zarr, and doesn't need GADM's polygon geometries at all (only
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

`REQUIRES` on gadm's **PREPARE** step -- only the integer-id mapping sidecar
(`GID_N_code_mapping.json`) is needed, not gadm's polygon geometries. Gadm's
PREPARE builds the rasterized output and writes the mapping sidecar directly;
`PipelineStep.GRID` doesn't exist anywhere. Scoped to this source's own
PREPARE step (only `_plan_prepare()` touches gadm), so FETCH runs unblocked
before gadm exists.
"""

from __future__ import annotations

import dataclasses
import json
import logging
import os
import tempfile
from typing import Dict, List, Optional

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.misc._fetch import ConfiguredFile, ConfiguredFilesFetchMixin
from src.data.sources.steps import Completion, PipelineStep, StepTarget, TargetSelection
from src.data.sources import verify

logger = logging.getLogger(__name__)

DEFAULT_DOI = "doi:10.7910/DVN/YUS575"

#: The dataset's one file this pipeline needs, hardcoded rather than
#: resolved via a live Dataverse API call (`GET .../api/datasets/
#: :persistentId?persistentId=...`) -- that endpoint sits behind an AWS WAF
#: bot challenge that blocks plain HTTP clients (confirmed directly, not
#: just suspected), so a live-call-based FETCH could never actually
#: succeed unattended. Looked up once via the dataset's file listing page
#: (`https://dataverse.harvard.edu/dataset.xhtml?persistentId=doi:10.7910/
#: DVN/YUS575`): the file labeled "PLAD_April_2024.tab" (Dataverse's
#: auto-converted tabular view of the uploaded Stata file) has id 10119325.
#: `dataFile.originalFileName` for that entry -- and the name this pipeline
#: expects on disk (`_resolve_plad_data_file()` below) -- is
#: "PLAD_April_2024.dta", even though the plain access URL (no
#: `?format=original`) actually serves Dataverse's tab-delimited bytes,
#: matching `_build_reg_fav_table()`'s `pd.read_table()` parse below: the
#: `.dta` extension on disk is inherited original-upload metadata, not a
#: claim about the byte format.
DEFAULT_FILE_ID = "10119325"
DEFAULT_FILENAME = "PLAD_April_2024.dta"


class PlaDSource(ConfiguredFilesFetchMixin, DataSource):
    """Political Leaders and Development: regional-favoritism (GID_N, year) table."""

    ID = "plad"
    ALIASES = ("harvard_plad", "harvard")
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)
    REQUIRES = ((PipelineStep.PREPARE, "gadm", PipelineStep.PREPARE),)

    DATA_SOURCE_NAME = "harvard"
    OUTPUT_PREFIX = "plad"  # hardcoded in the old code, see module docstring

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="plad")
        if cfg.year_range is None:
            cfg = dataclasses.replace(cfg, year_range=(1980, 2022))  # old PLADPreprocessor default
        super().__init__(ctx, cfg)

        file_id = cfg.raw.get("file_id", DEFAULT_FILE_ID)
        filename = cfg.raw.get("filename", DEFAULT_FILENAME)
        self.CONFIGURED_FILES: List[ConfiguredFile] = [
            ConfiguredFile(
                key="plad", url=f"https://dataverse.harvard.edu/api/access/datafile/{file_id}", name=filename
            )
        ]

        self.admin_level = cfg.raw.get("admin_level", 1)
        if self.admin_level not in (1, 2):
            raise ValueError("admin_level must be 1 or 2")

        self.temp_dir = cfg.temp_dir or tempfile.mkdtemp(prefix="plad_processor_")
        os.makedirs(self.temp_dir, exist_ok=True)

    @property
    def _gid_column(self) -> str:
        return f"GID_{self.admin_level}"

    def output_root(self, step: PipelineStep, *, namespace: str | None = None) -> str:
        if step is PipelineStep.PREPARE:
            # This source's PREPARE output lives at the on-disk location
            # `stage_2` (via layout's own PipelineStep.GRID convention), not
            # the generic `stage_1` a plain PREPARE would map to. Same
            # rationale as gadm/osm/country_classifications: PREPARE writes
            # to the GRID path.
            return layout.output_root(
                self.ctx.data_root,
                self.OUTPUT_PREFIX,
                PipelineStep.GRID,
                namespace=namespace,
                grid_id=self.ctx.grid_id,
                layout=self.ctx.layout,
            )
        return super().output_root(step, namespace=namespace)

    # list_remote_files/local_path/filename_to_entrypoint/get_all_entrypoints/
    # download/download_async/has_entrypoints: inherited from
    # ConfiguredFilesFetchMixin (src/data/sources/misc/_fetch.py), driven by
    # self.CONFIGURED_FILES set in __init__ above.

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
        if step is PipelineStep.PREPARE:
            return self._plan_prepare()
        raise AssertionError(f"unreachable: {step}")

    def _plan_prepare(self) -> List[StepTarget]:
        return self._discover_prepare()

    def _discover_prepare(self) -> List[StepTarget]:
        mapping_file = self._gid_mapping_file()
        if not os.path.exists(mapping_file):
            return []
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key=f"adm{self.admin_level}",
                output_path=os.path.join(
                    self.output_root(PipelineStep.PREPARE),
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
        if target.step is PipelineStep.PREPARE:
            return self._execute_prepare(target)
        raise AssertionError(f"unreachable: {target.step}")

    def _execute_fetch(self, target: StepTarget) -> bool:
        # FETCH is local-disk only now -- no HPC target required. `data
        # transfer` (separate, manual or auto per source config) is the only
        # thing that pushes to HPC.
        from src.data.common.fetch.driver import run_fetch

        return run_fetch(self, **self.cfg.raw.get("download", {}))

    # -- PREPARE: build the (GID_N, year) favored-unit table -----------------

    def _gid_mapping_file(self) -> str:
        from src.data.sources.misc.gadm import gid_mapping_path

        return gid_mapping_path(self.ctx.data_root, self.ctx.grid_id, self.ctx.layout, self._gid_column)

    def _resolve_plad_data_file(self) -> Optional[str]:
        raw_root = self.output_root(PipelineStep.FETCH)
        if not os.path.isdir(raw_root):
            return None
        for name in sorted(os.listdir(raw_root)):
            filename = name.lower()
            if "plad" in filename and filename.endswith(".dta"):
                return os.path.join(raw_root, name)
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

    def _execute_prepare(self, target: StepTarget) -> bool:
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
