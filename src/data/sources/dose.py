"""DOSE reported subnational economic output: fetch + prepare, no grid.

Wenz et al. 2023 (Sci Data 10:425), "DOSE -- Global dataset of reported
subnational economic output", v2.9 on Zenodo (record 13773040, 2024-09-17):
~1,660 first-level regions in 83 countries, 1953-2020, compiled from
statistical offices -- no satellite input, which is why the corridor fork
uses it as its growth measure other than lights
(docs/successors/fork-corridors-dose.md, Part 2). One ~15 MB CSV, so it
can come through the proxy as well as from the transfer node; like `ports`,
`download_async()` runs a `requests` download in an executor.

**Regions are keyed by GADM 3.6 `GID_1` codes; the pipeline uses GADM 4.1.**
Most codes carry over, but some countries were renumbered (GADM 4.1 added
Chile's Ñuble region, so `CHL.13_1` now means Ñuble, not Santiago) and some
have no 4.1 code at all (Colombia). PREPARE therefore matches each DOSE
region to a GADM 4.1 region in the same country (`match_regions`):

1. a name match (normalised `NAME_1`/`VARNAME_1`, similarity >= 0.8) that is
   clearly better than the code's own name (by > 0.1) overrides the code;
2. otherwise the same code, if it exists in GADM 4.1;
3. otherwise a name match >= 0.8;
4. otherwise unmatched. Two DOSE regions claiming one GADM region are both
   dropped as ambiguous (old/new Moroccan regions, Leningrad/St Petersburg).

Outputs, under `prepared/dose/adm/`:

- `dose_gadm_match.parquet`: one row per DOSE region -- `dose_GID_1`,
  `region`, `GID_1_gadm41` (string), `GID_1` (pipeline integer id, via
  gadm's `GID_1_code_mapping.json`), `match` (code/name/unmatched/ambiguous).
- `dose_gid1.parquet`: the matched (region, year) rows, keyed by `GID_1`
  (integer) and `year`, with population, output (LCU and USD, current and
  2015 prices, total and by sector where reported), DOSE's own annual
  temperature `T_a` and precipitation `P_a`, and `version`. **`T_a` is
  temperature: never use it as a control in a temperature regression.**
"""

from __future__ import annotations

import dataclasses
import difflib
import json
import logging
import os
import re
import unicodedata
from typing import Any, List

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import layout, registry
from src.data.sources.base import DataSource
from src.data.sources.misc._fetch import ConfiguredFile, ConfiguredFilesFetchMixin
from src.data.sources.steps import Completion, PipelineStep, StepTarget

logger = logging.getLogger(__name__)

DEFAULT_URL = "https://zenodo.org/api/records/13773040/files/DOSE_V2.9.csv/content"
DEFAULT_NAME = "DOSE_V2.9.csv"
NAME_MIN = 0.8      # name similarity accepted as a match
NAME_MARGIN = 0.1   # how much better a name must be to override the code

_STOPWORDS = {
    "region", "province", "state", "department", "the", "of", "oblast", "krai", "republic",
    "autonomous", "county", "district", "prefecture", "governorate",
}
_KEEP_COLUMNS = (
    "year", "pop", "grp_lcu", "grp_pc_lcu", "grp_pc_usd", "grp_pc_lcu_2015", "grp_pc_usd_2015",
    "grp_pc_lcu2015_usd", "ag_grp_pc_lcu_2015", "man_grp_pc_lcu_2015", "serv_grp_pc_lcu_2015",
    "cpi_2015", "deflator_2015", "StructChange", "version", "T_a", "P_a",
)


def _norm(name: Any) -> str:
    s = unicodedata.normalize("NFKD", str(name)).encode("ascii", "ignore").decode().lower()
    return " ".join(w for w in re.split(r"[^a-z]+", s) if w and w not in _STOPWORDS)


def _similarity(a: Any, b: Any) -> float:
    return difflib.SequenceMatcher(None, _norm(a), _norm(b)).ratio()


def match_regions(dose_regions, gadm):
    """Match DOSE regions (`GID_0`, `GID_1`, `region`) to GADM 4.1 level-1
    regions (`GID_0`, `GID_1`, `NAME_1`, `VARNAME_1`) -- rules in the module
    docstring. Returns one row per DOSE region."""
    import pandas as pd

    alts = {}
    by_country: dict = {}
    for r in gadm.itertuples():
        variants = [v for v in str(r.VARNAME_1).split("|") if v and v not in ("None", "nan")]
        alts[r.GID_1] = [r.NAME_1, *variants]
        by_country.setdefault(r.GID_0, []).append(r.GID_1)

    def score(region, gid):
        return max(_similarity(region, a) for a in alts[gid])

    rows = []
    for r in dose_regions.itertuples():
        scored = sorted(((score(r.region, g), g) for g in by_country.get(r.GID_0, [])), reverse=True)
        best_s, best = scored[0] if scored else (0.0, None)
        code = r.GID_1 if r.GID_1 in alts else None
        code_s = score(r.region, code) if code else 0.0
        if best is not None and best != code and best_s >= NAME_MIN and best_s > code_s + NAME_MARGIN:
            gid, how = best, "name"
        elif code:
            gid, how = code, "code"
        elif best is not None and best_s >= NAME_MIN:
            gid, how = best, "name"
        else:
            gid, how = None, "unmatched"
        rows.append(dict(GID_0=r.GID_0, dose_GID_1=r.GID_1, region=r.region, GID_1_gadm41=gid, match=how,
                         name_similarity=best_s if how == "name" else code_s))
    m = pd.DataFrame(rows)
    dup = m.GID_1_gadm41.notna() & m.duplicated("GID_1_gadm41", keep=False)
    m.loc[dup, "match"] = "ambiguous"
    m.loc[dup, "GID_1_gadm41"] = None
    return m


class DoseSource(ConfiguredFilesFetchMixin, DataSource):
    """DOSE regional output, matched to GADM 4.1 GID_1."""

    ID = "dose"
    STEPS = (PipelineStep.FETCH, PipelineStep.PREPARE)
    REQUIRES = ((PipelineStep.PREPARE, "gadm", PipelineStep.PREPARE),)

    DATA_SOURCE_NAME = "dose"

    def __init__(self, ctx: PipelineContext, cfg: SourceConfig):
        if cfg.data_path is None:
            cfg = dataclasses.replace(cfg, data_path="dose")
        super().__init__(ctx, cfg)

        url = cfg.raw.get("url", DEFAULT_URL)
        name = cfg.raw.get("name", DEFAULT_NAME)
        self.CONFIGURED_FILES: List[ConfiguredFile] = [ConfiguredFile(key="dose", url=url, name=name)]

    async def download_async(self, file_url: str, output_path: str, session: Any = None) -> None:
        """Synchronous `requests` download in an executor (module docstring)."""
        import asyncio

        loop = asyncio.get_event_loop()
        await loop.run_in_executor(None, self.download, file_url, output_path, None)

    # -- PREPARE ---------------------------------------------------------------

    def _raw_file_path(self) -> str:
        return os.path.join(self.output_root(PipelineStep.FETCH), self.CONFIGURED_FILES[0].name)

    def _gadm_root(self) -> str:
        return os.path.join(
            layout.output_root(self.ctx.data_root, "misc", PipelineStep.PREPARE, agg=layout.ADM_AGG), "gadm"
        )

    def _plan_prepare(self) -> List[StepTarget]:
        raw_file = self._raw_file_path()
        if not os.path.exists(raw_file):
            return []
        root = self.output_root(PipelineStep.PREPARE, agg=layout.ADM_AGG)
        return [
            StepTarget(
                source_id=self.ID, step=PipelineStep.PREPARE, key="dose",
                output_path=os.path.join(root, "dose_gid1.parquet"),
                inputs=(raw_file,), completion=Completion.PATH_EXISTS,
                meta={"match_path": os.path.join(root, "dose_gadm_match.parquet")},
            )
        ]

    def _execute_prepare(self, target: StepTarget) -> bool:
        import pandas as pd
        import pyogrio

        if not self.cfg.override and os.path.exists(target.output_path):
            logger.info("Skipping DOSE PREPARE, output already exists: %s", target.output_path)
            return True

        dose = pd.read_csv(target.inputs[0], low_memory=False)
        gadm = pyogrio.read_dataframe(
            os.path.join(self._gadm_root(), "gadm_levelADM_1_simplified.gpkg"),
            columns=["GID_0", "GID_1", "NAME_1", "VARNAME_1"], read_geometry=False,
        )
        with open(os.path.join(self._gadm_root(), "GID_1_code_mapping.json")) as f:
            gid_ids = json.load(f)

        match = match_regions(dose[["GID_0", "GID_1", "region"]].drop_duplicates("GID_1"), gadm)
        match["GID_1"] = match.GID_1_gadm41.map(gid_ids).astype("Int64")

        out = (
            dose.rename(columns={"GID_1": "dose_GID_1"})
            .merge(match.loc[match.GID_1.notna(), ["dose_GID_1", "GID_1", "GID_1_gadm41"]], on="dose_GID_1")
        )
        out = out[["GID_1", "GID_1_gadm41", "dose_GID_1", "GID_0", "region", *_KEEP_COLUMNS]]

        os.makedirs(os.path.dirname(target.output_path), exist_ok=True)
        match.to_parquet(target.meta["match_path"], index=False)
        tmp = f"{target.output_path}.tmp"
        out.to_parquet(tmp, index=False)
        os.replace(tmp, target.output_path)
        logger.info(
            "DOSE PREPARE complete: %d of %d regions matched (%s), %d region-years -> %s",
            int(match.GID_1.notna().sum()), len(match), match["match"].value_counts().to_dict(),
            len(out), target.output_path,
        )
        return True


registry.register(DoseSource.ID, __name__, DoseSource.__name__, DoseSource.STEPS, requires=DoseSource.REQUIRES)
