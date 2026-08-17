# 09 — Integrated Pipeline: merging download+preprocess, a standardized step vocabulary, splitting `misc`

This document supersedes the download/preprocess subsystem split and the `stage="annual"/"spatial"` naming
used throughout [`00-backbone-overview.md`](00-backbone-overview.md), [`04-ingest.md`](04-ingest.md),
[`05-migration.md`](05-migration.md), [`07-modis-ingest.md`](07-modis-ingest.md) and
[`08-hpc-transfer.md`](08-hpc-transfer.md). Those documents remain the historical record of the
backbone-grid/storage/convolution redesign; wherever they describe the download/preprocess boundary or the
`annual`/`spatial` stage names, this document is the current authority instead. Nothing in the grid, storage,
neighbourhood-engine, or tabularization design (docs 01-04) changes — this is a redesign of how per-source
acquisition and transformation code is organized and named, not of the science.

## 1. Why the old shape stopped fitting

`docs/design/00-08` were written against the repo's existing shape: a `src/data/download/` subsystem
(acquire raw files) and a separate `src/data/preprocess/` subsystem (turn raw files into analysis-ready
grids), with preprocess sources implementing two named sub-stages, `stage="annual"` (temporal compositing)
and `stage="spatial"` (reprojection onto the canonical grid). Direct inspection of the current code shows
four concrete ways this shape no longer describes the sources it needs to:

1. **MODIS has no download step.** `src/data/preprocess/sources/modis.py` streams directly from Microsoft
   Planetary Computer's STAC catalog *inside* its `stage="annual"`, combining acquisition and native-grid
   temporal compositing in one step. `preprocess/sources/factory.py::create_source()` returns `None` for
   it — there is no download-side counterpart to name.
2. **`misc` has no temporal composite**, so it invented a third stage name, `"vector"`, for OSM/GADM
   simplification. This is papered over by a hardcoded special case in `preprocess/workflow.py:121-126`:
   `if preprocessor_name.lower() == 'misc' and stage == 'annual': stage = 'vector'` — the workflow layer
   knows the name of one specific source.
3. **Static/point sources skip `"annual"` entirely.** `plad.py`, `berman_mining.py`, `snl_mining.py`
   implement only `stage="spatial"`. "Every source has both stages" was never actually true; the model just
   never had a way to say so.
4. **The two subsystems are glued by name-matching, not a contract.** `preprocess/sources/factory.py`'s
   `create_source()` imports the matching download-side class by string comparison, or returns `None` — and
   only the download side received the recent pluggable-registry refactor (`registry.py`, lazy-imported
   aliases); `preprocess/sources/factory.py::get_preprocessor_class()` is still a hardcoded if/elif chain,
   and `preprocess/workflow.py` (660 lines) is still a monolith.

**Verified, currently-latent consequences of the string-typed, two-subsystem model** (found by direct
grep/read during this redesign, not asserted):

- `get_preprocessor_class()` does not cover several aliases the download-side registry accepts.
  `--source eog_viirs` (the actual key in `orchestration/configs/data.yaml`) falls through to the generic
  branch, which tries to import `src.data.preprocess.sources.eog_viirs` and raises `ImportError`. Only
  `--source eog` works, and that alias appears in neither `data.yaml` nor `jobs.yaml`.
- `orchestration/slurm/jobs.yaml` names sources — `acag`, `esacci`, `viirs_annual`, `plad` — that do not all
  resolve against `data.yaml`'s actual keys: **no `acag` or `esacci` block exists in `data.yaml` at all**
  (confirmed by direct read of the file's `sources:` section), and the real keys are `eog_viirs` /
  `harvard_plad`, not `viirs_annual` / `plad`. Four of ACAG's and ESACCI's SLURM jobs have therefore never
  been runnable from the committed config.
- `misc`'s `worldbank_income_classes` config entry sets `subfolder: "hdi"` — the World Bank file lands in
  the HDI folder.
- `preprocess/workflow.py::handle_validate_task` (~180 lines) is dead: no preprocessor has a
  `preprocessing_index` attribute, and `UnifiedDataIndex` has no `validate_against_gcs` /
  `cleanup_missing_files` method. The download-side `TaskHandlers.handle_validate` is an explicit
  `return False` placeholder.
- `gcs_upload_path` is an `@abstractmethod` on `BaseDataSource`, implemented in all 8 download sources, and
  called from nowhere; `src/data/common/gcs/__init__.py` is 0 bytes.
- `HPCClient.execute_command` and `HPCClient.rsync_transfer` are each defined **twice** in the same class
  body (`client.py:94`/`423`, `129`/`543`) — the second definition silently shadows the first (already
  flagged, unfixed, in [`08-hpc-transfer.md`](08-hpc-transfer.md) §3).
- `berman_mining.py` defines `get_hpc_output_path` and `from_config` twice each — same shadowing pattern.
- `snl_mining`'s `data_path` is `"snf_mining"` in `data.yaml` but `"snl_mining"` in
  `orchestration/configs/debug/snl-mining-preprocess.yaml`.

None of this is a reason to distrust the *backbone* design (grid/storage/convolution) — it is a reason to
distrust the *plumbing* the backbone design was built on top of. This document redesigns the plumbing.

## 2. The step vocabulary

**Decision: replace `stage ∈ {"annual", "spatial", "vector"}` with a fixed three-value enum, `PipelineStep`,
named for the artefact each step produces, not the mechanism that produces it.**

This is the governing principle, and it is what makes every special case in §1 disappear: `"annual"` names
a *mechanism* (temporal compositing), which is why MODIS's streaming fetch had to be crammed into it,
`misc` had to invent `"vector"` because it composites nothing, and static sources had to skip a named stage
entirely. Naming steps by output artefact makes all three ordinary.

| Step | Produces | Meaning | Typical host |
|---|---|---|---|
| `FETCH` | raw bytes on local disk, registered in `UnifiedDataIndex` | Origin acquisition. Idempotent, per-file resumable. The only step that talks to an origin over an unreliable/authenticated channel. | any host with egress to the origin |
| `PREPARE` | one source-native intermediate per logical unit | Everything before the canonical grid: temporal compositing on the native grid (lights, GLASS, ACAG, ESACCI, ntl_harm, MODIS), archive extraction + geometry simplification (OSM, GADM), tabular cleaning into a panel (HDI/WB), DuckDB feature construction (snl_mining). Output is still in the source's own representation. | anywhere; may need origin egress if `FETCH` is absent (MODIS) |
| `GRID` | analysis-ready Zarr on the canonical EPSG:6933 GeoBox | Landing on the canonical grid: reprojection for raster inputs, rasterization for vector/tabular inputs. **This is the contract boundary** consumed by [`03-neighbourhood-engine.md`](03-neighbourhood-engine.md) and the assembly stage. | HPC compute node |

Rejected names, and why: `REGRID` (falsely implies the input is already gridded — false for
OSM/GADM/PLAD/HDI/WB); `RASTERIZE` (false for the raster sources, which reproject, not rasterize);
`SPATIAL` (the current name — every step is spatial, so it selects nothing); `INGEST`/`TRANSFORM` (too vague
to be a contract boundary anyone can rely on).

### Why exactly three, and why the ladder stops here

The ladder conceptually continues `fetch → prepare → grid → convolve → assemble`, but the last two have
**different cardinality**: they consume *all* sources at once and produce one shared artefact, whereas
`FETCH`/`PREPARE`/`GRID` are per-source. They are correctly not per-source steps and must not enter
`PipelineStep` — consistent with [`04-ingest.md`](04-ingest.md) §6's decision to tabularize exactly once,
at the very end. `src/data/common/neighbourhood/` currently has **no CLI verb at all** (only driven by
`scripts/validate_backbone_subset.py`); this document flags that as a known gap, out of scope here, worth a
future `run.py neighbourhood run`.

### Declaring step absence — structurally, not by name

```python
class MODISSource(DataSource):
    ID = "modis"
    STEPS = (PipelineStep.PREPARE, PipelineStep.GRID)   # no FETCH: nothing is landed raw
```

`DataSource.plan(step)` / `.execute(target)` raise `UnsupportedStepError(source_id, step, supported=STEPS)`
if `step not in STEPS` — enforced by the base class, so a subclass cannot silently omit the check. The CLI
validates `--step` against the registry's declared `steps` **before constructing the source**, so an
unsupported step is a clean user-facing error, not an import failure or a silent remap (the fate of
`misc`'s `"vector"` hack). **`--step` is required; there is no default.** The `misc`/`"vector"` hack existed
only because `stage` defaulted to `"annual"` — removing the default removes the class of bug.

### Cross-source dependencies are declared, not implicit

Today, `country_classifications`' grid step reads GADM's grid output plus a country-code mapping; PLAD's
prepare step reads GADM's prepared `.gpkg` via four hardcoded fallback path guesses; `snl_mining` reads two
GADM `.gpkg` files; `berman_mining`/`plad`/`snl_mining` all read the geobox cache under
`misc/processed/stage_1/misc`. These become a declared class attribute:

```python
REQUIRES: tuple[tuple[PipelineStep, str, PipelineStep], ...] = ((PipelineStep.GRID, "gadm", PipelineStep.GRID),)
```

Each entry is `(my_step, prereq_source_id, prereq_step)`: it gates only *my_step* of the declaring source, not
every step. Scoping per-step (rather than blocking the whole source uniformly, which was this design's original
shape) matters in practice: ecoregions' `REQUIRES` on gadm exists only for its GRID step's `gadm_gid3_dominant`
overlay target, so `data run --source ecoregions --step fetch` must not be blocked on gadm just because
ecoregions' *own later step* needs it. `SourceSpec.requires_for(step)` (`src/data/sources/registry.py`) is the
one place that narrows `REQUIRES` down to the entries relevant to a given step; both `_check_requires()`
(`src/cli/data/handlers.py`) and `orchestration/slurm/submit_chain.py`'s dependency-chaining call through it
rather than re-deriving the filter themselves.

The runner resolves each requirement through `layout.output_root()` before planning targets, failing with
`MissingPrerequisiteError` naming the exact missing artefact and the command that produces it. This is the
piece of sequencing that is currently pure human responsibility (no `--dependency=afterok` anywhere in
`orchestration/`) and it becomes machine-checkable at negligible cost.

## 3. On-disk layout: one function, paths do not move

`src/data/sources/layout.py` becomes the single place the existing `stage_1`/`stage_2`/`stage_2_ease6933`
numbering lives:

```python
def output_root(data_root, source_id, step, *, namespace=None, grid_id="legacy_4326") -> str
def raw_root(data_root, source_id, *, namespace=None) -> str
def index_path(local_index_dir, source_id) -> str
def marker_path(output_path) -> str
```

Legacy mapping (the default, byte-identical to today):

| Step | Path |
|---|---|
| `FETCH` | `<data_root>/<data_path>/` (per-source `data_path` from config, as today) |
| `PREPARE` | `<data_root>/<data_path>/processed/stage_1[/<namespace>]` |
| `GRID` | `<data_root>/<data_path>/processed/stage_2[/<namespace>]`, or `stage_2_ease6933` when `grid_id == "ease6933"` |

**Decision: do not rename physical directories as part of this refactor.** Reasons:

1. GLASS's prepare stage is a multi-day SLURM job (`jobs.yaml`: `time: "7-00:00:00"`) — re-running it for a
   directory rename is indefensible.
2. Keeping paths fixed is what makes the migration verifiable as a *pure refactor*: old and new code write
   the same path, so their outputs can be diffed directly (§13's validation gate), rather than trusted from
   reading code.
3. At least eight consumers hardcode these paths today: `src/data/assemble/constants.py`,
   `src/analysis/subsets/registry.py`, `src/data/common/geobox/geobox.py`, `plad.py`, `snl_mining`'s
   `geometry_path` config keys, `scripts/validate_backbone_subset.py`, `scripts/rechunk_zarr.py`, and
   `data.yaml`'s `assemble:` blocks.

The `stage_N` numbering becomes an internal implementation detail of one function, no longer a vocabulary
anyone reads or writes elsewhere. `grid_id` also finally implements the `grid: ease6933 | legacy_4326`
config switch [`05-migration.md`](05-migration.md) §1 recommended and that today only MODIS honours ad hoc.
A physical rename (`raw/`, `prepared/`, `grid/<grid_id>/`) was left as an explicitly separate, optional
`layout: v2` future task (§14) with the eight-consumer list above as its checklist — not on this
redesign's critical path.

**Update: `layout: v2` is now implemented**, additive and opt-in via `pipeline.layout` in `data.yaml`
(default `legacy`, unchanged from the table above). `output_root()`/`raw_root()`/`grid_store_path()`
(`src/data/sources/layout.py`) all take a `layout` parameter; every registered source honours it for
FETCH/PREPARE/GRID. Physical layout under `layout: v2`:

| Step | Path |
|---|---|
| `FETCH` | `<data_root>/raw/<data_path>[/<namespace>]` |
| `PREPARE` | `<data_root>/prepared/<data_path>[/<namespace>]` |
| `GRID` | `<data_root>/grid/<grid_id>/<family>.zarr` -- one store per variable family (§2's original decision was scoped to single-contributing-source families only; every registered GRID-capable source turned out to fit that shape, so no multi-source write-coordination mechanism was needed) |

Of the eight originally-listed consumers, `assemble:`'s `stage_3` blocks were found to be disconnected
from any code currently in `src/data/assemble/*.py` (no code produces or reads the `*_tabular.parquet`
filenames those blocks reference) and were deliberately left untouched rather than migrated -- flagged as
likely-dead config, not a live consumer. The other seven were updated to be `layout`-aware, plus two
previously-undocumented hardcoded consumers found during that work
(`src/analysis/subsets/resolve.py`'s classifications path, and a fragile `glass/source.py` hack that
derived a temp path by string-splitting a GRID output path on the literal substring `"stage_2"`).
A migration script (`scripts/migrate_legacy_layout.py`) physically
moves already-computed legacy-layout data into the new layout; dry-run by default, requires `--execute`.

**Update: `layout: v2` is now the only layout.** No source or orchestration config ever set
`pipeline.layout: v2` in production, so the "legacy" default and the `layout` parameter threaded
through `output_root()`/`raw_root()`/`grid_store_path()` were dead weight — removed entirely.
`src/data/sources/layout.py` now builds only the physical tree the table above describes (unconditionally,
no `layout` argument anywhere), `PipelineContext` no longer has a `layout` attribute, and
`scripts/migrate_legacy_layout.py` is the one-time tool for moving already-computed on-disk data from the
old `<data_path>/processed/stage_1`/`stage_2[_ease6933]` shape into it.

### Completion / resumability, generalized from the MODIS lesson

Commit `28d7132` moved MODIS off Zarr for its `PREPARE` output partly because "a killed-mid-write Zarr dir
looks complete to `os.path.exists`" — a temp-file + `os.replace` GeoTIFF write is atomic, a directory write
is not. Every other source today still uses the unsafe `os.path.exists` check. Generalize the lesson instead
of leaving it MODIS-only:

```python
@dataclass(frozen=True)
class StepTarget:
    source_id: str
    step: PipelineStep
    key: str                      # resumability identity: "2004", "2004/h18v04", "osm/land_mask"
    output_path: str
    inputs: tuple[str, ...] = ()
    completion: Completion = Completion.MARKER   # MARKER | PATH_EXISTS | NEVER
    meta: Mapping[str, Any] = field(default_factory=dict)
```

Directory outputs (Zarr) use `Completion.MARKER` — a sibling `.complete` file written only after `to_zarr`
returns. Single-file outputs (MODIS GeoTIFF, `.gpkg`, `.parquet`) use `PATH_EXISTS` with temp-file +
`os.replace` writes. Shared multi-year region-write stores (e.g. the reprojected timeseries zarrs, where
several targets write disjoint regions of one store) use a per-key marker under
`<store>/.complete/<key>`. The runner owns the skip/`--override` decision uniformly; sources stop
re-implementing `if not self.overwrite and os.path.exists(...)` (duplicated today under three different
config key spellings: `override`, `overwrite`, `force`).

## 4. Module / package layout

Replace `src/data/download/` + `src/data/preprocess/` with:

```
src/data/
  sources/                          # one directory per source, whole lifecycle
    steps.py            # PipelineStep, StepTarget, TransferUnit, Completion, UnsupportedStepError,
                         #   MissingPrerequisiteError
    layout.py            # the one place stage_N lives
    base.py               # DataSource ABC, RemoteFileCatalog Protocol, fetch mixins
    templates.py          # ArchiveRasterSource, VectorSource
    registry.py            # SourceSpec + resolve/create/describe/all_specs, lazy import
    acag/  esacci/  eog/  glass/  ntl_harm/  modis/       # bulk-archive + streaming raster sources
    osm/  gadm/  country_classifications/                 # the misc split, §8
    plad/  berman_mining/  snl_mining/                     # vector/point sources
  pipeline/                         # replaces download/workflow/ AND preprocess/workflow.py
    context.py  config.py  handlers.py  runner.py
  common/
    fetch/               async_downloader.py  http.py     # moved from download/async_downloader.py
    raster/              spatial.py  compositing.py  rasterize.py   # moved from preprocess/common/
    geobox/  hpc/  index/  neighbourhood/  dask/  spark/           # unchanged
  assemble/                                                        # unchanged, stays separate — §9
```

Every source is a **package even when it is one file**, so per-source growth (a `session.py`, a
`filenames.py`) never later forces a move — and the download-side and preprocess-side code for one source
finally sit in one directory.

**Dependency rule: `sources/* → common/*`, `pipeline/* → sources/*`, and never `sources/a → sources/b`.**
Today this is violated twice (`preprocess/sources/misc.py` imports `download/sources/misc.py`;
`preprocess/sources/factory.py::create_source` imports `EOGDataSource`/`MiscDataSource` directly). Under the
merge, both violations vanish by construction — cross-source coupling becomes `REQUIRES` on artefact paths,
never a class import.

### The `DataSource` contract

Replaces both `BaseDataSource` (download) and `AbstractPreprocessor` (preprocess):

```python
class DataSource(abc.ABC):
    ID: ClassVar[str]
    ALIASES: ClassVar[tuple[str, ...]] = ()
    STEPS: ClassVar[tuple[PipelineStep, ...]]
    REQUIRES: ClassVar[tuple[tuple[PipelineStep, str, PipelineStep], ...]] = ()

    def __init__(self, ctx: "PipelineContext", cfg: "SourceConfig"): ...

    @classmethod
    def from_config(cls, ctx, cfg) -> "DataSource": ...      # default: cls(ctx, cfg)

    @abc.abstractmethod
    def plan(self, step: PipelineStep, selection: "TargetSelection") -> list[StepTarget]: ...

    @abc.abstractmethod
    def execute(self, target: StepTarget) -> bool: ...

    def output_root(self, step) -> str                       # -> layout.output_root(...)
    def transfer_units(self, step) -> list[TransferUnit]      # default from output_root
    def close(self) -> None                                   # sessions/dask teardown
```

Dropped relative to the two ABCs it replaces: `gcs_upload_path` (dead abstractmethod, implemented 8×,
called from nowhere), `get_hpc_output_path(stage)` (subsumed by `layout`), the string-dispatch pair
`get_preprocessing_targets`/`process_target` (replaced by typed `plan`/`execute`), and per-source copies of
`_strip_remote_prefix` (replaced by the existing `src/config/runtime.py::strip_remote_prefix`, currently
reimplemented in 11 files).

**Decision: do not touch `UnifiedDataIndex` (1,294 lines) or `AsyncHPCDownloader` (613 lines) in this
refactor.** Both are large and have zero test coverage today. They duck-type on a fixed attribute set:
`DATA_SOURCE_NAME`, `data_path`, `has_entrypoints`, `list_remote_files(entrypoint)`, `get_file_hash(url)`,
`get_all_entrypoints()`, `filename_to_entrypoint(rel)`, `download_async(url, path, session)`,
`local_path(rel)`, `schema_dtypes`. Make that implicit contract explicit as a `typing.Protocol`,
`RemoteFileCatalog`, in `sources/base.py`; `FETCH`-capable sources satisfy it (`DATA_SOURCE_NAME` becomes a
`property` returning `ID`). Result: the two riskiest untested modules in the repo change by zero lines, and
the coupling becomes documented and test-enforced (§12).

### Two templates, and a rule against inventing a third

- **`ArchiveRasterSource`** — `acag`, `esacci`, `eog` (×3 aliases), `ntl_harm`, `glass`. Shared flow: read
  completed files from the parquet index → group by year → per-year native-grid composite via
  `common/raster/compositing.composite_to_annual` → write native Zarr (`PREPARE`); then
  `SpatialProcessor.process_spatial_standard` onto the canonical geobox (`GRID`). Each concrete source
  supplies only a filename→year parser, a per-file loader, a variables/attrs extractor, a resampling
  default, and nodata/packaging attrs. Collapses ~2,400 lines across five files to an estimated ~900.
- **`VectorSource`** — `osm`, `gadm`, `plad`, `berman_mining`. `PREPARE` = extract/clean/simplify into
  `.gpkg`/`.parquet`; `GRID` = rasterize onto the canonical geobox via the new `common/raster/rasterize.py`,
  extracted from the four existing near-duplicate "create empty zarr + tile-loop rasterize" implementations
  in `misc.py`, `plad.py`, `snl_mining.py`, `berman_mining.py`. This extraction is a prerequisite for the
  `misc` split (§8) — without it, OSM and GADM would each inherit their own copy.
- **`modis`, `country_classifications`, `snl_mining` subclass `DataSource` directly.** Explicit rule: *no
  abstraction for a population of one.* MODIS's STAC streaming, the HDI/WB tabular join, and snl_mining's
  DuckDB-driven tiled rasterization are each singletons; they call `common/raster/*` helpers directly rather
  than inheriting a template built for them alone.

### Where the shared engines live

| From | To | Change |
|---|---|---|
| `src/data/download/async_downloader.py` | `src/data/common/fetch/async_downloader.py` | import path only |
| — | `src/data/common/fetch/http.py` | new: aiohttp/retry/`Content-Length` helpers currently copy-pasted across `misc.py`, `ntl_harm.py`, `acag.py`, `glass/source.py` |
| `src/data/preprocess/common/spatial.py` | `src/data/common/raster/spatial.py` | none |
| `src/data/preprocess/common/compositing.py` | `src/data/common/raster/compositing.py` | none |
| — | `src/data/common/raster/rasterize.py` | new: extracted from the 4 duplicates above |

`common/` is already the established home for cross-cutting machinery (`hpc/`, `index/`, `geobox/`,
`neighbourhood/`); this placement keeps `sources/` containing *only* sources.

### The registry

Extends the download side's already-good lazy-import pattern to cover metadata available *without*
importing (so `data list` and the SLURM generator can validate `(source, step)` pairs without pulling
Selenium/pystac/duckdb into the process):

```python
@dataclass(frozen=True)
class SourceSpec:
    id: str
    aliases: tuple[str, ...]
    module: str
    class_name: str
    steps: tuple[PipelineStep, ...]
    requires: tuple[tuple[str, PipelineStep], ...] = ()

def resolve(name) -> SourceSpec        # alias -> spec, no import
def load(name) -> type[DataSource]     # imports the module, returns the class
def create(name, ctx, cfg) -> DataSource
def all_specs() -> tuple[SourceSpec, ...]
```

A test asserts spec metadata equals the loaded class's attributes — guarding against drift and doubling as
the "every registered module actually imports" smoke test whose absence is exactly why `--source eog_viirs`
breaking preprocess went unnoticed (§1).

## 5. Per-source migration mapping

| Source | `STEPS` | `REQUIRES` | Notes |
|---|---|---|---|
| `acag` | F,P,G | — | Migrated first — cleanest two-file merge, no auth/Selenium. The reference package shape. |
| `esacci` | F,P,G | — | Same pattern. |
| `eog_dmsp` / `eog_viirs` / `eog_dvnl` | F,P,G | — | One package, three registered ids distinguished by config, mirroring today's `NAMES` tuple. |
| `ntl_harm` | F,P,G | — | `_select_best_file_for_year` gets a real unit test (untested today). |
| `glass_modis` / `glass_avhrr` | F,P,G | — | Migrated last among raster sources (largest, dual-source). GLASS's `_calculate_statistics` computes the annual composite from raw daily data rather than its own `monthly_stats` — the naive-mean bug already flagged in [`07-modis-ingest.md`](07-modis-ingest.md) §4 as "do not copy this pattern." Fixed in a **separate, explicitly labelled commit after** the mechanical migration, not silently inside it. |
| `modis` (+ `modis_robustness_11a1`) | P,G | — | No `FETCH` — `PREPARE` streams from STAC. Already closest to the target shape; the reference for `transfer_units` and atomic writes. |
| `osm` | F,P,G | — | The `misc` split, part 1 (§8). |
| `gadm` | F,P,G | — | The `misc` split, part 2. Seeds `common/raster/rasterize.py`. |
| `country_classifications` | F,P,G | `("gadm", GRID)` | The `misc` split, part 3. First real use of `REQUIRES`. |
| `plad` | F,P,G | `("gadm", PREPARE)` | Four hardcoded GADM-path-guessing fallbacks collapse into one `REQUIRES` resolution. `--admin-level` CLI flag moves into config. |
| `berman_mining` | F,G | `("gadm", PREPARE)` | No `PREPARE` today (no persisted intermediate) — declared absent, not invented. Duplicate `get_hpc_output_path`/`from_config` definitions deleted. |
| `snl_mining` | P,G | `("gadm", PREPARE)` | Newly registered — see §6. |

## 6. `snl_mining`: formalize, with `FETCH` declared absent

**Decision: register it, with `STEPS = (PREPARE, GRID)`, and keep its notebooks as explicitly out-of-band
producers of the DuckDB input `PREPARE` depends on.**

Today `snl_mining` is unregistered on the download side — notebooks and a README only, because its
acquisition is a manual S&P Global `.xls` export plus an OpenAI batch-enrichment loop, genuinely not
automatable. Shoehorning that into `FETCH` would mean inventing a fake step. But its processing side is a
fully-implemented, SLURM-scheduled preprocessor with a real config block — a real source in every sense that
matters. Under a vocabulary where step absence is *declarable*, formalizing it costs nothing:

- The notebooks + README move to `src/data/sources/snl_mining/notebooks/`, so the whole lifecycle —
  including its manual part — lives in one directory.
- `plan(PREPARE)` starts with a precondition check on `duckdb_path`, raising `MissingPrerequisiteError`
  naming the notebook that produces it, replacing today's obscure downstream failure.
- Its internal DuckDB feature build (buffer tables, ADM count tables, R-tree indexes) — currently hidden
  inside one `"spatial"` stage — becomes an explicit, independently-resumable `PREPARE` step. This is the
  one place this redesign improves a source's actual resumability, not just relocates its code.
- The `snf_mining` vs `snl_mining` `data_path` inconsistency (§1) is fixed while here.

## 7. The `misc` split: three sources, not four

**Decision: three source packages — `osm`, `gadm`, `country_classifications` — with
`country_classifications` declaring two independently-fetched origins (UNDP HDI, World Bank income). Not a
four-way split.**

### Why not four

1. **They share a join key and one output artefact today.** The country-classifications processing reads
   both files, melts both, snapshots both at panel years, and merges them into one wide table on `iso3`
   (`misc.py:764-780`); the grid step writes all resulting boolean columns into **one** Zarr store via
   sequential `mode='a'` writes. A four-way split means either two output stores — a real downstream
   artefact change requiring edits in `src/analysis/`, out of this refactor's scope — or two sources racing
   `mode='a'` writes into one store, a concurrency hazard for no benefit.
2. **The downstream consumer already reads them together.** `src/analysis/subsets/registry.py` derives both
   `HDI_*` and `WB_*` subsets from the combined classifications parquet.
3. **A four-way split would recreate the exact coupling this redesign removes** — two sources whose outputs
   must be silently re-joined by a third, implicit combiner.

### Why this still satisfies "separate the sources in misc"

The objection this document answers is that OSM/GADM/HDI/WB are "distinguished only by config keys, not by
code structure." The step vocabulary already separates *origin* from *artefact*, so real separation lands
where it belongs:

- Each of the four origins gets its own config entry, its own raw subfolder (fixing the World Bank →
  `hdi`-subfolder bug from §1), and its own index rows — independently re-fetchable and versioned.
- **OSM and GADM — the two genuinely different *processing pipelines* — become fully separate packages**
  with separate `STEPS`, separate SLURM jobs, separate `plan`/`execute`.
- Inside `country_classifications`, HDI and WB become separate pure functions with separate tests:
  ```
  country_classifications/
    source.py       # DataSource: plan/execute, the iso3 join, the GADM-raster grid step
    hdi.py           # read_hdi(path) -> DataFrame   (csv, HDI thresholds, HDI_LO/ME/HI/VH)
    worldbank.py     # read_worldbank(path) -> DataFrame   (xlsx sheet, L/LM/UM/H)
  ```
  Each `read_*` is a pure function on a file path, independently unit-testable with a small fixture —
  neither is testable in isolation today.

### Escape hatch

Because step absence is declarable, a future full four-way split is a small local change: `hdi` and
`worldbank` become sources with `STEPS = (FETCH, PREPARE)`; `country_classifications` becomes
`STEPS = (GRID,)` with `REQUIRES = (("hdi", PREPARE), ("worldbank", PREPARE))`. Recorded here so the
decision is revisitable without a redesign (§14).

### The split is a code split, not a data migration

All three keep `data_path: misc` plus a `namespace` equal to today's subfolder, so every output path is
byte-identical and no downstream consumer changes. The one real migration: the parquet index filename is
derived from `data_path` today (shared by all four origins under one `MiscDataSource`), and must become
per-`source_id`. Handled by a new `data index --adopt-local`, which registers already-present local raw
files as `completed` without re-fetching.

## 8. Unified workflow and CLI

**One CLI domain, `data`, replaces `download` and `preprocess run`/`preprocess transfer`:**

```
run.py data list                                              # sources, aliases, steps, requires
run.py data plan         --config C --source S --step STEP [--years A B] [--key K ...]
run.py data index        --config C --source S [--rebuild] [--adopt-local]
run.py data run          --config C --source S --step STEP [--years A B] [--key K ...] [--override]
run.py data transfer     --config C --source S --step STEP [--direction push]
```

- **`index` stays a first-class verb**, not a flag on `--step fetch`: it is an idempotent catalog operation
  with different runtime and failure modes than downloading, and produces no artefact under `layout`.
- **`plan` is new**: prints the target list with per-target complete/pending status. It is the debugging
  tool the repo currently lacks, and it is what the migration-equivalence tests (§12) assert against.
- `--key` (repeatable, matching `StepTarget.key`) replaces both today's `--grid-cells` and `--subsource`,
  giving every source uniform single-target reruns.

**`assemble` and `analysis` stay separate CLI domains, deliberately.** The complaint this document answers
is about an artificial split *within one source's lifecycle*. `assemble` has different cardinality
(many-sources-in, one-panel-out), its own `assemble:` config keyed by output panel, and its own
`create`/`update` verbs — merging it would mean inventing a fake source. State this as a decision, not an
omission.

### `src/data/pipeline/` internals

- **`context.py` — `PipelineContext`**: `data_root`, `local_index_dir`, `ssh_target`, `key_file`,
  `staging_dir`, `temp_dir`, `grid_id`, dask settings, and the persistent-session registry (Selenium
  sessions for EOG/GLASS) lifted from the download side's `WorkflowContext` — the preprocess side's context
  never had this and didn't need it until sources merged.
- **`config.py` — `SourceConfig`**: an explicit dataclass with named fields plus a `raw: Mapping` escape
  hatch for source-private config blocks. Replaces the ~100 lines of kwargs-smearing duplicated across
  `PreprocessTaskHandlers.handle_preprocess`/`handle_validate` (an 11-step "copy everything into one flat
  dict with `remote_`/`gcs_`/`hpc_` prefixes" ritual, done twice).
- **`handlers.py`** — `build_index`, `run_step`, `transfer_step`, mirroring the download side's
  `TaskHandlers` (the pattern worth imitating, per direct comparison of the two subsystems).
- **`runner.py`** — resolves the source, validates `step in spec.steps`, checks `REQUIRES`, plans targets,
  filters complete targets unless `--override`, executes, aggregates results. **Signal handling moves from
  import time into an explicit install/restore inside `runner.py`** — today, importing
  `preprocess/workflow.py` installs a `SIGTERM` handler as a side effect of the import itself.

### HPC transfer under the new vocabulary

Structurally unchanged: `src/data/common/hpc/transfer.py` (`TransferManifest`, `transfer_unit`/
`transfer_units`, tar-or-direct-file push) stays exactly as generic as
[`08-hpc-transfer.md`](08-hpc-transfer.md) designed it. Only the hook name and vocabulary change:
`AbstractPreprocessor.get_transfer_units(stage)` → `DataSource.transfer_units(step)`; config
`transfer.stages: [annual]` → `transfer.steps: [prepare]`. MODIS's per-`(year, tile)` GeoTIFF override
survives verbatim, renamed. One necessary one-time step: rename the existing
`transfer_modis_annual.parquet` manifest to `transfer_modis_prepare.parquet` so MODIS does not silently
re-push ~6,700 already-transferred files. The duplicate `HPCClient` method definitions flagged in `08` §3
are deleted while this hook is touched anyway.

## 9. Orchestration fallout

### `orchestration/slurm/jobs.yaml`

- `stage:` → `step:` (`annual → prepare`, `spatial → grid`, `vector → prepare`).
- `subsource:` disappears: `misc-gadm-preprocess-spatial` → `source: gadm, step: grid`;
  `misc-osm-preprocess-spatial` → `source: osm, step: grid`.
- Fix the broken source names from §1 (`viirs_annual → eog_viirs`, `plad → plad` once the id matches); add
  the missing `data.yaml` blocks for `acag`/`esacci` so their four jobs become runnable for the first time.
- Add missing entries: `gadm-prepare`, `osm-prepare`, `country_classifications-prepare`,
  `country_classifications-grid`, `plad-prepare`, `modis-grid` — all currently hand-run or hand-maintained.
- New field **`host: slurm | egress`** (default `slurm`) and optional **`transfer_after: <step>`**.

### `orchestration/slurm/generate_slurm_scripts.py`

- Emits `run.py data run --source X --step Y`; drops `--subsource`/`--admin-level`.
- `host: egress` emits a plain shell script into `orchestration/scripts/` (optionally followed by
  `data transfer --step <transfer_after>`) instead of an `#SBATCH` script — **folding
  `orchestration/scripts/modis-ingest-annual.sh` into the generator.** Its reason for living outside the
  generator was "not a SLURM job"; that becomes a declared property (`host: egress`) rather than a directory
  convention, so **MODIS's `prepare` step is vocabulary-identical to every other source's `prepare` after
  this refactor** — what remains genuinely different (needing internet egress) is modeled as data, not as a
  naming exception.
- Validates at generation time: every job's `source` resolves in `registry.all_specs()`, every `step` is in
  that spec's declared `steps`, every `source` is a key in `data.yaml`. Converts the §1 findings from a
  class of silent runtime failure into a build-time failure, permanently.

### `orchestration/configs/data.yaml`

- `sources.misc` → `sources.osm` / `sources.gadm` / `sources.country_classifications`, each with an
  `origins:` list replacing the nested `sources:`/`files:` block; World Bank's subfolder fixed to
  `worldbank`.
- `transfer.stages` → `transfer.steps`.
- Add the missing `acag`/`esacci` blocks; fix `snl_mining.data_path`.
- New top-level `pipeline: {grid: legacy_4326 | ease6933, layout: legacy | v2}`.
- Delete confirmed-dead keys while here: `processing.demean_columns`, `processing.filter_land_only`,
  `processing.spark` ([`05-migration.md`](05-migration.md) §2).

## 10. Migration sequencing

Mirrors [`05-migration.md`](05-migration.md)'s style: dependency-ordered, additive by default, one hard
validation gate before deletion. Because `layout.py` reproduces today's physical paths exactly, new code
writes where old code wrote — so every source's migration is verifiable by comparing artefacts, not by
reading diffs.

| # | Task | Additive / in-place |
|---|---|---|
| 0 | Characterization tests against the *current* code: snapshot every source's `get_preprocessing_targets`/`get_hpc_output_path`. The migration oracle — nothing moves before this exists. | Additive |
| 1 | New scaffolding (`sources/{steps,layout,base,templates,registry}.py`, `pipeline/{context,config,handlers,runner}.py`), registry empty, nothing wired. | Additive |
| 2 | Move shared engines (`common/fetch/`, `common/raster/`) with re-export shims at old import paths; extract `rasterize.py` as new code. | Additive |
| 3 | Register `pipeline` CLI domain alongside still-working `download`/`preprocess`. | Additive |
| 4 | Migrate `acag` (pattern-setter). Gate: plan snapshot matches oracle; one year run under both codepaths produces identical Zarr. | Additive |
| 5 | Migrate `esacci`, `ntl_harm`, `eog`×3, `glass` (ascending risk); GLASS's compositing fix as a separate labelled follow-on commit. | Additive |
| 6 | Migrate `modis`, including the transfer-manifest rename. | Additive |
| 7 | The `misc` split (`osm`, `gadm`, `country_classifications`), then `plad`, `berman_mining`, `snl_mining`; one-time index re-registration via `data index --adopt-local`. | Additive code; one-time index re-registration |
| 8 | Orchestration + config changes; regenerate all SLURM scripts; `git rm` the old ones. | In place |
| 9 | **Hard validation gate**: one target per archetype (bulk-composite, streaming, vector, tabular-join, point) diffed old-code-vs-new-code output; full test suite green; generator `--check` clean. **Nothing in step 10 starts before this passes.** | — |
| 10 | Cutover: delete `src/data/download/`, `src/data/preprocess/`, the shims, `src/cli/download/`, `src/cli/preprocess/`; fold in `geobox_patch.py` (1,607 dead lines), dead `demean`/`gcs` wiring. | In place |
| 11 | *Optional, separate*: `layout: v2` physical rename, with the §3 consumer checklist. Not on the critical path. | Implemented |

**Status**: steps 0–10 complete. Step 9's hard gate passed (two archetype representatives waived by explicit
decision, not executed — see §14); step 10's cutover is done. Step 11 (`layout: v2`) is now implemented (§3),
additive and opt-in — was never on the critical path, but is no longer outstanding either.

**Overlap strategy**: rather than a runtime flag, the switch is *which CLI verb you invoke* —
`preprocess run --source acag` and `data run --source acag --step prepare` coexist for as long as both
modules exist, writing the same paths. Commit granularity: one commit per source package, each
self-contained and independently revertable.

## 11. Testing strategy

`tests/data/` today has zero coverage for `download/`, `common/hpc/`, `common/index/`, and nearly all of
`preprocess/sources/`. Add tests covering exactly the surfaces this refactor touches:

- `test_layout.py` — every path mapping, hardcoded from today's actual values.
- `test_registry.py` — every alias resolves; every registered module imports; spec metadata matches class
  attributes; every `data.yaml` source key resolves.
- `test_step_contract.py` — parameterized over all sources: undeclared-step errors, `REQUIRES` resolve,
  target-key uniqueness.
- One `test_plan.py` per source against a synthetic index fixture — the migration oracle from step 0.
- Parser unit tests for currently-unverified logic: GLASS filename parsing, `ntl_harm`'s best-file
  selection, HDI/WB threshold boundaries, MODIS tile↔bbox round-trip.
- `test_runner.py` — step validation, `MissingPrerequisiteError` content, completion-marker skip logic
  (including the killed-mid-write-directory case), signal handler *not* installed at import time.
- `test_fetch_protocol.py` — every `FETCH`-capable source satisfies `RemoteFileCatalog` — the one real
  coupling risk against the untouched `UnifiedDataIndex`/`AsyncHPCDownloader`.
- `test_jobs_yaml.py` — every `jobs.yaml` entry valid against the registry and `data.yaml`;
  `generate_slurm_scripts.py --check` reports no drift.

Explicitly not covered, stated as decisions: no live-network tests in the default suite (one opt-in
`@pytest.mark.network` per fetch source, listing-only, excluded from CI); no new tests for
`UnifiedDataIndex`/`HPCClient` internals (unchanged by this refactor — testing them properly is a separate
project); no full-panel end-to-end run in tests (that is the step-9 operational gate).

## 12. Dead code this redesign retires

| Item | Location | Lines |
|---|---|---|
| `handle_validate_task` (dead: no preprocessor has the attributes it checks) | `preprocess/workflow.py` | ~180 |
| `gcs_upload_path` (abstractmethod, 8 implementations, called nowhere) + `common/gcs/` | across download sources | ~60 |
| `create_source()` glue (obsolete by construction once sources own their whole lifecycle) | `preprocess/sources/factory.py` | ~35 |
| `_strip_remote_prefix` (11 copies → 1 shared helper) | across sources | ~40 |
| Duplicate `get_hpc_output_path`/`from_config` | `berman_mining.py` | ~15 |
| Duplicate `execute_command`/`rsync_transfer` | `hpc/client.py` | ~60 |
| `geobox_patch.py` (byte-identical vendored odc-geo copy, not imported anywhere — flagged in [`05-migration.md`](05-migration.md), unfixed until now) | `common/geobox/geobox_patch.py` | 1,607 |
| Dead `demean_columns`/`assemble demean` CLI+config wiring | `data.yaml`, CLI | — |

## 13. Verification

1. **Per-source migration gate**: `data plan --source X --step Y` diffed against the step-0
   characterization snapshot; one real target run through both old and new code, output diffed with a new
   `scripts/compare_step_output.py` (exact dims/dtypes/attrs, `assert_allclose` on values).
2. **Full test suite** green at every commit checkpoint.
3. **Generator drift check**: `generate_slurm_scripts.py --check` clean, including its new
   source/step/config validation — the regression test for the exact class of bug found during this
   redesign (§1).
4. **End-to-end validation gate before cutover**: one target per archetype, full `fetch→prepare→grid` run
   under the new code, compared byte-for-byte-equivalent to the existing on-disk artefact from the old code.
5. **CLI smoke test**: `run.py data list` enumerates all sources with correct steps/requires;
   `run.py data run --source acag` with no `--step` errors clearly rather than silently defaulting.

## 14. Open items

- **Step-9 hard gate: passed, cutover (step 10) complete.** As run for real on SLURM via
  `orchestration/slurm/validate-hard-gate-*.sh`: bulk-composite (`acag`), vector (`gadm`, `osm`), and point
  (`berman_mining`, `plad`, `snl_mining`) archetypes all have a real old-vs-new execution diff, all
  EQUIVALENT. Full test suite green and `generate_slurm_scripts.py --check` clean (23 jobs, no drift). Two
  archetype representatives were waived rather than execution-diffed, both by explicit decision rather than
  left silently unresolved:
  - **`country_classifications` (tabular-join archetype, its only representative).**
    `validate-hard-gate-country_classifications.sh`, run for real on SLURM, hit a genuine pre-existing OLD
    bug: `src/data/preprocess/sources/misc.py:645`'s
    `hdi.loc[:, "year"] = hdi["year"].str[4:].astype(int)` raised `TypeError` under this HPC's pandas
    (pyarrow-string-backed, ≥3.0) — OLD could not execute this step at all here, so no execution-based diff
    was possible without a throwaway legacy-pandas environment. Waived: the code-level comparison already in
    that script's header comment (OLD's broken in-place `.loc` cast vs. NEW's
    `src/data/sources/misc/hdi.py`, which already used plain bracket assignment with its own comment
    explaining the same incompatibility) was accepted as sufficient sign-off for this one step.
  - **`modis` (streaming archetype).** `validate-hard-gate-modis.sh` diffed GRID EQUIVALENT but never diffed
    PREPARE — it streams from Microsoft Planetary Computer's STAC API, which needs live internet access
    neither the sandbox nor (per the DuckDB extension-download failure that motivated
    `bootstrap_duckdb_extensions.sh`) SLURM compute nodes generally have. Waived: PREPARE was never exercised
    against OLD either (OLD has no equivalent streaming codepath for MODIS to diff against in the first
    place — MODIS's `STEPS = (PREPARE, GRID)` was new in this redesign, per §5's per-source mapping), so
    there was no OLD reference for this step to be equivalent *to*; GRID's diff plus code review stand as
    this archetype's sign-off.
  - Both waivers are conditions of the migration's own gate — declared here, not merely defaulted into by an
    inability to run OLD.
  - Cutover performed: `src/data/download/`, `src/data/preprocess/`, `src/cli/download/`,
    `src/cli/preprocess/`, `src/data/common/geobox/geobox_patch.py` (1,607 dead lines),
    `src/data/common/gcs/` (dead, unimported), `tests/data/preprocess/` (characterization tests against the
    now-deleted OLD code), and `orchestration/slurm/demean_modis.sh` (already flagged dead in
    `orchestration/slurm/jobs.yaml`) are all removed. The dead `assemble demean` CLI+config wiring (§12) is
    removed from `src/cli/assemble/{commands,handlers}.py`, `src/cli/main.py`, and `run.py`. Post-cutover:
    full test suite still green (same pre-existing, unrelated `test_summary_qos.py` failure), generator
    `--check` still clean, `run.py data list`/`--help`/`assemble --help` smoke-tested clean. The
    `validate-hard-gate-*.sh` pilot scripts are kept as-is (audit trail of how the gate was verified) even
    though their OLD-side invocations can no longer run post-cutover.
- The per-*variable* (not per-source) resampling override [`04-ingest.md`](04-ingest.md) §1 flags as
  necessary (lights need area-weighted sum, land cover needs nearest) — still unresolved; this redesign
  threads the hook through `common/raster/spatial.py` but does not retune values.
- GLASS's naive daily→annual mean (§5) — fixed in a follow-on commit, not this one.
- Whether `country_classifications` should later become four sources (§7's escape hatch).
- ~~The `layout: v2` physical directory rename (§3) — deferred, with its consumer checklist recorded.~~
  Implemented (§3's "Update" note); still opt-in, `assemble:`'s stage_3 blocks deliberately excluded
  (found disconnected from current `src/data/assemble/*.py` code, likely dead config).
- Job-dependency chaining (`--dependency=afterok`) — `REQUIRES` now makes the dependency graph derivable,
  but nothing yet emits SLURM dependency flags from it; a natural next step, not done here.
- A `run.py neighbourhood run` CLI verb — `src/data/common/neighbourhood/` has none today (§2).
