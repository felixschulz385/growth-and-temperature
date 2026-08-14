# 11. GLASS: static per-target FETCH (no crawl, no entrypoints)

**Status: implemented** (`src/data/sources/glass/source.py`,
`orchestration/configs/data.yaml`, `tests/data/sources/glass/
test_glass_fetch.py`) -- `_CrawlerMixin`/`crawler.py` deleted, both
`glass_modis`/`glass_avhrr` configs updated per §6.

Replaces GLASS's `RemoteFileCatalog`/entrypoint-crawl FETCH model
(`src/data/sources/glass/crawler.py::_CrawlerMixin`, `has_entrypoints=True`)
with a MODIS-shaped static target list (`src/data/sources/modis/source.py`'s
`_plan_fetch()`/`_execute_fetch()` pattern): every (year, day[, tile])
target is enumerable from config alone, so FETCH attempts each target
directly and logs the outcome, instead of crawling a directory tree first to
discover a required-files list and caching that discovery separately from
the download.

## 1. Problem with the current model

- **Two-phase, disk-cached discovery** (`catalog.required_files()` ->
  `_CrawlerMixin.get_all_entrypoints()` -> per-entrypoint
  `list_remote_files(entrypoint)`, cached to
  `_status/entrypoints/<key>.json`) exists only to avoid re-crawling an
  expensive remote listing across `data fetch` runs. For GLASS this crawl
  really is expensive (thousands of per-day HTTP requests), so the caching
  itself is justified -- but modeling "discover" and "download" as two
  separate steps adds an entrypoint-cache layer, entrypoint-level status
  sidecars, and (this session) a parallel `cached_entrypoint_counts()` /
  `STATIC_ENTRYPOINTS` path in `data summary` just to paper over the fact
  that discovery hasn't happened yet.
- **A real bug surfaces once `get_all_entrypoints()` is made static**
  (done this session, `src/data/sources/glass/source.py`): it generates a
  `{"year": y, "day": d}` entrypoint for every calendar day of every
  configured year, for *both* variants uniformly. `_CrawlerMixin.crawl()`
  only applies the `day` filter when it recurses into a matching
  **subfolder** href -- a **file** href found directly inside an
  already-matched year directory is yielded unconditionally, with no day
  check at all. If AVHRR's remote layout really is `<year>/<file>` (no day
  subdirectory -- see §3 below), every one of the ~365 per-day entrypoint
  crawls for a given year re-lists and re-yields that *entire* year
  directory, an unintended ~365x overcrawl per year. This plan removes the
  crawl-per-entrypoint model entirely, so the bug becomes moot rather than
  needing a standalone fix.
- **Unpredictable filename suffix remains unavoidable**: every example URL
  below carries a processing-date suffix (`...2022021.hdf`,
  `...2021259.hdf`, `...2021250.hdf`) that differs per file and cannot be
  derived from `(year, day, tile)` alone. Some live lookup per target (or
  per shared listing) is unavoidable -- the redesign's job is to stop
  persisting that lookup as a separate cached "entrypoint," not to
  eliminate it.

## 2. What "static targets, attempt-and-log" means

The (year, day[, tile]) space itself needs no discovery -- it is exactly
what the user has already specified:

- **GLASS MODIS**: day 055 of 2000 through day 365 of 2020, land tiles only.
- **GLASS AVHRR**: day 001 of 1992 through day 365 of 2020.

Both bounds are **not** calendar-year-aligned (MODIS starts mid-2000;
neither end lands on day 366 despite 2020 being a leap year), so the target
generator needs an explicit `(start_year, start_day) -> (end_year, end_day)`
walk, not a `year_range` cross `range(1, 366)` -- interior years use their
own real leap-aware day count, but the first and last year are clipped to
the given start/end day.

```python
def daterange_doy(start: tuple[int, int], end: tuple[int, int]) -> Iterator[tuple[int, int]]:
    """(year, day) pairs from start through end inclusive, leap-aware."""
    (y0, d0), (y1, d1) = start, end
    for year in range(y0, y1 + 1):
        days_in_year = 366 if calendar.isleap(year) else 365
        first_day = d0 if year == y0 else 1
        last_day = d1 if year == y1 else days_in_year
        for day in range(first_day, last_day + 1):
            yield year, day
```

**Confirmed**: `glass_avhrr`'s true range is **1992-2020** (day 001 of 1992
through day 365 of 2020), not the `[1982, 2021]` currently in
`orchestration/configs/data.yaml` -- that config value is wrong and must be
updated to `1992-2020` as part of this change (§6 step 2).

## 3. AVHRR directory layout: confirmed, no day subdirectory

`RAW_LISTING_DEPTH = 3` (`glass/source.py`) and its comment
(`"<year>/<day>/<file>"`) currently assume **both** variants nest files
under a day subdirectory. **Confirmed this is wrong for AVHRR**: it has no
day subdirectory -- files sit directly under `<year>/`:

```
https://glass.hku.hk/archive/LST/AVHRR/0.05D/1992/GLASS08B31.V40.A1992001.2021259.hdf
https://glass.hku.hk/archive/LST/AVHRR/0.05D/2020/GLASS08B31.V40.A2020366.2021250.hdf
```

MODIS keeps its day subdirectory
(`.../MODIS/Daily/1KM/2000/055/GLASS06A01...`). So the two variants take
different listing granularities (§4 step 2):

- **MODIS**: one listing GET per `(year, day)` -- the day directory --
  returns every land tile's file for that day.
- **AVHRR**: one listing GET per `year` -- the year directory -- returns
  every day's file for that year (~365 entries in one page). `RAW_LISTING_
  DEPTH` for AVHRR is therefore **2** (`<year>/<file>`), not 3 -- this
  constant needs to become per-variant rather than one shared class
  attribute (`GlassSource` already branches on `self.data_source_kind`
  elsewhere for exactly this kind of MODIS/AVHRR divergence).

## 4. Design

### 4.1 Land tile filter (MODIS variant only)

Reuse the existing MODIS sinusoidal tile-grid machinery
(`src/data/sources/modis/tiles.py::get_modis_sinusoidal_tiles`) rather than
computing a second land-tile allowlist: GLASS MODIS's `hXXvYY` tile ids are
the same sinusoidal grid `modis:`'s config already resolved via
`scripts/compute_modis_land_tiles.py` (`orchestration/configs/data.yaml`'s
282-tile `land_tiles: [...]` list under the `modis:` block). Point
`glass_modis:`'s own `land_tiles` at the same list (factor it into one
shared YAML value or Python constant both configs reference, so the two
never drift) instead of recomputing.

**Open question**: GLASS's own LST algorithm may not have identical
per-tile land/water coverage to MOD11/MOD21 (different sensor, different
processing chain) -- reusing `modis:`'s list is the cheap starting
assumption, not a verified match. If GLASS FETCH later shows systematic
gaps the land filter would explain, recompute
(`scripts/compute_modis_land_tiles.py` already takes a `land_polygons_path`
argument -- no new tooling needed, just a second invocation/config value if
the two ever need to diverge).

### 4.2 `_plan_fetch()` / `_execute_fetch()` (mirrors `ModisSource`)

`GlassSource` drops `RemoteFileCatalog` conformance for FETCH entirely
(same posture `ModisSource`'s own header docstring states: "does **not**
implement the full `RemoteFileCatalog` crawler protocol") and gains its own
`_plan_fetch()`/`_execute_fetch()`, dispatched from `plan()`/`execute()`
exactly like `ModisSource` already does.

```python
def _plan_fetch(self, selection: TargetSelection) -> List[StepTarget]:
    targets = []
    for year, day in daterange_doy(self.day_range_start, self.day_range_end):
        units = self.tiles if self.data_source_kind == "MODIS" else [None]
        for tile in units:
            key = f"{year}/{day:03d}/{tile}" if tile else f"{year}/{day:03d}"
            if not selection.matches_key(key):
                continue
            targets.append(StepTarget(
                source_id=self.cfg.source_id, step=PipelineStep.FETCH, key=key,
                output_path=self._raw_path(year, day, tile),
                completion=Completion.PATH_EXISTS,
                meta={"year": year, "day": day, "tile": tile},
            ))
    return targets
```

```python
def _execute_fetch(self, target: StepTarget) -> bool:
    status_dir = self.output_root(PipelineStep.FETCH)
    if not self.cfg.override and is_complete(target):
        return True

    year, day, tile = target.meta["year"], target.meta["day"], target.meta["tile"]
    try:
        listing = self._listing_for(year, day)  # memoized per run, see 4.3
    except requests.RequestException as exc:
        manifest.record_failure(status_dir, target.key, f"listing fetch failed: {exc}")
        return False

    match = self._match_in_listing(listing, year, day, tile)
    if match is None:
        # Listing loaded fine but this tile/day genuinely isn't there --
        # a real absence (sensor gap, non-land tile slipped through), not a
        # transient error. No point retrying against a directory that will
        # never populate.
        manifest.record_failure(status_dir, target.key, "not present in remote listing", permanent=True)
        return False

    rel_path, url = match
    try:
        self.download(url, target.output_path)
    except Exception as exc:
        manifest.record_failure(status_dir, target.key, f"download failed: {exc}")
        return False

    manifest.clear_failure(status_dir, target.key)
    return True
```

Failure handling deliberately distinguishes three cases (`manifest.
record_failure`'s existing `permanent` flag, unchanged):

| Outcome | `permanent` | Effect |
|---|---|---|
| listing fetch itself errors (network/5xx) | `False` | retried next run, same as any transient failure |
| listing loads, target not present in it | `True` | immediate `unavailable`, no wasted retries |
| download of a resolved URL fails | `False` | retried next run (could be transient) |

### 4.3 In-run listing memoization (not a disk cache)

One listing routinely resolves many sibling targets -- every land tile
sharing a MODIS day, or every day sharing an AVHRR year (§3). Re-fetching
that listing once per sibling target would multiply requests by ~280
(MODIS) or ~365 (AVHRR) for no reason. Memoize the parsed listing in a
plain in-memory dict for the lifetime of one `run_fetch()`/execute loop,
keyed by `(year, day)` for MODIS or `year` for AVHRR -- **not** written to
`_status/entrypoints/` or any other disk cache. This is the concrete
difference from the old model: the listing is a same-run implementation
detail of "attempt this target," not a cross-run persisted discovery
artifact.

```python
def _listing_for(self, year: int, day: int) -> dict[str, str]:
    cache_key = (year, day) if self.data_source_kind == "MODIS" else year
    if cache_key not in self._listing_cache:
        url = self._listing_url(year, day)  # day dir (MODIS) or year dir (AVHRR), per §3
        self._listing_cache[cache_key] = self._list_single_directory(url)
    return self._listing_cache[cache_key]
```

(`_list_single_directory` -- a plain non-recursive "GET one directory index,
parse hrefs" helper -- already exists in spirit as part of
`_CrawlerMixin.crawl()`; factor the single-page GET+parse out of that
recursive function rather than duplicating it. EOG's `_CrawlerMixin.
_list_single_directory()`, referenced by `eog/source.py`'s own docstring,
is precedent for exactly this shape already existing elsewhere in the
codebase.)

### 4.4 `data summary` comes along for free

Once `plan(FETCH)` returns real per-`(year,day[,tile])` `StepTarget`s,
`src/cli/data/handlers.py`'s existing routing already prefers that path
over the `RemoteFileCatalog` branch (`handlers.py:369-378`,
`_summarize_fetch_targets()` -- the same function MODIS already uses). No
GLASS-specific summary code is needed; deleting `has_entrypoints`/
`STATIC_ENTRYPOINTS` from `GlassSource` is what routes it there.

## 5. What gets deleted

- `GlassSource.has_entrypoints`, `STATIC_ENTRYPOINTS`, and its static
  `get_all_entrypoints()` (all added/changed this session) -- superseded.
- `GlassSource.filename_to_entrypoint()` stays (still useful for mapping a
  downloaded file back to its target key) but is no longer part of a
  `RemoteFileCatalog` contract GLASS no longer implements.
- `_CrawlerMixin`'s recursive `crawl()` and its entrypoint-based subfolder
  filtering, replaced by the single-directory listing helper (§4.3) --
  confirm no other source still depends on `_CrawlerMixin` before deleting
  it outright (currently GLASS-only, per `crawler.py`'s own header comment).
- GLASS's row, if any, in `tests/data/sources/test_fetch_protocol.py`'s
  `RemoteFileCatalog` parametrization -- mirrors MODIS's existing exclusion
  (`modis/source.py`'s header docstring already explains why MODIS is
  excluded from that same test).

## 6. Rollout

1. Update `orchestration/configs/data.yaml`'s `glass_avhrr.year_range` from
   `[1982, 2021]` to `[1992, 2020]` (§2), and split `RAW_LISTING_DEPTH` into
   a per-variant value (§3: 3 for MODIS, 2 for AVHRR).
2. Add `daterange_doy()` and the day-range config keys.
3. Port `_plan_fetch()`/`_execute_fetch()` per §4.
4. Delete the old crawler-based entrypoint path (§5).
5. Rewrite `tests/data/sources/glass/test_glass_plan.py` for the new FETCH
   shape (target enumeration, permanent-vs-retryable failure branching,
   listing memoization) -- the existing PREPARE-side tests
   (`test_glass_plan.py`'s annual-aggregation/reprojection coverage) are
   unaffected, since PREPARE still reads FETCH's raw output off local disk
   exactly as before.

Already-downloaded files on disk are unaffected by this migration --
`Completion.PATH_EXISTS`-based targets recognize them as complete on the
first run under the new code, same file layout, no data movement. There is
no entrypoint cache worth migrating (this session's GLASS entrypoint
sidecars are disposable; filesystem/status-sidecar truth over a stateful
ledger is already this project's established direction, see the DuckDB
ledger removal work preceding this doc).

## 7. Explicitly out of scope

- ESACCI/ACAG/ntl_harm/EOG's entrypoint status -- separate "decomplicate"
  thread, on hold (not started).
- PLAD / `commodity_prices`'s `prices_path` -- separate thread, not
  addressed here.
