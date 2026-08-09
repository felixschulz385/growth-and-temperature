# 07b — MODIS Outstanding Items

Live checklist for what's left on the MODIS ingest pipeline. Pulled from
[`06-open-questions.md`](06-open-questions.md)'s consolidated list (cross-referenced by number
below) plus items surfaced during later implementation. Update in place as items get resolved —
this doc tracks current status, unlike `06-open-questions.md`'s session-scoped writeup.

## Correctness-critical

Should resolve before trusting a production-scale ingest run; each of these silently corrupts
output if the current assumption is wrong.

- [x] **QC bit layout — fully resolved for both MOD11A1 and MOD21A2 (2026-08-09).**
      `decode_qc_valid_mask` ([`tiles.py`](../../src/data/sources/modis/tiles.py)) now takes a
      `product` argument selecting a per-product bit-value-to-error-K mapping. MOD11A1 confirmed
      against the MOD11 V6.1 Users' Guide (Wan, 2019) Table 13. MOD21A2 confirmed against the
      *correct* primary source, the MxD21 LST&E User Guide (Hulley et al., JPL, March 2019) Table
      12 — and this caught a real, previously-unknown bug: MOD21's LST-accuracy bits sit at the
      same position as MOD11's (bits 7&6) but mean the **opposite** thing (MOD11: increasing value
      = worse; MOD21: increasing value = better). Applying MOD11's mapping uniformly, as the code
      did before Table 12 was checked, would have silently inverted the quality filter on every
      pixel of the *primary* `21A2` product. Pinned by `tests/data/sources/modis/test_modis_qc.py`.
      Also caught, same document, Table 11: `BAND_SPECS["21A2"]`'s `view_angle`/`view_time` had a
      wrong offset/unmasked fill, the same bug class already fixed for `"11A1"` below.
      (`06-open-questions.md` #9)
- [x] **Platform authority — resolved (2026-08-09).** Queried 600 real STAC items directly (3
      collections x 5 regions x 4 years, both platforms represented): zero disagreements between
      `properties.platform` and the `MOD`/`MYD` id prefix. `_search_items`'s disagreement warning
      ([`source.py`](../../src/data/sources/modis/source.py)) is kept as a tripwire, not because a
      mismatch is expected. (`06-open-questions.md` #8)
- [x] **`odc.stac.load` scale/offset auto-apply — resolved (2026-08-09).** Loaded a real
      `modis-21A2-061` item with `odc-stac` 0.5.3 (same default kwargs `_load_tile_year` uses) and
      compared against a raw `rasterio` read of the same asset: values matched exactly, `uint16`,
      unscaled. `odc.stac.load()` does NOT auto-apply STAC scale/offset — the manual
      `raw * scale + offset` in `_load_tile_year`
      ([`source.py`](../../src/data/sources/modis/source.py):322-331) is required, not a
      double-application bug. (`06-open-questions.md` #11)
- [x] **MOD11 `Emis_31`/`Emis_32` offset — resolved (2026-08-09).** Confirmed `0.49` (matching
      21A2) via the same MOD11 V6.1 Users' Guide, Table 9. This caught a real bug — `BAND_SPECS`
      ([`source.py`](../../src/data/sources/modis/source.py)) had `offset: 0.0` hardcoded for
      `modis_robustness_11a1`'s emissivity bands, plus two adjacent wrong values the same table
      exposed (`Night_view_angl`'s fill/offset, `Night_view_time`'s fill). All corrected, pinned by
      `tests/data/sources/modis/test_modis_qc.py`. (`06-open-questions.md` #10)

## Completeness / scope

- [ ] **Land-tile allowlist: tooled, not yet run.** `compute_land_tiles()`
      ([`tiles.py`](../../src/data/sources/modis/tiles.py)) and
      `scripts/compute_modis_land_tiles.py` exist (added 2026-08-07), but `data.yaml`'s `modis`
      block still has no `land_tiles:` populated — needs the `osm` source's FETCH+PREPARE to have
      produced `land_polygons_simplified.gpkg` first, then the script run against it. Resolves the
      ~317-land-tile figure `06-open-questions.md` #12 flags as unverified.
- [x] **`modis_robustness_11a1`'s tiles/years — resolved (2026-08-09).** Replaced the 3 placeholder
      tiles with 5 chosen by forward-projecting one representative lon/lat per biome through the
      sinusoidal grid's own formulas (not guessed): `h12v09` (Amazon rainforest), `h18v06` (Sahara
      desert), `h18v04` (Central Europe temperate — same as the old placeholder), `h22v03` (Siberian
      boreal/taiga, near the clip edge), `h30v11` (Australian outback, southern hemisphere). Replaced
      `year_range: [2003, 2024]` (the *full* range, contradicting "not a full parallel backfill")
      with an explicit `years: [2004, 2014, 2023]` (early/mid/late mission) — required adding a new
      `years` config override to `ModisSource` ([`source.py`](../../src/data/sources/modis/source.py))
      since `year_range` can only express one contiguous span. Pinned by
      `tests/data/sources/modis/test_modis_plan.py`.
- [x] **Aqua-only start date — resolved (2026-08-09).** Queried Planetary Computer directly for
      every `MYD`-prefixed item in the 2002-05-04..2002-08-01 window (1037/632/1049 items for
      21A2/11A1/11A2) and took the true minimum by id-derived acquisition date. Real earliest
      granule: `modis-21A2-061` 2002-07-04, `modis-11A2-061` 2002-07-04, `modis-11A1-061`
      2002-07-28 — all ~2 months post-launch (commissioning). `data.yaml`'s `year_range: [2002,
      2025]` already safely covers this (2002 is just a partial year, handled gracefully), so no
      config change was needed. (`06-open-questions.md` #13)

## Operational

- [ ] **Real HPC transfer throughput/manifest schema unmeasured.** The `modis-fetch` push step's
      real-world timing for ~6,700 tile-year composites over the scicore transfer node has never
      been measured against a real batch. (`06-open-questions.md` #15)

## Unrelated, noticed in passing

- [x] **`glass-modis-preprocess-{annual,spatial}.sh` — resolved (2026-08-09), already fixed by a
      prior migration.** Those two scripts no longer exist; superseded by
      `orchestration/slurm/glass-modis-prepare.sh`/`glass-modis-grid.sh`, generated from `jobs.yaml`
      via `generate_slurm_scripts.py`, which correctly invoke `pipeline run --source glass_modis
      --step prepare`/`--step grid` (verified the subcommand registers and resolves). The old
      `--stage`-flag text survives only in `validate-hard-gate-modis.sh`'s comparison-log echo
      strings, documenting the prior invocation for a before/after diff, not a live script.
      (`06-open-questions.md` #14)
