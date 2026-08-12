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

- [x] **Land-tile allowlist — resolved (2026-08-11).** Ran `compute_modis_land_tiles.py` against
      GADM's raw ADM-hierarchy polygons (`data/misc/gadm/raw/gadm_410-gpkg/gadm_410.gpkg`, already
      present locally — 356,508 features covering every admin level down to the finest available
      per country) rather than waiting on the `osm` source's FETCH+PREPARE (which needs a ~920MB
      download plus a simplify pass neither has run yet); the script's own docstring says any
      land-distinguishing polygon layer works, not specifically OSM's. Found **282** land-covering
      tiles (of 648 within the 60-deg lat clip) — populated into `data.yaml`'s `modis` block's
      `land_tiles:` (the `modis_robustness_11a1` block uses an explicit `tiles:` override, so the
      allowlist doesn't apply there). Confirms the true figure is materially below the ~317
      `06-open-questions.md` #12 flagged as unverified. `tests/data/sources/modis/` (35 tests)
      still pass unaffected, since they exercise `compute_land_tiles()` against synthetic fixtures,
      not this config value.
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

- [x] **Real HPC transfer throughput — measured (2026-08-11); manifest schema — already resolved,
      not a separate open item.** Two parts to the original item:
      - **Manifest schema**: [`08-hpc-transfer.md`](08-hpc-transfer.md) §6 left open whether
        `UnifiedDataIndex` needs a schema addition for transfer-manifest use. Moot — the actual push
        path (`src/data/common/hpc/push.py`'s `HPCPusher`, per its module docstring "replacing both
        FETCH's inline tar/rsync/extract/verify ... and the separate `data transfer` path")
        superseded that whole design; MODIS FETCH tracks each (year, tile) unit's local/remote
        state directly in the generic ledger `artifacts` table (`ModisSource`'s module docstring,
        `source.py`), same as every other FETCH source. No new schema, no separate manifest format.
      - **Throughput**: ran a real push against the actual scicore transfer node
        (`transfer12.scicore.unibas.ch`, the `remote.ssh_target` this repo's sources already use)
        via the real `HPCPusher.push_batched()` — not a mock. 20 synthetic files, 40MB each (800MB
        total), sized to approximate a compressed float32 annual-composite GeoTIFF
        (`_write_annual_geotiff`, `source.py` — ~32 bands x 2400x2400 px, deflate-compressed;
        high-entropy synthetic bytes stand in for already-compressed real payload, since no
        production MODIS output exists locally to sample yet). Result: **800MB in 62.8s ≈ 12.7
        MB/s**, all 20 units verified present remotely, batch tar/rsync/extract/cleanup succeeded
        end-to-end, remote scratch dir removed after. At that rate, ~6,700 tile-year units of this
        size (~268GB) would take **roughly 5.9 hours single-threaded** — `push_units_concurrent`
        (thread-pooled) would cut this proportionally to worker count, not measured here.
        **Caveat, don't treat 12.7 MB/s as a hard production number**: this ingest host lacks
        an `rsync` binary, so `HPCClient.rsync_transfer` silently fell back to its PowerShell/`scp`
        path — the actual scicore-ingest-host codepath (which has `rsync`) wasn't exercised, and
        `scp` has no delta/compression benefit `rsync -z` would provide. Real annual-composite file
        sizes are also still an estimate (no production run has produced one yet to sample) rather
        than a measured distribution. Re-run this benchmark from wherever stage "annual" actually
        runs, with real `_write_annual_geotiff` output, before trusting the 5.9h figure for capacity
        planning. (`06-open-questions.md` #15)

## Unrelated, noticed in passing

- [x] **`glass-modis-preprocess-{annual,spatial}.sh` — resolved (2026-08-09), already fixed by a
      prior migration.** Those two scripts no longer exist; superseded by
      `orchestration/slurm/glass-modis-prepare.sh`/`glass-modis-grid.sh`, generated from `jobs.yaml`
      via `generate_slurm_scripts.py`, which correctly invoke `data run --source glass_modis
      --step prepare`/`--step grid` (verified the subcommand registers and resolves). The old
      `--stage`-flag text survives only in `validate-hard-gate-modis.sh`'s comparison-log echo
      strings, documenting the prior invocation for a before/after diff, not a live script.
      (`06-open-questions.md` #14)
