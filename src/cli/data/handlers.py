"""Handler functions for the ``data`` domain."""

from __future__ import annotations

import argparse
import dataclasses
import logging

from src.cli.common import setup_logging
from src.cli.config import load_config_with_env_vars
from src.data.pipeline.config import build_context, get_source_config
from src.data.sources import registry
from src.data.sources.steps import (
    STEP_ORDER,
    Completion,
    MissingPrerequisiteError,
    PipelineStep,
    TargetSelection,
    is_complete,
)

logger = logging.getLogger(__name__)


def handle_list(args: argparse.Namespace) -> None:
    """``data list`` -- enumerate every registered source."""
    setup_logging(args.log_level, debug=args.debug)
    for spec in sorted(registry.all_specs(), key=lambda s: s.id):
        steps = ", ".join(s.value for s in spec.steps)
        aliases = f" (aliases: {', '.join(spec.aliases)})" if spec.aliases else ""
        requires = (
            "; requires " + ", ".join(f"{my_step.value}<-{rid}:{rstep.value}" for my_step, rid, rstep in spec.requires)
            if spec.requires
            else ""
        )
        print(f"{spec.id}{aliases} -- steps: {steps}{requires}")


def _summarize_targets(targets: list) -> tuple[str, "bool | None"]:
    """One-line status for a (source, step)'s target list, plus whether that
    step is fully complete (`True`/`False`), or `None` if completeness isn't
    a meaningful concept for this step (the FETCH sync-missing pseudo-target
    below, which is `Completion.NEVER` by design and never "completes").

    FETCH steps (docs/design/09-integrated-pipeline.md §4) are universally
    modeled as a single `Completion.NEVER` "sync whatever is missing" pseudo-
    target -- `is_complete()` on it is always False by design, which would
    make every source look permanently 0% fetched. For that shape, report
    what's actually on disk (file count under the step's output root)
    instead of a meaningless complete/total ratio; everything else (PREPARE/
    GRID, and any FETCH step that deviates from the pseudo-target pattern)
    uses the normal is_complete() count.
    """
    if not targets:
        return "no targets", True  # nothing to do -> vacuously complete
    if len(targets) == 1 and targets[0].completion is Completion.NEVER:
        import os

        output_path = targets[0].output_path
        if os.path.isfile(output_path):
            count = 1
        elif os.path.isdir(output_path):
            count = sum(len(files) for _, _, files in os.walk(output_path))
        else:
            count = 0
        return ("no local data" if count == 0 else f"{count} file(s) fetched"), None
    complete = sum(1 for t in targets if is_complete(t))
    total = len(targets)
    pct = round(100 * complete / total) if total else 0
    return f"{complete}/{total} ({pct}%)", complete == total


def _print_source_summary(rows: dict) -> None:
    if not rows:
        print("No sources found.")
        return
    headers = ["source", *(step.value for step in STEP_ORDER), "verified"]
    widths = [max(len(headers[0]), *(len(name) for name in rows))]
    for i, step in enumerate(STEP_ORDER, start=1):
        widths.append(max(len(headers[i]), *(len(row[step.value]) for row in rows.values())))
    widths.append(max(len(headers[-1]), *(len(row["verified"]) for row in rows.values())))

    def fmt(cells: list) -> str:
        return "  ".join(cell.ljust(w) for cell, w in zip(cells, widths))

    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for name in sorted(rows):
        row = rows[name]
        print(fmt([name, *(row[step.value] for step in STEP_ORDER), row["verified"]]))


def _summarize_fetch(source, *, detailed: bool) -> str:
    """FETCH's own complete/outstanding/unavailable bucket counts
    (`src.data.common.fetch.manifest.plan_fetch`), for any
    `RemoteFileCatalog`-shaped source -- always reflects live disk state.
    `detailed=True` additionally splits `outstanding` into
    never-attempted vs. currently retrying, by peeking at each unit's status
    sidecar (`src.data.common.statusfile`)."""
    from src.data.common import statusfile
    from src.data.common.fetch import catalog, manifest
    from src.data.sources import layout

    raw_root = layout.raw_root(
        source.ctx.data_root, source.cfg.data_path, namespace=source.cfg.namespace, layout=source.ctx.layout
    )
    required = catalog.required_files(source, raw_root)
    if not required:
        return "no required files declared"

    listing = manifest.snapshot_local_listing(raw_root)
    plan = manifest.plan_fetch(required, listing, raw_root)
    base = f"{len(plan.complete)} complete, {len(plan.outstanding)} outstanding, {len(plan.unavailable)} unavailable"
    if not detailed:
        return base

    never_attempted = retrying = 0
    for req in plan.outstanding:
        status = statusfile.read(statusfile.status_path(raw_root, req.unit_id))
        if status and status.get("attempts"):
            retrying += 1
        else:
            never_attempted += 1
    return f"{base} (outstanding: {never_attempted} never attempted, {retrying} retrying)"


def _summarize_by_tile(source, target) -> "str | None":
    """Per-(tile, year) unit status breakdown for one `run_tiled_prepare()`-
    shaped PREPARE target -- `None` for anything else (a non-tiled PREPARE
    output, e.g. gadm's own vector-simplify-then-rasterize target doesn't
    look like this), so callers fall back to the normal collapsed summary.

    Recognized shape: `Completion.MARKER` (a tiled zarr, not a whole-file
    output) plus `target.meta["years"]` (every tiled source's `_plan_prepare()`
    persists this the same way -- see e.g. acag.py) and a `tile_size`
    instance attribute (every tiled source sets one, `cfg.raw.get("tile_size",
    tiling.DEFAULT_TILE_SIZE)`). Reads `src.data.common.prepare.driver
    .prepare_status()`'s per-unit status sidecars directly -- no run
    triggered."""
    if target.completion is not Completion.MARKER:
        return None
    years = target.meta.get("years")
    tile_size = getattr(source, "tile_size", None)
    if not years or tile_size is None:
        return None

    from src.data.common.geobox import get_target_geobox
    from src.data.common.prepare.driver import prepare_status

    geobox = get_target_geobox(source.ctx)
    counts = prepare_status(target.output_path, years, geobox, tile_size=tile_size)
    total = sum(counts.values())
    return f"{counts['complete']}/{total} complete, {counts['outstanding']} outstanding, {counts['unavailable']} unavailable"


def handle_summary(args: argparse.Namespace) -> None:
    """``data summary`` -- concise per-source, per-step data-availability
    overview. Builds each source directly (bypassing `_check_requires`, unlike
    `_build()`) since a summary should still show a source's own available
    steps even when an upstream REQUIRES dependency isn't complete yet."""
    from src.data.sources.base import RemoteFileCatalog

    setup_logging(args.log_level, debug=args.debug)
    config = load_config_with_env_vars(args.config)
    ctx = build_context(config)
    sources_cfg = config.get("sources", {}) or {}

    if getattr(args, "source", None):
        if args.source not in sources_cfg:
            raise KeyError(f"Source '{args.source}' not found in configuration. Available: {sorted(sources_cfg)}")
        names = [args.source]
    else:
        names = sorted(sources_cfg)

    rows: dict = {}
    for name in names:
        row = {step.value: "-" for step in STEP_ORDER}
        try:
            spec = registry.resolve(name)
            cfg = get_source_config(config, name)
            source = registry.create(name, ctx, cfg)
        except Exception as exc:
            rows[name] = {**{step.value: f"error: {exc}" for step in STEP_ORDER}, "verified": "error"}
            continue

        # Which of this source's own steps produces its final, verifiable
        # output: GRID if declared (a handful of sources still using it,
        # e.g. MODIS), else PREPARE, since PREPARE produces the final output
        # for every other source.
        if PipelineStep.GRID in spec.steps:
            final_step = PipelineStep.GRID
        elif PipelineStep.PREPARE in spec.steps:
            final_step = PipelineStep.PREPARE
        else:
            final_step = None

        had_error = False
        final_step_targets: list = []
        for step in STEP_ORDER:
            if step not in spec.steps:
                continue
            try:
                if step is PipelineStep.FETCH and hasattr(source, "verify_fetch"):
                    # ConfiguredFilesFetchMixin sources (osm/gadm/
                    # country_classifications/commodity_prices) fetch a
                    # small, fixed list of named files -- report exactly
                    # which are missing/mismatched instead of the bucket
                    # counts below, which can't tell "N files fetched" from
                    # "N files fetched under the wrong names."
                    result = source.verify_fetch()
                    row[step.value] = result.detail
                    continue
                if step is PipelineStep.FETCH and isinstance(source, RemoteFileCatalog):
                    row[step.value] = _summarize_fetch(source, detailed=getattr(args, "detailed", False))
                    continue
                targets = source.plan(step, TargetSelection())
                if step is PipelineStep.PREPARE and getattr(args, "by_tile", False):
                    # Per-target tile breakdown for whichever targets are
                    # tile-shaped (run_tiled_prepare()'s Completion.MARKER +
                    # meta["years"] + tile_size, see _summarize_by_tile()) --
                    # any non-tiled target in the same list (e.g. ecoregions'
                    # gadm_gid3_dominant sidecar) falls back to its own
                    # collapsed complete/total summary instead.
                    by_tile_parts = []
                    for t in targets:
                        detail = _summarize_by_tile(source, t)
                        if detail is None:
                            detail, _complete = _summarize_targets([t])
                        by_tile_parts.append(f"{t.key}: {detail}")
                    row[step.value] = "; ".join(by_tile_parts) if by_tile_parts else "no targets"
                else:
                    summary, _complete = _summarize_targets(targets)
                    row[step.value] = summary
                if step is final_step:
                    final_step_targets = targets
            except Exception as exc:
                row[step.value] = f"error: {exc}"
                had_error = True

        if had_error:
            row["verified"] = "error"
        elif final_step is None:
            row["verified"] = "-"
        elif not final_step_targets:
            row["verified"] = "-"
        else:
            complete_targets = [t for t in final_step_targets if is_complete(t)]
            if not complete_targets:
                row["verified"] = "pending"
            else:
                results = [source.verify_grid(t) for t in complete_targets]
                n_ok = sum(1 for r in results if r.ok)
                row["verified"] = "yes" if n_ok == len(results) else f"FAILED ({n_ok}/{len(results)})"
        source.close()
        rows[name] = row

    _print_source_summary(rows)


def _check_requires(spec: registry.SourceSpec, ctx, config, step: PipelineStep) -> None:
    """Gates only the `REQUIRES` entries scoped to *step* (`spec.requires_for`)
    -- e.g. ecoregions' FETCH runs unblocked even though its GRID entry needs
    gadm, since each `REQUIRES` triple now names which of *this* source's own
    steps it applies to.

    A required step's *actual* output location is whatever its own `plan()`
    says, not a bare `layout.output_root(data_path, step)` guess -- a source
    like gadm writes its PREPARE target's output to what `layout.output_root`
    would compute for GRID, so the two can disagree. Build the required
    source for real and check `is_complete()` against its own planned
    targets, the same ground truth `data summary`/the runner itself use,
    instead of re-deriving a path that can be wrong."""
    for requires_id, requires_step in spec.requires_for(step):
        requires_cfg = get_source_config(config, requires_id)
        requires_source = registry.create(requires_id, ctx, requires_cfg)
        try:
            targets = requires_source.plan(requires_step, TargetSelection())
            if targets and all(is_complete(t) for t in targets):
                continue
            expected = targets[0].output_path if targets else requires_source.output_root(requires_step)
            raise MissingPrerequisiteError(spec.id, requires_id, requires_step, expected)
        finally:
            requires_source.close()


def _selection_from_args(args: argparse.Namespace) -> TargetSelection:
    year_range = tuple(args.years) if getattr(args, "years", None) else None
    keys = tuple(args.keys) if getattr(args, "keys", None) else None
    return TargetSelection(year_range=year_range, keys=keys)


def _build(args: argparse.Namespace, step: PipelineStep):
    config = load_config_with_env_vars(args.config)
    ctx = build_context(config)
    _apply_cli_overrides(ctx, args)
    spec = registry.resolve(args.source)
    # Most aliases (docs/design/09-integrated-pipeline.md §4) are just
    # alternate spellings of the same config block (e.g. "esa_cci" ->
    # sources.esacci), so spec.id is the right lookup key. But a few aliases
    # name a distinct variant with its own config block sharing the same
    # implementation class (sources.eog_viirs/eog_dmsp/eog_dvnl all -> EogSource,
    # sources.modis_robustness_11a1 -> ModisSource) -- there, the config is keyed
    # by the alias itself, not spec.id. Prefer whichever of the two is an actual
    # key in the config so both patterns resolve correctly.
    sources_cfg = config.get("sources", {}) or {}
    config_id = args.source.lower() if args.source.lower() in sources_cfg else spec.id
    cfg = get_source_config(config, config_id)
    if getattr(args, "temp_dir", None):
        cfg = dataclasses.replace(cfg, temp_dir=args.temp_dir)
    _check_requires(spec, ctx, config, step)
    source = registry.create(args.source, ctx, cfg)
    return source, config


def _apply_cli_overrides(ctx, args: argparse.Namespace) -> None:
    """`data run`'s Dask-sizing flags override whatever `PipelineContext`
    built from the config file -- same override relationship the old
    `preprocess run --dask-threads/...` flags had over config, carried
    forward here (docs/design/09-integrated-pipeline.md §8). Other
    subcommands (plan/index/transfer) don't register these flags, so
    `getattr(..., None)` keeps this a no-op for them."""
    if getattr(args, "dask_threads", None) is not None:
        ctx.dask_threads = args.dask_threads
    if getattr(args, "dask_memory_limit", None) is not None:
        ctx.dask_memory_limit = args.dask_memory_limit
    if getattr(args, "dashboard_port", None) is not None:
        ctx.dashboard_port = args.dashboard_port


def handle_plan(args: argparse.Namespace) -> None:
    """``data plan`` -- print targets for (source, step) without running them."""
    setup_logging(args.log_level, debug=args.debug)
    step = PipelineStep(args.step)
    source, _ = _build(args, step)
    selection = _selection_from_args(args)

    targets = source.plan(step, selection)
    if not targets:
        print(f"No targets for source='{args.source}' step='{step.value}'.")
        return

    for target in targets:
        status = "complete" if is_complete(target) else "pending"
        print(f"[{status}] {target.key}  ->  {target.output_path}")
    source.close()


#: `transfer_mode` default for a source not explicitly configured either way
#: (`sources.<id>.transfer_mode: auto|manual` overrides this) -- the
#: high-disk-usage raster sources default to pushing to HPC automatically
#: right after a successful FETCH, everything else defaults to requiring an
#: explicit `data transfer` call.
AUTO_TRANSFER_DEFAULT_SOURCES = frozenset(
    {"modis", "modis_lst", "modis_robustness_11a1", "glass_modis", "glass_avhrr", "acag", "esacci", "ntl_harm", "eog"}
)


def _maybe_auto_transfer(source, step: PipelineStep) -> None:
    """Called after a successful `data run --step fetch` -- pushes whatever
    just landed locally to HPC if this source's `transfer_mode` resolves to
    `"auto"`. Silently skipped (not an error) when no HPC target is
    configured, since plenty of local/dev runs never push anywhere."""
    if step is not PipelineStep.FETCH:
        return
    default_mode = "auto" if getattr(source, "ID", None) in AUTO_TRANSFER_DEFAULT_SOURCES else "manual"
    mode = source.cfg.raw.get("transfer_mode", default_mode)
    if mode != "auto":
        return
    if not source.ctx.ssh_target:
        logger.info(
            "transfer_mode=auto for '%s' but remote.ssh_target is not configured -- skipping auto-transfer",
            getattr(source, "ID", "?"),
        )
        return

    from src.data.common.hpc.client import HPCClient

    client = HPCClient(target=source.ctx.ssh_target, key_file=source.ctx.key_file)
    results = _run_transfer_pass(argparse.Namespace(override=False), source, step, client)
    failed = [r for r in results if not r.ok]
    if failed:
        logger.warning(
            "Auto-transfer after fetch for '%s' had %d failure(s): %s",
            getattr(source, "ID", "?"), len(failed), [(r.unit_id, r.error) for r in failed],
        )
    elif results:
        logger.info("Auto-transfer after fetch for '%s': pushed %d unit(s)", getattr(source, "ID", "?"), len(results))


def handle_run(args: argparse.Namespace) -> None:
    """``data run`` -- execute a (source, step)'s pending targets."""
    setup_logging(args.log_level, debug=args.debug)
    step = PipelineStep(args.step)
    source, _ = _build(args, step)
    if args.override:
        source.cfg = dataclasses.replace(source.cfg, override=True)

    selection = _selection_from_args(args)

    targets = source.plan(step, selection)
    if not targets:
        logger.warning("No targets for source='%s' step='%s'.", args.source, step.value)
        return

    already_complete = {target.key for target in targets if is_complete(target)}

    failures = []
    for target in targets:
        if not source.cfg.override and target.key in already_complete:
            logger.info("Skipping %s -- already complete: %s", target.key, target.output_path)
            continue
        logger.info("Running %s/%s -> %s", args.source, target.key, target.output_path)
        try:
            ok = source.execute(target)
        except Exception:
            # A `False` return from execute() is already handled gracefully
            # here (logged, added to `failures`, loop continues) -- an
            # *exception* wasn't, so any transient per-target failure (a
            # flaky network read, a Planetary Computer signed URL that
            # expired between being issued and a deferred Dask read actually
            # running -- confirmed happening in practice on a real multi-
            # hour MODIS FETCH run) crashed the entire run instead of just
            # that one target. Individual real-world unreliability during a
            # ~6,700-tile-year run is expected, not exceptional; treat it the
            # same as a False return so the run keeps going.
            logger.exception("Target raised an exception: %s/%s", args.source, target.key)
            ok = False
        if not ok:
            logger.error("Target failed: %s/%s", args.source, target.key)
            failures.append(target.key)

    if not failures:
        _maybe_auto_transfer(source, step)
    source.close()

    if failures:
        raise RuntimeError(f"{len(failures)} target(s) failed for source='{args.source}' step='{step.value}': {failures}")


def _push_transfer_units(pusher, units: list, *, tar_max_files: int, tar_max_size_mb: int) -> list:
    """Route `TransferUnit`s to the right `HPCPusher` strategy.

    One unit (every source except MODIS's PREPARE, which is the only
    override of `transfer_units()`): `push_unit()` -- a single directory
    (tar+extract) or file (direct rsync).

    Many units, all single files sharing one output tree (MODIS's per-
    tile-year GeoTIFFs): batch-tar them via `push_batched()` -- pushing
    hundreds of files one rsync+extract round trip *each* would be far
    slower than amortizing over one shared tar. `remote_base_dir` is the
    longest common ancestor of every unit's `remote_path`; each unit's
    `PushUnit.remote_path` becomes the tar arcname relative to it, so nested
    structure (e.g. `<year>/<tile>.tif`) survives extraction intact.

    Anything else (mixed files/directories) -- concurrent per-unit pushes.
    """
    import os
    import posixpath

    from src.data.common.hpc.push import PushUnit

    if len(units) == 1:
        u = units[0]
        return [pusher.push_unit(PushUnit(unit_id=u.unit_id, local_path=u.local_path, remote_path=u.remote_path))]

    if len(units) > 1 and all(os.path.isfile(u.local_path) for u in units):
        # `remote_path` is always POSIX (the HPC target is a remote Linux
        # host regardless of the local OS) -- use `posixpath`, not `os.path`,
        # for every manipulation of it. On Windows `os.path` is `ntpath`:
        # `ntpath.commonpath`/`relpath` happily accept forward-slash input
        # but *emit* backslash-separated output, which then gets used
        # verbatim as a remote path/tar arcname (`mkdir -p foo\bar` on Linux
        # creates one literally-named `foo\bar` entry, not nested dirs).
        remote_base_dir = posixpath.commonpath([u.remote_path for u in units])
        push_units = [
            PushUnit(
                unit_id=u.unit_id, local_path=u.local_path,
                remote_path=posixpath.relpath(u.remote_path, remote_base_dir),
            )
            for u in units
        ]
        return pusher.push_batched(
            push_units, remote_base_dir, max_files=tar_max_files, max_bytes=tar_max_size_mb * 1024 * 1024,
        )

    push_units = [PushUnit(unit_id=u.unit_id, local_path=u.local_path, remote_path=u.remote_path) for u in units]
    return pusher.push_units_concurrent(push_units)


def _run_transfer_pass(args: argparse.Namespace, source, step: "PipelineStep", client) -> list:
    """One scan-and-push cycle: re-lists `source.transfer_units(step)` (a
    fresh filesystem scan for MODIS-style sources -- see its docstring),
    skips units that already exist on the HPC target (unless `--override`),
    and pushes the rest. Returns the `PushResult` list for whatever was
    actually pushed (empty if nothing was pending).

    The skip-check is a direct remote existence check (`check_paths_exist`,
    one batched round trip), not a cached belief -- always current, and
    doesn't need any local bookkeeping to stay in sync with what's really on
    HPC. Split out of `handle_transfer` so `--watch` mode (below) can call
    this repeatedly without duplicating the skip/push logic.
    """
    from src.data.common.hpc.push import HPCPusher, _full_remote_path

    units = source.transfer_units(step)
    if not units:
        return []

    if not args.override:
        full_paths = {u.unit_id: _full_remote_path(client, u.remote_path) for u in units}
        existence = client.check_paths_exist(list(full_paths.values()))
        pending = [u for u in units if not existence.get(full_paths[u.unit_id])]
        skipped = len(units) - len(pending)
        if skipped:
            logger.info("Skipping %d already-transferred unit(s)", skipped)
        units = pending

    if not units:
        return []

    logger.info("Transferring %d unit(s) for source '%s' step '%s'", len(units), args.source, step.value)
    pusher = HPCPusher(client)
    return _push_transfer_units(
        pusher, units,
        tar_max_files=source.cfg.raw.get("download", {}).get("tar_max_files", 100),
        tar_max_size_mb=source.cfg.raw.get("download", {}).get("tar_max_size_mb", 500),
    )


def handle_transfer(args: argparse.Namespace) -> None:
    """``data transfer`` -- push a step's local output to the HPC target.

    Generic across sources via `DataSource.transfer_units(step)`, driven by
    the unified `HPCPusher` (`common/hpc/push.py`). Already-pushed units are
    skipped via a direct remote existence check, not any local bookkeeping.

    `--watch`: instead of one scan-and-push pass, loop that pass on a
    `--poll-interval` timer until interrupted (Ctrl-C) -- for running
    alongside a concurrent `data run --step fetch` so newly-completed
    local output (e.g. MODIS's per-tile-year GeoTIFFs, written atomically via
    `os.replace` so `transfer_units()` never sees a partial file) gets pushed
    incrementally rather than requiring a separate manual `transfer` call
    after FETCH finishes.
    """
    setup_logging(args.log_level, debug=args.debug)
    if args.direction == "pull":
        raise NotImplementedError(
            "--direction pull is not implemented; included in the CLI for interface "
            "symmetry with the push direction, not because any current source needs it."
        )

    step = PipelineStep(args.step)
    source, _ = _build(args, step)

    if not source.ctx.ssh_target:
        source.close()
        raise ValueError("remote.ssh_target is not configured")

    from src.data.common.hpc.client import HPCClient

    client = HPCClient(target=source.ctx.ssh_target, key_file=source.ctx.key_file)

    if not getattr(args, "watch", False):
        results = _run_transfer_pass(args, source, step, client)
        source.close()
        if not results:
            logger.info("Nothing to transfer for source='%s' step='%s'.", args.source, step.value)
            return
        failures = [r for r in results if not r.ok]
        if failures:
            raise RuntimeError(
                f"Transfer failed for source='{args.source}' step='{step.value}': "
                f"{[(r.unit_id, r.error) for r in failures]}"
            )
        return

    import time

    poll_interval = getattr(args, "poll_interval", 30.0)
    logger.info(
        "Watching for source='%s' step='%s' output (poll every %.0fs) -- Ctrl-C to stop",
        args.source, step.value, poll_interval,
    )
    total_ok, total_failed = 0, 0
    try:
        while True:
            try:
                results = _run_transfer_pass(args, source, step, client)
            except Exception:
                logger.exception("Transfer pass failed; will retry after the next poll interval")
                results = []
            ok = [r for r in results if r.ok]
            failed = [r for r in results if not r.ok]
            total_ok += len(ok)
            total_failed += len(failed)
            if failed:
                logger.warning("Pass pushed %d unit(s), %d failed: %s", len(ok), len(failed), [(r.unit_id, r.error) for r in failed])
            elif ok:
                logger.info("Pass pushed %d unit(s)", len(ok))
            time.sleep(poll_interval)
    except KeyboardInterrupt:
        logger.info("Stopped watching (pushed %d unit(s) total, %d failed)", total_ok, total_failed)
    finally:
        source.close()
