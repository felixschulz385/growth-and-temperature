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
    local_completion_state,
    local_drift,
)

logger = logging.getLogger(__name__)


def handle_list(args: argparse.Namespace) -> None:
    """``data list`` -- enumerate every registered source."""
    setup_logging(args.log_level, debug=args.debug)
    for spec in sorted(registry.all_specs(), key=lambda s: s.id):
        steps = ", ".join(s.value for s in spec.steps)
        aliases = f" (aliases: {', '.join(spec.aliases)})" if spec.aliases else ""
        requires = (
            "; requires " + ", ".join(f"{rid}:{rstep.value}" for rid, rstep in spec.requires)
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


def handle_summary(args: argparse.Namespace) -> None:
    """``data summary`` -- concise per-source, per-step data-availability
    overview. Builds each source directly (bypassing `_check_requires`, unlike
    `_build()`) since a summary should still show a source's own available
    steps even when an upstream REQUIRES dependency isn't complete yet."""
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

        had_error = False
        grid_targets: list = []
        for step in STEP_ORDER:
            if step not in spec.steps:
                continue
            try:
                if step is PipelineStep.FETCH and hasattr(source, "verify_fetch"):
                    # ConfiguredFilesFetchMixin sources (osm/gadm/
                    # country_classifications) fetch a small, fixed list of
                    # named files -- report exactly which are missing/
                    # mismatched instead of the generic disk-walk count,
                    # which can't tell "N files fetched" from "N files
                    # fetched under the wrong names."
                    result = source.verify_fetch()
                    row[step.value] = result.detail
                    continue
                targets = source.plan(step, TargetSelection())
                summary, _complete = _summarize_targets(targets)
                row[step.value] = summary
                if step is PipelineStep.GRID:
                    grid_targets = targets
            except Exception as exc:
                row[step.value] = f"error: {exc}"
                had_error = True

        if had_error:
            row["verified"] = "error"
        elif PipelineStep.GRID not in spec.steps:
            row["verified"] = "-"
        elif not grid_targets:
            row["verified"] = "-"
        else:
            complete_targets = [t for t in grid_targets if is_complete(t)]
            if not complete_targets:
                row["verified"] = "pending"
            else:
                results = [source.verify_grid(t) for t in complete_targets]
                n_ok = sum(1 for r in results if r.ok)
                row["verified"] = "yes" if n_ok == len(results) else f"FAILED ({n_ok}/{len(results)})"
        source.close()
        rows[name] = row

    _print_source_summary(rows)


def _check_requires(spec: registry.SourceSpec, ctx, config) -> None:
    import os

    from src.data.common.ledger.paths import ledger_path
    from src.data.common.ledger.store import SourceLedger
    from src.data.sources import layout

    for requires_id, requires_step in spec.requires:
        requires_cfg = get_source_config(config, requires_id)
        expected = layout.output_root(
            ctx.data_root,
            requires_cfg.data_path,
            requires_step,
            namespace=requires_cfg.namespace,
            grid_id=ctx.grid_id,
            layout=ctx.layout,
        )

        # Prefer the prerequisite's own ledger when one exists
        # (docs/design/10-fetch-ledger.md §6): `step_complete()` knows about
        # HPC-verified state a bare local os.path.exists() can't see (e.g. a
        # prerequisite whose step ran on a different machine and was pushed,
        # not produced locally here). Falls back to the local-disk check
        # below when the prerequisite hasn't been through the ledger yet
        # (e.g. adopted from pre-ledger local output) -- same proxy as
        # always: "the directory has something in it," since the runner
        # itself checks the source's own StepTarget.completion when it plans.
        #
        # `requires_cfg` is a bare `SourceConfig`, never instantiated as a
        # `DataSource` here, so the misc-split sources' overridden
        # `data_path` property (gadm/osm/country_classifications/ecoregions,
        # all sharing `cfg.data_path="misc"` and disambiguating via
        # `cfg.namespace` -- see `DataSource.data_path`'s docstring,
        # src/data/sources/base.py) never applies to the raw config field.
        # Reconstruct the same combined string by hand -- mirrors `expected`
        # above, which already passes `namespace=requires_cfg.namespace` to
        # `layout.output_root()` for exactly this disambiguation.
        requires_data_path = (
            f"{requires_cfg.data_path}/{requires_cfg.namespace}" if requires_cfg.namespace else requires_cfg.data_path
        )
        requires_ledger_path = ledger_path(ctx.local_index_dir, requires_data_path)
        if requires_ledger_path and os.path.exists(requires_ledger_path):
            with SourceLedger.open(requires_ledger_path, data_path=requires_data_path, read_only=True) as ledger:
                if ledger.step_complete(requires_step.value):
                    continue

        if not os.path.exists(expected):
            raise MissingPrerequisiteError(spec.id, requires_id, requires_step, expected)


def _selection_from_args(args: argparse.Namespace) -> TargetSelection:
    year_range = tuple(args.years) if getattr(args, "years", None) else None
    keys = tuple(args.keys) if getattr(args, "keys", None) else None
    return TargetSelection(year_range=year_range, keys=keys)


def _build(args: argparse.Namespace):
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
    _check_requires(spec, ctx, config)
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


def _open_ledger_readonly(source):
    """Best-effort read-only `SourceLedger` for *source*, or None if
    `local_index_dir` isn't configured or no ledger file exists yet.
    Lets `is_complete()` honor `StepTarget.require_remote` (today: only
    MODIS's FETCH targets) from the CLI's hot paths, not just when a caller
    happens to pass one explicitly (docs/design/10-fetch-ledger.md §6)."""
    import os

    from src.data.common.ledger.paths import ledger_path
    from src.data.common.ledger.store import SourceLedger

    local_ledger_path = ledger_path(source.ctx.local_index_dir, source.data_path)
    if local_ledger_path is None or not os.path.exists(local_ledger_path):
        return None
    return SourceLedger.open(local_ledger_path, data_path=source.data_path, read_only=True)


def _heal_local_drift(source, step: PipelineStep, drifted: list[tuple[str, str]]) -> None:
    """Self-heal `local_drift()`-flagged rows: batch-correct the ledger's
    `local_state` to match on-disk reality -- the "automatically ... when
    conflict is detected" half of ledger-as-source-of-truth (the other half
    being an explicit `data reconcile`, which also re-discovers targets
    a bare disk-vs-ledger read can't catch, see `local_drift()`'s docstring).

    Called only after the read-only ledger connection used to *detect*
    drift has already been closed: DuckDB refuses a second same-process
    connection to one file whose `read_only` setting doesn't match an
    already-open one (confirmed empirically), so healing needs its own,
    separately-opened read-write connection, not a reuse of the read-only one.
    """
    if not drifted:
        return
    from src.data.common.ledger.paths import ledger_path
    from src.data.common.ledger.store import SourceLedger

    local_ledger_path = ledger_path(source.ctx.local_index_dir, source.data_path)
    if local_ledger_path is None:
        return
    logger.warning(
        "Local-disk drift detected for source='%s' step='%s': %d target(s) disagree with the ledger -- "
        "self-healing: %s",
        source.ID, step.value, len(drifted), [key for key, _ in drifted],
    )
    with SourceLedger.open(local_ledger_path, data_path=source.data_path) as ledger:
        ledger.set_local_states_batch(step.value, drifted)


def handle_plan(args: argparse.Namespace) -> None:
    """``data plan`` -- print targets for (source, step) without running them."""
    setup_logging(args.log_level, debug=args.debug)
    source, _ = _build(args)
    step = PipelineStep(args.step)
    selection = _selection_from_args(args)

    targets = source.plan(step, selection)
    if not targets:
        print(f"No targets for source='{args.source}' step='{step.value}'.")
        return
    ledger = _open_ledger_readonly(source)
    statuses: dict[str, bool] = {}
    drifted: list[tuple[str, str]] = []
    try:
        for target in targets:
            statuses[target.key] = is_complete(target, ledger=ledger)
            if ledger is not None and local_drift(target, ledger):
                drifted.append((target.key, local_completion_state(target)))
    finally:
        if ledger is not None:
            ledger.close()
    _heal_local_drift(source, step, drifted)

    for target in targets:
        status = "complete" if statuses[target.key] else "pending"
        print(f"[{status}] {target.key}  ->  {target.output_path}")
    source.close()


def handle_index(args: argparse.Namespace) -> None:
    """``data index`` -- build/refresh a FETCH-capable source's ledger
    crawl catalog (docs/design/10-fetch-ledger.md). ``--rebuild`` forces
    every entrypoint to be re-crawled, but -- unlike the old
    ``UnifiedDataIndex(rebuild=True)`` -- does not discard already-tracked
    download/transfer state; see `SourceLedger.reset_crawl_state()`.
    """
    setup_logging(args.log_level, debug=args.debug)
    source, _ = _build(args)
    if PipelineStep.FETCH not in source.STEPS:
        raise ValueError(f"Source '{args.source}' does not implement 'fetch'; nothing to index.")

    from src.data.sources.base import RemoteFileCatalog

    if not isinstance(source, RemoteFileCatalog):
        raise ValueError(
            f"Source '{args.source}' declares 'fetch' but has no crawlable remote file catalog to index "
            "(e.g. MODIS streams per-(year, tile) STAC queries instead of listing a flat file list) -- "
            "nothing to index; its fetch state is tracked directly via `data run --step fetch`."
        )

    from src.data.common.ledger import catalog
    from src.data.common.ledger.paths import ledger_path
    from src.data.common.ledger.store import SourceLedger

    local_ledger_path = ledger_path(source.ctx.local_index_dir, source.data_path)
    if local_ledger_path is None:
        raise ValueError("paths.local_index_dir is not configured -- cannot build/refresh a ledger.")

    with SourceLedger.open(local_ledger_path, data_path=source.data_path) as ledger:
        if args.rebuild:
            ledger.reset_crawl_state()
        files_indexed = catalog.refresh(ledger, source)

    logger.info("Indexed %s new file(s) for source '%s'", files_indexed, args.source)
    source.close()


def handle_run(args: argparse.Namespace) -> None:
    """``data run`` -- execute a (source, step)'s pending targets."""
    setup_logging(args.log_level, debug=args.debug)
    source, _ = _build(args)
    if args.override:
        source.cfg = dataclasses.replace(source.cfg, override=True)
    step = PipelineStep(args.step)
    selection = _selection_from_args(args)

    targets = source.plan(step, selection)
    if not targets:
        logger.warning("No targets for source='%s' step='%s'.", args.source, step.value)
        return

    # Completeness (incl. any StepTarget.require_remote gate) is resolved
    # once, up front, against a read-only ledger connection -- then closed
    # before any target executes. A source's own execute() may need its own
    # read-write ledger connection (today: MODIS's FETCH, tracking per-unit
    # local/remote state directly), and DuckDB allows only one read-write
    # connection per file at a time, so the two must never overlap.
    ledger = _open_ledger_readonly(source)
    drifted: list[tuple[str, str]] = []
    try:
        already_complete = {target.key for target in targets if is_complete(target, ledger=ledger)}
        if ledger is not None:
            drifted = [
                (target.key, local_completion_state(target)) for target in targets if local_drift(target, ledger)
            ]
    finally:
        if ledger is not None:
            ledger.close()
    _heal_local_drift(source, step, drifted)

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
    source.close()

    if failures:
        raise RuntimeError(f"{len(failures)} target(s) failed for source='{args.source}' step='{step.value}': {failures}")


def handle_reconcile(args: argparse.Namespace) -> None:
    """``data reconcile`` -- rebuild a source's DuckDB ledger from real
    on-disk/HPC filesystem state (docs/design/10-fetch-ledger.md §5). A
    manual/occasional operator command, not part of the normal fetch/run hot
    path: run once per source when adopting the ledger, or after any
    out-of-band filesystem surgery. Never converts the old
    `UnifiedDataIndex`/`TransferManifest` Parquet files -- the real
    filesystem/HPC state is ground truth.
    """
    setup_logging(args.log_level, debug=args.debug)
    config = load_config_with_env_vars(args.config)
    ctx = build_context(config)
    sources_cfg = config.get("sources", {}) or {}

    if args.source not in sources_cfg:
        raise KeyError(f"Source '{args.source}' not found in configuration. Available: {sorted(sources_cfg)}")

    import os

    from src.data.common.ledger.bootstrap import reconcile_fetch
    from src.data.common.ledger.paths import ledger_path
    from src.data.common.ledger.store import SourceLedger
    from src.data.sources import layout
    from src.data.sources.base import RemoteFileCatalog
    from src.data.sources.reconcile import reconcile_step

    spec = registry.resolve(args.source)
    cfg = get_source_config(config, args.source)
    source = registry.create(args.source, ctx, cfg)

    requested_steps = list(spec.steps) if getattr(args, "step", "all") == "all" else [PipelineStep(args.step)]
    requested_steps = [s for s in requested_steps if s in spec.steps]
    if not requested_steps:
        raise ValueError(f"Source '{args.source}' does not implement step '{args.step}'.")

    client = None
    if ctx.ssh_target:
        from src.data.common.hpc.client import HPCClient

        client = HPCClient(target=ctx.ssh_target, key_file=ctx.key_file)

    local_ledger_path = ledger_path(ctx.local_index_dir, source.data_path)
    if local_ledger_path is None:
        raise ValueError("paths.local_index_dir is not configured -- cannot open/create a ledger")

    with SourceLedger.open(local_ledger_path, data_path=source.data_path) as ledger:
        if client is not None:
            tmp_dir = os.path.join(ctx.staging_dir or ctx.local_index_dir, "reconcile_tmp")
            ledger.merge_from_remote(client, tmp_dir)

        for step in requested_steps:
            if step is PipelineStep.FETCH and isinstance(source, RemoteFileCatalog):
                # NOTE: `source.cfg.data_path`/`source.cfg.namespace` (the
                # resolved, post-__init__-default config), not
                # `source.data_path` -- that property is overridden by the
                # misc-split sources (gadm/osm/country_classifications) to a
                # combined "<data_path>/<namespace>" string purely for index-
                # file naming (base.py's own docstring), which would double
                # up the namespace segment if fed into raw_root() alongside
                # `namespace=` too.
                raw_root = layout.raw_root(
                    "", source.cfg.data_path, namespace=source.cfg.namespace, layout=ctx.layout
                )
                result = reconcile_fetch(ledger, source, raw_root=raw_root, client=client)
                logger.info(
                    "%s/fetch: discovered=%d verified_present=%d",
                    args.source, result["discovered"], result["verified_present"],
                )
            else:
                result = reconcile_step(source, step, ledger, client=client, remote_data_root=ctx.remote_data_root)
                logger.info(
                    "%s/%s: total=%d local_complete=%d remote_verified=%d",
                    args.source, step.value, result["total"], result["local_complete"], result["remote_verified"],
                )

    # `push_to_remote()` after the `with` block, not inside it -- scp'ing
    # the ledger's own `.duckdb` file while this process still holds it open
    # for read-write fails on Windows (`scp: open local ...: Broken pipe`,
    # same bug as `_run_transfer_pass()`'s docstring describes).
    if client is not None:
        ledger.push_to_remote(client)

    source.close()


def _push_transfer_units(pusher, units: list, *, tar_max_files: int, tar_max_size_mb: int) -> list:
    """Route `TransferUnit`s to the right `HPCPusher` strategy.

    One unit (every source except MODIS's PREPARE, which is the only
    override of `transfer_units()`): `push_unit()` -- a single directory
    (tar+extract) or file (direct rsync).

    Many units, all single files sharing one output tree (MODIS's per-
    tile-year GeoTIFFs): batch-tar them via `push_batched()` -- pushing
    hundreds of files one rsync+extract round trip *each* (as the old
    `transfer_units()` serial loop did) is exactly the inefficiency
    docs/design/10-fetch-ledger.md §1 flags. `remote_base_dir` is the
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


def _run_transfer_pass(args: argparse.Namespace, source, step: "PipelineStep", local_ledger_path: str, client) -> list:
    """One scan-and-push cycle: re-lists `source.transfer_units(step)` (a
    fresh filesystem scan for MODIS-style sources -- see its docstring),
    skips units the ledger already marked `VERIFIED` (unless `--override`),
    and pushes the rest. Returns the `PushResult` list for whatever was
    actually pushed (empty if nothing was pending).

    Split out of `handle_transfer` so `--watch` mode (below) can call this
    repeatedly without duplicating the skip/push/record logic.

    Three separate, short-lived ledger connections (via `open_with_retry()`),
    not one held for the whole pass:
    1. Read/write the small "what's pending" metadata, close.
    2. Do the actual push -- the slow part (real network I/O, tens of
       seconds+ for a batch) -- with NO ledger connection open at all, so a
       concurrent `data run --step fetch` (which now also only takes its
       own connection briefly per unit, see
       `ModisSource._ledger_ensure_artifact()`'s docstring) isn't locked out
       for the push's whole duration.
    3. Record the results, close -- *then* `push_to_remote()` the ledger
       file itself. Doing that last step while its own connection was still
       open (this function's previous shape) reliably failed on Windows
       (`scp: open local ...: Broken pipe`, confirmed against a real scicore
       push): the local `.duckdb` file was still held open by this same
       process's own DuckDB connection when `scp` tried to read it.
    """
    from src.data.common.hpc.push import HPCPusher
    from src.data.common.ledger.schema import RemoteState
    from src.data.common.ledger.store import SourceLedger

    units = source.transfer_units(step)
    if not units:
        return []

    with SourceLedger.open_with_retry(local_ledger_path, data_path=source.data_path) as ledger:
        for u in units:
            ledger.ensure_artifact(step.value, u.unit_id, local_path=u.local_path, remote_path=u.remote_path)

        if not args.override:
            states = ledger.remote_states(step.value, [u.unit_id for u in units])
            pending = [u for u in units if states.get(u.unit_id) != RemoteState.VERIFIED]
            skipped = len(units) - len(pending)
            if skipped:
                logger.info("Skipping %d already-transferred unit(s)", skipped)
            units = pending

    if not units:
        return []

    logger.info("Transferring %d unit(s) for source '%s' step '%s'", len(units), args.source, step.value)
    pusher = HPCPusher(client)
    results = _push_transfer_units(
        pusher, units,
        tar_max_files=source.cfg.raw.get("download", {}).get("tar_max_files", 100),
        tar_max_size_mb=source.cfg.raw.get("download", {}).get("tar_max_size_mb", 500),
    )

    with SourceLedger.open_with_retry(local_ledger_path, data_path=source.data_path) as ledger:
        ledger.record_push_batch(step.value, results)
    ledger.push_to_remote(client)  # after the `with` block -- connection is closed by now
    return results


def handle_transfer(args: argparse.Namespace) -> None:
    """``data transfer`` -- push a step's local output to the HPC target.

    docs/design/10-fetch-ledger.md §2 -- generic across sources via
    `DataSource.transfer_units(step)`, now driven by the same unified
    `HPCPusher` FETCH uses (`common/hpc/push.py`) instead of the old
    dedicated, duplicated `common/hpc/transfer.py`. Push status is tracked in
    the source's own DuckDB ledger, not a separate Parquet manifest.

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
            "symmetry with the push direction, not because any current source needs "
            "it (docs/design/10-fetch-ledger.md)."
        )

    source, _ = _build(args)
    step = PipelineStep(args.step)

    if not source.ctx.ssh_target:
        source.close()
        raise ValueError("remote.ssh_target is not configured")

    from src.data.common.hpc.client import HPCClient
    from src.data.common.ledger.paths import ledger_path

    local_ledger_path = ledger_path(source.ctx.local_index_dir, source.data_path)
    if local_ledger_path is None:
        source.close()
        raise ValueError("paths.local_index_dir is not configured -- cannot track transfer state")

    client = HPCClient(target=source.ctx.ssh_target, key_file=source.ctx.key_file)

    if not getattr(args, "watch", False):
        results = _run_transfer_pass(args, source, step, local_ledger_path, client)
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

    import duckdb

    poll_interval = getattr(args, "poll_interval", 30.0)
    logger.info(
        "Watching for source='%s' step='%s' output (poll every %.0fs) -- Ctrl-C to stop",
        args.source, step.value, poll_interval,
    )
    total_ok, total_failed = 0, 0
    try:
        while True:
            # `_run_transfer_pass` already opens its own short-lived,
            # retrying (`open_with_retry()`) connections internally -- this
            # `except duckdb.IOException` is a last-resort net for the rare
            # case every retry inside it is exhausted (a concurrently
            # running `data run --step fetch` holding the lock
            # unusually long), so this watch loop logs and tries again next
            # poll interval instead of crashing the whole watch command.
            try:
                results = _run_transfer_pass(args, source, step, local_ledger_path, client)
            except duckdb.IOException:
                logger.info("Ledger busy (FETCH holds it) -- will retry after the next poll interval")
                results = []
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
