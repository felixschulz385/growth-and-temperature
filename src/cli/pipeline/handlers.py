"""Handler functions for the ``pipeline`` domain."""

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
    """``pipeline list`` -- enumerate every registered source."""
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


def _summarize_targets(targets: list) -> str:
    """One-line status for a (source, step)'s target list.

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
        return "no targets"
    if len(targets) == 1 and targets[0].completion is Completion.NEVER:
        import os

        output_path = targets[0].output_path
        if os.path.isfile(output_path):
            count = 1
        elif os.path.isdir(output_path):
            count = sum(len(files) for _, _, files in os.walk(output_path))
        else:
            count = 0
        return "no local data" if count == 0 else f"{count} file(s) fetched"
    complete = sum(1 for t in targets if is_complete(t))
    total = len(targets)
    pct = round(100 * complete / total) if total else 0
    return f"{complete}/{total} ({pct}%)"


def _print_source_summary(rows: dict) -> None:
    if not rows:
        print("No sources found.")
        return
    headers = ["source", *(step.value for step in STEP_ORDER)]
    widths = [max(len(headers[0]), *(len(name) for name in rows))]
    for i, step in enumerate(STEP_ORDER, start=1):
        widths.append(max(len(headers[i]), *(len(row[step.value]) for row in rows.values())))

    def fmt(cells: list) -> str:
        return "  ".join(cell.ljust(w) for cell, w in zip(cells, widths))

    print(fmt(headers))
    print(fmt(["-" * w for w in widths]))
    for name in sorted(rows):
        print(fmt([name, *(rows[name][step.value] for step in STEP_ORDER)]))


def handle_summary(args: argparse.Namespace) -> None:
    """``pipeline summary`` -- concise per-source, per-step data-availability
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
            rows[name] = {step.value: f"error: {exc}" for step in STEP_ORDER}
            continue

        for step in STEP_ORDER:
            if step not in spec.steps:
                continue
            try:
                row[step.value] = _summarize_targets(source.plan(step, TargetSelection()))
            except Exception as exc:
                row[step.value] = f"error: {exc}"
        source.close()
        rows[name] = row

    _print_source_summary(rows)


def _check_requires(spec: registry.SourceSpec, ctx, config) -> None:
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
        # A directory existing with a MARKER-completion file, or any content
        # for a step whose target completion policy this handler cannot know
        # ahead of instantiating that other source -- treat "the directory has
        # something in it" as the observable proxy here; the runner itself
        # checks the source's own StepTarget.completion when it plans.
        import os

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
    """`pipeline run`'s Dask-sizing flags override whatever `PipelineContext`
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
    """``pipeline plan`` -- print targets for (source, step) without running them."""
    setup_logging(args.log_level, debug=args.debug)
    source, _ = _build(args)
    step = PipelineStep(args.step)
    selection = _selection_from_args(args)

    targets = source.plan(step, selection)
    if not targets:
        print(f"No targets for source='{args.source}' step='{step.value}'.")
        return
    for target in targets:
        status = "complete" if is_complete(target) else "pending"
        print(f"[{status}] {target.key}  ->  {target.output_path}")
    source.close()


def handle_index(args: argparse.Namespace) -> None:
    """``pipeline index`` -- build/refresh a FETCH-capable source's completion index."""
    setup_logging(args.log_level, debug=args.debug)
    source, _ = _build(args)
    if PipelineStep.FETCH not in source.STEPS:
        raise ValueError(f"Source '{args.source}' does not implement 'fetch'; nothing to index.")

    from src.data.common.index.unified_index import UnifiedDataIndex

    index = UnifiedDataIndex(
        bucket_name="",
        data_source=source,
        local_index_dir=source.ctx.local_index_dir,
        key_file=source.ctx.key_file,
        hpc_mode=bool(source.ctx.ssh_target),
    )
    files_indexed = index.build_index_from_source(
        data_source=source, rebuild=args.rebuild, only_missing_entrypoints=True
    )
    index.save()
    logger.info("Indexed %s file(s) for source '%s'", files_indexed, args.source)
    source.close()


def handle_run(args: argparse.Namespace) -> None:
    """``pipeline run`` -- execute a (source, step)'s pending targets."""
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

    failures = []
    for target in targets:
        if not source.cfg.override and is_complete(target):
            logger.info("Skipping %s -- already complete: %s", target.key, target.output_path)
            continue
        logger.info("Running %s/%s -> %s", args.source, target.key, target.output_path)
        ok = source.execute(target)
        if not ok:
            logger.error("Target failed: %s/%s", args.source, target.key)
            failures.append(target.key)
    source.close()

    if failures:
        raise RuntimeError(f"{len(failures)} target(s) failed for source='{args.source}' step='{step.value}': {failures}")


def handle_transfer(args: argparse.Namespace) -> None:
    """``pipeline transfer`` -- push a step's local output to the HPC target.

    docs/design/08-hpc-transfer.md, renamed per
    docs/design/09-integrated-pipeline.md §8 -- generic across sources via
    `DataSource.transfer_units(step)`. Structurally unchanged from the old
    `preprocess transfer`: still a thin CLI wrapper over
    `src/data/common/hpc/transfer.py::transfer_units`.
    """
    setup_logging(args.log_level, debug=args.debug)
    if args.direction == "pull":
        raise NotImplementedError(
            "--direction pull is not implemented; included in the CLI for interface "
            "symmetry with the push direction, not because any current source needs "
            "it (docs/design/08-hpc-transfer.md §4)."
        )

    source, _ = _build(args)
    step = PipelineStep(args.step)
    units = source.transfer_units(step)
    if not units:
        logger.warning("No transfer units for source='%s' step='%s'.", args.source, step.value)
        source.close()
        return

    if not source.ctx.ssh_target:
        raise ValueError("remote.ssh_target is not configured")

    import os

    from src.data.common.hpc.client import HPCClient
    from src.data.common.hpc.transfer import transfer_units

    manifest_path = None
    if source.ctx.local_index_dir:
        manifest_path = os.path.join(source.ctx.local_index_dir, f"transfer_{args.source}_{step.value}.parquet")

    logger.info("Transferring %d unit(s) for source '%s' step '%s'", len(units), args.source, step.value)
    unit_dicts = [{"unit_id": u.unit_id, "local_path": u.local_path, "remote_path": u.remote_path} for u in units]
    success = transfer_units(
        ssh_target=source.ctx.ssh_target,
        key_file=source.ctx.key_file,
        units=unit_dicts,
        manifest_path=manifest_path,
        override=args.override,
    )
    source.close()
    if not success:
        raise RuntimeError(f"Transfer failed for source='{args.source}' step='{step.value}'")
