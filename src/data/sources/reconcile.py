"""PREPARE/GRID bootstrap reconciliation against real local/HPC filesystem
state.

docs/design/10-fetch-ledger.md §5/§7. Reuses each source's own, unchanged
`plan()` as the authoritative "what should exist" list -- no source-specific
logic needed here. Lives under `src/data/sources/` (not `common/ledger/`)
because it needs `DataSource.plan()`/`StepTarget`/`is_complete()`; FETCH's
bootstrap (`common/ledger/bootstrap.py`) stays free of that dependency.
"""

from __future__ import annotations

import logging
import os
from typing import Any, Optional

from src.data.common.ledger.schema import LocalState, RemoteState
from src.data.common.ledger.store import SourceLedger
from src.data.sources.base import DataSource
from src.data.sources.steps import Completion, PipelineStep, TargetSelection, is_complete, marker_path

logger = logging.getLogger(__name__)


def reconcile_step(
    source: DataSource,
    step: PipelineStep,
    ledger: SourceLedger,
    *,
    client: Optional[Any] = None,
    remote_data_root: Optional[str] = None,
) -> dict[str, int]:
    """Reconcile *ledger*'s `artifacts` rows for `(source, step)` against
    real filesystem state. *step* must be PREPARE or GRID -- FETCH's
    catalog-shaped reconciliation is `common.ledger.bootstrap.reconcile_fetch`.

    Enumerates `source.plan(step, TargetSelection())` (every target this
    source currently believes should exist) as ground truth, checks local
    disk via the existing `is_complete()` machinery, and -- if *client* and
    *remote_data_root* are both given -- checks remote presence for every
    target in one batched round trip.
    """
    if step is PipelineStep.FETCH:
        raise ValueError("reconcile_step is for PREPARE/GRID; use common.ledger.bootstrap.reconcile_fetch for FETCH")

    targets = source.plan(step, TargetSelection())
    result = {"total": len(targets), "local_complete": 0, "remote_verified": 0}

    # key -> (probe path for existence-check, output-relative remote path to record)
    remote_targets: dict[str, tuple[str, str]] = {}
    for target in targets:
        ledger.ensure_artifact(step.value, target.key, local_path=target.output_path)

        if is_complete(target):
            ledger.set_local_state(step.value, target.key, LocalState.COMPLETE)
            result["local_complete"] += 1

        if client is None or remote_data_root is None:
            continue

        # MARKER-completion targets are directories with a sibling .complete
        # marker file -- probe the marker (a plain file) since
        # check_files_exist is a `[ -f ... ]` test and directories aren't
        # files. PATH_EXISTS targets probe their own (single-file) path.
        probe_path = marker_path(target.output_path) if target.completion is Completion.MARKER else target.output_path
        remote_output_path = os.path.relpath(target.output_path, remote_data_root)
        remote_probe_path = os.path.relpath(probe_path, remote_data_root)
        remote_targets[target.key] = (remote_probe_path, remote_output_path)
        ledger.ensure_artifact(step.value, target.key, remote_path=remote_output_path)

    if client is not None and remote_targets:
        probe_paths = [probe for probe, _ in remote_targets.values()]
        existence = client.check_files_exist(probe_paths)
        for key, (probe_path, _) in remote_targets.items():
            if existence.get(probe_path):
                ledger.set_remote_state(step.value, key, RemoteState.VERIFIED)
                result["remote_verified"] += 1

    logger.info(
        "Reconciled %s/%s: %d target(s), %d local-complete, %d remote-verified",
        source.ID, step.value, result["total"], result["local_complete"], result["remote_verified"],
    )
    return result
