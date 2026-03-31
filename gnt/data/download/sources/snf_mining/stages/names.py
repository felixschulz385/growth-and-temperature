"""Named stages for the SNF mining workflow."""

from __future__ import annotations

import logging
from enum import Enum
from typing import Collection

logger = logging.getLogger(__name__)


class Stage(str, Enum):
    """Selectable SNF mining workflow stages."""

    IDS = "ids"
    DETAIL_EXPORTS = "detail_exports"
    DETAIL_PARSE = "detail_parse"


ALL_STAGES: frozenset[Stage] = frozenset(Stage)


def resolve_stages(stages: Collection[Stage | str] | None) -> frozenset[Stage]:
    """Coerce *stages* to a frozenset of :class:`Stage` members."""
    if stages is None:
        logger.debug("No stages provided; defaulting to ALL_STAGES.")
        return ALL_STAGES

    result: set[Stage] = set()
    for stage in stages:
        if isinstance(stage, Stage):
            result.add(stage)
        else:
            result.add(Stage(str(stage).lower()))
    logger.debug("Resolved stages input %s to %s", stages, [stage.value for stage in result])
    return frozenset(result)
