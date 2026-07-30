"""Deprecated import path -- moved to src.data.common.raster.compositing.

docs/design/09-integrated-pipeline.md §4/§10 step 2: daily/periodic ->
annual compositing is shared library code used by every source's PREPARE
step, not owned by the preprocess subsystem being retired. This shim keeps
every `src.data.preprocess.sources.*` module working unmodified during the
migration; delete it in step 10's cutover alongside the rest of
`src/data/preprocess/`.
"""

from src.data.common.raster.compositing import *  # noqa: F401,F403
