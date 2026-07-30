"""Deprecated import path -- moved to src.data.sources.glass.crawler.

docs/design/09-integrated-pipeline.md §4/§10 step 5: the remote-listing
crawler is now owned by the merged `src/data/sources/glass/` package. This
shim keeps `src.data.download.sources.glass.source.GlassLSTDataSource` (still
used by the not-yet-deleted `src/data/preprocess/sources/glass.py`) working
unmodified; delete it in step 10's cutover alongside the rest of
`src/data/download/`.
"""

from src.data.sources.glass.crawler import *  # noqa: F401,F403
from src.data.sources.glass.crawler import _CrawlerMixin  # noqa: F401
