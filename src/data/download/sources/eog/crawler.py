"""Deprecated import path -- moved to src.data.sources.eog.crawler.

docs/design/09-integrated-pipeline.md §4/§10 step 5: the remote-listing
crawler is now owned by the merged `src/data/sources/eog/` package. This shim
keeps `src.data.download.sources.eog.source.EOGDataSource` (still used by the
not-yet-deleted `src/data/preprocess/sources/eog.py`) working unmodified;
delete it in step 10's cutover alongside the rest of `src/data/download/`.
"""

from src.data.sources.eog.crawler import *  # noqa: F401,F403
from src.data.sources.eog.crawler import _CrawlerMixin  # noqa: F401
