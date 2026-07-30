"""Deprecated import path -- moved to src.data.common.fetch.async_downloader.

docs/design/09-integrated-pipeline.md §4/§10 step 2: this module now lives in
`src/data/common/fetch/` as shared library code importable by any source, not
owned by the download subsystem being retired. This shim keeps
`src.data.preprocess.sources.*` and `src.data.download.workflow.handlers`
working unmodified during the migration; delete it in step 10's cutover
alongside the rest of `src/data/download/`.
"""

from src.data.common.fetch.async_downloader import *  # noqa: F401,F403
from src.data.common.fetch.async_downloader import AsyncHPCDownloader, run_async_download_workflow  # noqa: F401
