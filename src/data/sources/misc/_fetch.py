"""Shared FETCH mixin for the misc split: each source fetches a small, fixed
list of config-declared (url, name) files -- osm/gadm fetch exactly one,
country_classifications fetches two (HDI + World Bank).

docs/design/09-integrated-pipeline.md §7: real, shared logic across three
concrete sources (not a speculative abstraction for one), extracted from
`src/data/download/sources/misc.py::MiscDataSource`'s generic list-of-files
downloader -- simplified here since each split source's file list is fixed
by its own config block rather than an arbitrary externally-supplied list.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from src.data.sources.verify import VerificationResult

logger_name = __name__


@dataclass(frozen=True)
class ConfiguredFile:
    key: str
    url: str
    name: str


class ConfiguredFilesFetchMixin:
    """`self.CONFIGURED_FILES` (set by the subclass `__init__`) is the fixed
    list of files this source fetches. Satisfies the RemoteFileCatalog
    methods `UnifiedDataIndex`/`AsyncHPCDownloader` need."""

    CONFIGURED_FILES: List[ConfiguredFile]
    has_entrypoints = False

    def list_remote_files(self, entrypoint: Optional[dict] = None) -> List[Tuple[str, str]]:
        return [(f.name, f.url) for f in self.CONFIGURED_FILES]

    def get_file_hash(self, file_url: str) -> str:
        return hashlib.md5(file_url.encode("utf-8")).hexdigest()

    def get_all_entrypoints(self) -> List[Dict[str, Any]]:
        return []

    def filename_to_entrypoint(self, relative_path: str) -> Optional[Dict[str, Any]]:
        return None

    def local_path(self, relative_path: str) -> str:
        return os.path.join("data", self.DATA_SOURCE_NAME, relative_path)

    def download(self, file_url: str, output_path: str, session: Any = None) -> None:
        import requests

        s = session or requests.Session()
        with s.get(file_url, stream=True, timeout=60) as response:
            response.raise_for_status()
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            with open(output_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)

    async def download_async(self, file_url: str, output_path: str, session: Any = None) -> None:
        import aiofiles
        import aiohttp

        async def _do_download(sess: aiohttp.ClientSession):
            async with sess.get(file_url) as response:
                response.raise_for_status()
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                async with aiofiles.open(output_path, "wb") as f:
                    async for chunk in response.content.iter_chunked(8192):
                        await f.write(chunk)

        if session is None:
            connector = aiohttp.TCPConnector(limit=10, limit_per_host=5)
            timeout = aiohttp.ClientTimeout(total=600)
            async with aiohttp.ClientSession(connector=connector, timeout=timeout) as sess:
                await _do_download(sess)
        else:
            await _do_download(session)

    def verify_fetch(self) -> VerificationResult:
        """Checks every `self.CONFIGURED_FILES` entry is present at its exact
        expected local path under `output_root(FETCH)`.

        Exists because a source's FETCH directory can contain files (so the
        generic disk-walk-based "N file(s) fetched" summary looks fine) that
        don't match any `ConfiguredFile.name` -- e.g. a differently-named
        export, or a stale file left from before a config rename. PREPARE's
        own path check (`os.path.exists`, matching this exact filename) then
        silently plans zero targets with no indication why. This surfaces
        that mismatch directly instead of leaving it to be diagnosed by hand.
        """
        from src.data.sources.steps import PipelineStep

        root = self.output_root(PipelineStep.FETCH)
        missing = [f.name for f in self.CONFIGURED_FILES if not os.path.exists(os.path.join(root, f.name))]
        if not missing:
            return VerificationResult(True, f"ok: {len(self.CONFIGURED_FILES)} expected file(s) present")

        present: List[str] = []
        if os.path.isdir(root):
            for _, _, files in os.walk(root):
                present.extend(files)

        detail = f"missing {len(missing)}/{len(self.CONFIGURED_FILES)} expected file(s): {missing}"
        if present:
            detail += f" -- found instead: {sorted(present)[:10]}"
        else:
            detail += " -- FETCH directory is empty or doesn't exist"
        return VerificationResult(False, detail)
