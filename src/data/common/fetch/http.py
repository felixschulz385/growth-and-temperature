"""Shared aiohttp chunked-download-with-retry body for FETCH-capable
sources whose `download_async()` needs more than
`src.data.sources.misc._fetch.ConfiguredFilesFetchMixin`'s plain
(retry-less) streaming GET -- a transient network blip partway through a
large/slow download should be retried with backoff, not fail the whole
FETCH unit outright.

Was duplicated near line-for-line across acag.py/ntl_harm.py/
glass/source.py (~25 lines each: max_retries=3, `await
asyncio.sleep((attempt+1)*2)` on `aiohttp.ClientError`/
`asyncio.TimeoutError`, `iter_chunked(8192)`, `aiofiles`) before being
factored out here. Each source's own `aiohttp.ClientSession` construction
(connector limits, timeout) is deliberately NOT unified here -- those
numbers are genuinely tuned per host (e.g. glass's more conservative
`limit_per_host=2`/300s timeout vs acag's `limit_per_host=5`/600s), not
copy-paste.
"""

from __future__ import annotations

import asyncio
import os
from typing import Optional

import aiofiles
import aiohttp


async def download_with_retries(
    session: aiohttp.ClientSession,
    url: str,
    output_path: str,
    *,
    headers: Optional[dict] = None,
    max_retries: int = 3,
    chunk_size: int = 8192,
) -> None:
    """GET *url* and stream it to *output_path*, retrying up to
    *max_retries* times with linear backoff (`(attempt + 1) * 2` seconds)
    on `aiohttp.ClientError`/`asyncio.TimeoutError`. Re-raises the last
    error once retries are exhausted."""
    for attempt in range(max_retries):
        try:
            async with session.get(url, headers=headers) as resp:
                resp.raise_for_status()
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                async with aiofiles.open(output_path, "wb") as fh:
                    async for chunk in resp.content.iter_chunked(chunk_size):
                        await fh.write(chunk)
                return
        except (aiohttp.ClientError, asyncio.TimeoutError):
            if attempt < max_retries - 1:
                await asyncio.sleep((attempt + 1) * 2)
            else:
                raise
