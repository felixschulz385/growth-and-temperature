"""download_with_retries() -- the shared chunked-download-with-backoff body
extracted from acag.py/ntl_harm.py/glass/source.py's near-identical copies.

No pytest-asyncio in this repo's dependency set -- run the coroutine
directly via asyncio.run() inside a plain sync test, matching how the rest
of this codebase has no async test infrastructure either.
"""

import asyncio
import os

import aiohttp
import pytest

from src.data.common.fetch.http import download_with_retries


class _FakeResponse:
    def __init__(self, chunks, status_error=None):
        self._chunks = chunks
        self._status_error = status_error

    def raise_for_status(self):
        if self._status_error:
            raise self._status_error

    @property
    def content(self):
        return self

    async def iter_chunked(self, size):
        for chunk in self._chunks:
            yield chunk

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


class _FakeSession:
    """`responses`: list popped one-per-`.get()` call -- either a
    _FakeResponse or an exception instance/class to raise, simulating a
    transient failure followed by success."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.get_calls = []

    def get(self, url, headers=None):
        self.get_calls.append((url, headers))
        outcome = self._responses.pop(0)
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


def test_download_with_retries_writes_chunks_on_first_success(tmp_path):
    output_path = str(tmp_path / "out.bin")
    session = _FakeSession([_FakeResponse([b"hello ", b"world"])])

    asyncio.run(download_with_retries(session, "https://example.com/f", output_path))

    with open(output_path, "rb") as f:
        assert f.read() == b"hello world"
    assert session.get_calls == [("https://example.com/f", None)]


def test_download_with_retries_passes_headers_through(tmp_path):
    output_path = str(tmp_path / "out.bin")
    session = _FakeSession([_FakeResponse([b"data"])])

    asyncio.run(
        download_with_retries(session, "https://example.com/f", output_path, headers={"User-Agent": "x"})
    )
    assert session.get_calls == [("https://example.com/f", {"User-Agent": "x"})]


def test_download_with_retries_retries_on_client_error_then_succeeds(tmp_path, monkeypatch):
    output_path = str(tmp_path / "out.bin")
    session = _FakeSession([aiohttp.ClientError("transient"), _FakeResponse([b"ok"])])

    sleeps = []
    monkeypatch.setattr(asyncio, "sleep", lambda seconds: sleeps.append(seconds) or _noop())

    asyncio.run(download_with_retries(session, "https://example.com/f", output_path, max_retries=3))

    with open(output_path, "rb") as f:
        assert f.read() == b"ok"
    assert len(session.get_calls) == 2
    assert sleeps == [2]  # (attempt=0 + 1) * 2


async def _noop():
    return None


def test_download_with_retries_raises_after_exhausting_retries(tmp_path, monkeypatch):
    output_path = str(tmp_path / "out.bin")
    session = _FakeSession([aiohttp.ClientError("a"), aiohttp.ClientError("b")])
    monkeypatch.setattr(asyncio, "sleep", lambda seconds: _noop())

    with pytest.raises(aiohttp.ClientError):
        asyncio.run(download_with_retries(session, "https://example.com/f", output_path, max_retries=2))

    assert len(session.get_calls) == 2
    assert not os.path.exists(output_path)
