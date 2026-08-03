"""catalog.refresh() against fake RemoteFileCatalog-shaped sources.

docs/design/10-fetch-ledger.md. Replaces UnifiedDataIndex.build_index_from_source's
entrypoint-loop/missing-entrypoint-scan behavior -- tested here against the
new ledger only, no old-format fixtures.
"""

import pytest

from src.data.common.ledger import catalog
from src.data.common.ledger.store import SourceLedger


@pytest.fixture
def ledger(tmp_path):
    path = str(tmp_path / "gadm.duckdb")
    with SourceLedger.open(path, data_path="misc/gadm") as led:
        yield led


class _SimpleSource:
    has_entrypoints = False

    def __init__(self, files):
        self._files = files

    def list_remote_files(self, entrypoint=None):
        return self._files

    def get_file_hash(self, file_url):
        return file_url


def test_refresh_simple_source(ledger):
    source = _SimpleSource([("gadm_410-levels.zip", "https://example.com/gadm.zip")])
    added = catalog.refresh(ledger, source)
    assert added == 1
    assert ledger.stats("fetch")["total"] == 1

    # A second refresh with the same file list adds nothing new.
    assert catalog.refresh(ledger, source) == 0


class _EntrypointSource:
    has_entrypoints = True

    def __init__(self, entrypoints, files_by_year):
        self._entrypoints = entrypoints
        self._files_by_year = files_by_year
        self.crawl_calls = []

    def get_all_entrypoints(self):
        return self._entrypoints

    def list_remote_files(self, entrypoint=None):
        self.crawl_calls.append(entrypoint)
        return self._files_by_year.get(entrypoint["year"], [])

    def get_file_hash(self, file_url):
        return file_url


def test_refresh_entrypoint_source_crawls_only_missing(ledger):
    source = _EntrypointSource(
        entrypoints=[{"year": 2020}, {"year": 2021}],
        files_by_year={
            2020: [("2020/a.hdf", "https://x/2020/a.hdf")],
            2021: [("2021/b.hdf", "https://x/2021/b.hdf")],
        },
    )
    added = catalog.refresh(ledger, source)
    assert added == 2
    assert len(source.crawl_calls) == 2

    # Re-running with the same entrypoints crawls nothing new.
    added_again = catalog.refresh(ledger, source)
    assert added_again == 0
    assert len(source.crawl_calls) == 2  # unchanged -- no new crawl calls


def test_refresh_entrypoint_source_crawls_new_entrypoint_only(ledger):
    source = _EntrypointSource(
        entrypoints=[{"year": 2020}],
        files_by_year={2020: [("2020/a.hdf", "https://x/2020/a.hdf")]},
    )
    catalog.refresh(ledger, source)
    assert len(source.crawl_calls) == 1

    # A new entrypoint appears (e.g. this year's data went live).
    source._entrypoints = [{"year": 2020}, {"year": 2021}]
    source._files_by_year[2021] = [("2021/b.hdf", "https://x/2021/b.hdf")]
    added = catalog.refresh(ledger, source)
    assert added == 1
    assert len(source.crawl_calls) == 2  # only the new one triggered a crawl


def test_refresh_entrypoint_source_raises_without_any_entrypoints(ledger):
    source = _EntrypointSource(entrypoints=[], files_by_year={})
    with pytest.raises(ValueError):
        catalog.refresh(ledger, source)


def test_refresh_entrypoint_source_survives_one_crawl_failure(ledger):
    class _FlakySource(_EntrypointSource):
        def list_remote_files(self, entrypoint=None):
            if entrypoint["year"] == 2020:
                raise RuntimeError("network blip")
            return super().list_remote_files(entrypoint)

    source = _FlakySource(
        entrypoints=[{"year": 2020}, {"year": 2021}],
        files_by_year={2021: [("2021/b.hdf", "https://x/2021/b.hdf")]},
    )
    added = catalog.refresh(ledger, source)
    assert added == 1
    # The failed entrypoint is not marked crawled, so it's retried next time.
    missing = ledger.missing_entrypoints()
    assert missing == [{"year": 2020}]
