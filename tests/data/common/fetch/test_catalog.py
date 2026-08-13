import hashlib

from src.data.common import statusfile
from src.data.common.fetch import catalog


def _hash(url: str) -> str:
    return hashlib.md5(url.encode("utf-8")).hexdigest()


class _FlatSource:
    has_entrypoints = False
    get_file_hash = staticmethod(_hash)

    def __init__(self, files):
        self._files = files

    def list_remote_files(self, entrypoint=None):
        return self._files


class _EntrypointSource:
    has_entrypoints = True
    get_file_hash = staticmethod(_hash)

    def __init__(self, entrypoints, files_by_year, crawl_calls=None):
        self._entrypoints = entrypoints
        self._files_by_year = files_by_year
        self.crawl_calls = crawl_calls if crawl_calls is not None else []

    def get_all_entrypoints(self):
        return self._entrypoints

    def list_remote_files(self, entrypoint=None):
        self.crawl_calls.append(entrypoint)
        return self._files_by_year[entrypoint["year"]]


def test_flat_source_lists_remote_files_directly(tmp_path):
    source = _FlatSource([("a.nc", "https://x/a.nc"), ("b.nc", "https://x/b.nc")])
    required = catalog.required_files(source, str(tmp_path))
    assert [r.relative_path for r in required] == ["a.nc", "b.nc"]
    assert required[0].unit_id == _hash("https://x/a.nc")


def test_entrypoint_source_crawls_each_entrypoint_and_caches_result(tmp_path):
    calls = []
    source = _EntrypointSource(
        entrypoints=[{"year": 2020}, {"year": 2021}],
        files_by_year={2020: [("2020/a.nc", "https://x/2020/a.nc")], 2021: [("2021/b.nc", "https://x/2021/b.nc")]},
        crawl_calls=calls,
    )
    required = catalog.required_files(source, str(tmp_path))
    assert sorted(r.relative_path for r in required) == ["2020/a.nc", "2021/b.nc"]
    assert len(calls) == 2

    # Second call: both entrypoints now cached -- no new crawl calls.
    required_again = catalog.required_files(source, str(tmp_path))
    assert len(calls) == 2
    assert sorted(r.relative_path for r in required_again) == ["2020/a.nc", "2021/b.nc"]


def test_zero_file_entrypoint_result_is_not_cached_and_retried(tmp_path):
    calls = []
    source = _EntrypointSource(entrypoints=[{"year": 2020}], files_by_year={2020: []}, crawl_calls=calls)
    catalog.required_files(source, str(tmp_path))
    catalog.required_files(source, str(tmp_path))
    assert len(calls) == 2  # re-crawled both times, never cached as "done"


def test_refresh_entrypoints_forces_recrawl(tmp_path):
    calls = []
    source = _EntrypointSource(
        entrypoints=[{"year": 2020}], files_by_year={2020: [("2020/a.nc", "https://x/a.nc")]}, crawl_calls=calls
    )
    catalog.required_files(source, str(tmp_path))
    catalog.required_files(source, str(tmp_path), refresh_entrypoints=True)
    assert len(calls) == 2


def test_crawl_exception_is_swallowed_and_entrypoint_skipped(tmp_path):
    class _BoomSource(_EntrypointSource):
        def list_remote_files(self, entrypoint=None):
            raise RuntimeError("network boom")

    source = _BoomSource(entrypoints=[{"year": 2020}], files_by_year={})
    required = catalog.required_files(source, str(tmp_path))
    assert required == []


def test_entrypoint_key_uses_year_and_day_when_present():
    assert catalog.entrypoint_key({"year": 2020}) == "2020"
    assert catalog.entrypoint_key({"year": 2020, "day": 15}) == "2020_15"
