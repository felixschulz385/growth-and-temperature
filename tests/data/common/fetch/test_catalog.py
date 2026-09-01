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


def test_required_files_dedupes_a_file_shared_by_several_entrypoints(tmp_path):
    # EOG's 2012-2016 gas-flare survey: five yearly entrypoints, one
    # combined file. It must appear once, or duplicate downloads race on
    # the same "<path>.part".
    combined = ("flare_2012-2016.xlsx", "https://x/flare_2012-2016.xlsx")
    source = _EntrypointSource(
        entrypoints=[{"year": y} for y in (2012, 2013, 2014, 2015)],
        files_by_year={y: [combined] for y in (2012, 2013, 2014, 2015)},
    )
    required = catalog.required_files(source, str(tmp_path))
    assert [r.relative_path for r in required] == ["flare_2012-2016.xlsx"]


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


def test_zero_file_entrypoint_becomes_unavailable_after_max_attempts(tmp_path):
    from src.data.common.fetch import manifest

    calls = []
    source = _EntrypointSource(entrypoints=[{"year": 2020}], files_by_year={2020: []}, crawl_calls=calls)
    for _ in range(manifest.DEFAULT_MAX_ATTEMPTS):
        catalog.required_files(source, str(tmp_path))
    status = statusfile.read(statusfile.status_path(str(tmp_path), "2020"))
    assert status["status"] == manifest.STATUS_UNAVAILABLE
    assert status["attempts"] == manifest.DEFAULT_MAX_ATTEMPTS


def test_entrypoint_becoming_non_empty_clears_unavailable_status(tmp_path):
    from src.data.common.fetch import manifest

    empty_source = _EntrypointSource(entrypoints=[{"year": 2020}], files_by_year={2020: []})
    for _ in range(manifest.DEFAULT_MAX_ATTEMPTS):
        catalog.required_files(empty_source, str(tmp_path))
    assert statusfile.read(statusfile.status_path(str(tmp_path), "2020"))["status"] == manifest.STATUS_UNAVAILABLE

    recovered_source = _EntrypointSource(
        entrypoints=[{"year": 2020}], files_by_year={2020: [("2020/a.nc", "https://x/2020/a.nc")]}
    )
    catalog.required_files(recovered_source, str(tmp_path))
    assert statusfile.read(statusfile.status_path(str(tmp_path), "2020")) is None


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


class _NetworkAssertingEntrypointSource(_EntrypointSource):
    """Fails the test if a network-ish call (`list_remote_files`/
    `get_all_entrypoints`) is ever made -- for pinning `cached_required_files()`'s
    "never touches the network" contract."""

    def get_all_entrypoints(self):
        raise AssertionError("get_all_entrypoints() must not be called by cached_required_files()")

    def list_remote_files(self, entrypoint=None):
        raise AssertionError("list_remote_files() must not be called by cached_required_files()")


def test_cached_required_files_returns_none_for_non_entrypoint_source(tmp_path):
    source = _FlatSource([("a.nc", "https://x/a.nc")])
    assert catalog.cached_required_files(source, str(tmp_path)) is None


def test_cached_required_files_empty_when_nothing_cached_yet(tmp_path):
    source = _NetworkAssertingEntrypointSource(entrypoints=[{"year": 2020}], files_by_year={})
    assert catalog.cached_required_files(source, str(tmp_path)) == []


def test_cached_required_files_reads_only_the_entrypoint_cache_never_the_network(tmp_path):
    source = _EntrypointSource(
        entrypoints=[{"year": 2020}, {"year": 2021}],
        files_by_year={2020: [("2020/a.nc", "https://x/2020/a.nc")], 2021: [("2021/b.nc", "https://x/2021/b.nc")]},
    )
    # Populate the cache the normal way first (this call is allowed to crawl).
    catalog.required_files(source, str(tmp_path))

    network_free_source = _NetworkAssertingEntrypointSource(entrypoints=[], files_by_year={})
    required = catalog.cached_required_files(network_free_source, str(tmp_path))
    assert sorted(r.relative_path for r in required) == ["2020/a.nc", "2021/b.nc"]


class _YearOnlySource(_EntrypointSource):
    STATIC_ENTRYPOINTS = True

    def filename_to_entrypoint(self, relative_path):
        import re

        match = re.search(r"(\d{4})", relative_path)
        return {"year": int(match.group(1))} if match else None


def test_cached_entrypoint_counts_buckets_by_disk_presence_and_status(tmp_path):
    from src.data.common.fetch import manifest

    # 2020 never crawled -> outstanding. 2021 crawled empty repeatedly ->
    # unavailable. 2022 has a real file on disk -> complete, regardless of
    # whether it was ever crawled.
    empty_source = _YearOnlySource(entrypoints=[{"year": 2021}], files_by_year={2021: []})
    for _ in range(manifest.DEFAULT_MAX_ATTEMPTS):
        catalog.required_files(empty_source, str(tmp_path))

    source = _YearOnlySource(entrypoints=[{"year": 2020}, {"year": 2021}, {"year": 2022}], files_by_year={})
    listing = {"2022/a.nc": None}

    counts = catalog.cached_entrypoint_counts(source, str(tmp_path), listing)
    assert counts == (1, 1, 1)


def test_cached_entrypoint_counts_none_when_not_static(tmp_path):
    # _EntrypointSource declares no STATIC_ENTRYPOINTS -- e.g. ntl_harm,
    # whose get_all_entrypoints() hits the figshare API, so this must not
    # be called here (`_NetworkAssertingEntrypointSource` covers that).
    source = _EntrypointSource(entrypoints=[{"year": 2020}], files_by_year={})
    assert catalog.cached_entrypoint_counts(source, str(tmp_path), {}) is None


def test_cached_required_files_omits_uncrawled_entrypoints_silently(tmp_path):
    source = _EntrypointSource(
        entrypoints=[{"year": 2020}], files_by_year={2020: [("2020/a.nc", "https://x/2020/a.nc")]},
    )
    catalog.required_files(source, str(tmp_path))  # only 2020 gets cached

    network_free_source = _NetworkAssertingEntrypointSource(entrypoints=[], files_by_year={})
    required = catalog.cached_required_files(network_free_source, str(tmp_path))
    # 2021 was never crawled/cached -- silently absent, not an error and not
    # triggering a crawl to find out.
    assert [r.relative_path for r in required] == ["2020/a.nc"]
