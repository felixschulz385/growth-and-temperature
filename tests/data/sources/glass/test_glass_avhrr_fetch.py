"""GlassAvhrrSource's FETCH step (docs/design/11-glass-static-fetch.md): a
static per-(year, day) target list, attempted and logged directly -- no
crawl, no entrypoint cache. `daterange_doy()` (the target-space generator,
shared with GlassModisSource -- see `src/data/sources/glass/avhrr.py`) and
the listing-match/failure-branching logic are exercised directly here,
mirroring tests/data/sources/modis/test_modis_fetch_manifest.py's shape for
MODIS's own crawl-free FETCH step.

docs/design/12-glass-modis-rebuild.md §6: split off of the former
test_glass_fetch.py along the AVHRR/MODIS line -- these cases are AVHRR-
specific and unchanged in behavior; MODIS's own FETCH now has a completely
different (tile, year) target shape, covered by test_glass_modis_fetch.py.
"""

import os

import pytest
import requests

from src.data.common import statusfile
from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.glass.avhrr import GlassAvhrrSource, daterange_doy
from src.data.sources.steps import Completion, PipelineStep, TargetSelection


def _make_source(tmp_path, **extra_raw):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    base_url = "https://glass.hku.hk/archive/LST/AVHRR/0.05D/"
    day_range = extra_raw.pop("day_range", {"start": [2019, 364], "end": [2020, 1]})
    cfg = SourceConfig.from_dict("glass_avhrr", {"base_url": base_url, "day_range": day_range, **extra_raw})
    return GlassAvhrrSource(ctx, cfg), ctx


def test_daterange_doy_clips_first_and_last_year_leap_aware():
    assert list(daterange_doy((2019, 364), (2020, 2))) == [
        (2019, 364), (2019, 365),
        (2020, 1), (2020, 2),
    ]


def test_daterange_doy_single_year():
    assert list(daterange_doy((2021, 5), (2021, 7))) == [(2021, 5), (2021, 6), (2021, 7)]


def test_avhrr_plan_fetch_has_no_tile_dimension(tmp_path):
    source, _ = _make_source(tmp_path)
    targets = source.plan(PipelineStep.FETCH, TargetSelection())
    assert [t.key for t in targets] == ["2019/364", "2019/365", "2020/001"]
    assert all(t.meta["tile"] is None for t in targets)
    assert all(t.completion == Completion.PATH_EXISTS for t in targets)


def test_plan_fetch_recognizes_already_downloaded_file_as_complete(tmp_path):
    source, _ = _make_source(tmp_path)
    raw_root = source.output_root(PipelineStep.FETCH)
    os.makedirs(os.path.join(raw_root, "2019"), exist_ok=True)
    real_path = os.path.join(raw_root, "2019", "GLASS08B31.V40.A2019364.2021259.hdf")
    open(real_path, "w").close()

    targets = {t.key: t for t in source.plan(PipelineStep.FETCH, TargetSelection())}
    assert targets["2019/364"].output_path == real_path
    assert os.path.exists(targets["2019/364"].output_path)
    assert not os.path.exists(targets["2019/365"].output_path)


def test_execute_fetch_downloads_matched_file_and_clears_failure(tmp_path, monkeypatch):
    source, _ = _make_source(tmp_path)
    target = source.plan(PipelineStep.FETCH, TargetSelection(keys=("2019/364",)))[0]

    status_dir = source.output_root(PipelineStep.FETCH)
    statusfile.write(statusfile.status_path(status_dir, target.key), {"status": "retrying", "attempts": 1})

    href = "GLASS08B31.V40.A2019364.2021259.hdf"
    monkeypatch.setattr(source, "_listing_for", lambda year, day: [(href, f"https://x/{href}")])
    downloaded = {}
    monkeypatch.setattr(source, "download", lambda url, path, session=None: downloaded.setdefault("path", path))

    assert source.execute(target) is True
    assert downloaded["path"] == os.path.join(status_dir, "2019", href)
    assert statusfile.read(statusfile.status_path(status_dir, target.key)) is None


def test_execute_fetch_marks_permanently_unavailable_when_absent_from_listing(tmp_path, monkeypatch):
    source, _ = _make_source(tmp_path)
    target = source.plan(PipelineStep.FETCH, TargetSelection(keys=("2019/364",)))[0]
    monkeypatch.setattr(source, "_listing_for", lambda year, day: [("unrelated.hdf", "https://x/unrelated.hdf")])

    assert source.execute(target) is False
    status_dir = source.output_root(PipelineStep.FETCH)
    status = statusfile.read(statusfile.status_path(status_dir, target.key))
    assert status["status"] == "unavailable"


def test_execute_fetch_retries_on_transient_listing_error(tmp_path, monkeypatch):
    source, _ = _make_source(tmp_path)
    target = source.plan(PipelineStep.FETCH, TargetSelection(keys=("2019/364",)))[0]

    def _boom(year, day):
        raise requests.ConnectionError("network blip")

    monkeypatch.setattr(source, "_listing_for", _boom)

    assert source.execute(target) is False
    status_dir = source.output_root(PipelineStep.FETCH)
    status = statusfile.read(statusfile.status_path(status_dir, target.key))
    assert status["status"] == "retrying"


def test_execute_fetch_404_on_listing_is_permanent(tmp_path, monkeypatch):
    source, _ = _make_source(tmp_path)
    target = source.plan(PipelineStep.FETCH, TargetSelection(keys=("2019/364",)))[0]

    response = requests.Response()
    response.status_code = 404
    error = requests.HTTPError("404", response=response)

    def _boom(year, day):
        raise error

    monkeypatch.setattr(source, "_listing_for", _boom)

    assert source.execute(target) is False
    status_dir = source.output_root(PipelineStep.FETCH)
    status = statusfile.read(statusfile.status_path(status_dir, target.key))
    assert status["status"] == "unavailable"


def test_listing_is_memoized_across_sibling_targets(tmp_path, monkeypatch):
    source, _ = _make_source(tmp_path)
    calls = []

    def _fake_list(url):
        calls.append(url)
        return []

    monkeypatch.setattr(source, "_list_single_directory", _fake_list)
    source._listing_for(2019, 364)
    source._listing_for(2019, 364)
    assert calls == [source._listing_url(2019, 364)]
