"""EcoregionsSource.download(): paginated ArcGIS REST Feature Service query,
against a fake `requests.Session` -- no real network calls. Covers the two
failure modes confirmed live against the real service while building this
(see src/data/sources/ecoregions/source.py module docstring): a page whose
response can't be parsed (empirically, an oversized response gets silently
truncated past ~16MiB) triggers a page-size halving retry; a rate-limit
error delivered as HTTP 200 + a JSON error body (not a real 429 status)
triggers a sleep-and-retry at the same page size."""

import json
import os

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.ecoregions.source import _rate_limit_wait_seconds


def _make(tmp_path, **raw):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict("ecoregions", dict(raw))
    cls = registry.load("ecoregions")
    return cls(ctx, cfg), ctx


class _FakeResponse:
    def __init__(self, content: bytes, status: int = 200):
        self.content = content
        self.status_code = status

    def raise_for_status(self):
        pass


def _feature_geojson(n: int, start_id: int = 0) -> bytes:
    features = [
        {
            "type": "Feature",
            "properties": {
                "REALM": "Nearctic", "BIOME_NUM": 1, "BIOME_NAME": "Tundra",
                "ECO_ID": start_id + i, "ECO_NAME": f"Eco {start_id + i}",
            },
            "geometry": {"type": "Point", "coordinates": [float(i), 0.0]},
        }
        for i in range(n)
    ]
    return json.dumps({"type": "FeatureCollection", "features": features}).encode()


_RATE_LIMIT_BODY = json.dumps(
    {
        "error": {
            "code": 429,
            "message": "Unable to perform operation. Too many large geometry non-cacheable requests.",
            "details": ["API calls for large geometry quota exceeded (61)! maximum allowed (60) per Minute. Retry after 60 sec."],
        }
    }
).encode()


def test_rate_limit_wait_seconds_parses_real_error_body():
    assert _rate_limit_wait_seconds(_RATE_LIMIT_BODY) == 60


def test_rate_limit_wait_seconds_ignores_normal_feature_page():
    assert _rate_limit_wait_seconds(_feature_geojson(5)) is None


def test_download_pages_until_a_short_final_page(tmp_path, monkeypatch):
    source, _ = _make(tmp_path, page_size=3)
    monkeypatch.setattr("time.sleep", lambda s: None)

    class _FakeSession:
        def __init__(self):
            self.urls = []

        def get(self, url, timeout=None):
            self.urls.append(url)
            offset = int(url.split("resultOffset=")[1].split("&")[0])
            # 7 features total, page_size=3 -> pages of 3, 3, 1
            remaining = max(0, 7 - offset)
            n = min(3, remaining)
            return _FakeResponse(_feature_geojson(n, start_id=offset))

    session = _FakeSession()
    output_path = str(tmp_path / "out" / "ecoregions_raw.gpkg")
    source.download("https://example.test/query?f=geojson", output_path, session=session)

    assert os.path.exists(output_path)
    import geopandas as gpd

    gdf = gpd.read_file(output_path, engine="pyogrio")
    assert len(gdf) == 7
    assert len(session.urls) == 3  # offsets 0, 3, 6


def test_download_halves_page_size_on_unparseable_response(tmp_path, monkeypatch):
    source, _ = _make(tmp_path, page_size=4)
    monkeypatch.setattr("time.sleep", lambda s: None)

    class _FakeSession:
        def __init__(self):
            self.requested_sizes = []

        def get(self, url, timeout=None):
            size = int(url.split("resultRecordCount=")[1].split("&")[0])
            offset = int(url.split("resultOffset=")[1].split("&")[0])
            self.requested_sizes.append(size)
            if size == 4:
                return _FakeResponse(b"not valid json at all")
            # 3 features total -- a short final page once size=2 succeeds,
            # so the retry loop actually terminates.
            remaining = max(0, 3 - offset)
            return _FakeResponse(_feature_geojson(min(size, remaining)))

    session = _FakeSession()
    output_path = str(tmp_path / "out" / "ecoregions_raw.gpkg")
    source.download("https://example.test/query?f=geojson", output_path, session=session)

    assert session.requested_sizes[0] == 4  # first attempt at configured page_size
    assert 2 in session.requested_sizes  # halved after the failure


def test_download_sleeps_and_retries_same_page_on_rate_limit(tmp_path, monkeypatch):
    source, _ = _make(tmp_path, page_size=5)
    sleeps = []
    monkeypatch.setattr("time.sleep", lambda s: sleeps.append(s))

    calls = {"n": 0}

    class _FakeSession:
        def get(self, url, timeout=None):
            calls["n"] += 1
            if calls["n"] == 1:
                return _FakeResponse(_RATE_LIMIT_BODY)
            return _FakeResponse(_feature_geojson(2))  # short final page

    output_path = str(tmp_path / "out" / "ecoregions_raw.gpkg")
    source.download("https://example.test/query?f=geojson", output_path, session=_FakeSession())

    assert 60 in sleeps
    assert calls["n"] == 2


def test_download_raises_on_zero_features(tmp_path, monkeypatch):
    source, _ = _make(tmp_path, page_size=5)
    monkeypatch.setattr("time.sleep", lambda s: None)

    class _FakeSession:
        def get(self, url, timeout=None):
            return _FakeResponse(_feature_geojson(0))

    import pytest

    with pytest.raises(RuntimeError):
        source.download(
            "https://example.test/query?f=geojson", str(tmp_path / "out" / "ecoregions_raw.gpkg"), session=_FakeSession()
        )
