"""EogSource's VIIRS annual-composite entrypoint discovery (module docstring
in src/data/sources/eog/source.py): hardcoded year range (2012-2021), one
file per ingested variant (`average_masked`/`median_masked`/`cf_cvg` --
EogSource.VIIRS_VARIANTS) per year, selected out of that year's own
subdirectory, which also contains variants we don't ingest and
intermediate/rolling reprocessing periods. DMSP/DVNL keep the plain
(has_entrypoints=False) whole-directory crawl.
"""

import pytest

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources.eog.source import EogSource

_BASE_URLS = {
    "dmsp": "https://eogdata.mines.edu/wwwdata/dmsp/v4composites_rearrange/",
    "viirs": "https://eogdata.mines.edu/nighttime_light/annual/v21/",
    "dvnl": "https://eogdata.mines.edu/wwwdata/viirs_products/dvnl/",
}
_DATA_PATHS = {"dmsp": "eog/dmsp", "viirs": "eog/viirs", "dvnl": "eog/dvnl"}


@pytest.fixture(autouse=True)
def _no_real_eog_credentials_file(monkeypatch, tmp_path):
    from src.data.sources.eog import credentials as eog_credentials

    monkeypatch.setattr(eog_credentials, "DEFAULT_CREDENTIALS_PATH", tmp_path / "unused-eog-credentials.json")


@pytest.fixture(autouse=True)
def _no_real_sleep_between_year_crawls(monkeypatch):
    # _viirs_annual_listing() throttles its 10 real page loads with
    # time.sleep(1 + random()) (source.py) -- not needed against a mocked
    # _list_single_directory in these tests, and would otherwise cost every
    # test here several real seconds for no reason.
    from src.data.sources.eog import source as eog_source

    monkeypatch.setattr(eog_source.time, "sleep", lambda _seconds: None)


def _make_source(tmp_path, source_type="viirs", **raw):
    ctx = PipelineContext(data_root=str(tmp_path / "data_root"), local_index_dir=str(tmp_path / "index"))
    cfg = SourceConfig.from_dict(
        f"eog_{source_type}",
        {"data_path": _DATA_PATHS[source_type], "base_url": _BASE_URLS[source_type], **raw},
    )
    return EogSource(ctx, cfg)


def test_has_entrypoints_true_only_for_viirs_annual(tmp_path):
    assert _make_source(tmp_path, "viirs").has_entrypoints is True
    assert _make_source(tmp_path, "dmsp").has_entrypoints is False
    assert _make_source(tmp_path, "dvnl").has_entrypoints is False


def test_get_all_entrypoints_is_the_hardcoded_year_range(tmp_path):
    source = _make_source(tmp_path, "viirs")
    assert source.get_all_entrypoints() == [{"year": y} for y in range(2012, 2022)]


def test_get_all_entrypoints_empty_for_dmsp_dvnl(tmp_path):
    assert _make_source(tmp_path, "dmsp").get_all_entrypoints() == []
    assert _make_source(tmp_path, "dvnl").get_all_entrypoints() == []


# --- _viirs_annual_listing(): filename filtering ---------------------------


_DIRECTORY_HREFS = [
    # Canonical 2012 composite (end-of-year period), the three ingested
    # variants -- all kept.
    "VNL_v21_npp_201204-201212_global_vcmcfg_c202205302300.average_masked.dat.tif.gz",
    "VNL_v21_npp_201204-201212_global_vcmcfg_c202205302300.median_masked.dat.tif.gz",
    "VNL_v21_npp_201204-201212_global_vcmcfg_c202205302300.cf_cvg.dat.tif.gz",
    # Same period, a variant we don't ingest -- excluded.
    "VNL_v21_npp_201204-201212_global_vcmcfg_c202205302300.average.dat.tif.gz",
    # Intermediate/rolling reprocessing period spanning into 2013 -- doesn't
    # end in December of its own start year, excluded regardless of variant.
    "VNL_v21_npp_201204-201303_global_vcmcfg_c202205302300.average_masked.dat.tif.gz",
    # Canonical 2013 composite -- kept.
    "VNL_v21_npp_201301-201312_global_vcmcfg_c202205302301.average_masked.dat.tif.gz",
    # A file that isn't a VNL composite at all -- must not crash the regex match.
    "README.txt",
]


def _fake_entries(hrefs):
    return [(href, f"https://eogdata.mines.edu/nighttime_light/annual/v21/{href}") for href in hrefs]


def test_viirs_annual_listing_selects_ingested_variants_and_end_of_year_period(tmp_path, monkeypatch):
    source = _make_source(tmp_path, "viirs")
    monkeypatch.setattr(source, "_init_selenium_driver", lambda: None)
    monkeypatch.setattr(source, "_close_selenium_driver", lambda: None)
    # Every year's subdirectory happens to contain the same mixed listing --
    # each year's own end_month/end_year filter picks its own file out of it.
    monkeypatch.setattr(source, "_list_single_directory", lambda url: _fake_entries(_DIRECTORY_HREFS))

    listing = source._viirs_annual_listing()

    assert set(listing) == {2012, 2013}
    assert {href for href, _ in listing[2012]} == {
        "VNL_v21_npp_201204-201212_global_vcmcfg_c202205302300.average_masked.dat.tif.gz",
        "VNL_v21_npp_201204-201212_global_vcmcfg_c202205302300.median_masked.dat.tif.gz",
        "VNL_v21_npp_201204-201212_global_vcmcfg_c202205302300.cf_cvg.dat.tif.gz",
    }
    assert [href for href, _ in listing[2013]] == [
        "VNL_v21_npp_201301-201312_global_vcmcfg_c202205302301.average_masked.dat.tif.gz",
    ]


def test_viirs_annual_listing_ignores_years_outside_hardcoded_range(tmp_path, monkeypatch):
    source = _make_source(tmp_path, "viirs")
    monkeypatch.setattr(source, "_init_selenium_driver", lambda: None)
    monkeypatch.setattr(source, "_close_selenium_driver", lambda: None)
    out_of_range = "VNL_v21_npp_202201-202212_global_vcmcfg_c202301010000.average_masked.dat.tif.gz"
    monkeypatch.setattr(source, "_list_single_directory", lambda url: _fake_entries([out_of_range]))

    assert source._viirs_annual_listing() == {}


def test_viirs_annual_listing_crawls_each_years_subdirectory_once(tmp_path, monkeypatch):
    source = _make_source(tmp_path, "viirs")
    calls = []
    monkeypatch.setattr(source, "_init_selenium_driver", lambda: None)
    monkeypatch.setattr(source, "_close_selenium_driver", lambda: None)

    def _fake_list(url):
        calls.append(url)
        return _fake_entries(_DIRECTORY_HREFS)

    monkeypatch.setattr(source, "_list_single_directory", _fake_list)

    list(source.list_remote_files({"year": 2012}))
    list(source.list_remote_files({"year": 2013}))

    # One page load per year in the hardcoded range (2012-2021), not one per
    # requested year -- and the cache means a second request for an
    # already-crawled year adds no further calls.
    assert len(calls) == 10
    assert calls[0].endswith("/2012/")
    assert calls[1].endswith("/2013/")


def test_viirs_annual_listing_warns_and_keeps_first_on_multiple_matches(tmp_path, monkeypatch, caplog):
    import logging

    source = _make_source(tmp_path, "viirs")
    monkeypatch.setattr(source, "_init_selenium_driver", lambda: None)
    monkeypatch.setattr(source, "_close_selenium_driver", lambda: None)
    dup_a = "VNL_v21_npp_201204-201212_global_vcmcfg_c202205302300.average_masked.dat.tif.gz"
    dup_b = "VNL_v21_npp_201204-201212_global_vcmcfg_c202301010000.average_masked.dat.tif.gz"
    monkeypatch.setattr(source, "_list_single_directory", lambda url: _fake_entries([dup_a, dup_b]))

    with caplog.at_level(logging.WARNING):
        listing = source._viirs_annual_listing()

    assert len(listing[2012]) == 1
    assert listing[2012][0][0] == dup_a
    assert any("Multiple" in r.getMessage() for r in caplog.records)


def test_list_remote_files_yields_only_requested_year(tmp_path, monkeypatch):
    source = _make_source(tmp_path, "viirs")
    monkeypatch.setattr(source, "_init_selenium_driver", lambda: None)
    monkeypatch.setattr(source, "_close_selenium_driver", lambda: None)
    monkeypatch.setattr(source, "_list_single_directory", lambda url: _fake_entries(_DIRECTORY_HREFS))

    results_2012 = list(source.list_remote_files({"year": 2012}))
    assert len(results_2012) == 3  # average_masked + median_masked + cf_cvg
    assert all("201204-201212" in href for href, _ in results_2012)

    results_2015 = list(source.list_remote_files({"year": 2015}))
    assert results_2015 == []


# --- bare-year naming scheme (2013 on -- no month-range period) ------------


def test_viirs_annual_listing_matches_bare_year_naming_scheme(tmp_path, monkeypatch):
    # Live site behavior (not just a hypothetical): 2012 is named with a
    # month-range period ("201204-201212"), but 2013 on drop the period
    # entirely and use a bare calendar year, also switching the config
    # token from "vcmcfg" to "vcmslcfg" partway through.
    source = _make_source(tmp_path, "viirs")
    monkeypatch.setattr(source, "_init_selenium_driver", lambda: None)
    monkeypatch.setattr(source, "_close_selenium_driver", lambda: None)
    bare_year_hrefs = [
        "VNL_v21_npp_2013_global_vcmcfg_c202205302300.average.dat.tif.gz",
        "VNL_v21_npp_2013_global_vcmcfg_c202205302300.average_masked.dat.tif.gz",
        "VNL_v21_npp_2014_global_vcmslcfg_c202205302300.average_masked.dat.tif.gz",
    ]
    monkeypatch.setattr(source, "_list_single_directory", lambda url: _fake_entries(bare_year_hrefs))

    listing = source._viirs_annual_listing()

    assert set(listing) == {2013, 2014}
    assert listing[2013][0][0] == bare_year_hrefs[1]
    assert listing[2014][0][0] == bare_year_hrefs[2]


def test_filename_to_entrypoint_handles_both_naming_schemes(tmp_path):
    source = _make_source(tmp_path, "viirs")
    period_named = "VNL_v21_npp_201204-201212_global_vcmcfg_c202205302300.average_masked.dat.tif.gz"
    bare_year_named = "VNL_v21_npp_2014_global_vcmslcfg_c202205302300.average_masked.dat.tif.gz"

    assert source.filename_to_entrypoint(period_named) == {"year": 2012}
    assert source.filename_to_entrypoint(bare_year_named) == {"year": 2014}


def test_list_remote_files_falls_back_to_recursive_crawl_for_dmsp(tmp_path, monkeypatch):
    source = _make_source(tmp_path, "dmsp")
    called = []
    from src.data.sources.eog.crawler import _CrawlerMixin

    monkeypatch.setattr(_CrawlerMixin, "list_remote_files", lambda self, entrypoint=None: iter(called.append(1) or []))

    list(source.list_remote_files())
    assert called == [1]
