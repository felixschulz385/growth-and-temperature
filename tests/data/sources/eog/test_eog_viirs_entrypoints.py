"""EogSource's VIIRS annual-composite entrypoint discovery (module docstring
in src/data/sources/eog/source.py): hardcoded year range (2012-2021), one
`average_masked` file per year selected out of a flat directory that also
contains other variants and intermediate/rolling reprocessing periods.
DMSP/DVNL keep the plain (has_entrypoints=False) whole-directory crawl.
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
    # Canonical 2012 composite (end-of-year period, correct variant) -- kept.
    "VNL_v21_npp_201204-201212_global_vcmcfg_c202205302300.average_masked.dat.tif.gz",
    # Same period, wrong variant -- excluded.
    "VNL_v21_npp_201204-201212_global_vcmcfg_c202205302300.average.dat.tif.gz",
    "VNL_v21_npp_201204-201212_global_vcmcfg_c202205302300.cf_cvg.dat.tif.gz",
    # Intermediate/rolling reprocessing period spanning into 2013 -- doesn't
    # end in December of its own start year, excluded regardless of variant.
    "VNL_v21_npp_201204-201303_global_vcmcfg_c202205302300.average_masked.dat.tif.gz",
    # Canonical 2013 composite, correct variant -- kept.
    "VNL_v21_npp_201301-201312_global_vcmcfg_c202205302301.average_masked.dat.tif.gz",
    # A file that isn't a VNL composite at all -- must not crash the regex match.
    "README.txt",
]


def _fake_entries(hrefs):
    return [(href, f"https://eogdata.mines.edu/nighttime_light/annual/v21/{href}") for href in hrefs]


def test_viirs_annual_listing_selects_masked_variant_and_end_of_year_period(tmp_path, monkeypatch):
    source = _make_source(tmp_path, "viirs")
    monkeypatch.setattr(source, "_init_selenium_driver", lambda: None)
    monkeypatch.setattr(source, "_close_selenium_driver", lambda: None)
    monkeypatch.setattr(source, "_list_single_directory", lambda url: _fake_entries(_DIRECTORY_HREFS))

    listing = source._viirs_annual_listing()

    assert set(listing) == {2012, 2013}
    assert listing[2012] == [_fake_entries(_DIRECTORY_HREFS)[0]]
    assert listing[2013] == [_fake_entries(_DIRECTORY_HREFS)[4]]


def test_viirs_annual_listing_ignores_years_outside_hardcoded_range(tmp_path, monkeypatch):
    source = _make_source(tmp_path, "viirs")
    monkeypatch.setattr(source, "_init_selenium_driver", lambda: None)
    monkeypatch.setattr(source, "_close_selenium_driver", lambda: None)
    out_of_range = "VNL_v21_npp_202201-202212_global_vcmcfg_c202301010000.average_masked.dat.tif.gz"
    monkeypatch.setattr(source, "_list_single_directory", lambda url: _fake_entries([out_of_range]))

    assert source._viirs_annual_listing() == {}


def test_viirs_annual_listing_is_cached_across_years_in_one_process(tmp_path, monkeypatch):
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

    assert len(calls) == 1  # one crawl serves every year, not one per year


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
    assert len(results_2012) == 1
    assert "201204-201212" in results_2012[0][0]

    results_2015 = list(source.list_remote_files({"year": 2015}))
    assert results_2015 == []


def test_list_remote_files_falls_back_to_recursive_crawl_for_dmsp(tmp_path, monkeypatch):
    source = _make_source(tmp_path, "dmsp")
    called = []
    from src.data.sources.eog.crawler import _CrawlerMixin

    monkeypatch.setattr(_CrawlerMixin, "list_remote_files", lambda self, entrypoint=None: iter(called.append(1) or []))

    list(source.list_remote_files())
    assert called == [1]
