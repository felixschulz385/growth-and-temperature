"""decode_qc_valid_mask() and BAND_SPECS: pinned against the primary sources
that resolve docs/design/06-open-questions.md #9/#10 --
  - MOD11A1: Collection-6 MODIS LST Products Users' Guide (Wan, ERI/UCSB,
    June 2019), Table 13 (QC bit layout) and Table 9 (SDS scale/offset/fill).
  - MOD21A2: MxD21 LST&E User Guide (Hulley et al., JPL, March 2019),
    Table 12 (QC bit layout) and Table 11 (SDS scale/offset/fill).
Bits 7&6 sit at the same position in both products' QC field but mean the
*opposite* thing (MOD11: increasing value = worse LST error; MOD21:
increasing value = better) -- the tests below exercise both directions so a
future edit can't silently swap one product's mapping for the other's.
"""

import numpy as np
import pytest
import xarray as xr

from src.data.sources.modis import tiles as modis_util
from src.data.sources.modis.source import BAND_SPECS


def _qc(mandatory_qa: int, error_bits: int) -> xr.DataArray:
    value = (error_bits << 6) | mandatory_qa
    return xr.DataArray(np.array([[value]], dtype="uint8"), dims=("y", "x"))


@pytest.fixture(autouse=True)
def _reset_qc_warned(monkeypatch):
    monkeypatch.setattr(modis_util, "_QC_LAYOUT_WARNED", False)


def test_decode_qc_valid_mask_good_quality_within_threshold():
    qc = _qc(mandatory_qa=0b00, error_bits=0b00)  # good pixel, error <= 1K
    valid = modis_util.decode_qc_valid_mask(qc, max_lst_error_k=2.0, product="11A1")
    assert bool(valid.values[0, 0]) is True


def test_decode_qc_valid_mask_rejects_bad_mandatory_qa():
    qc = _qc(mandatory_qa=0b10, error_bits=0b00)  # cloud-affected pixel
    valid = modis_util.decode_qc_valid_mask(qc, max_lst_error_k=2.0, product="11A1")
    assert bool(valid.values[0, 0]) is False


def test_decode_qc_valid_mask_rejects_error_above_threshold():
    qc = _qc(mandatory_qa=0b00, error_bits=0b10)  # good pixel, error <= 3K
    valid = modis_util.decode_qc_valid_mask(qc, max_lst_error_k=2.0, product="11A1")
    assert bool(valid.values[0, 0]) is False  # 3K > 2.0K threshold


def test_decode_qc_valid_mask_confirmed_product_suppresses_warning(caplog):
    qc = _qc(mandatory_qa=0b00, error_bits=0b00)
    with caplog.at_level("WARNING"):
        modis_util.decode_qc_valid_mask(qc, product="11A1")
    assert not any("UNVERIFIED" in r.message for r in caplog.records)


def test_decode_qc_valid_mask_unconfirmed_product_still_warns(caplog):
    qc = _qc(mandatory_qa=0b00, error_bits=0b00)
    with caplog.at_level("WARNING"):
        modis_util.decode_qc_valid_mask(qc, product="bogus-future-product")
    assert any("UNVERIFIED" in r.message for r in caplog.records)


def test_decode_qc_valid_mask_21a2_suppresses_warning_now_confirmed(caplog):
    qc = _qc(mandatory_qa=0b00, error_bits=0b00)
    with caplog.at_level("WARNING"):
        modis_util.decode_qc_valid_mask(qc, product="21A2")
    assert not any("UNVERIFIED" in r.message for r in caplog.records)


def test_decode_qc_valid_mask_21a2_bit_meaning_is_inverted_from_11a1():
    # Same raw bits (error_bits=00), opposite verdict: for 11A1 that means
    # the *best* category (<=1K, passes a 2.0K threshold); for 21A2 it means
    # the *worst* category (>2K, unbounded, must fail a 2.0K threshold).
    qc = _qc(mandatory_qa=0b00, error_bits=0b00)
    assert bool(modis_util.decode_qc_valid_mask(qc, max_lst_error_k=2.0, product="11A1").values[0, 0]) is True
    assert bool(modis_util.decode_qc_valid_mask(qc, max_lst_error_k=2.0, product="21A2").values[0, 0]) is False


def test_decode_qc_valid_mask_21a2_best_category_passes():
    # error_bits=11 is 21A2's *best* category (<1K) -- opposite of 11A1
    # where 11 is the worst (>3K).
    qc = _qc(mandatory_qa=0b00, error_bits=0b11)
    assert bool(modis_util.decode_qc_valid_mask(qc, max_lst_error_k=2.0, product="21A2").values[0, 0]) is True


def test_band_specs_21a2_view_angle_matches_mxd21_users_guide_table_11():
    view_angle = BAND_SPECS["21A2"]["assets"]["view_angle"]
    assert view_angle["offset"] == -65.0
    assert view_angle["fill"] == 255


def test_band_specs_21a2_view_time_fill_matches_mxd21_users_guide_table_11():
    assert BAND_SPECS["21A2"]["assets"]["view_time"]["fill"] == 255


def test_band_specs_11a1_emis_offset_matches_users_guide_table_9():
    assets = BAND_SPECS["11A1"]["assets"]
    assert assets["emis_31"]["offset"] == 0.49
    assert assets["emis_32"]["offset"] == 0.49


def test_band_specs_11a1_view_angle_matches_users_guide_table_9():
    view_angle = BAND_SPECS["11A1"]["assets"]["view_angle"]
    assert view_angle["offset"] == -65.0
    assert view_angle["fill"] == 255


def test_band_specs_11a1_view_time_fill_matches_users_guide_table_9():
    assert BAND_SPECS["11A1"]["assets"]["view_time"]["fill"] == 255
