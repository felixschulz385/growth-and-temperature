from src.data.common.years import MAX_PLAUSIBLE_YEAR, MIN_PLAUSIBLE_YEAR, is_plausible_year


def test_is_plausible_year_accepts_bounds_inclusive():
    assert is_plausible_year(MIN_PLAUSIBLE_YEAR)
    assert is_plausible_year(MAX_PLAUSIBLE_YEAR)
    assert is_plausible_year(1950)


def test_is_plausible_year_rejects_out_of_range():
    assert not is_plausible_year(150)  # e.g. "150" typo'd for "1950"
    assert not is_plausible_year(MIN_PLAUSIBLE_YEAR - 1)
    assert not is_plausible_year(MAX_PLAUSIBLE_YEAR + 1)
