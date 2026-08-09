"""Shared plausible-year bounds for values parsed from untrusted/enriched
content (scraped, OCR'd, or LLM-imputed) before they become a zarr `time`
coordinate.

`src.data.sources.snl_mining.source.SnlMiningSource` was the first source to
derive its zarr time range from untrusted content (SNL's own opening/closing
year columns, optionally LLM-imputed) rather than validated config -- an
out-of-range value (e.g. "150" instead of "1950", a data-entry error) used to
reach `to_zarr(region="auto")`'s CF-time auto-region-detection deep inside
GRID, not PREPARE where the bad value actually lives, and fail there far more
cryptically (an `OutOfBoundsDatetime` chained into an unrelated missing-
`cftime` ImportError). Centralized here so the next source inferring time
bounds from similarly untrusted data starts protected, instead of needing to
rediscover and copy this exact bound-checking idiom by hand.
"""

from __future__ import annotations

MIN_PLAUSIBLE_YEAR = 1800
MAX_PLAUSIBLE_YEAR = 2100


def is_plausible_year(year: int) -> bool:
    return MIN_PLAUSIBLE_YEAR <= year <= MAX_PLAUSIBLE_YEAR
