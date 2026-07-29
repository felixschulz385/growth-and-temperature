"""Lights sources must default to area-weighted-sum resampling, not nearest.

docs/design/04-ingest.md §1: eog_viirs (raw VIIRS/DMSP) and ntl_harm
(harmonized DMSP-VIIRS) are both radiance fields and must resample by
flux-conserving sum by default -- unlike categorical/other sources sharing
`SpatialProcessor.process_spatial_standard`, which must keep "nearest".
"""

from src.data.preprocess.sources.eog import EOGPreprocessor
from src.data.preprocess.sources.ntl_harm import NTLHarmPreprocessor


def test_eog_preprocessor_defaults_to_sum_resampling(tmp_path):
    preprocessor = EOGPreprocessor(
        stage="spatial",
        year=2020,
        base_url="https://example.invalid/eog",
        data_path="eog/viirs",
        hpc_target=str(tmp_path),
    )
    assert preprocessor.resampling == "sum"


def test_eog_preprocessor_resampling_overridable(tmp_path):
    preprocessor = EOGPreprocessor(
        stage="spatial",
        year=2020,
        base_url="https://example.invalid/eog",
        data_path="eog/viirs",
        hpc_target=str(tmp_path),
        resampling="nearest",
    )
    assert preprocessor.resampling == "nearest"


def test_ntl_harm_preprocessor_defaults_to_sum_resampling(tmp_path):
    preprocessor = NTLHarmPreprocessor(
        stage="spatial",
        year=2020,
        hpc_target=str(tmp_path),
    )
    assert preprocessor.resampling == "sum"
