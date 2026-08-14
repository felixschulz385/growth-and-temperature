"""Every FETCH-capable source must satisfy RemoteFileCatalog -- the one real
coupling risk against the untouched UnifiedDataIndex/AsyncHPCDownloader
(docs/design/09-integrated-pipeline.md §4/§11).

This test exists because it would have caught a real bug: the first version
of every migrated source (acag/esacci/ntl_harm/eog/glass) only ever set
`self.cfg.data_path`, never a bare `self.data_path` attribute -- so
`UnifiedDataIndex.build_index_from_source(data_source=source, ...)` would
have raised `AttributeError` on `self.data_source.data_path` the moment any
FETCH step actually ran the indexing path. Fixed by adding a `data_path`
property to `DataSource` itself (src/data/sources/base.py) rather than
trusting each subclass to remember it.
"""

import pytest

from src.data.pipeline.config import SourceConfig
from src.data.pipeline.context import PipelineContext
from src.data.sources import registry
from src.data.sources.base import RemoteFileCatalog
from src.data.sources.steps import PipelineStep

_EXTRA_CONFIG: dict[str, dict] = {
    "eog": {"base_url": "https://example.invalid/eog"},
}

#: `source_id` to actually construct with, for sources whose bare `spec.id`
#: isn't itself a valid config-block key (EOG's `source_type` -- DMSP/VIIRS-
#: annual/DVNL -- is derived from `cfg.source_id`, src/data/sources/eog/
#: source.py, so it must be built with one of its real aliases, not the
#: generic "eog"). Defaults to `spec.id` when a source has no entry here.
_CONSTRUCT_AS: dict[str, str] = {
    "eog": "eog_viirs",
}

#: FETCH-capable sources that deliberately do NOT implement RemoteFileCatalog
#: (src/data/sources/modis/source.py module docstring): MODIS's FETCH streams
#: per-(year, tile) STAC queries from Planetary Computer, not a crawlable flat
#: file list, so there is no `list_remote_files()`/`download_async()` to
#: satisfy -- it tracks per-unit state directly in the ledger's generic
#: `artifacts` table instead (`_get_ledger()`/`_execute_fetch()`). GLASS
#: (docs/design/11-glass-static-fetch.md) is the same shape: a static
#: per-(year, day[, tile]) target list attempted-and-logged directly, not a
#: crawlable remote listing -- no `get_all_entrypoints()`/`list_remote_files()`
#: either.
_CRAWLER_PROTOCOL_EXEMPT = {"modis", "glass_modis", "glass_avhrr"}


def _fetch_capable_specs():
    return [
        spec
        for spec in registry.all_specs()
        if PipelineStep.FETCH in spec.steps and spec.id not in _CRAWLER_PROTOCOL_EXEMPT
    ]


@pytest.mark.parametrize("spec", _fetch_capable_specs(), ids=lambda s: s.id)
def test_fetch_capable_source_satisfies_remote_file_catalog(spec, tmp_path):
    cls = registry.load(spec.id)
    ctx = PipelineContext(data_root=str(tmp_path / "data"), local_index_dir=str(tmp_path / "index"))
    source_id = _CONSTRUCT_AS.get(spec.id, spec.id)
    cfg = SourceConfig.from_dict(source_id, {"data_path": spec.id, **_EXTRA_CONFIG.get(spec.id, {})})
    source = cls(ctx, cfg)

    assert isinstance(source, RemoteFileCatalog), f"{spec.id} does not satisfy RemoteFileCatalog"
    # isinstance() against a runtime_checkable Protocol only checks method/attribute
    # *presence*, not that a plain attribute (data_path, DATA_SOURCE_NAME,
    # has_entrypoints) actually holds a value rather than raising -- so also
    # exercise each attribute directly, which is exactly the check that would
    # have caught the data_path bug above.
    assert isinstance(source.data_path, str) and source.data_path
    assert isinstance(source.DATA_SOURCE_NAME, str) and source.DATA_SOURCE_NAME
    assert isinstance(source.has_entrypoints, bool)
