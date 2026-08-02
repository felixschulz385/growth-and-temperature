"""SourceConfig: the per-source config passed to `DataSource.from_config(ctx, cfg)`.

docs/design/09-integrated-pipeline.md §8: replaces the ~100 lines of
kwargs-smearing duplicated across PreprocessTaskHandlers.handle_preprocess/
handle_validate (src/data/preprocess/workflow.py, an 11-step "copy everything
into one flat dict with remote_/gcs_/hpc_ prefixes" ritual, done twice) with
one explicit dataclass plus a `raw` escape hatch for source-private config.

File loading and ${VAR}/${VAR:-default} expansion is NOT reimplemented here --
`src.cli.config.load_config_with_env_vars` already does that (including the
`*.local.yaml` deep-merge convention) and is what `src/cli/pipeline/handlers.py`
calls before building a SourceConfig/PipelineContext from the result.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from src.config.runtime import get_paths_config, get_remote_config
from src.data.pipeline.context import PipelineContext
from src.data.sources.layout import LEGACY_GRID_ID, LEGACY_LAYOUT

#: Fields SourceConfig models explicitly; everything else in a source's config
#: block lands in `raw` instead of being silently dropped.
_KNOWN_FIELDS = {"data_path", "namespace", "override", "temp_dir", "year_range", "type"}


@dataclass(frozen=True)
class SourceConfig:
    """Per-source config passed to `DataSource.from_config(ctx, cfg)`.

    `data_path` is deliberately `None`-able rather than defaulting to
    `source_id`: several old preprocessors have their own specific default
    when it's omitted from config (acag -> "acag/pm25", esacci ->
    "esacci/landcover", ntl_harm -> "ntl_harm/harmonized", glass -> derived
    from a MODIS/AVHRR path prefix constant) -- a generic source_id fallback
    here would silently paper over those and produce a different path than
    the old code. Each `DataSource.__init__` applies its own default (via
    `dataclasses.replace(cfg, data_path=...)`) when `cfg.data_path is None`,
    exactly mirroring its old preprocessor's default.
    """

    source_id: str
    data_path: str | None
    namespace: str | None = None
    override: bool = False
    temp_dir: str | None = None
    year_range: tuple[int, int] | None = None
    raw: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, source_id: str, data: Mapping[str, Any]) -> "SourceConfig":
        data = dict(data)
        year_range = data.get("year_range")
        return cls(
            source_id=source_id,
            data_path=data.get("data_path"),
            namespace=data.get("namespace"),
            override=bool(data.get("override", False)),
            temp_dir=data.get("temp_dir"),
            year_range=tuple(year_range) if year_range else None,
            raw={k: v for k, v in data.items() if k not in _KNOWN_FIELDS},
        )


def build_context(config: Mapping[str, Any]) -> PipelineContext:
    """Build a PipelineContext from a full pipeline config dict (the top-level
    `paths`/`remote`/`pipeline` blocks of data.yaml)."""
    paths = get_paths_config(dict(config))
    remote = get_remote_config(dict(config))
    pipeline_cfg = config.get("pipeline", {}) or {}

    return PipelineContext(
        data_root=paths.get("data_root"),
        local_index_dir=paths.get("local_index_dir"),
        ssh_target=remote.get("ssh_target"),
        key_file=remote.get("key_file"),
        grid_id=pipeline_cfg.get("grid", LEGACY_GRID_ID),
        layout=pipeline_cfg.get("layout", LEGACY_LAYOUT),
    )


def get_source_config(config: Mapping[str, Any], source_id: str) -> SourceConfig:
    """Look up `sources.<source_id>` in a full pipeline config dict and build
    its SourceConfig."""
    sources = config.get("sources", {}) or {}
    if source_id not in sources:
        raise KeyError(f"Source '{source_id}' not found in configuration. Available: {sorted(sources)}")
    return SourceConfig.from_dict(source_id, sources[source_id])
