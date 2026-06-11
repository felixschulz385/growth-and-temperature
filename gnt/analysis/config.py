"""
Unified configuration reader for ``orchestration/configs/analysis.xlsx``.

This is the single source of truth for all three concerns that previously
read the workbook independently:

* **Model execution** (workflow.py / run.py) — formula and data source.
* **SLURM submission** (submit.py) — table membership,
  wall-clock time budgets.
* **Table rendering** (generate_tables.py) — table display configuration,
  output formats.

Sheets consumed
---------------
Models           one row per model specification
Models in Tables table_name / model_name / order
Tables           (optional) per-table display overrides for generate_tables
"""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_ANALYSIS_DIR = Path(__file__).parent           # gnt/analysis/
PROJECT_ROOT = _ANALYSIS_DIR.parent.parent       # project root
RESULTS_DIR = PROJECT_ROOT / "output" / "analysis"
DEFAULT_EXCEL = PROJECT_ROOT / "orchestration" / "configs" / "analysis.xlsx"

# Default wall-clock budgets by assembled dataset and estimation method.
# These replace the deprecated per-model ``max_runtime`` workbook column.
# Values are derived from the 0.4.2 log screen summary as padded, rounded
# observed maxima by dataset/method.
DEFAULT_MODEL_MAX_RUNTIMES = {
    '500m': {
        'OLS': '0-20:15:00',
        'IV': '0-17:30:00',
    },
    '1km': {
        'OLS': '0-07:15:00',
        'IV': '0-02:45:00',
    },
    '5km': {
        'OLS': '0-00:30:00',
        'IV': '0-00:10:00',
    },
    '50km': {
        'OLS': '0-00:10:00',
        'IV': '0-00:05:00',
    },
    'adm2_1km': {
        'OLS': '0-00:05:00',
        'IV': '0-00:10:00',
    }
}
DEFAULT_MODEL_MAX_RUNTIME = '0-20:15:00'

RESOLUTION_DATA_SOURCE_MAP = {
    '500m': '500m',
    '1km': '1km',
    '5km': '5km',
    '50km': '50km',
    'ADM2': 'adm2_1km',
}

DEFAULT_TEMPORAL_EXTENTS = {
    '500m': '2012-2020',
    '1km': '2000-2020',
    '5km': '1991-2020',
    '50km': '2000-2020',
    'ADM2': '1992-2020',
}

FIXED_EFFECT_LABELS = {
    '0': 'NO',
    'NO': 'NO',
    'pixel_id': 'PX',
    'PX': 'PX',
    'pixel_id + country*year': 'PX+CY',
    'PX+CY': 'PX+CY',
    'pixel_id + year': 'PX+YR',
    'PX+YR': 'PX+YR',
    'pixel_id_1km': 'PX1K',
    'PX1K': 'PX1K',
    'pixel_id_1km + country*year': 'PX1K+CY',
    'PX1K+CY': 'PX1K+CY',
    'pixel_id_1km + year': 'PX1K+YR',
    'PX1K+YR': 'PX1K+YR',
    'pixel_id_5km': 'PX5K',
    'PX5K': 'PX5K',
    'pixel_id_5km + country*year': 'PX5K+CY',
    'PX5K+CY': 'PX5K+CY',
    'pixel_id_5km + year': 'PX5K+YR',
    'PX5K+YR': 'PX5K+YR',
    'pixel_id_50km': 'PX50K',
    'PX50K': 'PX50K',
    'pixel_id_50km + country*year': 'PX50K+CY',
    'PX50K+CY': 'PX50K+CY',
    'pixel_id_50km + year': 'PX50K+YR',
    'PX50K+YR': 'PX50K+YR',
    'GID_2': 'ADM2',
    'subdivision': 'ADM2',
    'ADM2': 'ADM2',
    'GID_2 + country*year': 'ADM2+CY',
    'subdivision + country*year': 'ADM2+CY',
    'ADM2+CY': 'ADM2+CY',
    'GID_2 + year': 'ADM2+YR',
    'subdivision + year': 'ADM2+YR',
    'ADM2+YR': 'ADM2+YR',
}

FIXED_EFFECT_TERMS = {
    'NO': [],
    'PX': ['pixel_id'],
    'PX+CY': ['pixel_id', 'country*year'],
    'PX+YR': ['pixel_id', 'year'],
    'PX1K': ['pixel_id_1km'],
    'PX1K+CY': ['pixel_id_1km', 'country*year'],
    'PX1K+YR': ['pixel_id_1km', 'year'],
    'PX5K': ['pixel_id_5km'],
    'PX5K+CY': ['pixel_id_5km', 'country*year'],
    'PX5K+YR': ['pixel_id_5km', 'year'],
    'PX50K': ['pixel_id_50km'],
    'PX50K+CY': ['pixel_id_50km', 'country*year'],
    'PX50K+YR': ['pixel_id_50km', 'year'],
    'ADM2': ['subdivision'],
    'ADM2+CY': ['subdivision', 'country*year'],
    'ADM2+YR': ['subdivision', 'year'],
}

CLUSTERING_LABELS = {
    'subdivision': 'ADM2',
    'ADM2': 'ADM2',
    'country': 'Country',
    'Country': 'Country',
}

CLUSTERING_COLUMNS = {
    'ADM2': 'subdivision',
    'Country': 'country',
}

FULL_SAMPLE_SPATIAL_EXTENT = 'full_sample'
CANONICAL_PARTITIONED_SPATIAL_EXTENT_RE = re.compile(r'^(HDI|WB)_[A-Z_]+_\d{4}$')
TABLE_SPATIAL_EXTENT_ALIASES = {
    ('HDI', 'LOW'): 'LO',
    ('HDI', '> LOW'): 'ME_HI_VH',
    ('HDI', 'GTLO'): 'ME_HI_VH',
    ('HDI', 'MEDIUM'): 'ME',
    ('HDI', 'HIGH'): 'HI',
    ('HDI', 'V HIGH'): 'VH',
    ('HDI', 'VERY HIGH'): 'VH',
    ('HDI', 'V. HIGH'): 'VH',
    ('WB', 'LOW'): 'LO',
    ('WB', '> LOW'): 'LM_UM_HI',
    ('WB', 'GTLO'): 'LM_UM_HI',
    ('WB', 'LOWER MIDDLE'): 'LM',
    ('WB', 'LM'): 'LM',
    ('WB', 'UPPER MIDDLE'): 'UM',
    ('WB', 'UM'): 'UM',
    ('WB', 'HIGH'): 'HI',
}


def normalize_resolution_label(resolution: Any) -> str:
    """Return canonical resolution label used in table metadata and paths."""
    if resolution is None or pd.isna(resolution):
        return ''

    raw = str(resolution).strip()
    if raw == 'adm2_1km':
        return 'ADM2'
    if raw in RESOLUTION_DATA_SOURCE_MAP:
        return raw
    raise ValueError(f"Unsupported resolution: {resolution!r}")


def resolve_data_source_from_resolution(resolution: Any) -> str:
    """Map a resolution label to the assembled parquet dataset stem."""
    label = normalize_resolution_label(resolution)
    return RESOLUTION_DATA_SOURCE_MAP[label]


def normalize_temporal_extent_label(temporal_extent: Any, resolution: Any) -> str:
    """Return canonical ``YYYY-YYYY`` temporal extent."""
    if temporal_extent is None or pd.isna(temporal_extent) or not str(temporal_extent).strip():
        return DEFAULT_TEMPORAL_EXTENTS[normalize_resolution_label(resolution)]
    return str(temporal_extent).strip()


def normalize_spatial_extent_label(spatial_extent: Any) -> str:
    """Return canonical spatial extent label used in metadata and paths."""
    if spatial_extent is None or pd.isna(spatial_extent):
        return FULL_SAMPLE_SPATIAL_EXTENT

    raw = str(spatial_extent).strip()
    if raw == '' or raw == 'nan':
        return FULL_SAMPLE_SPATIAL_EXTENT
    return raw


def resolve_table_spatial_extent_label(
    spatial_extent: Any,
    temporal_extent: Any,
    resolution: Any,
) -> str:
    """Resolve ``Models in Tables`` display labels to canonical subset ids."""
    raw = normalize_spatial_extent_label(spatial_extent)
    if raw == FULL_SAMPLE_SPATIAL_EXTENT:
        return raw
    if CANONICAL_PARTITIONED_SPATIAL_EXTENT_RE.fullmatch(raw):
        match = re.fullmatch(r'(HDI|WB)_GTLO_(\d{4})', raw)
        if match:
            family, year = match.groups()
            bucket = TABLE_SPATIAL_EXTENT_ALIASES[(family, 'GTLO')]
            return f'{family}_{bucket}_{year}'
        return raw

    normalized = re.sub(r'[\[\]]', ' ', raw)
    normalized = normalized.replace('_', ' ')
    normalized = re.sub(r'\s+', ' ', normalized).strip()
    if normalized.lower() == 'world':
        return FULL_SAMPLE_SPATIAL_EXTENT

    family_match = re.match(r'^(HDI|WB)\s+(.+)$', normalized, flags=re.IGNORECASE)
    if not family_match:
        return normalize_spatial_extent_label(spatial_extent)

    family = family_match.group(1).upper()
    bucket_label = family_match.group(2).strip().upper()
    bucket = TABLE_SPATIAL_EXTENT_ALIASES.get((family, bucket_label))
    if bucket is None:
        raise ValueError(
            f"Unsupported Models in Tables spatial extent: {spatial_extent!r}. "
            f"Expected World or {family} label."
        )

    temporal_label = normalize_temporal_extent_label(temporal_extent, resolution)
    start_year = int(temporal_label.split('-', 1)[0])
    return f'{family}_{bucket}_{start_year - 1}'


def build_temporal_extent_sql(temporal_extent: str) -> str:
    """Convert ``YYYY-YYYY`` to an inclusive SQL year filter."""
    match = re.fullmatch(r'(\d{4})\s*-\s*(\d{4})', str(temporal_extent).strip())
    if not match:
        raise ValueError(f"Invalid temporal extent: {temporal_extent!r}")
    start_year, end_year = match.groups()
    return f"(year >= {start_year}) AND (year <= {end_year})"


def normalize_fixed_effects_label(fixed_effects: Any) -> str:
    """Return canonical FE label used in table metadata and paths."""
    if fixed_effects is None or pd.isna(fixed_effects):
        return 'NO'

    raw = str(fixed_effects).strip()
    if raw == '' or raw == 'nan':
        return 'NO'
    try:
        return FIXED_EFFECT_LABELS[raw]
    except KeyError as exc:
        raise ValueError(f"Unsupported fixed effects specification: {fixed_effects!r}") from exc


def fixed_effect_terms_from_label(label: str) -> List[str]:
    """Return formula FE terms for a canonical FE label."""
    try:
        return FIXED_EFFECT_TERMS[label]
    except KeyError as exc:
        raise ValueError(f"Unsupported fixed effects label: {label!r}") from exc


def normalize_clustering_label(clustering: Any, resolution: Any) -> str:
    """Return canonical clustering label, applying resolution-based defaults."""
    if clustering is None or pd.isna(clustering) or not str(clustering).strip():
        resolution_label = normalize_resolution_label(resolution)
        return 'Country' if resolution_label in ('50km', 'ADM2') else 'ADM2'

    raw = str(clustering).strip()
    try:
        return CLUSTERING_LABELS[raw]
    except KeyError as exc:
        raise ValueError(f"Unsupported clustering specification: {clustering!r}") from exc


def cluster_column_from_label(label: str) -> str:
    """Return the physical clustering column for a canonical label."""
    try:
        return CLUSTERING_COLUMNS[label]
    except KeyError as exc:
        raise ValueError(f"Unsupported clustering label: {label!r}") from exc


def normalize_sql_query(query: str) -> str:
    """Convert workbook query syntax to SQL."""
    sql = str(query).strip()
    sql = sql.replace(' & ', ' AND ').replace(' | ', ' OR ')
    sql = re.sub(r'(?<![<>=!])==', '=', sql)
    return sql


def strip_temporal_conditions(query: str) -> str:
    """Remove simple year predicates from a workbook query string."""
    cleaned = str(query).strip()
    if not cleaned or cleaned == 'nan':
        return ''

    patterns = [
        r'\(\s*year\s*>=\s*\d{4}\s*\)\s*AND\s*\(\s*year\s*<=\s*\d{4}\s*\)',
        r'\(\s*year\s*>=\s*\d{4}\s*\)\s*AND\s*\(\s*year\s*<\s*\d{4}\s*\)',
        r'year\s*>=\s*\d{4}\s*AND\s*year\s*<=\s*\d{4}',
        r'year\s*>=\s*\d{4}\s*AND\s*year\s*<\s*\d{4}',
        r'\(?\s*year\s*[<>]=?\s*\d{4}\s*\)?',
    ]
    for pattern in patterns:
        cleaned = re.sub(pattern, '', cleaned, flags=re.IGNORECASE)

    cleaned = re.sub(r'\(\s*\)', '', cleaned)
    cleaned = re.sub(r'\s+(AND|OR)\s+(AND|OR)\s+', r' \1 ', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'^\s*(AND|OR)\s+', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+(AND|OR)\s*$', '', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned


def expand_data_source_path(data_source: str) -> str:
    """Resolve a dataset stem or explicit path to a parquet path."""
    data_source = str(data_source).strip()
    if not data_source.startswith('/') and not data_source.startswith('$'):
        return os.path.expandvars(f"data_nobackup/assembled/{data_source}.parquet")
    return os.path.expandvars(data_source)


# ---------------------------------------------------------------------------
# Runtime helpers (used by both config.py and slurm.py)
# ---------------------------------------------------------------------------

def parse_runtime_to_seconds(runtime_str: str) -> int:
    """Parse ``HH:MM:SS`` or ``D-HH:MM:SS`` to integer seconds."""
    runtime_str = str(runtime_str).strip()
    days = 0
    if '-' in runtime_str:
        day_part, time_part = runtime_str.split('-', 1)
        days = int(day_part)
    else:
        time_part = runtime_str

    parts = time_part.split(':')
    if len(parts) == 3:
        h, m, s = int(parts[0]), int(parts[1]), int(parts[2])
    elif len(parts) == 2:
        h, m, s = 0, int(parts[0]), int(parts[1])
    else:
        raise ValueError(f"Cannot parse runtime: {runtime_str!r}")

    return days * 86400 + h * 3600 + m * 60 + s


def seconds_to_slurm_time(seconds: int) -> str:
    """Convert integer seconds to ``D-HH:MM:SS`` SLURM format."""
    days, remainder = divmod(seconds, 86400)
    hours, remainder = divmod(remainder, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{days}-{hours:02d}:{minutes:02d}:{secs:02d}"


def normalize_runtime_dataset(data_source: Any) -> str:
    """Return the dataset key used by ``DEFAULT_MODEL_MAX_RUNTIMES``."""
    if data_source is None or pd.isna(data_source):
        return ''

    data_source_str = os.path.expandvars(str(data_source).strip())
    if not data_source_str or data_source_str == 'nan':
        return ''

    if data_source_str == 'ADM2':
        return 'adm2_1km'

    path = Path(data_source_str)
    if path.suffix == '.parquet':
        return path.stem
    return data_source_str


def infer_runtime_method(instruments: Any) -> str:
    """Infer whether a model should use the OLS or IV runtime budget."""
    if instruments is None or pd.isna(instruments):
        return 'OLS'

    instruments_str = str(instruments).strip()
    return 'IV' if instruments_str not in ('', '0', 'nan') else 'OLS'


def get_default_model_runtime(data_source: Any, instruments: Any) -> str:
    """Return the configured runtime for a model's dataset and method."""
    dataset = normalize_runtime_dataset(data_source)
    method = infer_runtime_method(instruments)
    return DEFAULT_MODEL_MAX_RUNTIMES.get(dataset, {}).get(
        method,
        DEFAULT_MODEL_MAX_RUNTIME,
    )


# ---------------------------------------------------------------------------
# AnalysisConfig
# ---------------------------------------------------------------------------

class AnalysisConfig:
    """Single interface for reading ``analysis.xlsx``.

    Parameters
    ----------
    excel_path:
        Path to the Excel workbook.  Defaults to
        ``orchestration/configs/analysis.xlsx``.
    results_dir:
        Base directory where analysis results live.  Defaults to
        ``output/analysis`` inside the project root.
    """

    def __init__(
        self,
        excel_path: Optional[str | Path] = None,
        results_dir: Optional[str | Path] = None,
    ):
        self.excel_path = Path(excel_path) if excel_path else DEFAULT_EXCEL
        if not self.excel_path.exists():
            raise FileNotFoundError(
                f"Analysis Excel workbook not found: {self.excel_path}"
            )
        self.base_path = Path(results_dir) if results_dir else RESULTS_DIR
        self._xl: Optional[pd.ExcelFile] = None
        self._cache: Dict[str, Any] = {}

    # ── lazy sheet accessors ────────────────────────────────────────────────

    def _xl_file(self) -> pd.ExcelFile:
        if self._xl is None:
            self._xl = pd.ExcelFile(self.excel_path)
        return self._xl

    def _sheet(self, name: str) -> Optional[pd.DataFrame]:
        key = f"_sheet_{name}"
        if key not in self._cache:
            xl = self._xl_file()
            self._cache[key] = (
                pd.read_excel(xl, sheet_name=name)
                if name in xl.sheet_names
                else None
            )
        return self._cache[key]

    # ── convenience properties ──────────────────────────────────────────────

    @property
    def df_models(self) -> pd.DataFrame:
        """``Models`` sheet as a raw DataFrame."""
        df = self._sheet('Models')
        if df is None:
            raise ValueError("'Models' sheet not found in workbook")
        return df

    @property
    def df_models_in_tables(self) -> pd.DataFrame:
        """``Models in Tables`` sheet as a raw DataFrame."""
        df = self._sheet('Models in Tables')
        if df is None:
            raise ValueError("'Models in Tables' sheet not found in workbook")
        return df

    def _numbered_models_in_tables(self) -> pd.DataFrame:
        """Return ``Models in Tables`` rows with a numeric ``order`` only."""
        rows = self.df_models_in_tables.copy()
        rows['_order_numeric'] = pd.to_numeric(rows.get('order'), errors='coerce')
        rows = rows[rows['_order_numeric'].notna()]
        return rows.sort_values('_order_numeric')

    @property
    def df_tables(self) -> Optional[pd.DataFrame]:
        """``Tables`` sheet, or *None* if the sheet does not exist."""
        return self._sheet('Tables')

    # ── Model specs ─────────────────────────────────────────────────────────

    def get_model_names(self) -> List[str]:
        """Return all model names defined in the ``Models`` sheet."""
        return self.df_models['model_name'].dropna().str.strip().tolist()

    def _get_model_row(self, model_name: str) -> pd.Series:
        """Return the workbook row for *model_name* from the ``Models`` sheet."""
        df = self.df_models
        rows = df[df['model_name'].str.strip() == model_name]
        if rows.empty:
            available = df['model_name'].str.strip().tolist()
            raise KeyError(
                f"Model '{model_name}' not found in 'Models' sheet. "
                f"Available: {available}"
            )
        return rows.iloc[0]

    def get_model_spec(
        self,
        model_name: str,
        *,
        fixed_effects: Optional[str] = None,
        resolution: Optional[str] = None,
        clustering: Optional[str] = None,
        temporal_extent: Optional[str] = None,
        spatial_extent: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Return the full specification dict for *model_name*.

        The returned dict mirrors the structure expected by
        :func:`~gnt.analysis.runner.run_duckreg`:

        .. code-block:: python

            {
                'description': str,
                'data_source': str,           # expanded path
                'formula': str,               # patsy-style
                'cluster1_col': str,          # optional
                'query': str,                 # optional SQL WHERE
                'spatial_extent': str,        # canonical subset shorthand
            }
        """
        row = self._get_model_row(model_name)

        # ── formula ─────────────────────────────────────────────────────────
        dependent = str(row['dependent']).strip()
        independent = str(row['independent']).strip()

        fe_label = normalize_fixed_effects_label(
            fixed_effects if fixed_effects is not None else row.get('fixed_effects', '0')
        )
        fe_terms = fixed_effect_terms_from_label(fe_label)

        instruments_raw = str(row.get('instruments', '0')).strip()
        has_iv = instruments_raw not in ('0', 'nan', '')

        formula = f"{dependent} ~ {independent}"
        if fe_terms:
            formula += " | " + " + ".join(fe_terms)
        if has_iv:
            formula += f" | {instruments_raw}"

        # ── data source ──────────────────────────────────────────────────────
        resolution_label = normalize_resolution_label(
            resolution if resolution is not None else row.get('data_source')
        )
        data_source = expand_data_source_path(
            resolve_data_source_from_resolution(resolution_label)
        )
        temporal_label = normalize_temporal_extent_label(temporal_extent, resolution_label)

        spec: Dict[str, Any] = {
            'description': (
                f"{row.get('section', 'Analysis')} - {row.get('subsection', model_name)}"
            ),
            'model_name': model_name,
            'data_source': data_source,
            'formula': formula,
            'fixed_effects_label': fe_label,
            'resolution': resolution_label,
            'temporal_extent': temporal_label,
            'spatial_extent': resolve_table_spatial_extent_label(
                spatial_extent,
                temporal_label,
                resolution_label,
            ),
        }

        cluster_label = normalize_clustering_label(
            clustering if clustering is not None else row.get('clustering'),
            resolution_label,
        )
        spec['clustering'] = cluster_label
        spec['cluster1_col'] = cluster_column_from_label(cluster_label)

        query = strip_temporal_conditions(str(row.get('query', '')).strip())
        if query and query != 'nan':
            spec['query'] = query

        spec['variant_path'] = [
            model_name,
            fe_label,
            resolution_label,
            temporal_label,
            spec['spatial_extent'],
            cluster_label,
        ]

        return spec

    def get_all_model_specs(self) -> Dict[str, Dict[str, Any]]:
        """Return a ``{model_name: spec_dict}`` mapping for all models."""
        return {name: self.get_model_spec(name) for name in self.get_model_names()}

    # ── Table membership ────────────────────────────────────────────────────

    def get_all_table_names(self) -> List[str]:
        """Return unique table names in ``Models in Tables`` sheet order."""
        return (
            self._numbered_models_in_tables()['table_name']
            .dropna()
            .unique()
            .tolist()
        )

    def get_models_for_table(self, table_name: str) -> List[str]:
        """Return model names for *table_name* sorted by the ``order`` column."""
        mit = self._numbered_models_in_tables()
        rows = mit[mit['table_name'] == table_name]
        if rows.empty:
            available = self.get_all_table_names()
            raise KeyError(
                f"Table '{table_name}' not found in 'Models in Tables'. "
                f"Available: {available}"
            )
        return rows['model_name'].tolist()

    def get_table_model_specs(self, table_name: str) -> List[Dict[str, Any]]:
        """Return ordered variant-aware model specs for *table_name*."""
        mit = self._numbered_models_in_tables()
        rows = mit[mit['table_name'] == table_name]
        if rows.empty:
            available = self.get_all_table_names()
            raise KeyError(
                f"Table '{table_name}' not found in 'Models in Tables'. "
                f"Available: {available}"
            )

        specs: List[Dict[str, Any]] = []
        for _, row in rows.iterrows():
            specs.append(
                self.get_model_spec(
                    str(row['model_name']).strip(),
                    fixed_effects=row.get('Fixed Effects'),
                    resolution=row.get('Resolution'),
                    clustering=row.get('Clustering'),
                    temporal_extent=row.get('Temporal Extent'),
                    spatial_extent=row.get('Spatial Extent'),
                )
            )
        return specs

    # ── Runtime budgets (SLURM) ─────────────────────────────────────────────

    def get_model_runtime_seconds(self, model_name: str) -> int:
        """Return the derived runtime budget for *model_name* in seconds."""
        row = self._get_model_row(model_name)
        runtime_str = get_default_model_runtime(
            row.get('data_source'),
            row.get('instruments'),
        )
        return parse_runtime_to_seconds(runtime_str)

    def get_model_runtime_seconds_for_spec(self, spec: Dict[str, Any]) -> int:
        """Return the derived runtime budget for a fully resolved model spec."""
        row = self._get_model_row(spec['model_name'])
        runtime_str = get_default_model_runtime(
            resolve_data_source_from_resolution(spec.get('resolution', row.get('data_source'))),
            row.get('instruments'),
        )
        return parse_runtime_to_seconds(runtime_str)

    def get_table_runtime_seconds(self, table_name: str) -> int:
        """Return the summed derived runtime for all models in *table_name*."""
        return sum(
            self.get_model_runtime_seconds_for_spec(spec)
            for spec in self.get_table_model_specs(table_name)
        )

    def get_table_runtime_slurm(self, table_name: str) -> str:
        """Return SLURM-formatted combined runtime for *table_name*."""
        return seconds_to_slurm_time(self.get_table_runtime_seconds(table_name))

    # ── Table display config (generate_tables) ──────────────────────────────

    def get_models_for_table_with_labels(self, table_name: str) -> List[Tuple[str, Optional[str]]]:
        """Return ``[(model_name, model_label), …]`` for *table_name*, sorted by ``order``.

        ``model_label`` is *None* when the column is absent or the cell is empty.
        """
        mit = self._numbered_models_in_tables()
        rows = mit[mit['table_name'] == table_name]
        if rows.empty:
            available = self.get_all_table_names()
            raise KeyError(
                f"Table '{table_name}' not found in 'Models in Tables'. "
                f"Available: {available}"
            )
        result: List[Tuple[str, Optional[str]]] = []
        for _, row in rows.iterrows():
            name = str(row['model_name']).strip()
            label_raw = row.get('model_label')
            label: Optional[str] = (
                str(label_raw).strip()
                if label_raw is not None and pd.notna(label_raw) and str(label_raw).strip()
                else None
            )
            result.append((name, label))
        return result

    def get_table_display_config(self, table_name: str) -> Dict[str, Any]:
        """Return a ``create_regression_table``-compatible kwarg dict.

        Reads per-table display overrides from the optional ``Tables`` sheet.
        The ``model_paths`` key always contains the ordered list of model names
        for *table_name*.

        ``Tables`` sheet columns recognised:
        ``caption``, ``notes`` (``|``-separated list), ``show_first_stage``,
        ``table_environment``, ``table_size`` (alias: ``fontsize``), ``decimals``, ``stars``,
        ``include_ci``, ``scale``, ``model_display_names`` (comma/semicolon-separated),
        ``output_formats``, ``show_stats``,
        ``select_coefs_keys`` / ``select_coefs_labels`` (semicolon-separated),
        ``select_instruments_keys`` / ``select_instruments_labels``

        ``Models in Tables`` sheet columns recognised:
        ``model_label`` — used as column display names when
        ``model_display_names`` is not set in the ``Tables`` sheet.
        ``Spatial Extent`` — accepts ``World`` plus human-readable
        ``HDI …`` / ``WB …`` labels and resolves them to year-stamped
        canonical sample splits using ``start_year - 1``.
        ``Dependent Variable`` — rendered as a grouped header row above
        ``model_label``/model display names when consecutive values match.
        Any additional columns beyond ``order``, ``table_name``,
        ``model_name``, ``model_label``, ``Dependent Variable``,
        ``Instrument`` are collected into a
        ``custom_rows`` dict (``{column_name: [value_per_model]}``) and
        appended to the bottom block of the rendered table.
        """
        model_label_pairs = self.get_models_for_table_with_labels(table_name)
        models = [name for name, _ in model_label_pairs]
        labels = [label for _, label in model_label_pairs]
        config: Dict[str, Any] = {
            'model_paths': [
                {
                    'model_name': spec['model_name'],
                    'fixed_effects': spec['fixed_effects_label'],
                    'resolution': spec['resolution'],
                    'temporal_extent': spec['temporal_extent'],
                    'spatial_extent': spec['spatial_extent'],
                    'clustering': spec['clustering'],
                }
                for spec in self.get_table_model_specs(table_name)
            ]
        }

        # Use model_label column from Models in Tables as default display names
        if any(lbl is not None for lbl in labels):
            config['model_names'] = [
                lbl if lbl is not None else name
                for name, lbl in model_label_pairs
            ]

        mit = self._numbered_models_in_tables()
        mit_rows = mit[mit['table_name'] == table_name]

        if 'Dependent Variable' in mit_rows.columns:
            dep_values = []
            for _, row in mit_rows.iterrows():
                val = row.get('Dependent Variable')
                dep_values.append(
                    str(val).strip()
                    if val is not None and pd.notna(val) and str(val).strip()
                    else ''
                )

            if any(dep_values):
                header_row: List[Tuple[str, slice]] = []
                start = 0
                while start < len(dep_values):
                    end = start + 1
                    while end < len(dep_values) and dep_values[end] == dep_values[start]:
                        end += 1
                    header_row.append((dep_values[start], slice(start, end)))
                    start = end

                existing_header_rows = config.get('header_rows') or []
                config['header_rows'] = [header_row] + list(existing_header_rows)

        if self.df_tables is None:
            return config

        trows = self.df_tables[self.df_tables['table_name'] == table_name]
        if trows.empty:
            return config

        r = trows.iloc[0]

        # scalar string overrides
        for col in ('caption', 'show_first_stage', 'table_environment', 'table_size'):
            val = r.get(col)
            if val is not None and pd.notna(val):
                config[col] = str(val).strip()

        # 'fontsize' is a user-friendly alias for 'table_size' (takes precedence)
        fontsize_val = r.get('fontsize')
        if fontsize_val is not None and pd.notna(fontsize_val):
            config['table_size'] = str(fontsize_val).strip()

        scale_val = r.get('scale')
        if scale_val is not None and pd.notna(scale_val):
            config['table_scale'] = str(scale_val).strip()

        # notes: pipe-separated → list of strings
        notes_val = r.get('notes')
        if notes_val is not None and pd.notna(notes_val):
            notes_parts = [s.strip() for s in str(notes_val).split('|') if s.strip()]
            config['notes'] = notes_parts if len(notes_parts) > 1 else notes_parts[0]

        # scalar int override
        for col in ('decimals',):
            val = r.get(col)
            if val is not None and pd.notna(val):
                config[col] = int(val)

        # scalar bool overrides (handle both Python bool and Excel string "True"/"False")
        for col in ('stars', 'include_ci'):
            val = r.get(col)
            if val is not None and pd.notna(val):
                if isinstance(val, bool):
                    config[col] = val
                elif isinstance(val, str):
                    config[col] = val.strip().lower() not in ('false', '0', 'no', '')
                else:
                    config[col] = bool(val)

        # comma-or-semicolon separated list helper
        def _split(col_name: str, sep: str = None) -> Optional[List[str]]:
            val = r.get(col_name)
            if val is None or pd.isna(val):
                return None
            raw = str(val).strip()
            if not raw:
                return None
            # auto-detect separator: prefer ';' when present, else ','
            delimiter = sep or (';' if ';' in raw else ',')
            return [s.strip() for s in raw.split(delimiter) if s.strip()]

        # model display names from Tables sheet override model_label
        disp = _split('model_display_names')
        if disp:
            config['model_names'] = disp
        fmts = _split('output_formats')
        if fmts:
            config['output_formats'] = fmts
        stats = _split('show_stats')
        if stats:
            config['show_stats'] = stats

        # select_coefs: zip keys and labels into a dict
        coef_keys = _split('select_coefs_keys', sep=';')
        coef_labels = _split('select_coefs_labels', sep=';')
        if coef_keys:
            if coef_labels and len(coef_labels) == len(coef_keys):
                config['select_coefs'] = dict(zip(coef_keys, coef_labels))
            else:
                config['select_coefs'] = {k: k for k in coef_keys}

        # select_instruments: zip keys and labels into a dict
        inst_keys = _split('select_instruments_keys', sep=';')
        inst_labels = _split('select_instruments_labels', sep=';')
        if inst_keys:
            if inst_labels and len(inst_labels) == len(inst_keys):
                config['select_instruments'] = dict(zip(inst_keys, inst_labels))
            else:
                config['select_instruments'] = {k: k for k in inst_keys}

        # Extra columns in Models in Tables → custom_rows at bottom of table
        _mit_standard_cols = {
            'order', 'table_name', 'model_name', 'model_label',
            'Dependent Variable', 'Instrument',
        }
        extra_cols = [
            c for c in self.df_models_in_tables.columns
            if c not in _mit_standard_cols
        ]
        if extra_cols:
            extra_custom: Dict[str, List] = {}
            for col in extra_cols:
                vals = []
                for _, row in mit_rows.iterrows():
                    v = row.get(col)
                    vals.append(
                        str(v).strip() if v is not None and pd.notna(v) else ''
                    )
                # Only include the column when at least one model has a value
                if any(v != '' for v in vals):
                    extra_custom[col] = vals
            if extra_custom:
                # Merge with any custom_rows already in config
                existing = config.get('custom_rows')
                if existing is None:
                    config['custom_rows'] = extra_custom
                elif isinstance(existing, dict):
                    config['custom_rows'] = [existing, extra_custom]
                else:
                    config['custom_rows'] = list(existing) + [extra_custom]

        return config

    # ── Convenience / diagnostics ───────────────────────────────────────────

    def get_missing_models(self, analysis_type: str = 'duckreg') -> Dict[str, List[str]]:
        """Return ``{table_name: [missing_model_names]}`` for models without results.

        A model is considered *missing* when its result directory does not
        exist under ``self.base_path``.
        """
        missing: Dict[str, List[str]] = {}
        for table in self.get_all_table_names():
            absent = [
                spec['model_name']
                for spec in self.get_table_model_specs(table)
                if not (self.base_path / analysis_type / Path(*spec['variant_path'])).exists()
            ]
            if absent:
                missing[table] = absent
        return missing

    def table_runtime_summary(self) -> pd.DataFrame:
        """Return a DataFrame with runtime info for every table.

        Columns: ``table_name``, ``n_models``, ``total_seconds``,
        ``slurm_time``.
        """
        rows = []
        for table in self.get_all_table_names():
            models = self.get_models_for_table(table)
            secs = self.get_table_runtime_seconds(table)
            rows.append({
                'table_name': table,
                'n_models': len(models),
                'total_seconds': secs,
                'slurm_time': seconds_to_slurm_time(secs),
            })
        return pd.DataFrame(rows)

    def as_workflow_config(self) -> Dict[str, Any]:
        """Return a dict compatible with the legacy ``workflow.py`` ``load_config`` format."""
        return {
            'analyses': {
                'duckreg': {
                    'specifications': self.get_all_model_specs(),
                    'defaults': {},
                }
            },
            'output': {'base_path': str(self.base_path)},
        }
