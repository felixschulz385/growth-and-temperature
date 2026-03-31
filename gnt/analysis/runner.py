"""
Analysis execution layer.

Provides:
* :func:`load_subset` — load country IDs from a subset JSON file
* :func:`build_geographic_query` — build a SQL ``WHERE`` clause from subset / country config
* :func:`run_duckreg` — execute a single model via duckreg and persist results
* :func:`cleanup_analysis_results` — prune old result files, keeping the latest per version group
"""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

from .config import AnalysisConfig, PROJECT_ROOT


# ---------------------------------------------------------------------------
# Geographic helpers
# ---------------------------------------------------------------------------

def load_subset(subset_name: str, subsets_dir: Optional[Path] = None) -> List[int]:
    """Load country IDs from a subset JSON file.

    Subset files live in ``data_nobackup/subsets/``.  Two-letter uppercase
    codes (e.g. ``'AF'``) are resolved to ``continent_af.json``.

    Parameters
    ----------
    subset_name:
        Subset identifier: bare name, two-letter continent code, or filename
        ending in ``.json``.
    subsets_dir:
        Override the default subset directory.

    Raises
    ------
    FileNotFoundError
        When the subset directory or file does not exist.
    """
    logger = logging.getLogger(__name__)
    if subsets_dir is None:
        subsets_dir = PROJECT_ROOT / "data_nobackup" / "subsets"

    if not subsets_dir.exists():
        raise FileNotFoundError(f"Subsets directory not found: {subsets_dir}")

    if len(subset_name) == 2 and subset_name.isupper():
        subset_file = subsets_dir / f"continent_{subset_name.lower()}.json"
    elif subset_name.endswith('.json'):
        subset_file = subsets_dir / subset_name
    else:
        subset_file = subsets_dir / f"{subset_name}.json"

    if not subset_file.exists():
        available = [f.stem for f in subsets_dir.glob("*.json")]
        raise FileNotFoundError(
            f"Subset '{subset_name}' not found. Available: {available}"
        )

    with open(subset_file) as fh:
        data = json.load(fh)

    country_ids: List[int] = data['country_ids']
    logger.info(
        f"Loaded subset '{data.get('name', subset_name)}': {len(country_ids)} countries"
    )
    return country_ids


def build_geographic_query(spec_config: Dict[str, Any]) -> Optional[str]:
    """Build a Pandas-style query string for geographic filtering.

    Reads ``subset``, ``countries``, and ``country_col`` from *spec_config*.
    Returns ``None`` when no geographic filter is requested.
    """
    subset_name = spec_config.get('subset')
    country_filter = spec_config.get('countries')
    country_col = spec_config.get('country_col', 'country')

    queries = []
    if subset_name:
        ids = load_subset(subset_name)
        if ids:
            queries.append(f"{country_col}.isin({ids})")

    if country_filter:
        if isinstance(country_filter, list):
            queries.append(f"{country_col}.isin({country_filter})")
        else:
            queries.append(f"{country_col} == {country_filter}")

    return " & ".join(f"({q})" for q in queries) if queries else None


# ---------------------------------------------------------------------------
# Main execution function
# ---------------------------------------------------------------------------

def run_duckreg(
    config: AnalysisConfig,
    spec_name: str,
    output_dir: Optional[str] = None,
    verbose: bool = True,
    dataset_override: Optional[str] = None,
) -> Any:
    """Run a single DuckReg model and optionally persist results.

    Parameters
    ----------
    config:
        :class:`~gnt.analysis.config.AnalysisConfig` instance.
    spec_name:
        Name of the model specification (must exist in the ``Models`` sheet).
    output_dir:
        If given, results JSON, CSV, and TXT are written here under
        ``duckreg/<spec_name>/``.
    verbose:
        Log the analysis summary block (default ``True``).
    dataset_override:
        Use this path as the data source instead of the one in the spec.

    Returns
    -------
    duckreg model object
    """
    from duckreg import duckreg
    from duckreg.utils.summary import format_model_summary

    logger = logging.getLogger(__name__)

    spec_config = config.get_model_spec(spec_name)
    defaults = config.get_defaults()
    settings = {**defaults, **spec_config.get('settings', {})}

    data_source = dataset_override or spec_config['data_source']
    formula = spec_config.get('formula')
    if not formula:
        raise ValueError(f"Model '{spec_name}' has no formula")

    # ── build SQL WHERE clause ────────────────────────────────────────────────
    sql_where: Optional[str] = None
    subset_name = spec_config.get('subset')
    if subset_name:
        country_col = spec_config.get('country_col', 'country')
        country_ids = load_subset(subset_name)
        sql_where = f"{country_col} IN ({','.join(map(str, country_ids))})"

    user_query = spec_config.get('query')
    if user_query:
        user_sql = user_query.replace(' & ', ' AND ').replace(' | ', ' OR ')
        sql_where = f"({sql_where}) AND ({user_sql})" if sql_where else user_sql

    # ── scratch DB path ───────────────────────────────────────────────────────
    wd = os.environ.get('WD', str(PROJECT_ROOT))
    slurm_job_id = os.environ.get('SLURM_JOB_ID', 'local')
    scratch_path = (
        Path(wd) / 'scratch_nobackup' / slurm_job_id / 'duckreg' / f'analysis_{spec_name}'
    )
    scratch_path.mkdir(exist_ok=True, parents=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    db_name = str(scratch_path / f'duckreg_{spec_name}_{timestamp}.db')

    # ── SE method ─────────────────────────────────────────────────────────────
    fitter = settings.get('fitter', 'duckdb')
    se_method_raw = settings.get('se_method', 'HC1')
    cluster_col = spec_config.get('cluster1_col')
    if cluster_col and isinstance(se_method_raw, str) and se_method_raw.startswith('CRV'):
        se_method: Any = {se_method_raw: cluster_col}
    else:
        se_method = se_method_raw

    # ── DuckDB resource kwargs ────────────────────────────────────────────────
    threads = int(settings.get('threads', 1))
    memory_limit = settings.get('memory_limit')
    if isinstance(memory_limit, str):
        memory_limit = os.path.expandvars(memory_limit)
    max_temp_directory_size = settings.get('max_temp_directory_size')
    if isinstance(max_temp_directory_size, str):
        max_temp_directory_size = os.path.expandvars(max_temp_directory_size)

    if verbose:
        info_lines = [
            "=== DuckReg analysis summary ===",
            f"Description: {spec_config['description']}",
            f"Data source:  {data_source}",
            f"Formula:      {formula}",
        ]
        if sql_where:
            info_lines.append(f"Filter:       {sql_where}")
        info_lines.append("Settings:")
        for k, v in settings.items():
            info_lines.append(f"  {k}: {v}")
        info_lines.append(f"Fitter: {fitter}, SE method: {se_method}")
        info_lines.append(f"Database:     {db_name}")
        info_lines.append("=" * 30)
        logger.info("\n" + "\n".join(info_lines))

    # ── fit ───────────────────────────────────────────────────────────────────
    _n_bootstraps = settings.get('n_bootstraps', 0)
    bootstrap_config = {'n': _n_bootstraps} if _n_bootstraps and _n_bootstraps > 0 else None

    resource_kwargs: Dict[str, Any] = {'threads': threads}
    if memory_limit is not None:
        resource_kwargs['memory_limit'] = memory_limit
    if max_temp_directory_size is not None:
        resource_kwargs['max_temp_directory_size'] = max_temp_directory_size

    model = duckreg(
        formula=formula,
        data=data_source,
        subset=sql_where,
        bootstrap=bootstrap_config,
        round_strata=settings.get('round_strata'),
        seed=settings.get('seed', 42),
        fe_method=settings.get('fe_method', 'demean'),
        db_name=db_name,
        se_method=se_method,
        fitter=fitter,
        **resource_kwargs,
    )

    logger.info("Analysis complete!")

    model_summary = model.summary()
    text_output = format_model_summary(model_summary, spec_config, precision=4)
    print("\n" + text_output)

    # ── persist results ───────────────────────────────────────────────────────
    if output_dir:
        output_path = Path(output_dir) / 'duckreg' / spec_name
        output_path.mkdir(parents=True, exist_ok=True)

        results_container: Dict[str, Any] = {
            'analysis_metadata': {
                'analysis_type': 'duckreg',
                'spec_name': spec_name,
                'description': spec_config['description'],
                'formula': formula,
                'data_source': data_source,
                'query': sql_where,
                'settings': settings,
            },
            **model_summary,
        }

        version_info = model_summary.get('version_info', {})
        duckreg_version = version_info.get('duckreg_version', 'unknown')
        logger.info(f"Results computed with duckreg version: {duckreg_version}")

        if model_summary.get('first_stage'):
            logger.info(
                f"First stage results stored for "
                f"{len(model_summary['first_stage'])} endogenous variable(s)"
            )

        (output_path / f'results_{timestamp}.txt').write_text(text_output)

        coefficients_df = model.summary_df()
        if not coefficients_df.empty:
            coefficients_df.to_csv(output_path / 'coefficients.csv')

        for endog, fs_dict in model_summary.get('first_stage', {}).items():
            if isinstance(fs_dict, dict) and 'coef_names' in fs_dict:
                fs_df = pd.DataFrame({
                    'variable': fs_dict.get('coef_names', []),
                    'coefficient': fs_dict.get('coefficients', []),
                    'std_error': fs_dict.get('std_errors', []),
                    't_stat': fs_dict.get('t_statistics', []),
                    'p_value': fs_dict.get('p_values', []),
                })
                if 'ci_lower' in fs_dict and 'ci_upper' in fs_dict:
                    fs_df['ci_lower'] = fs_dict['ci_lower']
                    fs_df['ci_upper'] = fs_dict['ci_upper']
                safe = endog.replace(' ', '_').replace('/', '_')
                fs_df.to_csv(output_path / f'first_stage_{safe}.csv', index=False)

        with open(output_path / f'results_{timestamp}.json', 'w') as fh:
            json.dump(results_container, fh, indent=2)

        logger.info(f"Results saved to: {output_path}")

    return model


# ---------------------------------------------------------------------------
# Cleanup
# ---------------------------------------------------------------------------

def cleanup_analysis_results(
    output_dir: str,
    dry_run: bool = False,
    analysis_type: str = 'duckreg',
) -> None:
    """Prune stale result files, keeping the latest per minor version.

    For each model directory the function groups result JSON files by the
    minor component of the duckreg version (``X`` in ``y.X.z``) and deletes
    all but the most recent file within each group.

    Parameters
    ----------
    output_dir:
        Base output directory (e.g. ``output/analysis``).
    dry_run:
        When *True*, log what would be deleted without touching any files.
    analysis_type:
        Sub-directory to scan (default ``'duckreg'``).
    """
    logger = logging.getLogger(__name__)

    output_path = Path(output_dir) / analysis_type
    if not output_path.exists():
        logger.warning(f"Output directory not found: {output_path}")
        return

    total_deleted = 0
    total_kept = 0

    for spec_dir in output_path.iterdir():
        if not spec_dir.is_dir():
            continue

        result_files = list(spec_dir.glob("results_*.json"))
        if not result_files:
            continue

        # version_minor → {timestamp: [associated files]}
        version_groups: Dict[int, Dict[datetime, List[Path]]] = {}

        for jf in result_files:
            try:
                with open(jf) as fh:
                    data = json.load(fh)

                vi = data.get('version_info', {})
                ver = vi.get('duckreg_version', '0.0.0')
                if ver == 'unknown':
                    ver = '0.0.0'

                parts_v = ver.split('.')
                minor = int(parts_v[1]) if len(parts_v) >= 2 and parts_v[1].isdigit() else 0

                stem_parts = jf.stem.split('_', 1)
                if len(stem_parts) != 2 or len(stem_parts[1]) != 15:
                    logger.warning(f"Unexpected filename: {jf.name}")
                    continue
                ts = datetime.strptime(stem_parts[1], '%Y%m%d_%H%M%S')

                version_groups.setdefault(minor, {}).setdefault(ts, [])

                associated = [jf]
                txt = spec_dir / f"results_{stem_parts[1]}.txt"
                if txt.exists():
                    associated.append(txt)
                version_groups[minor][ts].extend(associated)

            except Exception as exc:
                logger.error(f"Error processing {jf}: {exc}")

        for minor, ts_map in version_groups.items():
            if not ts_map:
                continue
            latest_ts = max(ts_map)
            ver_label = f"0.{minor}.x"
            for ts, files in ts_map.items():
                if ts == latest_ts:
                    logger.info(
                        f"Keeping {spec_dir.name} (v{ver_label}, {ts}): {len(files)} files"
                    )
                    total_kept += len(files)
                else:
                    for fpath in files:
                        if dry_run:
                            logger.info(f"Would delete: {fpath.relative_to(output_path)}")
                        else:
                            fpath.unlink()
                            logger.debug(f"Deleted: {fpath.relative_to(output_path)}")
                        total_deleted += 1

    if dry_run:
        logger.info(
            f"Dry run complete: would delete {total_deleted} files, keep {total_kept} files"
        )
    else:
        logger.info(
            f"Cleanup complete: deleted {total_deleted} files, kept {total_kept} files"
        )
