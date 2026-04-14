"""
Table rendering for analysis results.

This module is responsible for **display only** — it never reads the Excel
config or touches the file system beyond loading pre-computed result JSON
files.  Configuration is consumed via :class:`~gnt.analysis.config.AnalysisConfig`.

Public API
----------
* :class:`RegressionTableData` — data container passed to formatters
* :class:`TableFormatterFactory` — ``get_formatter('html' | 'latex')``
* :func:`create_regression_table` — build a table string from model names
* :func:`generate_table_from_config` — build a table from :class:`AnalysisConfig`
* :func:`save_generated_tables` — write table files to disk
* :func:`create_tables_download_zip` — zip every file in the tables directory
* :func:`generate_all_tables` — generate and save every table in the config
* :func:`summarize_tables` — print a status overview of all tables
"""

from __future__ import annotations

import json
import warnings
import zipfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd

from .config import AnalysisConfig, PROJECT_ROOT, RESULTS_DIR
from .results import (
    find_latest_model_result,
    get_coefficient_data,
    get_model_date,
    get_model_version,
    is_2sls_model,
    load_models_by_name,
)

warnings.filterwarnings('ignore')

# Default output directory for table partials.  Use the top‑level
# output/analysis/tables directory rather than the old `_includes` package
# subfolder.  This change mirrors the modification in
# generate_tables.py and keeps all generated artefacts together.
OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis" / "tables"

# ── Raw-LaTeX marker (passes LaTeX snippets through the escaping layer) ──────
RAW_LATEX_MARKER = "\x00RAW_LATEX\x00"


def _mark_raw_latex(text: str) -> str:
    return f"{RAW_LATEX_MARKER}{text}{RAW_LATEX_MARKER}"


def _is_raw_latex(text: str) -> bool:
    return text.startswith(RAW_LATEX_MARKER) and text.endswith(RAW_LATEX_MARKER)


def _extract_raw_latex(text: str) -> str:
    if _is_raw_latex(text):
        return text[len(RAW_LATEX_MARKER):-len(RAW_LATEX_MARKER)]
    return text


# ---------------------------------------------------------------------------
# Data container
# ---------------------------------------------------------------------------

@dataclass
class RegressionTableData:
    """All information needed to render a single regression table."""
    model_names: List[str]
    coefficient_rows: List[Dict[str, Any]]
    stat_rows: List[Dict[str, Any]]
    custom_row_groups: List[Dict[str, Any]]
    caption: Optional[str]
    include_ci: bool
    stars: bool
    notes: Optional[Union[str, List[str]]]
    header_rows: Optional[List[Dict[str, Any]]]
    decimals: int
    first_stage_rows: Optional[List[Dict[str, Any]]] = None
    show_first_stage: str = "default"   # "none" | "default" | "full"
    has_2sls: bool = False
    table_environment: str = "table"
    table_size: str = "footnotesize"


# ---------------------------------------------------------------------------
# Formatter base class and implementations
# ---------------------------------------------------------------------------

class TableFormatter(ABC):
    """Abstract base class for table formatters."""

    @abstractmethod
    def format(self, data: RegressionTableData) -> str: ...

    @abstractmethod
    def get_file_extension(self) -> str: ...

    # ── shared helpers ───────────────────────────────────────────────────────

    def _build_notes_text(self, data: RegressionTableData) -> str:
        parts: List[str] = []
        if data.notes:
            notes = [data.notes] if isinstance(data.notes, str) else list(data.notes)
            for note in notes:
                parts.append(note if note.endswith('.') else note + '.')
        parts.append(
            'Standard errors in parentheses.'
            if not data.include_ci
            else '95% confidence intervals in brackets.'
        )
        return ' '.join(parts)

    def _get_significance_note(self, data: RegressionTableData) -> str:
        return "* p<0.05, ** p<0.01, *** p<0.001" if data.stars else ""


class HtmlTableFormatter(TableFormatter):
    """Formats regression tables as HTML."""

    def get_file_extension(self) -> str:
        return ".html"

    def format(self, data: RegressionTableData) -> str:
        parts: List[str] = []
        if data.caption:
            parts.append(f'<div class="table-caption">{data.caption}</div>')
        parts += [
            '<table class="regression-table">',
            self._build_header(data),
            self._build_body(data),
            self._build_footer(data),
            '</table>',
        ]
        return '\n'.join(parts)

    def _build_header(self, data: RegressionTableData) -> str:
        parts = ['<thead>']
        if data.header_rows:
            for row_def in data.header_rows:
                parts.append('<tr><th></th>')
                col_idx = 0
                for label, col_slice in row_def.items():
                    if label:
                        span = col_slice.stop - col_slice.start
                        parts.append(
                            f'<th colspan="{span}" class="header-span">{label}</th>'
                        )
                        col_idx = col_slice.stop
                    elif isinstance(col_slice, slice):
                        for _ in range(col_slice.start, col_slice.stop):
                            parts.append('<th></th>')
                        col_idx = col_slice.stop
                for _ in range(col_idx, len(data.model_names)):
                    parts.append('<th></th>')
                parts.append('</tr>')
        parts.append('<tr class="model-names-row"><th></th>')
        for name in data.model_names:
            parts.append(f'<th>{name}</th>')
        parts += ['</tr>', '</thead>']
        return '\n'.join(parts)

    def _build_body(self, data: RegressionTableData) -> str:
        parts = ['<tbody>']
        n_cols = len(data.model_names)

        # first stage
        if data.show_first_stage == "full" and data.first_stage_rows:
            for i, row in enumerate(data.first_stage_rows):
                is_hdr = row.pop('_is_header', False)
                if is_hdr:
                    parts += [
                        '<tr class="first-stage-header">',
                        f'<td colspan="{n_cols + 1}"><strong>{row["Variable"]}</strong></td>',
                        '</tr>',
                    ]
                else:
                    css = ' class="first-stage-row"' if i % 2 == 0 else ''
                    parts.append(f'<tr{css}><td>{row["Variable"]}</td>')
                    for name in data.model_names:
                        parts.append(f'<td>{row.get(name, "")}</td>')
                    parts.append('</tr>')
            if data.has_2sls:
                parts.append(
                    f'<tr class="stage-separator">'
                    f'<td colspan="{n_cols + 1}"><strong>Second Stage</strong></td></tr>'
                )

        # coefficient rows
        for i, row in enumerate(data.coefficient_rows):
            css = ' class="coef-row"' if i % 2 == 0 else ''
            parts.append(f'<tr{css}><td>{row["Variable"]}</td>')
            for name in data.model_names:
                parts.append(f'<td>{row.get(name, "")}</td>')
            parts.append('</tr>')

        # stat rows
        for i, row in enumerate(data.stat_rows):
            css = ' class="stats-section"' if i == 0 else ''
            parts.append(f'<tr{css}><td>{row["Variable"]}</td>')
            for name in data.model_names:
                parts.append(f'<td>{row.get(name, "")}</td>')
            parts.append('</tr>')

        # custom row groups
        for gi, row_group in enumerate(data.custom_row_groups):
            for ri, (label, values) in enumerate(row_group.items()):
                css = ''
                if gi == 0 and ri == 0:
                    css = ' class="custom-section"'
                elif gi > 0 and ri == 0:
                    css = ' class="custom-group-section"'
                parts.append(f'<tr{css}><td>{label}</td>')
                if isinstance(values, (list, tuple)):
                    for v in values:
                        parts.append(f'<td>{v}</td>')
                else:
                    for _ in data.model_names:
                        parts.append(f'<td>{values}</td>')
                parts.append('</tr>')

        parts.append('</tbody>')
        return '\n'.join(parts)

    def _build_footer(self, data: RegressionTableData) -> str:
        text = 'Notes: ' + self._build_notes_text(data)
        sig = self._get_significance_note(data)
        if sig:
            text += ' ' + sig.replace('<', '&lt;')
        return (
            f'<tfoot><tr><td colspan="{len(data.model_names) + 1}">'
            f'{text}</td></tr></tfoot>'
        )


class LatexTableFormatter(TableFormatter):
    """Formats regression tables as LaTeX."""

    def get_file_extension(self) -> str:
        return ".tex"

    def _escape_latex(self, text: str) -> str:
        if _is_raw_latex(text):
            return _extract_raw_latex(text)
        for char, repl in {
            '&': r'\&', '%': r'\%', '$': r'\$', '#': r'\#',
            '_': r'\_', '{': r'\{', '}': r'\}',
            '<': r'$<$', '>': r'$>$',
            '~': r'\textasciitilde{}', '^': r'\textasciicircum{}',
        }.items():
            text = text.replace(char, repl)
        return text

    def format(self, data: RegressionTableData) -> str:
        col_spec = 'l' + 'c' * len(data.model_names)
        parts = [
            f'\\begin{{{data.table_environment}}}[htbp]',
            r'\centering',
            f'\\{data.table_size}',
        ]
        if data.caption:
            parts.append(f'\\caption{{{self._escape_latex(data.caption)}}}')
        parts += [
            f'\\begin{{tabular}}{{{col_spec}}}',
            r'\toprule',
            self._build_header(data),
            self._build_body(data),
            r'\bottomrule',
            r'\end{tabular}',
            self._build_footer(data),
            f'\\end{{{data.table_environment}}}',
        ]
        return '\n'.join(parts)

    def _build_header(self, data: RegressionTableData) -> str:
        parts: List[str] = []
        n_models = len(data.model_names)
        if data.header_rows:
            for row_def in data.header_rows:
                row_parts = ['']
                col_idx = 0
                for label, col_slice in row_def.items():
                    if label:
                        span = col_slice.stop - col_slice.start
                        row_parts.append(
                            f'\\multicolumn{{{span}}}{{c}}{{{self._escape_latex(label)}}}'
                        )
                        col_idx = col_slice.stop
                while col_idx < n_models:
                    row_parts.append('')
                    col_idx += 1
                parts.append(' & '.join(row_parts) + r' \\')
                parts.append(r'\cmidrule(lr){2-' + str(n_models + 1) + '}')
        row_parts = [''] + [self._escape_latex(n) for n in data.model_names]
        parts += [' & '.join(row_parts) + r' \\', r'\midrule']
        return '\n'.join(parts)

    def _build_body(self, data: RegressionTableData) -> str:
        parts: List[str] = []
        n_cols = len(data.model_names)

        if data.show_first_stage == "full" and data.first_stage_rows:
            for row in data.first_stage_rows:
                is_hdr = row.pop('_is_header', False)
                if is_hdr:
                    parts += [
                        r'\midrule',
                        f'\\multicolumn{{{n_cols + 1}}}{{l}}'
                        f'{{\\textit{{{self._escape_latex(row["Variable"])}}}}} \\\\',
                    ]
                else:
                    rp = [self._escape_latex(row['Variable'])]
                    for name in data.model_names:
                        rp.append(row.get(name, ''))
                    parts.append(' & '.join(rp) + r' \\')

        if data.has_2sls:
            parts += [
                r'\midrule',
                f'\\multicolumn{{{n_cols + 1}}}{{l}}{{\\textit{{Second Stage}}}} \\\\',
            ]

        for row in data.coefficient_rows:
            rp = [self._escape_latex(row['Variable'])]
            for name in data.model_names:
                rp.append(row.get(name, ''))
            parts.append(' & '.join(rp) + r' \\')

        if data.stat_rows:
            parts.append(r'\midrule')
            for row in data.stat_rows:
                rp = [self._escape_latex(row['Variable'])]
                for name in data.model_names:
                    rp.append(row.get(name, ''))
                parts.append(' & '.join(rp) + r' \\')

        for gi, row_group in enumerate(data.custom_row_groups):
            parts.append(
                r'\midrule'
                if gi == 0
                else r'\cmidrule(lr){1-' + str(n_cols + 1) + '}'
            )
            for label, values in row_group.items():
                rp = [self._escape_latex(label)]
                if isinstance(values, (list, tuple)):
                    rp.extend(self._escape_latex(str(v)) for v in values)
                else:
                    rp.extend([self._escape_latex(str(values))] * n_cols)
                parts.append(' & '.join(rp) + r' \\')

        return '\n'.join(parts)

    def _build_footer(self, data: RegressionTableData) -> str:
        text = self._build_notes_text(data)
        sig = self._get_significance_note(data)
        if sig:
            text += ' ' + sig
        return (
            '\\begin{threeparttable}\n'
            '\\begin{tablenotes}\\small\n'
            f'\\item \\textit{{Notes:}} {self._escape_latex(text)}\n'
            '\\end{tablenotes}\n'
            '\\end{threeparttable}'
        )


class TableFormatterFactory:
    """Factory for creating table formatters."""

    _formatters: Dict[str, type] = {
        'html':  HtmlTableFormatter,
        'latex': LatexTableFormatter,
        'tex':   LatexTableFormatter,
    }

    @classmethod
    def get_formatter(cls, format_type: str) -> TableFormatter:
        fmt = format_type.lower()
        if fmt not in cls._formatters:
            raise ValueError(
                f"Unknown format: {fmt!r}. Available: {list(cls._formatters)}"
            )
        return cls._formatters[fmt]()

    @classmethod
    def register_formatter(cls, format_type: str, formatter_class: type) -> None:
        cls._formatters[format_type.lower()] = formatter_class


# ---------------------------------------------------------------------------
# Data extraction helpers
# ---------------------------------------------------------------------------

def _extract_coefficient_data(
    models: List[Dict[str, Any]],
    model_names: List[str],
    select_coefs: Optional[Dict[str, str]],
    decimals: int,
    stars: bool,
    include_ci: bool,
) -> List[Dict[str, Any]]:
    """Build coefficient + SE/CI rows."""
    if select_coefs is not None:
        all_vars: List[str] = list(select_coefs.keys())
    else:
        all_vars = []
        for m in models:
            cd = get_coefficient_data(m)
            for v in cd.get('names', cd.get('coef_names', [])):
                if v not in all_vars:
                    all_vars.append(v)

    # Group vars by display label so vars sharing a label are merged into
    # one row (first non-empty value per model column wins).
    # raw_latex_labels tracks which labels come from select_coefs (need no escaping).
    label_to_vars: Dict[str, List[str]] = {}
    raw_latex_labels: set = set()
    for var in all_vars:
        if select_coefs and var in select_coefs:
            disp = select_coefs[var]
            raw_latex_labels.add(disp)
        else:
            disp = var
        label_to_vars.setdefault(disp, []).append(var)

    rows: List[Dict[str, Any]] = []
    for disp_label, vars_for_label in label_to_vars.items():
        marked = _mark_raw_latex(disp_label) if disp_label in raw_latex_labels else disp_label
        coef_row: Dict[str, Any] = {'Variable': marked}
        se_row:   Dict[str, Any] = {'Variable': ''}

        for i, model in enumerate(models):
            cd = get_coefficient_data(model)
            var_names = cd.get('names', cd.get('coef_names', []))
            estimates = cd.get('estimates', cd.get('coefficients', []))

            # Among all vars sharing this label, take the first that appears in
            # this model's coefficient list.
            matched_idx: Optional[int] = None
            for var in vars_for_label:
                if var in var_names:
                    matched_idx = var_names.index(var)
                    break

            if matched_idx is not None:
                coef = estimates[matched_idx]
                coef_str = f"{coef:.{decimals}f}"
                if stars and 'p_values' in cd:
                    p = cd['p_values'][matched_idx]
                    coef_str += '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                coef_row[model_names[i]] = coef_str
                if 'std_errors' in cd:
                    if include_ci:
                        lo = cd.get('conf_int_lower', [None] * (matched_idx + 1))[matched_idx]
                        hi = cd.get('conf_int_upper', [None] * (matched_idx + 1))[matched_idx]
                        se_row[model_names[i]] = (
                            f"[{lo:.{decimals}f}, {hi:.{decimals}f}]"
                            if lo is not None and hi is not None
                            else ""
                        )
                    else:
                        se_row[model_names[i]] = f"({cd['std_errors'][matched_idx]:.{decimals}f})"
                else:
                    se_row[model_names[i]] = ""
            else:
                coef_row[model_names[i]] = ""
                se_row[model_names[i]] = ""

        rows.extend([coef_row, se_row])
    return rows


def _extract_first_stage_data(
    models: List[Dict[str, Any]],
    model_names: List[str],
    decimals: int,
    stars: bool,
    include_ci: bool,
    select_coefs: Optional[Dict[str, str]] = None,
    select_instruments: Optional[Dict[str, str]] = None,
) -> List[Dict[str, Any]]:
    """Build one merged first-stage coefficient block for 2SLS models."""
    instrument_filter = select_instruments or select_coefs

    all_vars: List[str] = []
    for m in models:
        if not is_2sls_model(m):
            continue
        for fs_data in m.get('first_stage', {}).values():
            for inst in fs_data.get('instrument_names', []):
                if (not instrument_filter or inst in instrument_filter) and inst not in all_vars:
                    all_vars.append(inst)

    if not all_vars:
        return []

    hdr: Dict[str, Any] = {'Variable': 'First Stage', '_is_header': True}
    for name in model_names:
        hdr[name] = ''

    rows: List[Dict[str, Any]] = [hdr]
    for var in all_vars:
        if instrument_filter and var in instrument_filter:
            disp = instrument_filter[var]
            if any(c in disp for c in ('$', '\\', '{', '}')):
                disp = _mark_raw_latex(disp)
        else:
            disp = var
        coef_row: Dict[str, Any] = {'Variable': f'  {disp}'}
        se_row:   Dict[str, Any] = {'Variable': ''}

        for i, model in enumerate(models):
            if not is_2sls_model(model):
                coef_row[model_names[i]] = se_row[model_names[i]] = ""
                continue

            coef_entries: List[str] = []
            se_entries: List[str] = []
            for fs_data in model.get('first_stage', {}).values():
                coef_names = fs_data.get('coef_names', [])
                if var not in coef_names:
                    continue

                idx = coef_names.index(var)
                c = fs_data['coefficients'][idx]
                cs = f"{c:.{decimals}f}"
                if stars and 'p_values' in fs_data:
                    p = fs_data['p_values'][idx]
                    cs += '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else ''
                coef_entries.append(cs)

                if 'std_errors' in fs_data and not include_ci:
                    se_entries.append(f"({fs_data['std_errors'][idx]:.{decimals}f})")

            coef_row[model_names[i]] = "; ".join(coef_entries)
            se_row[model_names[i]] = "; ".join(se_entries)

        rows.extend([coef_row, se_row])

    return rows


_STAT_KEYS = {
    'n_obs':           ['n_obs', 'n_observations'],
    'n_observations':  ['n_obs', 'n_observations'],
    'r_squared':       ['r_squared'],
    'adj_r_squared':   ['adj_r_squared'],
    'n_compressed_rows': ['n_compressed', 'n_compressed_rows'],
    'n_compressed':    ['n_compressed', 'n_compressed_rows'],
    'compression_ratio': ['compression_ratio'],
    'df_model':        ['df_model'],
    'df_resid':        ['df_resid'],
}
_INT_STATS = {'n_obs', 'n_observations', 'n_compressed', 'n_compressed_rows', 'df_model', 'df_resid'}
_FLOAT_STATS = {'r_squared', 'adj_r_squared', 'compression_ratio'}


def _extract_stat_rows(
    models: List[Dict[str, Any]],
    model_names: List[str],
    show_stats: List[str],
    decimals: int,
    show_first_stage: str = "default",
    include_version_info: bool = True,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    if include_version_info:
        ver_row: Dict[str, Any] = {'Variable': 'Model Version'}
        date_row: Dict[str, Any] = {'Variable': 'Computed Date'}
        for i, m in enumerate(models):
            ver_row[model_names[i]] = get_model_version(m)
            date_row[model_names[i]] = get_model_date(m)
        rows.extend([ver_row, date_row])

    for stat_key in show_stats:
        stat_row: Dict[str, Any] = {'Variable': stat_key.replace('_', ' ').title()}
        for i, model in enumerate(models):
            sample = model.get('sample_info', {})
            mstats = model.get('model_statistics', {})
            coefs  = model.get('coefficients', {})
            val = None
            for key in _STAT_KEYS.get(stat_key, [stat_key]):
                for src in (sample, mstats, coefs):
                    if key in src and src[key] is not None:
                        val = src[key]
                        break
                if val is not None:
                    break
            if val is not None:
                if stat_key in _INT_STATS:
                    stat_row[model_names[i]] = f"{int(val):,}"
                elif stat_key in _FLOAT_STATS:
                    stat_row[model_names[i]] = f"{val:.{decimals}f}"
                else:
                    stat_row[model_names[i]] = str(val)
            else:
                stat_row[model_names[i]] = ""
        rows.append(stat_row)

    if show_first_stage in ("default", "full"):
        for endog in sorted({
            ev
            for m in models if is_2sls_model(m)
            for ev in m.get('first_stage', {})
        }):
            frow: Dict[str, Any] = {'Variable': f'First Stage F ({endog})'}
            for i, m in enumerate(models):
                if not is_2sls_model(m):
                    frow[model_names[i]] = ""
                    continue
                fs = m.get('first_stage', {}).get(endog, {})
                f_stat = fs.get('f_statistic')
                frow[model_names[i]] = f"{f_stat:.{decimals}f}" if f_stat is not None else ""
            rows.append(frow)

    return rows


def _normalize_custom_rows(
    custom_rows: Optional[Union[Dict, List]],
    model_names: List[str],
) -> List[Dict[str, Any]]:
    if not custom_rows:
        return []
    groups = [custom_rows] if isinstance(custom_rows, dict) else list(custom_rows)
    for g in groups:
        for label, vals in g.items():
            if isinstance(vals, (list, tuple)) and len(vals) != len(model_names):
                raise ValueError(
                    f"Custom row '{label}' has {len(vals)} values "
                    f"but there are {len(model_names)} models"
                )
    return groups


# ---------------------------------------------------------------------------
# Public table creation functions
# ---------------------------------------------------------------------------

def create_regression_table(
    model_specs: Union[List[str], List[Dict]],
    base_path: Union[str, Path],
    model_names: Optional[List[str]] = None,
    show_stats: Optional[List[str]] = None,
    include_ci: bool = False,
    decimals: int = 3,
    stars: bool = True,
    caption: Optional[str] = None,
    select_coefs: Optional[Dict[str, str]] = None,
    select_instruments: Optional[Dict[str, str]] = None,
    custom_rows: Optional[Union[Dict, List]] = None,
    notes: Optional[Union[str, List[str]]] = None,
    header_rows: Optional[List[Dict]] = None,
    show_first_stage: str = "default",
    table_environment: str = "table",
    table_size: str = "footnotesize",
    output_format: str = 'html',
) -> str:
    """Render a regression table from a list of model specifications.

    Parameters
    ----------
    model_specs:
        Model names (str) or dicts with ``'name'``/``'model_name'`` key and
        optional ``'analysis_type'``.
    base_path:
        Base results directory (``output/analysis``).
    model_names:
        Display names for column headers.  Auto-generated when *None*.
    show_stats:
        Which statistics to show below coefficients.  Defaults to
        ``['n_observations', 'n_compressed']``.
    show_first_stage:
        ``'none'`` | ``'default'`` (F-stats only) | ``'full'`` (all coefficients).
    output_format:
        ``'html'`` | ``'latex'`` | ``'tex'``.
    """
    models = load_models_by_name(model_specs, base_path)
    if not models:
        return (
            "<p>No models to display</p>"
            if output_format == 'html'
            else "% No models to display"
        )

    has_2sls = any(is_2sls_model(m) for m in models)

    if model_names is None:
        model_names = [
            s if isinstance(s, str) else (s.get('name') or s.get('model_name', f'Model {i+1}'))
            for i, s in enumerate(model_specs)
        ]
    elif len(model_names) != len(models):
        raise ValueError(
            f"model_names length ({len(model_names)}) != "
            f"number of models ({len(models)})"
        )

    if show_stats is None:
        show_stats = ['n_observations', 'n_compressed']

    coef_rows = _extract_coefficient_data(
        models, model_names, select_coefs, decimals, stars, include_ci
    )
    stat_rows = _extract_stat_rows(
        models, model_names, show_stats, decimals, show_first_stage,
        include_version_info=True
    )
    custom_row_groups = _normalize_custom_rows(custom_rows, model_names)

    first_stage_rows = None
    if show_first_stage == "full":
        first_stage_rows = _extract_first_stage_data(
            models, model_names, decimals, stars, include_ci,
            select_coefs, select_instruments
        )

    table_data = RegressionTableData(
        model_names=model_names,
        coefficient_rows=coef_rows,
        stat_rows=stat_rows,
        custom_row_groups=custom_row_groups,
        caption=caption,
        include_ci=include_ci,
        stars=stars,
        notes=notes,
        header_rows=header_rows,
        decimals=decimals,
        first_stage_rows=first_stage_rows,
        show_first_stage=show_first_stage,
        has_2sls=has_2sls,
        table_environment=table_environment,
        table_size=table_size,
    )

    return TableFormatterFactory.get_formatter(output_format).format(table_data)


def generate_table_from_config(
    table_name: str,
    config: AnalysisConfig,
    output_formats: Optional[List[str]] = None,
) -> Dict[str, str]:
    """Render a table from :class:`~gnt.analysis.config.AnalysisConfig`.

    Parameters
    ----------
    table_name:
        Table name as defined in the ``Models in Tables`` sheet.
    config:
        :class:`AnalysisConfig` instance.
    output_formats:
        Formats to render.  Overrides the ``Tables`` sheet value; falls back
        to ``['html']``.

    Returns
    -------
    dict
        ``{format: rendered_string}``
    """
    table_cfg = config.get_table_display_config(table_name)

    fmts = output_formats or table_cfg.pop('output_formats', ['html'])
    model_specs: List[str] = table_cfg.pop('model_paths')

    results: Dict[str, str] = {}
    for fmt in fmts:
        results[fmt] = create_regression_table(
            model_specs=model_specs,
            base_path=config.base_path,
            output_format=fmt,
            **table_cfg,
        )
    return results


def save_generated_tables(
    table_name: str,
    generated_tables: Dict[str, str],
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Write rendered table strings to files.

    Parameters
    ----------
    table_name:
        Used in the output filename: ``table_<table_name>.<ext>``.
    generated_tables:
        ``{format: rendered_string}`` as returned by
        :func:`generate_table_from_config`.
    output_dir:
        Target directory.  Defaults to ``output/analysis/tables``.
    """
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    _ext = {'html': '.html', 'latex': '.tex', 'tex': '.tex'}
    for fmt, content in generated_tables.items():
        path = out / f"table_{table_name}{_ext.get(fmt, f'.{fmt}')}"
        path.write_text(content)
        print(f"  Generated: {path}")


def create_tables_download_zip(
    output_dir: Optional[Union[str, Path]] = None,
    zip_name: str = "download.zip",
) -> Path:
    """Create a zip archive containing every file in the tables directory."""
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    zip_path = out / zip_name
    files = sorted(
        path for path in out.rglob("*")
        if path.is_file() and path.resolve() != zip_path.resolve()
    )

    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in files:
            archive.write(path, path.relative_to(out).as_posix())

    print(f"  Generated: {zip_path}")
    return zip_path


# ---------------------------------------------------------------------------
# High-level orchestration
# ---------------------------------------------------------------------------

def summarize_tables(config: AnalysisConfig) -> None:
    """Print a status overview of every table defined in the Excel config.

    For each model the function reports the last run date, duckreg version
    used, and whether results exist.
    """
    table_names = config.get_all_table_names()

    print(f"\n{'=' * 80}")
    print("  Analysis Tables Summary")
    print(f"  Excel  : {config.excel_path}")
    print(f"  Results: {config.base_path}")
    print(f"{'=' * 80}")

    col_w = 52
    for table_name in table_names:
        models = config.get_models_for_table(table_name)
        print(f"\n  Table: {table_name}  ({len(models)} model{'s' if len(models) != 1 else ''})")
        print(
            f"  {'Model':<{col_w}}  {'Last Run':<12}  {'Version':<12}  Status"
        )
        print(f"  {'-' * col_w}  {'-' * 12}  {'-' * 12}  ------")
        for model in models:
            try:
                path = find_latest_model_result(model, config.base_path)
                with open(path) as fh:
                    data = json.load(fh)
                date = get_model_date(data)
                ver  = get_model_version(data)
                status = "ok"
            except FileNotFoundError:
                date = ver = 'N/A'
                status = "missing"
            except Exception as exc:
                date = ver = 'error'
                status = f"error: {exc}"
            print(f"  {model:<{col_w}}  {date:<12}  {ver:<12}  {status}")

    print()


def generate_all_tables(
    config: AnalysisConfig,
    table_names: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    output_formats: Optional[List[str]] = None,
) -> None:
    """Generate, save, and archive all (or a named subset of) tables.

    Parameters
    ----------
    config:
        :class:`AnalysisConfig` instance.
    table_names:
        Limit generation to these table names.  All tables when *None*.
    output_dir:
        Target directory for generated files.
    output_formats:
        Override the output formats for every table.
    """
    names = table_names or config.get_all_table_names()
    for name in names:
        print(f"Generating table: {name}")
        try:
            generated = generate_table_from_config(name, config, output_formats)
            save_generated_tables(name, generated, output_dir)
        except Exception as exc:
            print(f"  Error generating '{name}': {exc}")
    create_tables_download_zip(output_dir)


# Backward-compatible alias
generate_main_table = generate_all_tables
