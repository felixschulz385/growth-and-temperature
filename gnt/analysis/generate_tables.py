"""
Generate table partials for Jekyll site in multiple formats (HTML, LaTeX).
Run this script to regenerate all tables from analysis results.
"""

import argparse
import json
import pandas as pd
from pathlib import Path
from abc import ABC, abstractmethod
from typing import Optional, Union, Dict, List, Any
from dataclasses import dataclass
import warnings
import os
import re
warnings.filterwarnings('ignore')

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "output" / "analysis"
# default location where generated table files are placed.  Updated to the
# shared analysis output directory rather than the legacy `_includes` folder
# under the Python package.  This makes the tables easier to find from outside
# the `gnt` package and aligns with the new static output layout.
OUTPUT_DIR = PROJECT_ROOT / "output" / "analysis" / "tables"
CONFIG_DIR = Path(__file__).parent

# Special marker for raw LaTeX content (won't be escaped)
RAW_LATEX_MARKER = "\x00RAW_LATEX\x00"


def _mark_raw_latex(text: str) -> str:
    """Mark text as raw LaTeX to prevent escaping."""
    return f"{RAW_LATEX_MARKER}{text}{RAW_LATEX_MARKER}"


def _is_raw_latex(text: str) -> bool:
    """Check if text is marked as raw LaTeX."""
    return text.startswith(RAW_LATEX_MARKER) and text.endswith(RAW_LATEX_MARKER)


def _extract_raw_latex(text: str) -> str:
    """Extract raw LaTeX content, removing markers."""
    if _is_raw_latex(text):
        return text[len(RAW_LATEX_MARKER):-len(RAW_LATEX_MARKER)]
    return text


# Ensure output directory exists (create parent directories as needed)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


@dataclass
class RegressionTableData:
    """Data structure holding all information needed to render a regression table."""
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
    first_stage_rows: Optional[List[Dict[str, Any]]] = None  # For full first stage display
    show_first_stage: str = "default"  # "none", "default", "full"
    has_2sls: bool = False  # Indicates whether any loaded model is a 2SLS/IV model
    table_environment: str = "table"  # LaTeX table environment (e.g., 'table', 'sidewaystable')
    table_size: str = "footnotesize"  # LaTeX text size command (e.g., 'footnotesize', 'small', 'normalsize')


class TableFormatter(ABC):
    """Abstract base class for table formatters."""
    
    @abstractmethod
    def format(self, data: RegressionTableData) -> str:
        """Format the table data into the target format."""
        pass
    
    @abstractmethod
    def get_file_extension(self) -> str:
        """Return the file extension for this format."""
        pass
    
    def _build_notes_text(self, data: RegressionTableData) -> str:
        """Build the notes text (shared logic)."""
        notes_parts = []
        
        if data.notes:
            if isinstance(data.notes, str):
                note = data.notes
                if not note.endswith('.'):
                    note += '.'
                notes_parts.append(note)
            elif isinstance(data.notes, (list, tuple)):
                for note in data.notes:
                    if not note.endswith('.'):
                        note += '.'
                    notes_parts.append(note)
        
        se_note = 'Standard errors in parentheses.' if not data.include_ci else '95% confidence intervals in brackets.'
        notes_parts.append(se_note)
        
        return ' '.join(notes_parts)
    
    def _get_significance_note(self, data: RegressionTableData) -> str:
        """Get significance stars note."""
        if data.stars:
            return "* p<0.05, ** p<0.01, *** p<0.001"
        return ""


class HtmlTableFormatter(TableFormatter):
    """Formats regression tables as HTML."""
    
    get_file_extension = lambda self: ".html"
    
    def format(self, data: RegressionTableData) -> str:
        html_parts = []
        
        if data.caption:
            html_parts.append(f'<div class="table-caption">{data.caption}</div>')
        
        html_parts.append('<table class="regression-table">')
        html_parts.append(self._build_header(data))
        html_parts.append(self._build_body(data))
        html_parts.append(self._build_footer(data))
        html_parts.append('</table>')
        
        return '\n'.join(html_parts)
    
    def _build_header(self, data: RegressionTableData) -> str:
        parts = ['<thead>']
        
        # Add header rows if provided
        if data.header_rows:
            for header_row_def in data.header_rows:
                parts.append('<tr>')
                parts.append('<th></th>')
                
                col_idx = 0
                for header_label, col_slice in header_row_def.items():
                    if header_label:
                        colspan = col_slice.stop - col_slice.start
                        parts.append(f'<th colspan="{colspan}" class="header-span">{header_label}</th>')
                        col_idx = col_slice.stop
                    else:
                        if isinstance(col_slice, slice):
                            for _ in range(col_slice.start, col_slice.stop):
                                parts.append('<th></th>')
                            col_idx = col_slice.stop
                
                if col_idx < len(data.model_names):
                    for _ in range(col_idx, len(data.model_names)):
                        parts.append('<th></th>')
                
                parts.append('</tr>')
        
        # Main model names header row
        parts.append('<tr class="model-names-row">')
        parts.append('<th></th>')
        for name in data.model_names:
            parts.append(f'<th>{name}</th>')
        parts.append('</tr>')
        parts.append('</thead>')
        
        return '\n'.join(parts)
    
    def _build_body(self, data: RegressionTableData) -> str:
        parts = ['<tbody>']
        
        # First stage rows (if show_first_stage == "full")
        if data.show_first_stage == "full" and data.first_stage_rows:
            for i, row in enumerate(data.first_stage_rows):
                is_header = row.pop('_is_header', False)
                if is_header:
                    css_class = ' class="first-stage-header"'
                    parts.append(f'<tr{css_class}>')
                    parts.append(f'<td colspan="{len(data.model_names) + 1}"><strong>{row["Variable"]}</strong></td>')
                    parts.append('</tr>')
                else:
                    css_class = ' class="first-stage-row"' if i % 2 == 0 else ''
                    parts.append(f'<tr{css_class}>')
                    parts.append(f'<td>{row["Variable"]}</td>')
                    for name in data.model_names:
                        parts.append(f'<td>{row.get(name, "")}</td>')
                    parts.append('</tr>')
            
            # Add separator only if any model is 2SLS
            if data.has_2sls:
                parts.append(f'<tr class="stage-separator"><td colspan="{len(data.model_names) + 1}"><strong>Second Stage</strong></td></tr>')
        
        # If no explicit first-stage block was printed but models include 2SLS, we do NOT print a Second Stage header here
        # Coefficient rows (second stage for 2SLS)
        for i, row in enumerate(data.coefficient_rows):
            css_class = ' class="coef-row"' if i % 2 == 0 else ''
            parts.append(f'<tr{css_class}>')
            parts.append(f'<td>{row["Variable"]}</td>')
            for name in data.model_names:
                parts.append(f'<td>{row.get(name, "")}</td>')
            parts.append('</tr>')
        
        # Stats section
        if data.stat_rows:
            for i, row in enumerate(data.stat_rows):
                css_class = ' class="stats-section"' if i == 0 else ''
                parts.append(f'<tr{css_class}>')
                parts.append(f'<td>{row["Variable"]}</td>')
                for name in data.model_names:
                    parts.append(f'<td>{row.get(name, "")}</td>')
                parts.append('</tr>')
        
        # Custom row groups
        for group_idx, row_group in enumerate(data.custom_row_groups):
            for row_idx, (label, values) in enumerate(row_group.items()):
                css_class = ''
                if group_idx == 0 and row_idx == 0:
                    css_class = ' class="custom-section"'
                elif group_idx > 0 and row_idx == 0:
                    css_class = ' class="custom-group-section"'
                
                parts.append(f'<tr{css_class}>')
                parts.append(f'<td>{label}</td>')
                
                if isinstance(values, (list, tuple)):
                    for val in values:
                        parts.append(f'<td>{val}</td>')
                else:
                    for _ in data.model_names:
                        parts.append(f'<td>{values}</td>')
                parts.append('</tr>')
        
        parts.append('</tbody>')
        return '\n'.join(parts)
    
    def _build_footer(self, data: RegressionTableData) -> str:
        colspan = len(data.model_names) + 1
        notes_text = 'Notes: ' + self._build_notes_text(data)
        
        sig_note = self._get_significance_note(data)
        if sig_note:
            notes_text += ' ' + sig_note.replace('<', '&lt;')
        
        return f'<tfoot><tr><td colspan="{colspan}">{notes_text}</td></tr></tfoot>'


class LatexTableFormatter(TableFormatter):
    """Formats regression tables as LaTeX."""
    
    get_file_extension = lambda self: ".tex"
    
    def format(self, data: RegressionTableData) -> str:
        n_cols = len(data.model_names) + 1
        col_spec = 'l' + 'c' * len(data.model_names)
        
        parts = []
        parts.append(f'\\begin{{{data.table_environment}}}[htbp]')
        parts.append(r'\centering')
        parts.append(f'\\{data.table_size}')
        
        if data.caption:
            parts.append(f'\\caption{{{self._escape_latex(data.caption)}}}')
        
        parts.append(f'\\begin{{tabular}}{{{col_spec}}}')
        parts.append(r'\toprule')
        
        parts.append(self._build_header(data))
        parts.append(self._build_body(data))
        
        parts.append(r'\bottomrule')
        parts.append(r'\end{tabular}')
        parts.append(self._build_footer(data))
        parts.append(f'\\end{{{data.table_environment}}}')
        
        return '\n'.join(parts)
    
    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters, unless marked as raw LaTeX."""
        # Check if this is marked as raw LaTeX
        if _is_raw_latex(text):
            return _extract_raw_latex(text)
        
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\textasciicircum{}',
        }
        for char, replacement in replacements.items():
            text = text.replace(char, replacement)
        return text
    
    def _build_header(self, data: RegressionTableData) -> str:
        parts = []
        
        # Add header rows if provided
        if data.header_rows:
            for header_row_def in data.header_rows:
                row_parts = ['']
                col_idx = 0
                
                for header_label, col_slice in header_row_def.items():
                    if header_label:
                        colspan = col_slice.stop - col_slice.start
                        start_col = col_slice.start + 2  # +2 for 1-indexing and label column
                        end_col = start_col + colspan - 1
                        row_parts.append(f'\\multicolumn{{{colspan}}}{{c}}{{{self._escape_latex(header_label)}}}')
                        col_idx = col_slice.stop
                
                # Fill remaining columns
                while col_idx < len(data.model_names):
                    row_parts.append('')
                    col_idx += 1
                
                parts.append(' & '.join(row_parts) + r' \\')
                parts.append(r'\cmidrule(lr){2-' + str(len(data.model_names) + 1) + '}')
        
        # Model names row
        row_parts = [''] + [self._escape_latex(name) for name in data.model_names]
        parts.append(' & '.join(row_parts) + r' \\')
        parts.append(r'\midrule')
        
        return '\n'.join(parts)
    
    def _build_body(self, data: RegressionTableData) -> str:
        parts = []
        
        # First stage rows (if show_first_stage == "full")
        if data.show_first_stage == "full" and data.first_stage_rows:
            for row in data.first_stage_rows:
                is_header = row.pop('_is_header', False)
                if is_header:
                    parts.append(r'\midrule')
                    parts.append(f'\\multicolumn{{{len(data.model_names) + 1}}}{{l}}{{\\textit{{{self._escape_latex(row["Variable"])}}}}} \\\\')
                else:
                    # Extract variable name and handle raw LaTeX markers
                    var_name = row['Variable']
                    var_escaped = self._escape_latex(var_name)
                    
                    row_parts = [var_escaped]
                    for name in data.model_names:
                        row_parts.append(row.get(name, ''))
                    parts.append(' & '.join(row_parts) + r' \\')
        
        # Add separator for second stage only if any model is 2SLS
        if data.has_2sls:
            parts.append(r'\midrule')
            parts.append(f'\\multicolumn{{{len(data.model_names) + 1}}}{{l}}{{\\textit{{Second Stage}}}} \\\\')
        
        # Coefficient rows (second stage for 2SLS)
        for row in data.coefficient_rows:
            # Extract and properly escape variable name (handles raw LaTeX)
            var_name = row['Variable']
            var_escaped = self._escape_latex(var_name)
            
            row_parts = [var_escaped]
            for name in data.model_names:
                row_parts.append(row.get(name, ''))
            parts.append(' & '.join(row_parts) + r' \\')
        
        # Stats section
        if data.stat_rows:
            parts.append(r'\midrule')
            for row in data.stat_rows:
                row_parts = [self._escape_latex(row['Variable'])]
                for name in data.model_names:
                    row_parts.append(row.get(name, ''))
                parts.append(' & '.join(row_parts) + r' \\')
        
        # Custom row groups
        for group_idx, row_group in enumerate(data.custom_row_groups):
            if group_idx == 0:
                parts.append(r'\midrule')
            else:
                parts.append(r'\cmidrule(lr){1-' + str(len(data.model_names) + 1) + '}')
            
            for label, values in row_group.items():
                row_parts = [self._escape_latex(label)]
                if isinstance(values, (list, tuple)):
                    row_parts.extend([self._escape_latex(str(v)) for v in values])
                else:
                    row_parts.extend([self._escape_latex(str(values))] * len(data.model_names))
                parts.append(' & '.join(row_parts) + r' \\')
        
        return '\n'.join(parts)
    
    def _build_footer(self, data: RegressionTableData) -> str:
        notes_text = self._build_notes_text(data)
        sig_note = self._get_significance_note(data)
        if sig_note:
            notes_text += ' ' + sig_note
        
        n_cols = len(data.model_names) + 1
        return f'\\begin{{threeparttable}}\n\\begin{{tablenotes}}\\small\n\\item \\textit{{Notes:}} {self._escape_latex(notes_text)}\n\\end{{tablenotes}}\n\\end{{threeparttable}}'


class TableFormatterFactory:
    """Factory for creating table formatters."""
    
    _formatters = {
        'html': HtmlTableFormatter,
        'latex': LatexTableFormatter,
        'tex': LatexTableFormatter,
    }
    
    @classmethod
    def get_formatter(cls, format_type: str) -> TableFormatter:
        """Get a formatter for the specified format type."""
        format_type = format_type.lower()
        if format_type not in cls._formatters:
            raise ValueError(f"Unknown format type: {format_type}. Available: {list(cls._formatters.keys())}")
        return cls._formatters[format_type]()
    
    @classmethod
    def register_formatter(cls, format_type: str, formatter_class: type):
        """Register a new formatter type."""
        cls._formatters[format_type.lower()] = formatter_class


def _find_latest_model_result(model_name: str, analysis_type: str, base_path: Union[str, Path]) -> Path:
    """
    Find the most recent result file for a model.
    
    Parameters:
    -----------
    model_name : str
        Name of the model specification (e.g., 'ntlharm_regfav_twfe_replicate')
    analysis_type : str
        Type of analysis (e.g., 'duckreg', 'online_rls', 'online_2sls')
    base_path : str or Path
        Base output directory from analysis config
    
    Returns:
    --------
    Path : Path to the most recent results JSON file
    """
    base_path = Path(base_path)
    model_dir = base_path / analysis_type / model_name
    
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No results found for model '{model_name}' in {analysis_type} at {model_dir}"
        )
    
    # Find all results_YYYYMMDD_HHMMSS.json files
    result_files = sorted(model_dir.glob("results_*.json"))
    
    if not result_files:
        raise FileNotFoundError(
            f"No result files found in {model_dir}. Expected files like 'results_YYYYMMDD_HHMMSS.json'"
        )
    
    latest = result_files[-1]  # Last one is most recent (sorted by timestamp)
    return latest


def _load_models_by_name(model_specs: List[Dict[str, str]], base_path: Union[str, Path]) -> List[Dict]:
    """
    Load model data from result files by model name.
    
    Parameters:
    -----------
    model_specs : list of dict
        List of dicts with keys 'name' (model specification name) and 'analysis_type' (e.g., 'duckreg')
        Optional 'analysis_type' defaults to 'duckreg' if not provided
    base_path : str or Path
        Base output directory from analysis config
    
    Returns:
    --------
    list : List of loaded model data dictionaries. If a model result can't be found, a placeholder dict
           is returned for that model and its name is printed in the "Unavailable models" list.
    """
    models = []
    unavailable = []
    
    for spec in model_specs:
        if isinstance(spec, str):
            # Simple string format - treat as model name
            model_name = spec
            analysis_type = 'duckreg'  # Default
        elif isinstance(spec, dict):
            model_name = spec.get('name') or spec.get('model_name')
            analysis_type = spec.get('analysis_type', 'duckreg')
            
            if not model_name:
                raise ValueError(f"Model spec must have 'name' or 'model_name' key: {spec}")
        else:
            raise TypeError(f"Model spec must be string or dict, got {type(spec)}")
        
        try:
            # Find latest result file
            result_path = _find_latest_model_result(model_name, analysis_type, base_path)
            
            # Load the result
            with open(result_path, 'r') as f:
                model_data = json.load(f)
            
            models.append(model_data)
        except FileNotFoundError:
            # Record unavailable model but keep a placeholder so table keeps the column
            unavailable.append(f"{model_name} (analysis_type={analysis_type})")
            placeholder = {
                '__missing__': True,
                'model_spec': model_name,
                'metadata': {'analysis_type': analysis_type}
            }
            models.append(placeholder)
        except Exception as e:
            # Non-fatal: record as unavailable with error message and append placeholder
            unavailable.append(f"{model_name} (error: {e})")
            placeholder = {
                '__missing__': True,
                'model_spec': model_name,
                'metadata': {'analysis_type': analysis_type}
            }
            models.append(placeholder)
    
    if unavailable:
        print("Unavailable models:", ", ".join(unavailable))
    
    return models


def _get_model_metadata(model: Dict) -> Dict:
    """
    Get metadata from model, handling both 'metadata' and 'analysis_metadata' keys.
    Returns the metadata dict, or empty dict if neither key exists.
    """
    return model.get('analysis_metadata', model.get('metadata', {}))


def _get_model_version(model: Dict) -> str:
    """
    Extract version information from model.
    Handles both old format (metadata/timestamp) and new format (version_info).
    Returns version in x.x.x format, or 0.0.0 if not in valid format.
    """
    import re
    
    # Try new format first
    version_info = model.get('version_info', {})
    if version_info:
        version = version_info.get('duckreg_version', '')
        if version and re.match(r'^\d+\.\d+\.\d+$', str(version)):
            return str(version)
    
    # Try old format - look in coefficients or model_statistics
    coef_data = model.get('coefficients', {})
    if isinstance(coef_data, dict) and 'duckreg_version' in coef_data:
        version = coef_data.get('duckreg_version', '')
        if version and re.match(r'^\d+\.\d+\.\d+$', str(version)):
            return str(version)
    
    # Check model_statistics for estimator info
    model_stats = model.get('model_statistics', {})
    estimator_type = model_stats.get('estimator_type', '')
    if estimator_type and re.match(r'^\d+\.\d+\.\d+$', str(estimator_type)):
        return str(estimator_type)
    
    return '0.0.0'


def _get_model_date(model: Dict) -> str:
    """
    Extract computation date from model.
    Handles both old format (metadata/timestamp) and new format (version_info/computed_at).
    Returns date in YYYY-MM-DD format.
    """
    # Try new format first
    version_info = model.get('version_info', {})
    if version_info and 'computed_at' in version_info:
        timestamp = version_info['computed_at']
        # Parse ISO format timestamp and extract date
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        except:
            return timestamp.split('T')[0] if 'T' in timestamp else timestamp
    
    # Try coefficients block
    coef_data = model.get('coefficients', {})
    if isinstance(coef_data, dict) and 'computed_at' in coef_data:
        timestamp = coef_data['computed_at']
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        except:
            return timestamp.split('T')[0] if 'T' in timestamp else timestamp
    
    # Try old format in metadata
    metadata = _get_model_metadata(model)
    if 'timestamp' in metadata:
        timestamp = metadata['timestamp']
        try:
            from datetime import datetime
            dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            return dt.strftime('%Y-%m-%d')
        except:
            return timestamp.split('T')[0] if 'T' in timestamp else timestamp
    
    return 'N/A'


def _is_2sls_model(model: Dict) -> bool:
    """Check if model is a 2SLS/IV model."""
    return 'first_stage' in model and model.get('first_stage')


def _get_coefficient_data(model: Dict) -> Dict:
    """
    Get coefficient data from model, handling both OLS and 2SLS cases.
    For 2SLS, the main 'coefficients' block contains second stage results.
    """
    # Check if this is a 2SLS model - main coefficients ARE the second stage
    if _is_2sls_model(model):
        return model.get('coefficients', {})
    
    # For non-2SLS models, check for nested structure or direct coefficients
    metadata = _get_model_metadata(model)
    if metadata.get('analysis_type') == 'online_2sls':
        return model.get('second_stage', {}).get('coefficients', {})
    
    return model.get('coefficients', {})


def _extract_coefficient_data(models: List[Dict], model_names: List[str], 
                               select_coefs: Optional[Dict], decimals: int,
                               stars: bool, include_ci: bool) -> List[Dict]:
    """Extract and format coefficient rows from models (second stage for 2SLS)."""
    # Determine variables to include
    if select_coefs is not None:
        all_vars = list(select_coefs.keys())
    else:
        all_vars = []
        for model in models:
            coef_data = _get_coefficient_data(model)
            
            # Handle both 'names' and 'coef_names' keys
            var_names = coef_data.get('names', coef_data.get('coef_names', []))
            for var in var_names:
                if var not in all_vars:
                    all_vars.append(var)
    
    rows = []
    for var in all_vars:
        # Use select_coefs mapping if provided, marking as raw LaTeX to prevent escaping
        if select_coefs and var in select_coefs:
            display_name = _mark_raw_latex(select_coefs[var])
        else:
            display_name = var
        
        # Coefficient row
        coef_row = {'Variable': display_name}
        # SE row
        se_row = {'Variable': ''}
        
        for i, model in enumerate(models):
            coef_data = _get_coefficient_data(model)
            
            # Handle both 'names' and 'coef_names' keys
            var_names = coef_data.get('names', coef_data.get('coef_names', []))
            # Handle both 'estimates' and 'coefficients' keys
            estimates = coef_data.get('estimates', coef_data.get('coefficients', []))
            
            if var in var_names:
                idx = var_names.index(var)
                coef = estimates[idx]
                
                coef_str = f"{coef:.{decimals}f}"
                if stars and 'p_values' in coef_data:
                    p_val = coef_data['p_values'][idx]
                    if p_val < 0.001:
                        coef_str += "***"
                    elif p_val < 0.01:
                        coef_str += "**"
                    elif p_val < 0.05:
                        coef_str += "*"
                
                coef_row[model_names[i]] = coef_str
                
                if 'std_errors' in coef_data:
                    if include_ci:
                        ci_lower = coef_data.get('conf_int_lower', [None])[idx]
                        ci_upper = coef_data.get('conf_int_upper', [None])[idx]
                        if ci_lower is not None and ci_upper is not None:
                            se_row[model_names[i]] = f"[{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]"
                        else:
                            se_row[model_names[i]] = ""
                    else:
                        se = coef_data['std_errors'][idx]
                        se_row[model_names[i]] = f"({se:.{decimals}f})"
                else:
                    se_row[model_names[i]] = ""
            else:
                coef_row[model_names[i]] = ""
                se_row[model_names[i]] = ""
        
        rows.append(coef_row)
        rows.append(se_row)
    
    return rows


def _extract_first_stage_data(models: List[Dict], model_names: List[str],
                               decimals: int, stars: bool, include_ci: bool,
                               select_coefs: Optional[Dict] = None,
                               select_instruments: Optional[Dict] = None) -> List[Dict]:
    """Extract first stage coefficient rows for 2SLS models.
    
    Parameters:
    -----------
    select_coefs : dict, optional
        Map of instrument names to display names. If provided, only these instruments
        are shown in the first stage.
    select_instruments : dict, optional
        Alias for select_coefs (for consistency with second stage coefficient selection).
        If both provided, select_instruments takes precedence.
    """
    # Use select_instruments if provided, otherwise fall back to select_coefs
    instrument_filter = select_instruments or select_coefs
    
    # Collect all first stage variable names across models
    all_vars = []
    endog_var_names = []
    
    for model in models:
        if not _is_2sls_model(model):
            continue
        
        first_stage = model.get('first_stage', {})
        for endog_var, fs_data in first_stage.items():
            if endog_var not in endog_var_names:
                endog_var_names.append(endog_var)
            
            # Get instrument names (the key coefficients in first stage)
            instrument_names = fs_data.get('instrument_names', [])
            for inst in instrument_names:
                # Only add if not filtered by select_instruments
                if not instrument_filter or inst in instrument_filter:
                    if inst not in all_vars:
                        all_vars.append(inst)
    
    if not all_vars:
        return []
    
    rows = []
    
    # Add header row for first stage section
    for endog_var in endog_var_names:
        # Header for this endogenous variable's first stage
        header_row = {'Variable': f'First Stage: {endog_var}', '_is_header': True}
        for name in model_names:
            header_row[name] = ''
        rows.append(header_row)
        
        for var in all_vars:
            # Use select_instruments/select_coefs mapping if provided
            if instrument_filter and var in instrument_filter:
                display_name = instrument_filter[var]
                # If it contains LaTeX, mark it to prevent escaping in formatter
                if any(c in display_name for c in ['$', '\\', '{', '}']):
                    display_name = _mark_raw_latex(display_name)
            else:
                display_name = var
            
            # Coefficient row - display_name is already properly marked or plain
            coef_row = {'Variable': f'  {display_name}'}
            # SE row  
            se_row = {'Variable': ''}
            
            for i, model in enumerate(models):
                if not _is_2sls_model(model):
                    coef_row[model_names[i]] = ""
                    se_row[model_names[i]] = ""
                    continue
                
                first_stage = model.get('first_stage', {})
                fs_data = first_stage.get(endog_var, {})
                
                coef_names = fs_data.get('coef_names', [])
                coefficients = fs_data.get('coefficients', [])
                
                if var in coef_names:
                    idx = coef_names.index(var)
                    coef = coefficients[idx]
                    
                    coef_str = f"{coef:.{decimals}f}"
                    if stars and 'p_values' in fs_data:
                        p_val = fs_data['p_values'][idx]
                        if p_val < 0.001:
                            coef_str += "***"
                        elif p_val < 0.01:
                            coef_str += "**"
                        elif p_val < 0.05:
                            coef_str += "*"
                    
                    coef_row[model_names[i]] = coef_str
                    
                    if 'std_errors' in fs_data:
                        if include_ci:
                            se_row[model_names[i]] = ""  # CI not typically shown for first stage
                        else:
                            se = fs_data['std_errors'][idx]
                            se_row[model_names[i]] = f"({se:.{decimals}f})"
                    else:
                        se_row[model_names[i]] = ""
                else:
                    coef_row[model_names[i]] = ""
                    se_row[model_names[i]] = ""
            
            rows.append(coef_row)
            rows.append(se_row)
    
    return rows


def _extract_stat_rows(models: List[Dict], model_names: List[str],
                        show_stats: List[str], decimals: int,
                        show_first_stage: str = "default",
                        include_version_info: bool = True) -> List[Dict]:
    """Extract model statistics rows, including first stage F-stats for 2SLS."""
    stat_mapping = {
        'n_obs': ['n_obs', 'n_observations'],
        'n_observations': ['n_obs', 'n_observations'],
        'r_squared': ['r_squared'],
        'adj_r_squared': ['adj_r_squared'],
        'n_compressed_rows': ['n_compressed', 'n_compressed_rows'],
        'n_compressed': ['n_compressed', 'n_compressed_rows'],
        'compression_ratio': ['compression_ratio'],
        'df_model': ['df_model'],
        'df_resid': ['df_resid']
    }
    
    stat_rows = []
    
    # Add version and date rows if requested
    if include_version_info:
        # Version row
        version_row = {'Variable': 'Model Version'}
        for i, model in enumerate(models):
            version = _get_model_version(model)
            version_row[model_names[i]] = version
        stat_rows.append(version_row)
        
        # Date row
        date_row = {'Variable': 'Computed Date'}
        for i, model in enumerate(models):
            date = _get_model_date(model)
            date_row[model_names[i]] = date
        stat_rows.append(date_row)
    
    for stat_key in show_stats:
        stat_row = {'Variable': stat_key.replace('_', ' ').title()}
        
        for i, model in enumerate(models):
            # Check multiple locations for statistics
            sample_info = model.get('sample_info', {})
            model_stats = model.get('model_statistics', {})
            coef_data = model.get('coefficients', {})
            
            val = None
            if stat_key in stat_mapping:
                # Try each possible key in order of preference
                for key in stat_mapping[stat_key]:
                    # Check sample_info first (new format)
                    if key in sample_info and sample_info[key] is not None:
                        val = sample_info[key]
                        break
                    # Check model_statistics (old format)
                    if key in model_stats and model_stats[key] is not None:
                        val = model_stats[key]
                        break
                    # Check coefficients block
                    if key in coef_data and coef_data[key] is not None:
                        val = coef_data[key]
                        break
            else:
                # Check for exact key match
                if stat_key in sample_info and sample_info[stat_key] is not None:
                    val = sample_info[stat_key]
                elif stat_key in model_stats and model_stats[stat_key] is not None:
                    val = model_stats[stat_key]
                elif stat_key in coef_data and coef_data[stat_key] is not None:
                    val = coef_data[stat_key]
            
            if val is not None:
                if stat_key in ['r_squared', 'adj_r_squared', 'compression_ratio']:
                    stat_row[model_names[i]] = f"{val:.{decimals}f}"
                elif stat_key in ['n_obs', 'n_observations', 'n_compressed', 'n_compressed_rows', 'df_model', 'df_resid']:
                    stat_row[model_names[i]] = f"{int(val):,}"
                else:
                    stat_row[model_names[i]] = str(val)
            else:
                stat_row[model_names[i]] = ""
        
        stat_rows.append(stat_row)
    
    # Add first stage F-statistics if show_first_stage is "default" or "full"
    if show_first_stage in ("default", "full"):
        # Check if any model is 2SLS
        has_2sls = any(_is_2sls_model(model) for model in models)
        
        if has_2sls:
            # Collect all endogenous variables
            endog_vars = set()
            for model in models:
                if _is_2sls_model(model):
                    first_stage = model.get('first_stage', {})
                    endog_vars.update(first_stage.keys())
            
            for endog_var in sorted(endog_vars):
                # Create F-stat row
                f_stat_row = {'Variable': f'First Stage F ({endog_var})'}
                
                for i, model in enumerate(models):
                    if not _is_2sls_model(model):
                        f_stat_row[model_names[i]] = ""
                        continue
                    
                    first_stage = model.get('first_stage', {})
                    fs_data = first_stage.get(endog_var, {})
                    f_stat = fs_data.get('f_statistic')
                    
                    if f_stat is not None:
                        f_stat_row[model_names[i]] = f"{f_stat:.{decimals}f}"
                    else:
                        f_stat_row[model_names[i]] = ""
                
                stat_rows.append(f_stat_row)
    
    return stat_rows


def _normalize_custom_rows(custom_rows: Optional[Union[Dict, List]], 
                           model_names: List[str]) -> List[Dict]:
    """Normalize custom rows to list of dicts format."""
    if not custom_rows:
        return []
    
    if isinstance(custom_rows, dict):
        custom_row_groups = [custom_rows]
    else:
        custom_row_groups = custom_rows
    
    # Validate
    for row_group in custom_row_groups:
        for label, values in row_group.items():
            if isinstance(values, (list, tuple)) and len(values) != len(model_names):
                raise ValueError(
                    f"Custom row '{label}' has {len(values)} values but there are {len(model_names)} models"
                )
    
    return custom_row_groups


class ExcelConfigLoader:
    """Load table configurations from the analysis Excel workbook.

    Reads the following sheets from ``orchestration/configs/analysis.xlsx``:

    * ``Models in Tables`` — columns ``table_name``, ``model_name``, ``order``
    * ``Models``           — one row per model (provides model-level metadata)
    * ``Tables`` (optional) — one row per table with display overrides:
        ``table_name``, ``caption``, ``model_display_names`` (comma-separated),
        ``output_formats`` (comma-separated), ``show_stats`` (comma-separated),
        ``decimals``, ``stars``, ``include_ci``, ``notes``,
        ``show_first_stage``, ``table_environment``, ``table_size``
    """

    DEFAULT_EXCEL = PROJECT_ROOT / 'orchestration' / 'configs' / 'analysis.xlsx'

    def __init__(self, excel_path: Union[str, Path] = None):
        self.excel_path = Path(excel_path) if excel_path else self.DEFAULT_EXCEL
        if not self.excel_path.exists():
            raise FileNotFoundError(f"Analysis Excel not found: {self.excel_path}")
        self._df_mit, self._df_models, self._df_tables = self._load_sheets()
        self.base_path = RESULTS_DIR

    # ── sheet loading ────────────────────────────────────────────────────────

    def _load_sheets(self):
        xl = pd.ExcelFile(self.excel_path)
        df_mit = pd.read_excel(xl, sheet_name='Models in Tables')
        df_models = pd.read_excel(xl, sheet_name='Models')
        df_tables = (
            pd.read_excel(xl, sheet_name='Tables')
            if 'Tables' in xl.sheet_names
            else None
        )
        return df_mit, df_models, df_tables

    # ── public API ───────────────────────────────────────────────────────────

    def get_all_table_names(self) -> List[str]:
        """Return unique table names in sheet order."""
        return list(self._df_mit['table_name'].dropna().unique())

    def get_models_for_table(self, table_name: str) -> List[str]:
        """Return model names for *table_name*, sorted by the ``order`` column."""
        rows = self._df_mit[
            self._df_mit['table_name'] == table_name
        ].sort_values('order')
        if rows.empty:
            available = self.get_all_table_names()
            raise KeyError(
                f"Table '{table_name}' not found in 'Models in Tables'. "
                f"Available: {available}"
            )
        return rows['model_name'].tolist()

    def get_table_config(self, table_name: str) -> Dict[str, Any]:
        """Return a ``create_regression_table``-compatible config dict."""
        models = self.get_models_for_table(table_name)
        config: Dict[str, Any] = {'model_paths': models}

        if self._df_tables is None:
            return config

        trow = self._df_tables[self._df_tables['table_name'] == table_name]
        if trow.empty:
            return config

        r = trow.iloc[0]

        # Scalar overrides
        for col in ('caption', 'notes', 'show_first_stage', 'table_environment', 'table_size'):
            val = r.get(col)
            if val is not None and pd.notna(val):
                config[col] = str(val).strip()

        # 'fontsize' is a user-friendly alias for 'table_size' (takes precedence)
        fontsize_val = r.get('fontsize')
        if fontsize_val is not None and pd.notna(fontsize_val):
            config['table_size'] = str(fontsize_val).strip()

        for col in ('decimals',):
            val = r.get(col)
            if val is not None and pd.notna(val):
                config[col] = int(val)

        for col in ('stars', 'include_ci'):
            val = r.get(col)
            if val is not None and pd.notna(val):
                config[col] = bool(val)

        # Comma-separated list columns
        def _split(col_name: str) -> Optional[List[str]]:
            val = r.get(col_name)
            if val is not None and pd.notna(val):
                return [s.strip() for s in str(val).split(',') if s.strip()]
            return None

        model_display_names = _split('model_display_names')
        if model_display_names:
            config['model_names'] = model_display_names

        output_formats = _split('output_formats')
        if output_formats:
            config['output_formats'] = output_formats

        show_stats = _split('show_stats')
        if show_stats:
            config['show_stats'] = show_stats

        return config

    @property
    def models_df(self) -> pd.DataFrame:
        """Raw ``Models`` sheet DataFrame."""
        return self._df_models


def create_regression_table(model_specs: Union[List[str], List[Dict]], 
                            base_path: Union[str, Path],
                            model_names: List[str] = None,
                            show_stats: List[str] = None,
                            include_ci: bool = False, 
                            decimals: int = 3, 
                            stars: bool = True, 
                            caption: str = None, 
                            select_coefs: Dict = None,
                            select_instruments: Dict = None,
                            custom_rows: Union[Dict, List] = None, 
                            notes: Union[str, List[str]] = None, 
                            header_rows: List[Dict] = None,
                            show_first_stage: str = "default",
                            table_environment: str = "table",
                            table_size: str = "footnotesize",
                            output_format: str = 'html') -> str:
    """
    Create a regression table in the specified format from model names.
    
    Parameters:
    -----------
    model_specs : list of str or list of dict
        Model specifications. Can be simple model names or dicts with 'name' and 'analysis_type'.
        Example: ['ntlharm_regfav_twfe_replicate', 'ntlharm_regfav_twfe_HDI']
    base_path : str or Path
        Base output directory where analysis results are stored
    model_names : list, optional
        Display names for each model column. If None, uses model spec names.
    show_stats : list, optional
        Model statistics to display.
    include_ci : bool, optional
        Include confidence intervals instead of std errors.
    decimals : int, optional
        Number of decimal places.
    stars : bool, optional
        Add significance stars.
    caption : str, optional
        Table caption.
    select_coefs : dict, optional
        Map coefficient names to display names (second stage).
    select_instruments : dict, optional
        Map instrument names to display names (first stage). If omitted, shows all instruments.
    custom_rows : dict or list, optional
        Custom rows below statistics.
    notes : str or list, optional
        Custom notes for footer.
    header_rows : list, optional
        Multi-level header definitions.
    show_first_stage : str, optional
        How to display first stage for 2SLS models:
        - "none": Don't show any first stage information
        - "default": Show only F-statistics in the stats section
        - "full": Show full first stage coefficients above second stage + F-stats
        Default is "default".
    table_environment : str, optional
        LaTeX table environment to use. Examples: 'table', 'sidewaystable', 'table*'.
        Default is 'table'.
    table_size : str, optional
        LaTeX text size command for the table. Examples: 'tiny', 'scriptsize', 'footnotesize', 
        'small', 'normalsize', 'large', 'Large', 'LARGE', 'huge', 'Huge'.
        Default is 'footnotesize'.
    output_format : str, optional
        Output format: 'html', 'latex', or 'tex'. Default is 'html'.
    
    Returns:
    --------
    str : Formatted table string
    """
    # Load models by name
    models = _load_models_by_name(model_specs, base_path)
    
    if not models:
        return "<p>No models to display</p>" if output_format == 'html' else "% No models to display"
    
    # Determine if any model is 2SLS/IV (used to control "Second Stage" headers)
    has_2sls = any(_is_2sls_model(m) for m in models)
    
    # Generate model names
    if model_names is None:
        # Use specification names or model names from config
        model_names = []
        for spec in model_specs:
            if isinstance(spec, str):
                model_names.append(spec)
            elif isinstance(spec, dict):
                model_names.append(spec.get('name', f"Model {len(model_names) + 1}"))
    elif len(model_names) != len(models):
        raise ValueError(f"Number of model names ({len(model_names)}) must match number of models ({len(models)})")
    
    # Set defaults for show_stats
    if show_stats is None:
        show_stats = ['n_observations', 'n_compressed']
    
    # Extract data
    coefficient_rows = _extract_coefficient_data(models, model_names, select_coefs, decimals, stars, include_ci)
    stat_rows = _extract_stat_rows(models, model_names, show_stats, decimals, show_first_stage, include_version_info=True)
    custom_row_groups = _normalize_custom_rows(custom_rows, model_names)
    
    # Extract first stage data if needed
    first_stage_rows = None
    if show_first_stage == "full":
        first_stage_rows = _extract_first_stage_data(models, model_names, decimals, stars, include_ci, 
                                                     select_coefs, select_instruments)
    
    # Build data structure
    table_data = RegressionTableData(
        model_names=model_names,
        coefficient_rows=coefficient_rows,
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
        table_size=table_size
    )
    
    # Get formatter and format
    formatter = TableFormatterFactory.get_formatter(output_format)
    return formatter.format(table_data)


def generate_table_from_config(
    table_name: str,
    config_loader: ExcelConfigLoader = None,
    output_formats: List[str] = None,
) -> Dict[str, str]:
    """Generate a table from the Excel config.

    Parameters
    ----------
    table_name:
        Name of the table as defined in the ``Models in Tables`` sheet.
    config_loader:
        ``ExcelConfigLoader`` instance.  If *None* a default one is created.
    output_formats:
        Override the output formats for this call (e.g. ``['html', 'latex']``).
        If *None*, uses the value from the ``Tables`` sheet or falls back to
        ``['html']``.

    Returns
    -------
    dict
        ``{format: rendered_string}``
    """
    if config_loader is None:
        config_loader = ExcelConfigLoader()

    table_config = config_loader.get_table_config(table_name)

    # Determine output formats
    if output_formats is None:
        output_formats = table_config.pop('output_formats', ['html'])
    else:
        table_config.pop('output_formats', None)

    model_specs = table_config.pop('model_paths', None)
    if not model_specs:
        raise ValueError(
            f"Table '{table_name}' has no models in 'Models in Tables' sheet"
        )

    results = {}
    for fmt in output_formats:
        results[fmt] = create_regression_table(
            model_specs=model_specs,
            base_path=config_loader.base_path,
            **table_config,
            output_format=fmt,
        )

    return results


def save_generated_tables(
    table_name: str,
    generated_tables: Dict[str, str],
    output_dir: Optional[Union[str, Path]] = None,
) -> None:
    """Save generated tables to files.

    Parameters
    ----------
    table_name:
        Name of the table (used in the output filename).
    generated_tables:
        ``{format: table_string}`` mapping returned by
        :func:`generate_table_from_config`.
    output_dir:
        Output directory.  Defaults to ``output/analysis/tables``.
    """
    out = Path(output_dir) if output_dir else OUTPUT_DIR
    out.mkdir(parents=True, exist_ok=True)

    _extensions = {'html': '.html', 'latex': '.tex', 'tex': '.tex'}

    for fmt, table_str in generated_tables.items():
        ext = _extensions.get(fmt, f'.{fmt}')
        output_file = out / f"table_{table_name}{ext}"
        output_file.write_text(table_str)
        print(f"  Generated: {output_file}")


# ── High-level entry points ──────────────────────────────────────────────────

def summarize_tables(loader: ExcelConfigLoader = None) -> None:
    """Print a status overview of every table defined in the Excel config.

    For each model the function looks up the latest result file and reports:
    - the computation date
    - the duckreg version used
    - whether results are present
    """
    if loader is None:
        loader = ExcelConfigLoader()

    table_names = loader.get_all_table_names()

    print(f"\n{'=' * 80}")
    print(f"  Analysis Tables Summary")
    print(f"  Excel  : {loader.excel_path}")
    print(f"  Results: {loader.base_path}")
    print(f"{'=' * 80}")

    col_w = 52
    for table_name in table_names:
        models = loader.get_models_for_table(table_name)
        print(f"\n  Table: {table_name}  ({len(models)} model{'s' if len(models) != 1 else ''})")
        print(f"  {'Model':<{col_w}}  {'Last Run':<12}  {'Version':<12}  Status")
        print(f"  {'-' * col_w}  {'-' * 12}  {'-' * 12}  ------")
        for model_name in models:
            try:
                result_path = _find_latest_model_result(model_name, 'duckreg', loader.base_path)
                with open(result_path) as fh:
                    data = json.load(fh)
                date = _get_model_date(data)
                version = _get_model_version(data)
                status = "ok"
            except FileNotFoundError:
                date = 'N/A'
                version = 'N/A'
                status = "missing"
            except Exception as exc:
                date = 'error'
                version = 'error'
                status = f"error: {exc}"
            print(f"  {model_name:<{col_w}}  {date:<12}  {version:<12}  {status}")

    print()


def generate_all_tables(
    loader: ExcelConfigLoader = None,
    table_names: Optional[List[str]] = None,
    output_dir: Optional[Union[str, Path]] = None,
    output_formats: Optional[List[str]] = None,
) -> None:
    """Generate and save all (or a subset of) tables from the Excel config.

    Parameters
    ----------
    loader:
        ``ExcelConfigLoader`` instance.  Created with defaults when *None*.
    table_names:
        Specific tables to generate.  All tables when *None*.
    output_dir:
        Output directory for the generated files.
    output_formats:
        Override output formats for every table (e.g. ``['html', 'latex']``).
    """
    if loader is None:
        loader = ExcelConfigLoader()

    names = table_names or loader.get_all_table_names()

    for table_name in names:
        print(f"Generating table: {table_name}")
        try:
            generated = generate_table_from_config(
                table_name, loader, output_formats=output_formats
            )
            save_generated_tables(table_name, generated, output_dir)
        except Exception as exc:
            print(f"  Error generating '{table_name}': {exc}")


# Keep the old name as an alias for backward compatibility.
generate_main_table = generate_all_tables


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Work with analysis tables defined in orchestration/configs/analysis.xlsx.\n\n"
            "Modes:\n"
            "  summary   Print a status overview of every table and model.\n"
            "  generate  Render and save table files (default)."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        '--mode',
        choices=['summary', 'generate'],
        default='generate',
        help="Operation mode (default: generate)",
    )
    parser.add_argument(
        '--excel',
        default=None,
        metavar='PATH',
        help="Path to analysis.xlsx (default: orchestration/configs/analysis.xlsx)",
    )
    parser.add_argument(
        '--tables',
        nargs='*',
        metavar='TABLE',
        default=None,
        help="Specific table names to process (default: all tables)",
    )
    parser.add_argument(
        '--output-dir',
        default=None,
        metavar='DIR',
        help="Output directory for generated table files (default: output/analysis/tables)",
    )
    parser.add_argument(
        '--formats',
        nargs='*',
        choices=['html', 'latex', 'tex'],
        metavar='FMT',
        default=None,
        help="Output formats: html, latex, tex (default: from Tables sheet or html)",
    )

    args = parser.parse_args()

    # Delegate to the canonical implementation in gnt.analysis.tables / config
    try:
        from gnt.analysis.config import AnalysisConfig
        from gnt.analysis.tables import summarize_tables as _summarize, generate_all_tables as _gen_all
    except ImportError:
        # Fallback: use local ExcelConfigLoader when package isn't installed
        from gnt.analysis.generate_tables import ExcelConfigLoader as AnalysisConfig  # type: ignore
        _summarize = summarize_tables  # type: ignore
        _gen_all = generate_all_tables  # type: ignore

    try:
        cfg = AnalysisConfig(args.excel)
    except FileNotFoundError as exc:
        print(f"Error: {exc}")
        return 1

    if args.mode == 'summary':
        _summarize(cfg)
        return 0

    # generate mode
    _gen_all(
        config=cfg,
        table_names=args.tables,
        output_dir=args.output_dir,
        output_formats=args.formats,
    )
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
