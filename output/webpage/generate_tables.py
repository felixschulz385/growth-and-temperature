"""
Generate HTML table partials for Jekyll site.
Run this script to regenerate all tables from analysis results.
"""

import json
import pandas as pd
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "output" / "analysis"
OUTPUT_DIR = Path(__file__).parent / "_includes"

# Ensure output directory exists
OUTPUT_DIR.mkdir(exist_ok=True)



def create_regression_table(model_paths, model_names=None, show_stats=['n_obs', 'r_squared', 'adj_r_squared'], 
                           include_ci=False, decimals=3, stars=True, caption=None, select_coefs=None,
                           custom_rows=None, notes=None, header_rows=None):
    """
    Create a beautiful HTML regression table similar to texreg in R.
    
    Parameters:
    -----------
    model_paths : str or Path or list
        Path(s) to JSON result file(s). Can be a single path or list of paths.
    model_names : list, optional
        Custom names for each model column. If None, uses "Model 1", "Model 2", etc.
    show_stats : list, optional
        Which model statistics to display at bottom. Options: 'n_obs', 'r_squared', 
        'adj_r_squared', 'df_model', 'df_resid', 'n_compressed_rows', 'compression_ratio'
    include_ci : bool, optional
        Whether to include confidence intervals in parentheses instead of std errors
    decimals : int, optional
        Number of decimal places for coefficients
    stars : bool, optional
        Whether to add significance stars (* p<0.05, ** p<0.01, *** p<0.001)
    caption : str, optional
        Table caption
    select_coefs : dict, optional
        Dictionary mapping original coefficient names to display names.
        If provided, only these coefficients are shown in the order specified.
        Example: {"nb_mines_a": "Number of Mines", "Intercept": "Constant"}
    custom_rows : dict or list of dict, optional
        Dictionary or list of dictionaries for custom rows below statistics.
        Keys are row labels, values are either single values or lists.
        If a list of dicts is provided, each dict is a separate group with soft borders.
        Example: [
            {"Fixed Effects": "Pixel + Year", "Clustering": "ADM2"},
            {"Sample": ["Full", "Full"], "Period": ["1992-2009", "2000-2010"]}
        ]
    notes : str or list, optional
        Custom notes to display in table footer before significance stars.
        Can be a single string or list of strings (each becomes a separate sentence).
        Example: "All models include time trends." or 
                ["Robust standard errors.", "Weighted by population."]
    header_rows : list of dict, optional
        List of header row definitions for multi-level headers with colspan.
        Each dict maps header names to column slices.
        Example: [
            {"Data Source": slice(0, 2), "Sample": slice(2, 4)},
            {}  # Empty dict for the model names row
        ]
    
    Returns:
    --------
    str : HTML table string
    """
    # Convert single path to list
    if isinstance(model_paths, (str, Path)):
        model_paths = [Path(model_paths)]
    else:
        model_paths = [Path(p) for p in model_paths]
    
    # Load all models
    models = []
    for path in model_paths:
        if path.exists():
            with open(path, 'r') as f:
                models.append(json.load(f))
        else:
            raise FileNotFoundError(f"Model file not found: {path}")
    
    if not models:
        return "<p>No models to display</p>"
    
    # Generate model names
    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]
    elif len(model_names) != len(models):
        raise ValueError(f"Number of model names ({len(model_names)}) must match number of models ({len(models)})")
    
    # Extract all unique variables across models
    if select_coefs is not None:
        # Use specified coefficients in specified order
        all_vars = list(select_coefs.keys())
    else:
        # Extract all unique variables across models
        all_vars = []
        for model in models:
            # Handle 2SLS second stage
            if model['metadata']['analysis_type'] == 'online_2sls':
                coef_data = model.get('second_stage', {}).get('coefficients', {})
            else:
                coef_data = model.get('coefficients', {})
            
            var_names = coef_data.get('names', [])
            for var in var_names:
                if var not in all_vars:
                    all_vars.append(var)
    
    # Build coefficient rows
    rows = []
    for var in all_vars:
        # Determine display name
        display_name = select_coefs[var] if select_coefs else var
        row_data = {'Variable': display_name}
        
        for i, model in enumerate(models):
            # Get coefficients for this model
            if model['metadata']['analysis_type'] == 'online_2sls':
                coef_data = model.get('second_stage', {}).get('coefficients', {})
            else:
                coef_data = model.get('coefficients', {})
            
            var_names = coef_data.get('names', [])
            
            if var in var_names:
                idx = var_names.index(var)
                coef = coef_data['estimates'][idx]
                
                # Format coefficient with stars if available
                coef_str = f"{coef:.{decimals}f}"
                if stars and 'p_values' in coef_data:
                    p_val = coef_data['p_values'][idx]
                    if p_val < 0.001:
                        coef_str += "***"
                    elif p_val < 0.01:
                        coef_str += "**"
                    elif p_val < 0.05:
                        coef_str += "*"
                
                row_data[model_names[i]] = coef_str
                
                # Add standard error or CI row
                if 'std_errors' in coef_data:
                    if include_ci:
                        ci_lower = coef_data['conf_int_lower'][idx]
                        ci_upper = coef_data['conf_int_upper'][idx]
                        row_data[f"{model_names[i]}_se"] = f"[{ci_lower:.{decimals}f}, {ci_upper:.{decimals}f}]"
                    else:
                        se = coef_data['std_errors'][idx]
                        row_data[f"{model_names[i]}_se"] = f"({se:.{decimals}f})"
                else:
                    row_data[f"{model_names[i]}_se"] = ""
            else:
                row_data[model_names[i]] = ""
                row_data[f"{model_names[i]}_se"] = ""
        
        rows.append(row_data)
    
    # Create DataFrame for coefficients
    df_rows = []
    for row in rows:
        # Add coefficient row
        coef_row = {'Variable': row['Variable']}
        for name in model_names:
            coef_row[name] = row.get(name, "")
        df_rows.append(coef_row)
        
        # Add SE/CI row
        se_row = {'Variable': ''}
        for name in model_names:
            se_row[name] = row.get(f"{name}_se", "")
        df_rows.append(se_row)
    
    df = pd.DataFrame(df_rows)
    
    # Add model statistics
    stat_rows = []
    
    # Stat key mapping for different model types
    stat_mapping = {
        'n_obs': ['n_obs', 'n_observations'],
        'r_squared': ['r_squared'],
        'adj_r_squared': ['adj_r_squared'],
        'n_compressed_rows': ['n_compressed_rows'],
        'compression_ratio': ['compression_ratio'],
        'df_model': ['df_model'],
        'df_resid': ['df_resid']
    }
    
    for stat_key in show_stats:
        stat_row = {'Variable': stat_key.replace('_', ' ').title()}
        
        for i, model in enumerate(models):
            stats = model.get('model_statistics', {})
            
            # Try all possible keys for this statistic
            val = None
            if stat_key in stat_mapping:
                for key in stat_mapping[stat_key]:
                    if key in stats and stats[key] is not None:
                        val = stats[key]
                        break
            elif stat_key in stats and stats[stat_key] is not None:
                val = stats[stat_key]
            
            if val is not None:
                if stat_key in ['r_squared', 'adj_r_squared', 'compression_ratio']:
                    stat_row[model_names[i]] = f"{val:.{decimals}f}"
                elif stat_key in ['n_obs', 'n_observations', 'n_compressed_rows', 'df_model', 'df_resid']:
                    stat_row[model_names[i]] = f"{int(val):,}"
                else:
                    stat_row[model_names[i]] = str(val)
            else:
                stat_row[model_names[i]] = ""
        
        stat_rows.append(stat_row)
    
    if stat_rows:
        df_stats = pd.DataFrame(stat_rows)
        df = pd.concat([df, df_stats], ignore_index=True)
    
    # Track where custom rows start (for styling)
    custom_rows_start_idx = len(df)
    custom_row_group_starts = []  # Track start of each group
    
    # Add custom rows if provided
    if custom_rows:
        # Normalize to list of dicts
        if isinstance(custom_rows, dict):
            custom_row_groups = [custom_rows]
        else:
            custom_row_groups = custom_rows
        
        for group_idx, row_group in enumerate(custom_row_groups):
            # Track start of this group for styling
            if group_idx > 0:
                custom_row_group_starts.append(len(df))
            
            for label, values in row_group.items():
                custom_row = {'Variable': label}
                
                # Handle single value or list of values
                if isinstance(values, (list, tuple)):
                    if len(values) != len(model_names):
                        raise ValueError(f"Custom row '{label}' has {len(values)} values but there are {len(model_names)} models")
                    for i, name in enumerate(model_names):
                        custom_row[name] = str(values[i])
                else:
                    # Single value applied to all models
                    for name in model_names:
                        custom_row[name] = str(values)
                
                df = pd.concat([df, pd.DataFrame([custom_row])], ignore_index=True)
    
    # Build HTML table (CSS is in separate file)
    html_parts = []
    
    if caption:
        html_parts.append(f'<div class="table-caption">{caption}</div>')
    
    html_parts.append('<table class="regression-table">')
    
    # Header
    html_parts.append('<thead>')
    
    # Add header rows if provided
    if header_rows:
        for i, header_row_def in enumerate(header_rows):
            html_parts.append('<tr>')
            html_parts.append('<th></th>')  # Empty cell for row labels column
            
            col_idx = 0
            for header_label, col_slice in header_row_def.items():
                if header_label:  # Non-empty header label
                    colspan = col_slice.stop - col_slice.start
                    html_parts.append(f'<th colspan="{colspan}" class="header-span">{header_label}</th>')
                    col_idx = col_slice.stop
                else:
                    # Empty cells without border
                    if isinstance(col_slice, slice):
                        for _ in range(col_slice.start, col_slice.stop):
                            html_parts.append('<th></th>')
                        col_idx = col_slice.stop
            
            # Fill remaining columns if needed
            if col_idx < len(model_names):
                for _ in range(col_idx, len(model_names)):
                    html_parts.append('<th></th>')
            
            html_parts.append('</tr>')
    
    # Main model names header row
    html_parts.append('<tr class="model-names-row">')
    html_parts.append('<th></th>')  # Empty for variable names column
    for name in model_names:
        html_parts.append(f'<th>{name}</th>')
    html_parts.append('</tr>')
    html_parts.append('</thead>')
    
    # Body - coefficients, stats, and custom rows
    html_parts.append('<tbody>')
    stat_start_idx = len(df_rows)
    
    for idx, row in df.iterrows():
        if idx == stat_start_idx:
            html_parts.append('<tr class="stats-section">')
        elif idx == custom_rows_start_idx:
            html_parts.append('<tr class="custom-section">')
        elif idx in custom_row_group_starts:
            html_parts.append('<tr class="custom-group-section">')
        else:
            # Add coef-row class only to coefficient rows (not SE rows)
            if idx < stat_start_idx and idx % 2 == 0:
                html_parts.append('<tr class="coef-row">')
            else:
                html_parts.append('<tr>')
        
        for col in df.columns:
            html_parts.append(f'<td>{row[col]}</td>')
        html_parts.append('</tr>')
    html_parts.append('</tbody>')
    
    # Footer with notes
    html_parts.append('<tfoot><tr><td colspan="' + str(len(model_names) + 1) + '">')
    
    # Build notes section as a single string
    notes_text = 'Notes: '
    
    # Add custom notes if provided
    if notes:
        if isinstance(notes, str):
            notes_text += notes
            if not notes.endswith('.'):
                notes_text += '.'
        elif isinstance(notes, (list, tuple)):
            for i, note in enumerate(notes):
                notes_text += note
                if not note.endswith('.'):
                    notes_text += '.'
                if i < len(notes) - 1:
                    notes_text += ' '
        notes_text += ' '
    
    # Add standard error/CI note
    notes_text += 'Standard errors in parentheses.' if not include_ci else '95% confidence intervals in brackets.'
    
    # Add significance stars note
    if stars:
        notes_text += ' * p&lt;0.05, ** p&lt;0.01, *** p&lt;0.001'
    
    html_parts.append(notes_text)
    html_parts.append('</td></tr></tfoot>')
    
    html_parts.append('</table>')
    
    return '\n'.join(html_parts)


def generate_main_table():
    """Generate the main regression table."""
    html_table = create_regression_table(
        [
            "../analysis/duckreg/ntlharm_regfav_twfe_replicate/results_20251124_115759.json", 
            "../analysis/duckreg/ntlharm_regfav_twfe_HDI/results_20251124_142533.json",
            "../analysis/duckreg/ntlharm_regfav_twfe_af/results_20251124_142202.json",
            "../analysis/duckreg/ntlharm_mines_twfe/results_20251124_144038.json"
        ],
        model_names=["(1.1)", "(1.2)", "(1.3)", "(2)"],
        caption="Effect of Mining and Regime Favorability on Nighttime Lights",
        select_coefs={
            "reg_fav": "Leader's Birthregion",
            "nb_mines_a": "Number of Active Mines",
        },
        show_stats=['n_observations', 'n_compressed_rows'],
        custom_rows=[
            {
                "Fixed Effects": "Pixel + Country×Year",
                "Clustering": "ADM2",
            },
            {
                "Resolution": "5km",
                "Spatial Extent": ["H&R", "HDI low or medium", "Africa", "Africa"],
                "Temporal Extent": ["1992-2009", "1992-2021", "1992-2021", "2000-2010"]
            }
        ],
        notes=[
            "Dependent variable is log(nighttime lights + 0.01)",
            "Both models use Mundlak fixed effects estimation with bootstrapped standard errors (999 iterations)"
        ],
        include_ci=False,
        decimals=3,
        header_rows=[
            {
                "Birthregion": slice(0, 3),
                "Mining": slice(3, 4)
            }
        ]
    )
    
    # Write to file
    output_file = OUTPUT_DIR / "table_main.html"
    with open(output_file, 'w') as f:
        f.write(html_table)
    print(f"Generated: {output_file}")

if __name__ == "__main__":
    print("Generating HTML table partials...")
    generate_main_table()
    print("Done!")
