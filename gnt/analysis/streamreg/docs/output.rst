.. filepath: /scicore/home/meiera/schulz0022/projects/growth-and-temperature/docs/output.rst

Output Formatting System
========================

StreamReg provides a flexible output system for saving regression results in multiple formats.
The system is object-oriented: results know how to save themselves.

Overview
--------

After fitting a model, results can be saved using:

.. code-block:: python

    model.fit(data)
    model.results_.save("output/my_analysis")

This creates a timestamped directory with comprehensive results in multiple formats.

Output Directory Structure
--------------------------

Example output structure::

    output/
    └── my_analysis_20240115_143022/
        ├── README.md                 # Overview and quick results
        ├── summary.txt               # Full formatted report
        ├── results.json              # Machine-readable results
        ├── coefficients.csv          # Coefficient table
        ├── table.tex                 # LaTeX table
        ├── diagnostics.txt           # Detailed diagnostics
        └── config_snapshot.json      # Configuration used

Available Formats
-----------------

JSON Format (``results.json``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Machine-readable format containing all results:

.. code-block:: python

    model.results_.save_json(output_dir)

Contents:

* All coefficients with standard errors, t-statistics, p-values
* Model fit statistics (R², adjusted R², RSS, RMSE)
* F-statistics and degrees of freedom
* First stage results (for 2SLS)
* Cluster diagnostics
* Metadata

Example structure:

.. code-block:: json

    {
      "model_type": "ols",
      "cluster_type": "two_way",
      "n_obs": 150000,
      "n_features": 5,
      "r_squared": 0.4523,
      "adj_r_squared": 0.4519,
      "f_statistic": 617.34,
      "f_pvalue": 0.0,
      "coefficients": {
        "temperature": {
          "estimate": 0.0234,
          "std_error": 0.0045,
          "t_statistic": 5.20,
          "p_value": 0.0001
        }
      }
    }

CSV Format (``coefficients.csv``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Spreadsheet-compatible coefficient table:

.. code-block:: python

    model.results_.save_csv(output_dir)

Columns:

* ``feature``: Variable name
* ``coefficient``: Estimated coefficient
* ``std_error``: Standard error
* ``t_statistic``: t-statistic
* ``p_value``: p-value
* ``sig``: Significance stars (*** p<0.01, ** p<0.05, * p<0.10)
* ``n_obs``: Number of observations
* ``r_squared``: R-squared

Summary Report (``summary.txt``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Human-readable text report:

.. code-block:: python

    model.results_.save_summary(
        output_dir,
        spec_config={'description': 'My analysis'}
    )

Sections:

1. **Header**: Analysis description, model type, data source, timestamp
2. **Model Statistics**: N, R², adjusted R², F-statistic, RSS, RMSE
3. **Coefficient Table**: Formatted coefficient estimates with significance
4. **Cluster Diagnostics**: Cluster counts, sizes, balance, warnings
5. **First Stage Results**: For 2SLS models

Example output::

    ================================================================================
    REGRESSION ANALYSIS RESULTS
    ================================================================================
    
    Analysis: Climate-Growth Baseline
    Model Type: OLS
    Date: 2024-01-15 14:30:22
    
    --------------------------------------------------------------------------------
    
    MODEL STATISTICS
    --------------------------------------------------------------------------------
    Observations:                    150,000
    Features:                              5
    R-squared:                        0.4523
    Adjusted R-squared:               0.4519
    F-statistic:                     617.340
    Standard Error Type:            two_way
    
    --------------------------------------------------------------------------------
    
    COEFFICIENT ESTIMATES
    --------------------------------------------------------------------------------
                         coefficient  std_error  t_statistic   p_value sig
    intercept               -0.0123     0.0089      -1.3820    0.1670    
    temperature              0.0234     0.0045       5.2000    0.0001 ***
    rainfall                 0.0156     0.0023       6.7826    0.0000 ***
    temperature_squared     -0.0012     0.0003      -4.0000    0.0001 ***
    
    Significance levels: *** p<0.01, ** p<0.05, * p<0.10

LaTeX Format (``table.tex``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Publication-ready LaTeX table:

.. code-block:: python

    model.results_.save_latex(output_dir)

Generates a complete ``table`` environment with:

* Coefficients with significance stars
* Standard errors in parentheses
* Model statistics (N, R²)
* Notes on standard error type

Can be directly included in LaTeX documents:

.. code-block:: latex

    \input{output/my_analysis_20240115_143022/table.tex}

Diagnostics (``diagnostics.txt``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Detailed diagnostic information:

.. code-block:: python

    model.results_.save_diagnostics(
        output_dir,
        spec_config={
            'formula': 'y ~ x1 + x2',
            'cluster1_col': 'country'
        }
    )

Contents:

* Specification details (formula, features, target)
* Clustering configuration
* Feature engineering transformations
* Estimation settings
* Detailed coefficient information with confidence intervals

README (``README.md``)
~~~~~~~~~~~~~~~~~~~~~~~

Quick overview in Markdown:

.. code-block:: python

    model.results_.save_readme(
        output_dir,
        spec_config={'description': 'My analysis'},
        timestamp='20240115_143022'
    )

Includes:

* Analysis metadata (model type, N, R², SE type)
* File descriptions
* Specification (formula)
* Top 5 coefficients by magnitude
* Cluster information

Configuration Snapshot (``config_snapshot.json``)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Records the exact configuration used:

.. code-block:: python

    from gnt.analysis.streamreg.output import ConfigFormatter
    
    ConfigFormatter.save(
        spec_config={'formula': 'y ~ x'},
        full_config={'analyses': {...}},
        run_dir=output_dir,
        timestamp='20240115_143022'
    )

Ensures reproducibility by recording all settings.

Using the Save Interface
-------------------------

Save All Formats
~~~~~~~~~~~~~~~~

.. code-block:: python

    # Default: saves all formats
    run_dir = model.results_.save(
        output_dir="output/my_analysis",
        spec_name="baseline"
    )
    
    print(f"Results saved to: {run_dir}")

Save with Configuration Context
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    spec_config = {
        'description': 'Climate-growth baseline specification',
        'formula': 'gdp_growth ~ temperature + rainfall',
        'data_source': 'data/climate_growth.parquet',
        'cluster1_col': 'country',
        'cluster2_col': 'year'
    }
    
    full_config = {
        'analyses': {
            'online_rls': {
                'defaults': {'alpha': 0.001}
            }
        }
    }
    
    model.results_.save(
        output_dir="output",
        spec_name="baseline",
        spec_config=spec_config,
        full_config=full_config
    )

Save Selected Formats
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Only JSON and CSV for programmatic analysis
    model.results_.save(
        output_dir="output",
        formats=['json', 'csv']
    )
    
    # Only publication outputs
    model.results_.save(
        output_dir="output",
        formats=['latex', 'summary']
    )

Individual Format Methods
~~~~~~~~~~~~~~~~~~~~~~~~~

For fine-grained control:

.. code-block:: python

    from pathlib import Path
    
    output_dir = Path("output/custom")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save individual formats
    model.results_.save_json(output_dir)
    model.results_.save_csv(output_dir)
    model.results_.save_latex(output_dir)
    
    # With configuration
    spec_config = {'description': 'My analysis'}
    model.results_.save_summary(output_dir, spec_config)
    model.results_.save_diagnostics(output_dir, spec_config)
    model.results_.save_readme(output_dir, spec_config, timestamp='20240115')

Formatter Classes
-----------------

The system uses specialized formatter classes:

.. code-block:: python

    from gnt.analysis.streamreg.output import (
        JSONFormatter,
        CSVFormatter,
        SummaryFormatter,
        LaTeXFormatter,
        DiagnosticsFormatter,
        READMEFormatter,
        ConfigFormatter
    )

Each formatter has a static ``format_and_save()`` method:

.. code-block:: python

    # Direct formatter usage
    JSONFormatter.format_and_save(model.results_, output_dir)
    CSVFormatter.format_and_save(model.results_, output_dir)
    SummaryFormatter.format_and_save(model.results_, output_dir, spec_config)

Custom Formatters
-----------------

You can create custom formatters by subclassing ``BaseFormatter``:

.. code-block:: python

    from gnt.analysis.streamreg.output import BaseFormatter
    import json
    
    class CustomJSONFormatter(BaseFormatter):
        @staticmethod
        def format_and_save(results, run_dir):
            output_file = run_dir / "custom_results.json"
            
            # Custom data structure
            custom_data = {
                'model': results.model_type,
                'significant_coefficients': [
                    {
                        'name': name,
                        'coef': float(results.coefficients[i]),
                        'pval': float(results.p_values[i])
                    }
                    for i, name in enumerate(results.feature_names)
                    if results.p_values[i] < 0.05
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(custom_data, f, indent=2)
    
    # Use custom formatter
    CustomJSONFormatter.format_and_save(model.results_, output_dir)

Best Practices
--------------

1. **Always use spec_name**: Makes outputs easy to identify

   .. code-block:: python
   
       model.results_.save("output", spec_name="baseline_model")

2. **Include configuration**: Ensures reproducibility

   .. code-block:: python
   
       model.results_.save("output", spec_config=config, full_config=full_config)

3. **Version your outputs**: Timestamping is automatic

4. **Save frequently**: Don't lose results from long-running jobs

   .. code-block:: python
   
       model.fit(data)
       model.results_.save("output/checkpoints", spec_name=f"iter_{i}")

5. **Use selective formats**: For intermediate results

   .. code-block:: python
   
       # Quick checkpoint
       model.results_.save("output/temp", formats=['json'])

Integration Examples
--------------------

Loading Results for Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import json
    from pathlib import Path
    
    # Load JSON results
    results_file = Path("output/baseline_20240115_143022/results.json")
    with open(results_file) as f:
        results_data = json.load(f)
    
    # Extract specific information
    r_squared = results_data['r_squared']
    temp_coef = results_data['coefficients']['temperature']['estimate']
    
    print(f"R²: {r_squared:.4f}")
    print(f"Temperature coefficient: {temp_coef:.4f}")

Comparing Multiple Specifications
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    import pandas as pd
    from pathlib import Path
    
    # Load coefficient tables from multiple specs
    specs = ['baseline', 'with_controls', 'robustness']
    
    comparison = []
    for spec in specs:
        spec_dir = next(Path("output").glob(f"{spec}_*"))
        csv_file = spec_dir / "coefficients.csv"
        df = pd.read_csv(csv_file)
        df['specification'] = spec
        comparison.append(df)
    
    comparison_df = pd.concat(comparison)
    
    # Pivot for side-by-side comparison
    pivot = comparison_df.pivot(
        index='feature',
        columns='specification',
        values='coefficient'
    )
    print(pivot)

Creating Publication Tables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Generate LaTeX tables for multiple models
    models = {
        'Model 1': model1,
        'Model 2': model2,
        'Model 3': model3
    }
    
    for name, model in models.items():
        model.results_.save(
            f"output/{name.replace(' ', '_')}",
            formats=['latex']
        )
    
    # Manually combine in LaTeX document
    # or use LaTeX packages like booktabs for multi-model tables