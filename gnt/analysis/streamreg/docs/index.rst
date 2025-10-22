.. filepath: /scicore/home/meiera/schulz0022/projects/growth-and-temperature/docs/index.rst

StreamReg: Streaming Regression for Large Datasets
===================================================

**StreamReg** is a Python library for efficient online regression estimation on datasets that don't fit in memory. 
It provides OLS and 2SLS (Two-Stage Least Squares) estimators with cluster-robust standard errors, 
optimized for large partitioned parquet datasets.

.. image:: https://readthedocs.org/projects/streamreg-gnt/badge/?version=latest
    :target: https://streamreg-gnt.readthedocs.io/en/latest/?badge=latest
    :alt: Documentation Status

Features
--------

* **Online Learning**: Process data in chunks without loading entire datasets into memory
* **Parallel Processing**: Efficient multi-core processing of partitioned datasets
* **Cluster-Robust SE**: Support for one-way and two-way clustered standard errors
* **Formula Interface**: R-style formulas with support for transformations and interactions
* **Flexible Output**: Save results in multiple formats (JSON, CSV, LaTeX, text reports)
* **2SLS Support**: Two-stage least squares for instrumental variable estimation
* **Feature Engineering**: Built-in support for polynomials, interactions, and custom transformations

Quick Start
-----------

Installation::

    pip install -e .

Basic OLS Example:

.. code-block:: python

    from gnt.analysis.streamreg.api import OLS
    
    # Initialize model with formula
    model = OLS(formula="gdp_growth ~ temperature + precipitation + I(temperature^2)")
    
    # Fit with cluster-robust standard errors
    model.fit("data/dataset.parquet", cluster=['country', 'year'])
    
    # View results
    print(model.summary())
    
    # Save results
    model.results_.save("output/my_analysis", spec_name="baseline")

2SLS Example:

.. code-block:: python

    from gnt.analysis.streamreg.api import TwoSLS
    
    # Initialize with instruments after |
    model = TwoSLS(
        formula="gdp_growth ~ temperature + rainfall | historical_temp + elevation",
        endogenous=['temperature']
    )
    
    # Fit model
    model.fit("data/dataset.parquet", cluster='country')
    
    # View first and second stage results
    print(model.summary(stage='all'))
    
    # Save comprehensive results
    model.results_.save("output/iv_analysis")

Contents
--------

.. toctree::
   :maxdepth: 2
   :caption: User Guide
   
   installation
   usage
   output
   
.. toctree::
   :maxdepth: 2
   :caption: API Reference
   
   api
   modules

.. toctree::
   :maxdepth: 1
   :caption: Additional Information
   
   changelog
   contributing

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`