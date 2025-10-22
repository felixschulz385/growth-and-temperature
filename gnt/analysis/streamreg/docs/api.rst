.. filepath: /scicore/home/meiera/schulz0022/projects/growth-and-temperature/docs/api.rst

API Reference
=============

This page documents the public API for StreamReg.

High-Level Estimators
---------------------

OLS
~~~

.. autoclass:: gnt.analysis.streamreg.api.OLS
   :members:
   :undoc-members:
   :show-inheritance:

TwoSLS
~~~~~~

.. autoclass:: gnt.analysis.streamreg.api.TwoSLS
   :members:
   :undoc-members:
   :show-inheritance:

Results
-------

RegressionResults
~~~~~~~~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.results.RegressionResults
   :members:
   :undoc-members:
   :show-inheritance:

Data Interface
--------------

StreamData
~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.data.StreamData
   :members:
   :undoc-members:
   :show-inheritance:

DatasetInfo
~~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.data.DatasetInfo
   :members:
   :undoc-members:
   :show-inheritance:

Formula Parsing
---------------

FormulaParser
~~~~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.formula.FormulaParser
   :members:
   :undoc-members:
   :show-inheritance:

Feature Engineering
-------------------

FeatureTransformer
~~~~~~~~~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.transforms.FeatureTransformer
   :members:
   :undoc-members:
   :show-inheritance:

Output Formatters
-----------------

BaseFormatter
~~~~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.output.BaseFormatter
   :members:
   :undoc-members:
   :show-inheritance:

JSONFormatter
~~~~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.output.JSONFormatter
   :members:
   :undoc-members:
   :show-inheritance:

CSVFormatter
~~~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.output.CSVFormatter
   :members:
   :undoc-members:
   :show-inheritance:

SummaryFormatter
~~~~~~~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.output.SummaryFormatter
   :members:
   :undoc-members:
   :show-inheritance:

LaTeXFormatter
~~~~~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.output.LaTeXFormatter
   :members:
   :undoc-members:
   :show-inheritance:

DiagnosticsFormatter
~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.output.DiagnosticsFormatter
   :members:
   :undoc-members:
   :show-inheritance:

READMEFormatter
~~~~~~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.output.READMEFormatter
   :members:
   :undoc-members:
   :show-inheritance:

ConfigFormatter
~~~~~~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.output.ConfigFormatter
   :members:
   :undoc-members:
   :show-inheritance:

Low-Level Estimators
--------------------

OnlineRLS
~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.estimators.ols.OnlineRLS
   :members:
   :undoc-members:
   :show-inheritance:

Online2SLS
~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.estimators.iv.Online2SLS
   :members:
   :undoc-members:
   :show-inheritance:

Orchestrators
-------------

ParallelOLSOrchestrator
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.estimators.ols.ParallelOLSOrchestrator
   :members:
   :undoc-members:
   :show-inheritance:

TwoSLSOrchestrator
~~~~~~~~~~~~~~~~~~

.. autoclass:: gnt.analysis.streamreg.estimators.iv.TwoSLSOrchestrator
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

Convenience Functions
~~~~~~~~~~~~~~~~~~~~~

.. autofunction:: gnt.analysis.streamreg.api.ols

.. autofunction:: gnt.analysis.streamreg.api.twosls