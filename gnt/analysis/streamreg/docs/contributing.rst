.. filepath: /scicore/home/meiera/schulz0022/projects/growth-and-temperature/docs/contributing.rst

Contributing
============

We welcome contributions to StreamReg! This guide will help you get started.

Development Setup
-----------------

1. Clone the repository::

    git clone https://github.com/your-org/growth-and-temperature.git
    cd growth-and-temperature

2. Create a development environment::

    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate

3. Install in development mode with test dependencies::

    pip install -e ".[dev]"

4. Run tests::

    pytest tests/

Code Style
----------

We follow PEP 8 style guidelines with some modifications:

* Line length: 100 characters
* Use type hints for function signatures
* Use Google-style docstrings

Example function:

.. code-block:: python

    def compute_statistic(
        data: np.ndarray,
        method: str = 'robust'
    ) -> float:
        """
        Compute a statistical measure from data.
        
        Args:
            data: Input data array
            method: Computation method ('robust' or 'classical')
        
        Returns:
            Computed statistic value
        
        Raises:
            ValueError: If method is not recognized
        """
        if method not in ['robust', 'classical']:
            raise ValueError(f"Unknown method: {method}")
        
        # Implementation
        return result

Testing
-------

Writing Tests
~~~~~~~~~~~~~

* Place tests in ``tests/`` directory
* Name test files ``test_*.py``
* Use pytest fixtures for common setup
* Aim for >80% code coverage

Example test:

.. code-block:: python

    import pytest
    import numpy as np
    from gnt.analysis.streamreg.api import OLS
    
    def test_ols_simple():
        """Test basic OLS functionality."""
        # Setup
        X = np.random.randn(100, 2)
        y = X @ np.array([1.5, -0.5]) + np.random.randn(100) * 0.1
        
        # Fit model
        model = OLS(formula="y ~ x1 + x2")
        model.fit(pd.DataFrame({'y': y, 'x1': X[:, 0], 'x2': X[:, 1]}))
        
        # Assert
        assert model.n_obs_ == 100
        assert model.r_squared_ > 0.9

Running Tests
~~~~~~~~~~~~~

Run all tests::

    pytest

Run with coverage::

    pytest --cov=gnt.analysis.streamreg --cov-report=html

Run specific test file::

    pytest tests/test_ols.py

Documentation
-------------

Building Documentation
~~~~~~~~~~~~~~~~~~~~~~

1. Install documentation dependencies::

    pip install sphinx sphinx_rtd_theme

2. Build HTML documentation::

    cd docs
    make html

3. View documentation::

    open _build/html/index.html

Writing Documentation
~~~~~~~~~~~~~~~~~~~~~

* Use reStructuredText (RST) format
* Follow existing documentation structure
* Include code examples for new features
* Update API reference for new classes/functions

Docstring Format
~~~~~~~~~~~~~~~~

Use Google-style docstrings:

.. code-block:: python

    def my_function(arg1: int, arg2: str = 'default') -> bool:
        """
        Short description of the function.
        
        Longer description with more details about what the function does,
        how it works, and any important considerations.
        
        Args:
            arg1: Description of arg1
            arg2: Description of arg2 with default value
        
        Returns:
            Description of return value
        
        Raises:
            ValueError: When arg1 is negative
            TypeError: When arg2 is not a string
        
        Example:
            >>> result = my_function(42, 'test')
            >>> print(result)
            True
        """
        # Implementation
        pass

Pull Request Process
--------------------

1. **Create a branch**::

    git checkout -b feature/my-new-feature

2. **Make your changes**

   * Write code
   * Add tests
   * Update documentation

3. **Commit your changes**::

    git add .
    git commit -m "Add feature: description"

4. **Push to GitHub**::

    git push origin feature/my-new-feature

5. **Open a Pull Request**

   * Describe your changes
   * Reference any related issues
   * Ensure CI passes

Code Review Guidelines
----------------------

When reviewing code:

* Check for correctness and edge cases
* Verify test coverage
* Ensure documentation is updated
* Check code style consistency
* Consider performance implications
* Look for potential bugs or security issues

Reporting Issues
----------------

When reporting bugs, please include:

1. **System information**

   * OS and version
   * Python version
   * Package versions (``pip list``)

2. **Steps to reproduce**

   * Minimal code example
   * Expected vs actual behavior
   * Error messages and stack traces

3. **Additional context**

   * Data characteristics (if relevant)
   * Any workarounds you've tried

Feature Requests
----------------

When proposing new features:

1. **Describe the use case**

   * What problem does it solve?
   * Who would benefit?

2. **Propose an API**

   * How would users interact with the feature?
   * Example code

3. **Consider alternatives**

   * Are there existing solutions?
   * Why is this approach better?

Areas for Contribution
----------------------

Good first issues:

* Documentation improvements
* Additional examples
* Test coverage improvements
* Bug fixes

More advanced contributions:

* New estimators (panel models, GMM)
* Performance optimizations
* Additional output formats
* Extended formula syntax

Getting Help
------------

* Open an issue for questions
* Join our discussions
* Check existing documentation

Thank you for contributing to StreamReg!