.. filepath: /scicore/home/meiera/schulz0022/projects/growth-and-temperature/docs/installation.rst

Installation
============

Requirements
------------

StreamReg requires Python 3.8 or later and the following dependencies:

* numpy >= 1.20.0
* pandas >= 1.3.0
* scipy >= 1.7.0
* pyarrow >= 5.0.0
* tqdm >= 4.60.0
* pyyaml >= 5.4.0

Optional dependencies for development:

* pytest >= 6.0.0
* sphinx >= 4.0.0
* sphinx_rtd_theme >= 1.0.0

Installation from Source
-------------------------

1. Clone the repository::

    git clone https://github.com/your-org/growth-and-temperature.git
    cd growth-and-temperature

2. Install in development mode::

    pip install -e .

This will install the package and all required dependencies.

Installation with pip (when published)
---------------------------------------

Once published to PyPI, you can install directly::

    pip install streamreg-gnt

Verifying Installation
----------------------

To verify the installation, run:

.. code-block:: python

    import gnt.analysis.streamreg as sr
    from gnt.analysis.streamreg.api import OLS, TwoSLS
    
    print("StreamReg version:", sr.__version__ if hasattr(sr, '__version__') else "0.1.0")
    print("OLS available:", OLS is not None)
    print("TwoSLS available:", TwoSLS is not None)

Setting up for HPC Environments
--------------------------------

If running on an HPC cluster (e.g., with SLURM), StreamReg will automatically 
detect available CPU resources from environment variables:

* ``SLURM_CPUS_PER_TASK``
* ``SLURM_NTASKS``
* ``SLURM_JOB_CPUS_PER_NODE``

Example SLURM submission script::

    #!/bin/bash
    #SBATCH --job-name=streamreg_analysis
    #SBATCH --ntasks=1
    #SBATCH --cpus-per-task=16
    #SBATCH --mem=64G
    #SBATCH --time=04:00:00
    
    module load python/3.9
    source venv/bin/activate
    
    python -m gnt.analysis.entrypoint online_rls -s baseline_spec

Troubleshooting
---------------

**Import errors**

If you encounter import errors, ensure the project root is in your Python path::

    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent))

**Memory issues**

For very large datasets, adjust the chunk size::

    model = OLS(formula="y ~ x", chunk_size=5000)  # Smaller chunks
    model.fit(data)

**Parallel processing issues**

If parallel processing fails, try reducing the number of workers::

    model = OLS(formula="y ~ x", n_workers=4)  # Explicit worker count
    model.fit(data)