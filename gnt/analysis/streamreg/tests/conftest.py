"""
Pytest configuration and fixtures for streamreg tests.
"""
import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import tempfile
import shutil


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    dirpath = tempfile.mkdtemp()
    yield Path(dirpath)
    shutil.rmtree(dirpath)


@pytest.fixture
def simple_data():
    """Generate simple regression data."""
    np.random.seed(42)
    n = 100
    df = pd.DataFrame({
        'y': np.random.randn(n),
        'x1': np.random.randn(n),
        'x2': np.random.randn(n)
    })
    df['y'] = 2 * df['x1'] + 3 * df['x2'] + 0.5 * np.random.randn(n)
    return df


@pytest.fixture
def clustered_data():
    """Generate data with cluster structure."""
    np.random.seed(42)
    n_clusters = 10
    n_per_cluster = 20
    n = n_clusters * n_per_cluster
    
    df = pd.DataFrame({
        'cluster': np.repeat(np.arange(n_clusters), n_per_cluster),
        'x1': np.random.randn(n),
        'x2': np.random.randn(n)
    })
    
    # Add cluster-specific effects
    cluster_effects = np.random.randn(n_clusters)
    df['cluster_effect'] = cluster_effects[df['cluster'].values]
    df['y'] = 2 * df['x1'] + 3 * df['x2'] + df['cluster_effect'] + 0.1 * np.random.randn(n)
    
    return df
