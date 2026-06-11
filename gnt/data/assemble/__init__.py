"""
Data assembly module for the GNT system.

This module provides functionality to merge multiple datasets using a tile-by-tile
approach based on configuration specifications, reading directly from zarr files
and outputting to parquet format with automatic scaling applied.
"""

from gnt.data.assemble.workflow import run_assembly, run_workflow_with_config

__all__ = ['run_assembly', 'run_workflow_with_config']
