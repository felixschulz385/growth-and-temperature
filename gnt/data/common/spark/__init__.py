"""
Spark utilities for the GNT data system.
"""

from .client import create_spark_session, SparkSessionContextManager

__all__ = ['create_spark_session', 'SparkSessionContextManager']
