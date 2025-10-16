import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Dict, Any
from dataclasses import dataclass
import logging
import pyarrow.parquet as pq

logger = logging.getLogger(__name__)


@dataclass
class DatasetInfo:
    """Metadata about a dataset."""
    n_rows: int
    n_cols: int
    columns: List[str]
    numeric_columns: List[str]
    source_type: str  # 'parquet', 'dataframe', 'partitioned'
    source_path: Optional[Path] = None
    partitions: Optional[List[Path]] = None


class StreamData:
    """
    Unified data interface for streaming regression.
    
    Supports:
    - Pandas DataFrame (in-memory)
    - Single parquet file
    - Partitioned parquet dataset
    """
    
    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        chunk_size: int = 10000
    ):
        """
        Initialize data source.
        
        Parameters:
        -----------
        data : str, Path, or DataFrame
            Data source - can be path to parquet or DataFrame
        chunk_size : int
            Size of chunks for streaming
        """
        self.chunk_size = chunk_size
        self._setup_data_source(data)
    
    def _setup_data_source(self, data: Union[str, Path, pd.DataFrame]):
        """Setup data source and extract metadata."""
        if isinstance(data, pd.DataFrame):
            self._setup_dataframe(data)
        else:
            path = Path(data)
            if not path.exists():
                raise FileNotFoundError(f"Data source not found: {path}")
            
            if path.is_dir():
                self._setup_partitioned_parquet(path)
            elif path.suffix == '.parquet':
                self._setup_single_parquet(path)
            else:
                raise ValueError(f"Unsupported data source: {path}")
    
    def _setup_dataframe(self, df: pd.DataFrame):
        """Setup from DataFrame."""
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.info = DatasetInfo(
            n_rows=len(df),
            n_cols=len(df.columns),
            columns=df.columns.tolist(),
            numeric_columns=numeric_cols,
            source_type='dataframe'
        )
        self._dataframe = df
        self._parquet_file = None
        
        logger.info(f"Loaded DataFrame: {self.info.n_rows:,} rows, {self.info.n_cols} columns")
    
    def _setup_single_parquet(self, path: Path):
        """Setup from single parquet file."""
        self._parquet_file = pq.ParquetFile(path)
        schema = self._parquet_file.metadata.schema
        
        columns = [field.name for field in schema]
        # Read small sample to determine numeric columns
        sample_df = next(self._parquet_file.iter_batches(batch_size=100)).to_pandas()
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.info = DatasetInfo(
            n_rows=self._parquet_file.metadata.num_rows,
            n_cols=len(columns),
            columns=columns,
            numeric_columns=numeric_cols,
            source_type='parquet',
            source_path=path
        )
        self._dataframe = None
        
        logger.info(f"Loaded parquet: {self.info.n_rows:,} rows, {self.info.n_cols} columns")
    
    def _setup_partitioned_parquet(self, path: Path):
        """Setup from partitioned parquet dataset."""
        from gnt.analysis.models.online_RLS import discover_partitions
        
        partitions = discover_partitions(path)
        
        # Read first partition for schema
        first_parquet = pq.ParquetFile(partitions[0])
        schema = first_parquet.metadata.schema
        columns = [field.name for field in schema]
        
        # Estimate total rows
        total_rows = sum(
            pq.ParquetFile(p).metadata.num_rows 
            for p in partitions
        )
        
        # Read small sample to determine numeric columns
        sample_df = next(first_parquet.iter_batches(batch_size=100)).to_pandas()
        numeric_cols = sample_df.select_dtypes(include=[np.number]).columns.tolist()
        
        self.info = DatasetInfo(
            n_rows=total_rows,
            n_cols=len(columns),
            columns=columns,
            numeric_columns=numeric_cols,
            source_type='partitioned',
            source_path=path,
            partitions=partitions
        )
        self._dataframe = None
        self._parquet_file = None
        
        logger.info(f"Loaded partitioned dataset: {self.info.n_rows:,} rows (estimated), "
                   f"{len(partitions)} partitions")
    
    def iter_chunks(self, columns: Optional[List[str]] = None):
        """
        Iterate over data in chunks.
        
        Parameters:
        -----------
        columns : list of str, optional
            Columns to load. If None, loads all columns.
        
        Yields:
        -------
        DataFrame chunks
        """
        if self.info.source_type == 'dataframe':
            # Yield chunks from DataFrame
            for i in range(0, self.info.n_rows, self.chunk_size):
                chunk = self._dataframe.iloc[i:i+self.chunk_size]
                if columns:
                    chunk = chunk[columns]
                yield chunk
        
        elif self.info.source_type == 'parquet':
            # Yield chunks from single parquet
            for batch in self._parquet_file.iter_batches(batch_size=self.chunk_size):
                chunk = batch.to_pandas()
                if columns:
                    chunk = chunk[columns]
                yield chunk
        
        elif self.info.source_type == 'partitioned':
            # This is handled differently for parallel processing
            raise NotImplementedError(
                "Partitioned datasets should use parallel processing. "
                "Access partitions directly via .info.partitions"
            )
    
    def validate_columns(self, required_cols: List[str]) -> None:
        """Validate that required columns exist."""
        missing = [col for col in required_cols if col not in self.info.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def get_numeric_columns(self, exclude: Optional[List[str]] = None) -> List[str]:
        """Get list of numeric columns, optionally excluding some."""
        exclude = exclude or []
        return [col for col in self.info.numeric_columns if col not in exclude]
