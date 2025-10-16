import numpy as np
import pandas as pd
from pathlib import Path
from typing import Union, List, Optional, Dict, Any, Iterator, Tuple
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
    
    All parallel processing is handled internally - users don't need to know about partitioning.
    """
    
    def __init__(
        self,
        data: Union[str, Path, pd.DataFrame],
        chunk_size: int = 10000
    ):
        """Initialize data source."""
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
        
        logger.debug(f"Loaded DataFrame: {self.info.n_rows:,} rows, {self.info.n_cols} columns")
    
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
        
        logger.debug(f"Loaded parquet: {self.info.n_rows:,} rows, {self.info.n_cols} columns")
    
    def _setup_partitioned_parquet(self, path: Path):
        """Setup from partitioned parquet dataset."""
        partitions = self._discover_partitions(path)
        
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
        
        logger.info(f"Loaded partitioned dataset: {total_rows:,} rows, {len(partitions)} partitions")
    
    def _discover_partitions(self, path: Path) -> List[Path]:
        """Discover and validate partition files."""
        logger.debug(f"Discovering partitions in {path}")
        
        # Find all .parquet files recursively
        all_files = list(path.rglob("*.parquet"))
        
        if not all_files:
            raise ValueError(f"No parquet files found in {path}")
        
        # Filter valid partitions
        valid_partitions = []
        for file in all_files:
            try:
                # Check file size
                if file.stat().st_size < 1024:
                    logger.debug(f"Skipping small partition: {file.name}")
                    continue
                valid_partitions.append(file)
            except Exception as e:
                logger.warning(f"Cannot access partition {file}: {e}")
        
        if not valid_partitions:
            raise ValueError(f"No valid partitions found in {path}")
        
        logger.debug(f"Found {len(valid_partitions)} valid partitions")
        return sorted(valid_partitions)
    
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
            # Yield chunks from all partitions sequentially
            import pyarrow.parquet as pq
            
            for partition_file in self.info.partitions:
                try:
                    parquet_file = pq.ParquetFile(partition_file)
                    for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                        chunk = batch.to_pandas()
                        if columns:
                            chunk = chunk[columns]
                        yield chunk
                except Exception as e:
                    logger.warning(f"Failed to read partition {partition_file.name}: {e}")
                    continue
    
    def iter_chunks_parallel(self, columns: Optional[List[str]] = None, 
                            n_workers: Optional[int] = None) -> Iterator[Tuple[int, pd.DataFrame]]:
        """
        Iterate over chunks in parallel (across partitions if applicable).
        
        Parameters:
        -----------
        columns : list of str, optional
            Columns to load
        n_workers : int, optional
            Number of parallel workers (only relevant for partitioned data)
        
        Yields:
        -------
        tuple: (chunk_id, chunk_df)
        """
        if self.info.source_type == 'partitioned':
            # For partitioned data, yield chunks from all partitions
            chunk_id = 0
            for partition_idx, partition_file in enumerate(self.info.partitions):
                try:
                    parquet_file = pq.ParquetFile(partition_file)
                    for batch in parquet_file.iter_batches(batch_size=self.chunk_size):
                        chunk = batch.to_pandas()
                        if columns:
                            chunk = chunk[columns]
                        yield (chunk_id, chunk)
                        chunk_id += 1
                except Exception as e:
                    logger.warning(f"Failed to read partition {partition_file.name}: {e}")
                    continue
        else:
            # For non-partitioned data, use iter_chunks
            chunk_id = 0
            for chunk in self.iter_chunks(columns=columns):
                yield (chunk_id, chunk)
                chunk_id += 1
    
    def estimate_n_chunks(self) -> int:
        """Estimate total number of chunks."""
        return max(1, self.info.n_rows // self.chunk_size)
    
    def supports_parallel(self) -> bool:
        """Check if data source supports efficient parallel processing."""
        return self.info.source_type == 'partitioned'
    
    def validate_columns(self, required_cols: List[str]) -> None:
        """Validate that required columns exist."""
        missing = [col for col in required_cols if col not in self.info.columns]
        if missing:
            raise ValueError(f"Missing required columns: {missing}")
    
    def get_numeric_columns(self, exclude: Optional[List[str]] = None) -> List[str]:
        """Get list of numeric columns, optionally excluding some."""
        exclude = exclude or []
        return [col for col in self.info.numeric_columns if col not in exclude]
    
    def get_schema_sample(self) -> pd.DataFrame:
        """Get a small sample for schema validation."""
        if self.info.source_type == 'dataframe':
            return self._dataframe.head(100)
        elif self.info.source_type == 'parquet':
            return next(self._parquet_file.iter_batches(batch_size=100)).to_pandas()
        elif self.info.source_type == 'partitioned':
            first_file = pq.ParquetFile(self.info.partitions[0])
            return next(first_file.iter_batches(batch_size=100)).to_pandas()
