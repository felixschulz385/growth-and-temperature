import abc
from typing import Dict, Any, List, Tuple
import logging
import os
import xarray as xr

logger = logging.getLogger(__name__)

class AbstractPreprocessor(abc.ABC):
    """
    Abstract base class for geodata preprocessors.
    Enforces the interface for common preprocessing methods.
    """

    def __init__(self, **kwargs):
        self.config = kwargs

    @abc.abstractmethod
    def get_preprocessing_targets(self, stage: str, year_range: Tuple[int, int] = None) -> List[Dict[str, Any]]:
        pass

    @abc.abstractmethod
    def get_hpc_output_path(self, stage: str) -> str:
        pass

    @abc.abstractmethod
    def process_target(self, target: Dict[str, Any]) -> bool:
        pass

    @classmethod
    @abc.abstractmethod
    def from_config(cls, config: Dict[str, Any]):
        pass

    def get_transfer_units(self, stage: str) -> List[Dict[str, Any]]:
        """Local paths produced by *stage* that should be pushed to the HPC target.

        Optional hook (docs/design/08-hpc-transfer.md §2) — only sources whose
        stage output is produced somewhere other than the HPC target need to
        override this. Default: derive one transfer unit from
        ``get_hpc_output_path(stage)``, mapping the local stage root onto the
        same relative path under the remote target (local ``hpc_target`` and
        the remote SSH target's base path are the same conceptual data root,
        just local vs. remote — see docs/design/08-hpc-transfer.md §1).
        Sources with a finer per-unit output layout (e.g. per-tile-year)
        should override this for finer-grained transfer resumability.
        """
        local_path = self.get_hpc_output_path(stage)
        hpc_root = self._strip_remote_prefix(self.config.get("hpc_target"))
        if hpc_root:
            remote_path = os.path.relpath(local_path, hpc_root)
        else:
            remote_path = os.path.basename(os.path.normpath(local_path))
        return [{
            "unit_id": stage,
            "local_path": local_path,
            "remote_path": remote_path,
        }]

    def _strip_remote_prefix(self, path):
        """Remove scp/ssh prefix like user@host: from paths. Override in subclasses if needed."""
        if isinstance(path, str):
            import re
            return re.sub(r"^[^@]+@[^:]+:", "", path)
        return path