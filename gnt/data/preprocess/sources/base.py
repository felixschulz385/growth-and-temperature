import abc
from typing import Dict, Any, List, Tuple


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