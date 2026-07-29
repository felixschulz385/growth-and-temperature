from abc import ABC, abstractmethod
import requests

class BaseDataSource(ABC):
    @abstractmethod
    def list_remote_files(self, entrypoint: dict) -> list[str]:
        ...

    @abstractmethod
    def local_path(self, remote_file: str) -> str:
        ...

    @abstractmethod
    def download(self, remote_file: str, local_path: str, session: requests.Session = None) -> None:
        ...

    @abstractmethod
    def gcs_upload_path(self, base_url: str, relative_path: str) -> str:
        ...

    def get_authenticated_session(self) -> requests.Session:
        """Optional: return an authenticated session if needed."""
        return None

    @abstractmethod
    def filename_to_entrypoint(self, relative_path: str) -> dict:
        ...
    
    def get_all_entrypoints(self):
        """
        Returns a list of dictionaries containing entrypoint information
        by recursively examining the directory structure.
        
        Returns:
            A list of dicts with entrypoint information
        """
        raise NotImplementedError("This method should be implemented by subclasses")