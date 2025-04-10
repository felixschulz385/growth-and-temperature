from abc import ABC, abstractmethod
import requests

class BaseDataSource(ABC):
    @abstractmethod
    def list_remote_files(self) -> list[str]:
        ...

    @abstractmethod
    def local_path(self, remote_file: str) -> str:
        ...

    @abstractmethod
    def download(self, remote_file: str, local_path: str, session: requests.Session = None) -> None:
        ...

    @abstractmethod
    def gcs_upload_path(self, relative_path: str, base_url: str, gcs_prefix: str) -> str:
        ...

    def get_authenticated_session(self) -> requests.Session:
        """Optional: return an authenticated session if needed."""
        return None
