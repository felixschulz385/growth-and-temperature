# download/base.py
from abc import ABC, abstractmethod

class BaseDataSource(ABC):
    @abstractmethod
    def list_remote_files(self) -> list[str]:
        """Return a list of remote files available for download."""

    @abstractmethod
    def local_path(self, remote_file: str) -> str:
        """Return the local path where a remote file should be stored."""

    @abstractmethod
    def download(self, remote_file: str, local_path: str) -> None:
        """Download the remote file to the given local path."""

    @abstractmethod
    def gcs_upload_path(self, relative_path: str, base_url: str, gcs_prefix: str) -> str:
        """Return the path in GCS where the file should be uploaded."""
