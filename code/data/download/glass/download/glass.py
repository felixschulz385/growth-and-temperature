# download/glass.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

from .base import BaseDataSource

class GLASSDataSource(BaseDataSource):
    def __init__(self, base_url: str, file_extensions=None):
        self.DATA_SOURCE_NAME = "glass"
        self.base_url = base_url
        self.file_extensions = file_extensions or [".hdf"]

    def list_remote_files(self) -> list[tuple[str, str]]:
        file_urls = []

        res = requests.get(self.base_url)
        soup = BeautifulSoup(res.text, "html.parser")

        for link in soup.find_all("a"):
            href = link.get("href")
            if not href or not href.endswith("/"):
                continue
            year_url = urljoin(self.base_url, href)

            year_page = requests.get(year_url)
            year_soup = BeautifulSoup(year_page.text, "html.parser")

            for file_link in year_soup.find_all("a"):
                file_href = file_link.get("href")
                if any(file_href.endswith(ext) for ext in self.file_extensions):
                    file_url = urljoin(year_url, file_href)
                    file_name = file_url.replace(self.base_url, "")  # relative path
                    file_urls.append((file_name, file_url))

        return file_urls

    def local_path(self, relative_path: str) -> str:
        # Assuming a local directory structure that mirrors the remote one
        return os.path.join("data", relative_path)
    
    def download(self, file_url: str, output_path: str) -> None:
        r = requests.get(file_url, stream=True)
        r.raise_for_status()

        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

    def gcs_upload_path(self, base_url: str, relative_path: str) -> str:
        parsed = urlparse(base_url)
        parts = parsed.path.strip("/").split("/")

        datatype = "/".join(parts[1:]) if len(parts) > 2 else "unknown"

        filename = os.path.basename(relative_path)
        return f"{self.DATA_SOURCE_NAME}/{datatype}/{filename}"

