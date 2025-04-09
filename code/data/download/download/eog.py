# download/glass.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

from .base import BaseDataSource

class EOGDataSource(BaseDataSource):
    def __init__(self, base_url: str, file_extensions: list[str] = None):
        self.DATA_SOURCE_NAME = "eog"
        self.base_url = base_url
        self.file_extensions = file_extensions

    def list_remote_files(self) -> list[tuple[str, str]]:
        
        def collect_files_from_url(url: str, relative_prefix: str = "") -> list[tuple[str, str]]:
            collected_files = []

            page = requests.get(url)
            soup = BeautifulSoup(page.text, "html.parser")

            for link_td in soup.find_all("td", {"class": "indexcolname"}):
                href = link_td.a.get("href")
                if not href:
                    continue
                # Skip parent directory links
                if urlparse(url).path.startswith(href) and href.count("/") == urlparse(url).path.count("/") - 1:
                    continue
                
                full_url = urljoin(url, href)

                if href.endswith("/"):  # It's a directory, recurse into it
                    new_prefix = os.path.join(relative_prefix, href)
                    collected_files.extend(collect_files_from_url(full_url, new_prefix))
                elif any(href.endswith(ext) for ext in self.file_extensions):  # It's a file
                    relative_path = os.path.join(relative_prefix, href)
                    collected_files.append((relative_path, full_url))

            return collected_files

        return collect_files_from_url(self.base_url)

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

