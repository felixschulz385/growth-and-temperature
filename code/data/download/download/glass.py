# download/glass.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os
import time

from .base import BaseDataSource

class GLASSDataSource(BaseDataSource):
    def __init__(self, base_url: str, file_extensions=None):
        self.DATA_SOURCE_NAME = "glass"
        self.base_url = base_url
        self.file_extensions = file_extensions or [".hdf"]

    def list_remote_files(self):
        def crawl(url: str, relative_path: str = ""):
            time.sleep(.5)
            res = requests.get(url)
            soup = BeautifulSoup(res.text, "html.parser")

            for link in soup.find_all("a"):
                href = link.get("href")
                if not href or href in ("../", "./"):
                    continue

                full_url = urljoin(url, href)
                new_relative_path = relative_path + href

                if href.endswith("/"):
                    yield from crawl(full_url, new_relative_path)
                else:
                    if any(href.endswith(ext) for ext in self.file_extensions):
                        yield (new_relative_path, full_url)

        return crawl(self.base_url)

    def local_path(self, relative_path: str) -> str:
        # Assuming a local directory structure that mirrors the remote one
        return os.path.join("data", relative_path)
    
    def download(self, file_url: str, output_path: str, session: requests.Session = None) -> None:
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

