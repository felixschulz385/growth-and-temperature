# download/harvard.py
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import os

from .base import BaseDataSource

class HarvardDataSource(BaseDataSource):
    def __init__(self, base_url: str, file_extensions: list[str] = None):
        self.DATA_SOURCE_NAME = "harvard"
        self.base_url = base_url
        self.file_extensions = file_extensions

    def list_remote_files(self) -> list[tuple[str, str]]:
        api_url = f"https://dataverse.harvard.edu/api/datasets/:persistentId?persistentId=doi:{self.base_url}"
        response = requests.get(api_url)
        response.raise_for_status()
        dataset_info = response.json()
        
        files = dataset_info['data']['latestVersion']['files']
        result = []
        
        for file in files:
            label = file["label"]
            if any(label.endswith(ext) for ext in self.file_extensions):
                relative_path = f"{self.base_url}/{file['dataFile']['originalFileName']}"
                file_id = file['dataFile']['id']
                full_url = f"https://dataverse.harvard.edu/api/access/datafile/{file_id}"
                result.append((relative_path, full_url))
        
        return result

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

    # def download_legacy(self, file_url: str, output_path: str, session: requests.Session = None) -> None:
    #     from selenium import webdriver
    #     from selenium.webdriver.chrome.options import Options
    #     from selenium.webdriver.common.by import By
    #     import time

    #     chrome_options = Options()
    #     chrome_options.add_argument("--headless")
    #     chrome_options.add_argument("--no-sandbox")
    #     chrome_options.add_argument("--disable-dev-shm-usage")

    #     # This disables downloads
    #     prefs = {
    #         "download.prompt_for_download": False,
    #         "download.default_directory": output_path,
    #     }
    #     chrome_options.add_experimental_option("prefs", prefs)
    #     driver = webdriver.Chrome(options=chrome_options)

    #     download_url = file_url
    #     driver.get(download_url)

    #     time.sleep(1)

    #     driver.find_element(By.CLASS_NAME, "btn-access-file").click()

    #     time.sleep(1)

    #     driver.find_element(By.CLASS_NAME, "btn-download").click()

    #     time.sleep(1)

    #     driver.quit()