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
                
    def get_authenticated_session(self) -> requests.Session:
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        import time

        # Use Selenium to log in and get cookies
        username = os.environ.get("EOG_USERNAME")
        password = os.environ.get("EOG_PASSWORD")

        if not username or not password:
            raise ValueError("EOG_USERNAME and EOG_PASSWORD must be set in environment variables.")

        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")

        # This disables downloads
        prefs = {
            "download.prompt_for_download": False,
            "download.default_directory": "/dev/null",  # not writable in most cases
            "download_restrictions": 3  # 0 = allow all, 3 = block all
        }
        chrome_options.add_experimental_option("prefs", prefs)
        driver = webdriver.Chrome(options=chrome_options)
        
        login_prompting_url = "https://eogdata.mines.edu/wwwdata/dmsp/v4composites_rearrange/F10_1992/F101992.v4b.global.avg_vis.tif"
        driver.get(login_prompting_url)

        # Fill login form (adjust selectors as needed)
        driver.find_element("id", "username").send_keys(username)
        driver.find_element("id", "password").send_keys(password)
        driver.find_element("name", "login").click()

        # Wait for login & redirect
        time.sleep(5)

        # Download using cookies from the session
        cookies = driver.get_cookies()
        driver.quit()

        session = requests.Session()
        for c in cookies:
            session.cookies.set(c['name'], c['value'])

        return session

    def download(self, file_url: str, output_path: str, session: requests.Session = None) -> None:
        if session is None:
            raise ValueError("Authenticated session must be provided for download.")

        r = session.get(file_url, stream=True)
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

