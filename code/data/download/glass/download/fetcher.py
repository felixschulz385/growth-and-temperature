import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin
import os

def get_all_files(base_url, file_extensions=None):
    """Recursively get all file URLs from base_url that match file_extensions."""
    file_extensions = file_extensions or [".hdf"]
    file_urls = []

    # Get list of year directories
    res = requests.get(base_url)
    soup = BeautifulSoup(res.text, "html.parser")

    for link in soup.find_all("a"):
        href = link.get("href")
        if not href or not href.endswith("/"):
            continue
        year_url = urljoin(base_url, href)

        # Get files in the year directory
        year_page = requests.get(year_url)
        year_soup = BeautifulSoup(year_page.text, "html.parser")

        for file_link in year_soup.find_all("a"):
            file_href = file_link.get("href")
            if any(file_href.endswith(ext) for ext in file_extensions):
                file_url = urljoin(year_url, file_href)
                file_name = file_url.replace(base_url, "")  # Use relative path
                file_urls.append((file_name, file_url))

    return file_urls

def download_file(file_url, output_dir):
    filename = os.path.basename(file_url)
    local_path = os.path.join(output_dir, filename)

    r = requests.get(file_url, stream=True)
    r.raise_for_status()

    with open(local_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)

    return local_path