import pytest
from unittest.mock import patch, Mock
from download.eog import EOGDataSource

sample_html = """
<html>
  <body>
    <table>
      <tr><td class="indexcolname"><a href="subdir/">subdir/</a></td></tr>
      <tr><td class="indexcolname"><a href="file1.tif">file1.tif</a></td></tr>
      <tr><td class="indexcolname"><a href="file2.txt">file2.txt</a></td></tr>
    </table>
  </body>
</html>
"""

# Top-level HTML contains a subdirectory and a file
html_root = """
<html>
  <body>
    <table>
      <tr><td class="indexcolname"><a href="subdir/">subdir/</a></td></tr>
      <tr><td class="indexcolname"><a href="file1.tif">file1.tif</a></td></tr>
    </table>
  </body>
</html>
"""

# HTML for subdir/ contains one nested file
html_subdir = """
<html>
  <body>
    <table>
      <tr><td class="indexcolname"><a href="nested.tif">nested.tif</a></td></tr>
    </table>
  </body>
</html>
"""

@pytest.fixture
def mock_requests_get():
    with patch("download.glass.requests.get") as mock_get:
        yield mock_get

def test_list_remote_files_recursively_collects_files(mock_requests_get):
    base_url = "https://example.com/data/"
    allowed_extensions = [".tif"]
    data_source = EOGDataSource(base_url, allowed_extensions)

    def mock_get(url):
        mock_response = Mock()
        if url == base_url:
            mock_response.text = html_root
        elif url == url + "subdir/":
            mock_response.text = html_subdir
        elif url == "https://example.com/data/subdir/":
            mock_response.text = html_subdir
        else:
            raise ValueError(f"Unexpected URL: {url}")
        return mock_response

    mock_requests_get.side_effect = mock_get

    result = data_source.list_remote_files()

    expected = [
        ("file1.tif", "https://example.com/data/file1.tif"),
        ("subdir/nested.tif", "https://example.com/data/subdir/nested.tif")
    ]
    assert sorted(result) == sorted(expected)
