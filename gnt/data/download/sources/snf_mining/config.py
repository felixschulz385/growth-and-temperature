"""
Constants and configuration for the S&P Global SNF Mining scraper.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent
# Repository root: gnt/data/download/sources/snf_mining -> 5 levels up
_REPO_ROOT = _THIS_DIR.parents[4]

# Raw data lives at  <repo_root>/data/snf_mining/raw  (mirrors misc.py convention)
DATA_DIR: Path = _REPO_ROOT / "data" / "snf_mining" / "raw"

# Relative path used for GCS-style addressing (same pattern as MiscDataSource)
DATA_SOURCE_PATH: str = "snf_mining/raw"

# Database stored inside the raw data directory so everything is co-located
DEFAULT_DB_PATH: Path = DATA_DIR / "snf_mining.duckdb"

# Export download location for screener XLSX files
EXPORT_DIR: Path = DATA_DIR / "exports"

# ---------------------------------------------------------------------------
# S&P Global / Capital IQ URLs
# ---------------------------------------------------------------------------
BASE_URL = "https://www.capitaliq.spglobal.com"
LOGIN_URL = f"{BASE_URL}/web/login?ignoreIDMContext=1"
SCREENER_URL = (
    f"{BASE_URL}/web/client?auth=inherit"
    "#office/screener?perspective=243327"
)
PROFILE_URL_TEMPLATE = (
    f"{BASE_URL}/web/client#metalsAndMiningProperty/profile?ID={{mine_id}}"
)

# Stable key that identifies the screener perspective stored in screener_state
SCREENER_KEY: str = "spglobal_snf_243327"

# ---------------------------------------------------------------------------
# Selenium CSS / XPath selectors
# ---------------------------------------------------------------------------
SEL = {
    # Login page
    "username_input": "//input[@autocomplete='username']",
    "submit_btn": "//input[@type='submit']",
    "password_input": "//input[@type='password']",

    # Screener
    "run_screen_btn": "//button[normalize-space(.)='Run Screen']",
    "pager_container": "div.ui-iggrid-pagedropdowncontainer",
    "next_page_btn": "div.ui-iggrid-nextpage[title='go to the next page']",
    "next_page_clickable": "div.ui-iggrid-nextpage",
    "data_row": "tr[data-id]",

    # Export controls
    "export_select": "select[name='snlInput154']",
    "export_button": "//button[contains(@class,'snl-widgets-input-button') and .//span[text()='Export']]",
    "download_modal": "div.modal-dialog.snl-views-office-download-report",
    "download_link": "a.downloadLink",

    # Profile / detail page (extend as blocks are added)
    "profile_icon": "button[data-testid='userflyout-icon'], button[aria-label='Icon button'] span[data-icon='user']",
    "sign_out_link": "//a[normalize-space(.)='Sign out']",

    # Optional chrome / UX elements
    "cookie_banner": "onetrust-button-group-parent",
    "cookie_accept": "onetrust-accept-btn-handler",
    "hamburger_menu": "button[aria-label='Open navigation']",
}

# ---------------------------------------------------------------------------
# Timing
# ---------------------------------------------------------------------------
DEFAULT_WAIT_SECONDS = 10
PAGE_TURN_WAIT_SECONDS = 10
SHORT_WAIT_SECONDS = 5
SMALL_SLEEP_SECONDS = 0.35
DOWNLOAD_WAIT_SECONDS = 90
MAP_LOAD_WAIT_SECONDS = 30
PERIODIC_BROWSER_RESTART_MINE_INTERVAL = 250

# ---------------------------------------------------------------------------
# Browser viewport
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_WIDTH = 1920
DEFAULT_WINDOW_HEIGHT = 1080
