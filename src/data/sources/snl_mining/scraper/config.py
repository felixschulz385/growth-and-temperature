"""
Constants and configuration for the S&P Global SNL Mining scraper.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Filesystem
# ---------------------------------------------------------------------------
_THIS_DIR = Path(__file__).parent
# Repository root: src/data/sources/snl_mining/scraper -> 5 levels up
_REPO_ROOT = _THIS_DIR.parents[4]

# This tool is standalone (not pipeline-wired -- see module docstring in
# src/data/sources/snl_mining/source.py on why FETCH is declared absent), so
# its output lives under the gitignored /data convention rather than through
# the pipeline's own layout module. It shares data/raw/snl_mining with the
# pipeline source's own
# manual-export DuckDB on purpose -- both are "parsed xlsx", just from
# different intake paths, and DEFAULT_DB_PATH below is the single merged
# database both sides read/write.
DATA_DIR: Path = _REPO_ROOT / "data" / "raw" / "snl_mining"

# Merged database: this scraper's own mines/mine_subsection_*/detail_* tables
# plus the manual-import notebook's properties/property_texts/property_llm_years/
# etc. tables, all in one file.
DEFAULT_DB_PATH: Path = DATA_DIR / "database.duckdb"

# Export download location for screener XLSX files
EXPORT_DIR: Path = DATA_DIR / "scraping"

# Ephemeral working state (Selenium user-data dirs) -- not durable output, so
# it lives outside data/ entirely, under the repo's gitignored scratch
# convention (see /scratch_nobackup in .gitignore) rather than being mistaken
# for real scraper data.
SCRATCH_DIR: Path = _REPO_ROOT / "scratch_nobackup" / "snl_mining"

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
SCREENER_KEY: str = "spglobal_snl_243327"

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
DOWNLOAD_WAIT_SECONDS = 30
MAP_LOAD_WAIT_SECONDS = 30
PERIODIC_BROWSER_RESTART_MINE_INTERVAL = 250

# ---------------------------------------------------------------------------
# Browser viewport
# ---------------------------------------------------------------------------
DEFAULT_WINDOW_WIDTH = 1920
DEFAULT_WINDOW_HEIGHT = 1080
