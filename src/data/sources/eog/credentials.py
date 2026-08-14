"""EOG login credentials -- read from a git-ignored JSON file
(`orchestration/secrets/eog.credentials.json`, matching the S&P Global
scraper's identical `orchestration/secrets/spglobal.credentials.json`
convention, `src/data/sources/snl_mining/scraper/session/auth.py::
load_credentials()`) instead of requiring `EOG_USERNAME`/`EOG_PASSWORD` to
already be exported in the shell -- a machine just needs this one file to
exist once, rather than every shell that runs `data summary`/`data run`
needing the environment configured.

Falls back to `EOG_USERNAME`/`EOG_PASSWORD` environment variables when the
file is absent, so existing shell-based setups (and tests) keep working
unchanged.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CREDENTIALS_PATH = REPO_ROOT / "orchestration" / "secrets" / "eog.credentials.json"


def load_eog_credentials(path: "str | Path | None" = None) -> Tuple[Optional[str], Optional[str]]:
    """`(username, password)` -- from *path* (or `DEFAULT_CREDENTIALS_PATH`
    if *path* is `None`) if it exists and has both fields, else from
    `EOG_USERNAME`/`EOG_PASSWORD`. `(None, None)` if neither source has
    them."""
    credentials_path = Path(path) if path is not None else DEFAULT_CREDENTIALS_PATH
    if credentials_path.exists():
        with open(credentials_path, "r", encoding="utf-8") as fh:
            credentials = json.load(fh)
        username = credentials.get("username")
        password = credentials.get("password")
        if username and password:
            return username, password
        logger.warning(
            "EOG credentials file at %s is missing 'username'/'password' -- falling back to "
            "EOG_USERNAME/EOG_PASSWORD environment variables.",
            credentials_path,
        )
    return os.environ.get("EOG_USERNAME"), os.environ.get("EOG_PASSWORD")
