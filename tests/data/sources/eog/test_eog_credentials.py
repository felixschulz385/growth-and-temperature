"""load_eog_credentials(): git-ignored orchestration/secrets/eog.credentials.json
takes priority over EOG_USERNAME/EOG_PASSWORD, which is the fallback when
the file is absent or incomplete."""

import json

from src.data.sources.eog.credentials import load_eog_credentials


def test_returns_env_vars_when_no_file(tmp_path, monkeypatch):
    monkeypatch.setenv("EOG_USERNAME", "env-user")
    monkeypatch.setenv("EOG_PASSWORD", "env-pass")
    username, password = load_eog_credentials(tmp_path / "does-not-exist.json")
    assert (username, password) == ("env-user", "env-pass")


def test_returns_none_none_when_nothing_set(tmp_path, monkeypatch):
    monkeypatch.delenv("EOG_USERNAME", raising=False)
    monkeypatch.delenv("EOG_PASSWORD", raising=False)
    assert load_eog_credentials(tmp_path / "does-not-exist.json") == (None, None)


def test_file_takes_priority_over_env_vars(tmp_path, monkeypatch):
    monkeypatch.setenv("EOG_USERNAME", "env-user")
    monkeypatch.setenv("EOG_PASSWORD", "env-pass")
    path = tmp_path / "eog.credentials.json"
    path.write_text(json.dumps({"username": "file-user", "password": "file-pass"}))
    assert load_eog_credentials(path) == ("file-user", "file-pass")


def test_incomplete_file_falls_back_to_env_vars(tmp_path, monkeypatch, caplog):
    import logging

    monkeypatch.setenv("EOG_USERNAME", "env-user")
    monkeypatch.setenv("EOG_PASSWORD", "env-pass")
    path = tmp_path / "eog.credentials.json"
    path.write_text(json.dumps({"username": "file-user"}))  # no password

    with caplog.at_level(logging.WARNING):
        result = load_eog_credentials(path)

    assert result == ("env-user", "env-pass")
    assert any("missing" in r.getMessage() for r in caplog.records)


def test_default_path_used_when_none_given(tmp_path, monkeypatch):
    from src.data.sources.eog import credentials as eog_credentials

    monkeypatch.setattr(eog_credentials, "DEFAULT_CREDENTIALS_PATH", tmp_path / "eog.credentials.json")
    monkeypatch.setenv("EOG_USERNAME", "env-user")
    monkeypatch.setenv("EOG_PASSWORD", "env-pass")
    assert load_eog_credentials() == ("env-user", "env-pass")
