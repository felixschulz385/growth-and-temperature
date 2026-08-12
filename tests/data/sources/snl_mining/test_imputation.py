"""imputation.py: LLM year-imputation module (converted from
snl_mining_openai_enrichment.ipynb). `openai` is a real dependency now
tracked in environment.yml, but not every dev environment has re-run
`conda env create` since -- skip cleanly at collection time if it isn't
installed rather than failing the whole suite.
"""

import pytest

pytest.importorskip("openai")

import duckdb

from src.data.sources.snl_mining.imputation import load_fused_property_texts


def _make_db(tmp_path):
    path = str(tmp_path / "test.duckdb")
    con = duckdb.connect(path)
    con.execute(
        "CREATE TABLE property_texts (property_id VARCHAR, source_property_key VARCHAR, field_name VARCHAR, raw_text VARCHAR)"
    )
    con.execute("CREATE TABLE detail_work_history_events (mine_id VARCHAR, event_sequence INTEGER, event_text VARCHAR)")
    return con


def test_fused_text_prefers_scraped_over_manual(tmp_path):
    # detail_work_history_events reconstructs the same underlying narrative
    # as property_texts (module docstring), un-truncated by Excel's cell
    # limit -- scraped wins when both exist.
    con = _make_db(tmp_path)
    con.execute("INSERT INTO property_texts VALUES ('m1', 'spglobal_manual:m1', 'full_work_history', 'old manual text')")
    con.executemany(
        "INSERT INTO detail_work_history_events VALUES (?, ?, ?)",
        [("m1", 1, "Opened in 1990."), ("m1", 2, "Closed in 2010.")],
    )
    result = load_fused_property_texts(con)
    row = result.loc[result["property_id"] == "m1"].iloc[0]
    assert row["raw_text"] == "Opened in 1990. Closed in 2010."
    assert row["source_property_key"] == "spglobal_manual:m1"


def test_fused_text_falls_back_to_manual_when_no_scraped_text(tmp_path):
    con = _make_db(tmp_path)
    con.execute("INSERT INTO property_texts VALUES ('m2', 'spglobal_manual:m2', 'full_work_history', 'only manual text')")
    result = load_fused_property_texts(con)
    row = result.loc[result["property_id"] == "m2"].iloc[0]
    assert row["raw_text"] == "only manual text"
    assert row["source_property_key"] == "spglobal_manual:m2"


def test_fused_text_tags_scraper_only_mine(tmp_path):
    # A mine with scraped text but no manual properties/property_texts row
    # at all (the ~425 scraped-only mines found on real data) still needs a
    # non-null source_property_key for prepare_candidates' merge key.
    con = _make_db(tmp_path)
    con.executemany(
        "INSERT INTO detail_work_history_events VALUES (?, ?, ?)",
        [("m3", 1, "Discovered in 1985.")],
    )
    result = load_fused_property_texts(con)
    row = result.loc[result["property_id"] == "m3"].iloc[0]
    assert row["raw_text"] == "Discovered in 1985."
    assert row["source_property_key"] == "spglobal_scraped:m3"
