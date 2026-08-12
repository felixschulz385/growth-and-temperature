# SNL Mining Notebooks

This folder contains the SNL mining download-side notebooks that replace the older combined experiment notebook.

## Notebooks

### `snl_mining_manual_xls_to_duckdb.ipynb`

Purpose:
- ingest the manually exported S&P Global mining `.xls`
- standardize the manual export into stable relational tables
- write the cleaned result to DuckDB

Expected output:
- `data/raw/snl_mining/database.duckdb` -- shared with the scraper (`scraper/config.py`'s `DEFAULT_DB_PATH` points at the same file): this notebook's tables and the scraper's own `detail_*`/`mines`/etc. tables live in one merged database.

Tables written:
- `source_files`
- `properties`
- `property_texts`
- `property_work_history_events`
- `raw_property_records`

Status:
- transitional / likely to be deprecated later once the richer scraper (`src/data/sources/snl_mining/scraper/`, run via `scripts/debug_snl_mining_scraper.py`) is the main ingestion path

### `snl_mining_openai_enrichment.ipynb`

**Superseded for routine use** by `scripts/run_snl_mining_imputation.py` / `src/data/sources/snl_mining/imputation.py` -- the same `MineYearBatchEngine`, now an importable module with a CLI wrapper instead of notebook cells, reading fused scraped+manual work-history text (`imputation.load_fused_property_texts`) instead of the manual `property_texts` table alone. Keep this notebook for interactive one-off probing/debugging (e.g. testing a prompt change against a single mine before running the script for real).

Purpose:
- load standardized mining property text from the manual-export DuckDB
- prepare and manage OpenAI batch requests
- periodically re-check batch progress and advance the queue
- write model-imputed opening and closing years into the DuckDB table `property_llm_years`

Expected outputs:
- `data/raw/snl_mining/imputation/mine_year_extract_manifest.parquet`
- `data/raw/snl_mining/database.duckdb` table `property_llm_years`
- `data/raw/snl_mining/imputation/batch_requests/*.jsonl`
- `data/raw/snl_mining/imputation/batch_outputs/*.jsonl`

Stored fields in `property_llm_years`:
- `property_id`
- `llm_opening_year`
- `llm_opening_status`
- `llm_opening_evidence`
- `llm_closing_year`
- `llm_closing_status`
- `llm_closing_evidence`
- `api_input_tokens`
- `api_output_tokens`

## Recommended order

1. Run `snl_mining_manual_xls_to_duckdb.ipynb`
2. Run the one-row probe in `snl_mining_openai_enrichment.ipynb`
3. Build the manifest and submit batches incrementally
4. Either rerun the refresh/ingest/submit cells periodically or use the periodic monitor helper
5. Query `property_llm_years` from the DuckDB database when ingestion is complete

## Notes

- The current OpenAI workflow extracts year-level opening and closing imputations, not full calendar dates.
- The main downstream enrichment artifact is the DuckDB table `property_llm_years`, keyed by `property_id`.
- The periodic monitor helper keeps the notebook session active while polling. If you do not want to keep the notebook open, submit a batch, close the notebook, and return later to refresh and ingest results.
