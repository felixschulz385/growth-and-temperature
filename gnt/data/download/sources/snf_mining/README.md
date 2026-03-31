# SNF Mining Notebooks

This folder contains the SNF mining download-side notebooks that replace the older combined experiment notebook.

## Notebooks

### `snf_mining_manual_xls_to_duckdb.ipynb`

Purpose:
- ingest the manually exported S&P Global mining `.xls`
- standardize the manual export into stable relational tables
- write the cleaned result to DuckDB

Expected output:
- `data/snf_mining/processed/stage_0/manual_xls/snf_mining_manual_export.duckdb`

Tables written:
- `source_files`
- `properties`
- `property_texts`
- `property_work_history_events`
- `raw_property_records`

Status:
- transitional / likely to be deprecated later once the richer scraper is the main ingestion path

### `snf_mining_openai_enrichment.ipynb`

Purpose:
- load standardized mining property text from the manual-export DuckDB
- prepare and manage OpenAI batch requests
- periodically re-check batch progress and advance the queue
- write model-imputed opening and closing years into the DuckDB table `property_llm_years`

Expected outputs:
- `data/snf_mining/processed/stage_0/llm/mine_year_extract_manifest.parquet`
- `data/snf_mining/processed/stage_0/manual_xls/snf_mining_manual_export.duckdb` table `property_llm_years`
- `data/snf_mining/processed/stage_0/llm/batch_requests/*.jsonl`
- `data/snf_mining/processed/stage_0/llm/batch_outputs/*.jsonl`

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

1. Run `snf_mining_manual_xls_to_duckdb.ipynb`
2. Run the one-row probe in `snf_mining_openai_enrichment.ipynb`
3. Build the manifest and submit batches incrementally
4. Either rerun the refresh/ingest/submit cells periodically or use the periodic monitor helper
5. Query `property_llm_years` from the DuckDB database when ingestion is complete

## Notes

- The current OpenAI workflow extracts year-level opening and closing imputations, not full calendar dates.
- The main downstream enrichment artifact is the DuckDB table `property_llm_years`, keyed by `property_id`.
- The periodic monitor helper keeps the notebook session active while polling. If you do not want to keep the notebook open, submit a batch, close the notebook, and return later to refresh and ingest results.
