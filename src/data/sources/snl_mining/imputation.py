"""LLM-based year-imputation for `snl_mining`, converted from the interactive
`notebooks/snl_mining_openai_enrichment.ipynb` into an importable module so
it can be driven by `scripts/run_snl_mining_imputation.py`. Not a pipeline
`STEPS` member -- like the two source notebooks it supersedes for routine use
(`notebooks/README.md`), this is a genuinely async, hours-long external
OpenAI Batch API call, and `PipelineStep` only has `FETCH`/`PREPARE`/`GRID`
(no source has ever needed a fourth). The notebook stays for interactive
one-off probing/debugging.

Text source: fused scraped + manual work-history text (`load_fused_property_
texts`), not the manual `property_texts` table alone -- the scraper's
`detail_work_history_events` reconstructs the same underlying narrative
(verified: for the 20,674 mine_ids present in both, 38% byte-identical, the
rest differing only by re-join whitespace), un-truncated by Excel's
~32,767-char cell limit that clips the manual `full_work_history` field for
long-history mines (203 mines hit that cap; for those the scraped
reconstruction is materially more complete, up to 212 events vs one clipped
blob). `property_texts` stays the fallback for the ~4,104 mines that only
have manual text.

Output: unchanged target -- DuckDB table `property_llm_years` inside the
shared `database.duckdb` (same file the scraper and PREPARE both use),
delete-then-insert keyed by `property_id`, plus a CSV export to the sibling
`csv/property_llm_years.csv` (same convention as the scraper's own `csv/`
exports, `scraper/storage/regularized.py::export_regularized_tables_to_csv`).
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path
from typing import Optional

import duckdb
import pandas as pd
from openai import OpenAI

from .scraper.config import DATA_DIR, DEFAULT_DB_PATH

logger = logging.getLogger(__name__)

LLM_RESULT_TABLE = "property_llm_years"
DEFAULT_OUTPUT_DIR: Path = DATA_DIR / "imputation"
DEFAULT_CSV_DIR: Path = DATA_DIR / "csv"


def load_fused_property_texts(con) -> pd.DataFrame:
    """One row per mine_id with `full_work_history` text, preferring the
    scraper's `detail_work_history_events` (reconstructed by concatenating
    `event_text` in `event_sequence` order) and falling back to the manual
    `property_texts` table (module docstring). Shaped like the original
    `property_texts` query (`property_id`, `source_property_key`,
    `field_name`, `raw_text`) so it's a drop-in for
    `MineYearBatchEngine.prepare_candidates`'s merge, which keys on
    `(property_id, source_property_key)` against `properties`.
    `source_property_key` is carried through from `properties` where a
    manual row exists (unchanged from today); mines only present via the
    scraped text get a new `spglobal_scraped:<property_id>` tag so the merge
    key is never null.
    """
    query = """
        WITH scraped_text AS (
            SELECT mine_id AS property_id,
                string_agg(event_text, ' ' ORDER BY event_sequence) AS raw_text
            FROM detail_work_history_events
            GROUP BY mine_id
        ),
        manual_text AS (
            SELECT property_id, source_property_key, raw_text
            FROM property_texts WHERE field_name = 'full_work_history'
        )
        SELECT
            COALESCE(s.property_id, m.property_id) AS property_id,
            COALESCE(m.source_property_key, 'spglobal_scraped:' || s.property_id) AS source_property_key,
            'full_work_history' AS field_name,
            COALESCE(s.raw_text, m.raw_text) AS raw_text
        FROM scraped_text s
        FULL OUTER JOIN manual_text m ON s.property_id = m.property_id
    """
    return con.execute(query).df()


def load_properties(con) -> pd.DataFrame:
    return con.execute("SELECT * FROM properties").df()


def export_property_llm_years_csv(con, csv_dir: Path = DEFAULT_CSV_DIR) -> Path:
    """Dump `property_llm_years` to `<csv_dir>/property_llm_years.csv` via
    DuckDB's native `COPY ... TO`, same convention as the scraper's own
    `csv/` exports."""
    csv_dir = Path(csv_dir)
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "property_llm_years.csv"
    escaped = str(csv_path).replace("'", "''")
    con.execute(f"COPY {LLM_RESULT_TABLE} TO '{escaped}' (HEADER, DELIMITER ',')")
    return csv_path


class MineYearBatchEngine:
    """Plan, submit, track, and ingest year-extraction jobs for mining properties.

    The manifest is stored on disk as parquet so the queue state survives process
    restarts. Parsed model outputs are written into a DuckDB table so downstream
    consumers can query them without relying on standalone parquet artifacts.
    """

    ACTIVE_BATCH_STATUSES = {"validating", "in_progress", "finalizing", "cancelling"}
    TERMINAL_BATCH_STATUSES = {"completed", "failed", "cancelled", "expired"}
    RESULT_COLUMNS = [
        "property_id",
        "llm_opening_year",
        "llm_opening_status",
        "llm_opening_evidence",
        "llm_closing_year",
        "llm_closing_status",
        "llm_closing_evidence",
        "api_input_tokens",
        "api_output_tokens",
    ]

    def __init__(self, client: OpenAI, model: str, project_root: Path, live_service_tier: str = "flex") -> None:
        """Initialize the engine with the OpenAI client and prompt configuration."""
        self.client = client
        self.model = model
        self.project_root = project_root.resolve()
        self.live_service_tier = live_service_tier
        self.year_like_pattern = re.compile(r"\b(?:18|19|20)\d{2}\b")
        self.system_prompt = """
Extract opening and closing years for a mining property from S&P Global work history text.

Return JSON only, matching the provided schema exactly.

Field meanings:
- pid: property id from the input
- oy: opening year
- os: opening status
- oe: short opening evidence excerpt
- cy: closing year
- cs: closing status
- ce: short closing evidence excerpt

Status values:
- explicit: the year is directly stated for the requested event
- inferred: the best reading supports the year, but it is not directly stated for the requested event
- not_found: no usable evidence for the event
- ambiguous: conflicting or unclear evidence

Rules:
1. Extract only years, not full dates.
2. Opening means the original opening or operational start year.
3. Closing means the actual closure or end-of-operations year.
4. Opening and closing are independent; one may be present without the other.
5. Treat start-up, opened, commenced operations, first production, or entered production as opening evidence when they clearly refer to the property.
6. Treat closed, ceased operations, shut down, production ended, or permanently suspended as closing evidence when they clearly refer to the property.
7. Do not confuse reopening, restart, expansion, ownership changes, feasibility studies, permitting, temporary suspensions, care and maintenance, or sale events with original opening or permanent final closure unless the text clearly says so.
8. Evidence excerpts should be short verbatim spans when available.
9. Be conservative. Prefer ambiguous or not_found over overconfident guesses.
""".strip()
        # Keep the user prefix byte-stable so prompt caching has a better chance to help.
        self.user_prompt_prefix = "Extract opening and closing years for this mining property.\n\n"
        self.schema = {
            "type": "json_schema",
            "name": "mine_year_extract",
            "strict": True,
            "schema": {
                "type": "object",
                "additionalProperties": False,
                "properties": {
                    "pid": {"type": "string"},
                    "oy": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "os": {"type": "string", "enum": ["explicit", "inferred", "not_found", "ambiguous"]},
                    "oe": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                    "cy": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
                    "cs": {"type": "string", "enum": ["explicit", "inferred", "not_found", "ambiguous"]},
                    "ce": {"anyOf": [{"type": "string"}, {"type": "null"}]},
                },
                "required": ["pid", "oy", "os", "oe", "cy", "cs", "ce"],
            },
        }

    def _utc_now(self) -> str:
        """Return a stable UTC timestamp string for manifest updates."""
        return pd.Timestamp.now("UTC").isoformat()

    def _format_batch_errors(self, batch) -> str | None:
        """Collapse batch-level errors into one manifest-friendly string."""
        errors = getattr(batch, "errors", None)
        if not errors or not getattr(errors, "data", None):
            return None
        parts = []
        for err in errors.data:
            code = getattr(err, "code", "unknown_error")
            message = getattr(err, "message", "")
            parts.append(f"{code}: {message}".strip())
        return " | ".join(parts)[:1000] if parts else None

    def _to_manifest_path(self, path: Path | str) -> str:
        """Store manifest paths relative to the project root whenever possible."""
        candidate = Path(path)
        if not candidate.is_absolute():
            return candidate.as_posix()
        try:
            return candidate.resolve().relative_to(self.project_root).as_posix()
        except ValueError:
            return candidate.resolve().as_posix()

    def _resolve_manifest_path(self, path_value: Path | str) -> Path:
        """Resolve a manifest path value against the project root."""
        candidate = Path(path_value)
        return candidate if candidate.is_absolute() else self.project_root / candidate

    def build_user_payload(self, property_id: str, work_history_text: str) -> str:
        """Build the stable user payload sent to the model."""
        return self.user_prompt_prefix + f"property_id: {property_id}\nfull_work_history: {work_history_text}"

    def estimate_input_tokens(self, property_id: str, work_history_text: str, fixed_overhead_tokens: int = 0) -> int:
        """Approximate prompt tokens for planning against batch queue limits."""
        payload = self.build_user_payload(property_id, work_history_text)
        return math.ceil(len(payload) / 4) + fixed_overhead_tokens

    def split_for_batches(
        self,
        query_plan: pd.DataFrame,
        fixed_overhead_tokens: int,
        max_enqueued_input_tokens: int = 1_800_000,
        max_requests_per_batch: int = 50_000,
    ) -> list[pd.DataFrame]:
        """Split the query plan into chunks that respect queue and row limits."""
        chunks: list[pd.DataFrame] = []
        current_rows: list[dict] = []
        current_tokens = 0

        for row in query_plan.itertuples(index=False):
            row_tokens = self.estimate_input_tokens(
                str(row.property_id),
                row.raw_text,
                fixed_overhead_tokens=fixed_overhead_tokens,
            )
            would_exceed_tokens = current_rows and current_tokens + row_tokens > max_enqueued_input_tokens
            would_exceed_rows = current_rows and len(current_rows) >= max_requests_per_batch

            if would_exceed_tokens or would_exceed_rows:
                chunks.append(pd.DataFrame(current_rows))
                current_rows = []
                current_tokens = 0

            current_rows.append({col: getattr(row, col) for col in query_plan.columns})
            current_tokens += row_tokens

        if current_rows:
            chunks.append(pd.DataFrame(current_rows))

        return chunks

    def prepare_candidates(self, property_texts: pd.DataFrame, properties: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        """Filter the standardized tables down to rows worth sending to the model."""
        base = property_texts.merge(
            properties[["property_id", "source_property_key", "actual_start_up_year", "actual_closure_year"]],
            on=["property_id", "source_property_key"],
            how="left",
        ).copy()
        base["has_text"] = base["raw_text"].fillna("").str.strip().ne("")
        base["has_year_like"] = base["raw_text"].fillna("").str.contains(self.year_like_pattern)
        base["has_ground_truth"] = base["actual_start_up_year"].notna() | base["actual_closure_year"].notna()

        candidates = base.loc[base["has_text"] & (base["has_year_like"] | base["has_ground_truth"])].copy()
        candidates["query_reason"] = "year_like_only"
        candidates.loc[candidates["has_ground_truth"], "query_reason"] = "ground_truth_only"
        candidates.loc[candidates["has_year_like"] & candidates["has_ground_truth"], "query_reason"] = "year_like_and_ground_truth"

        summary = pd.DataFrame([{
            "all_property_text_rows": len(property_texts),
            "rows_with_text": int(base["has_text"].sum()),
            "rows_with_year_like": int((base["has_text"] & base["has_year_like"]).sum()),
            "rows_with_ground_truth": int((base["has_text"] & base["has_ground_truth"]).sum()),
            "llm_query_rows": len(candidates),
        }])
        return candidates, summary

    def select_probe(self, candidates: pd.DataFrame, preferred_property_id: str | None = None) -> pd.DataFrame:
        """Pick one row for a cheap live sanity check before the full batch run."""
        if preferred_property_id is not None:
            probe = candidates.loc[candidates["property_id"] == preferred_property_id].head(1).copy()
            if not probe.empty:
                return probe
        return candidates.head(1).copy()

    def query_one_live(self, property_id: str, work_history_text: str) -> tuple[dict, dict | None, str]:
        """Run a single synchronous probe request through the Responses API."""
        response = self.client.responses.create(
            model=self.model,
            service_tier=self.live_service_tier,
            input=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": self.build_user_payload(property_id, work_history_text)},
            ],
            reasoning={"effort": "low"},
            text={"verbosity": "low", "format": self.schema},
        )
        if not response.output_text:
            raise ValueError("The model returned no structured text output. Inspect the raw response for refusals or incomplete results.")
        usage = response.usage.model_dump() if getattr(response, "usage", None) else None
        return json.loads(response.output_text), usage, response.output_text

    def build_batch_requests(self, query_plan: pd.DataFrame) -> list[dict]:
        """Convert one chunk into `/v1/responses` batch request records."""
        requests = []
        for row in query_plan.itertuples(index=False):
            requests.append({
                "custom_id": f"mine-{row.property_id}",
                "method": "POST",
                "url": "/v1/responses",
                "body": {
                    "model": self.model,
                    "input": [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": self.build_user_payload(str(row.property_id), row.raw_text)},
                    ],
                    "reasoning": {"effort": "low"},
                    "text": {"verbosity": "low", "format": self.schema},
                },
            })
        return requests

    def write_request_jsonl(self, query_plan: pd.DataFrame, path: Path) -> Path:
        """Write one batch chunk to the JSONL format expected by the Batch API."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for request in self.build_batch_requests(query_plan):
                handle.write(json.dumps(request, ensure_ascii=True) + "\n")
        return path

    def write_frame_parquet(self, frame: pd.DataFrame, path: Path) -> Path:
        """Persist a dataframe as parquet using DuckDB to avoid optional pandas engines."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with duckdb.connect() as con:
            con.register("frame_df", frame)
            con.execute(f"COPY frame_df TO '{str(path)}' (FORMAT PARQUET)")
        return path

    def load_frame_parquet(self, path: Path) -> pd.DataFrame:
        """Load a parquet file through DuckDB."""
        with duckdb.connect() as con:
            return con.execute(f"SELECT * FROM read_parquet('{str(path)}')").df()

    def create_manifest(
        self,
        query_plan_chunks: list[pd.DataFrame],
        manifest_path: Path,
        request_dir: Path,
        output_dir: Path,
        fixed_overhead_tokens: int,
        overwrite: bool = False,
    ) -> pd.DataFrame:
        """Create or load the batch manifest and all request JSONL files."""
        if manifest_path.exists() and not overwrite:
            return self.load_manifest(manifest_path)

        request_dir.mkdir(parents=True, exist_ok=True)
        output_dir.mkdir(parents=True, exist_ok=True)

        rows = []
        for chunk_index, chunk in enumerate(query_plan_chunks, start=1):
            estimated_input_tokens = sum(
                self.estimate_input_tokens(str(pid), text, fixed_overhead_tokens=fixed_overhead_tokens)
                for pid, text in zip(chunk["property_id"], chunk["raw_text"])
            )
            request_path = request_dir / f"mine_year_extract_batch_{chunk_index:03d}.jsonl"
            output_path = output_dir / f"mine_year_extract_batch_{chunk_index:03d}_output.jsonl"
            self.write_request_jsonl(chunk, request_path)
            rows.append({
                "chunk_index": chunk_index,
                "request_path": self._to_manifest_path(request_path),
                "output_path": self._to_manifest_path(output_path),
                "manifest_status": "pending",
                "api_status": pd.NA,
                "batch_id": pd.NA,
                "input_file_id": pd.NA,
                "output_file_id": pd.NA,
                "error_file_id": pd.NA,
                "request_rows": int(len(chunk)),
                "completed_rows": 0,
                "estimated_input_tokens": int(estimated_input_tokens),
                "attempt_count": 0,
                "last_error": pd.NA,
                "submitted_at": pd.NA,
                "completed_at": pd.NA,
                "updated_at": self._utc_now(),
            })

        manifest = pd.DataFrame(rows)
        self.save_manifest(manifest, manifest_path)
        return manifest

    def save_manifest(self, manifest: pd.DataFrame, manifest_path: Path) -> Path:
        """Persist the manifest after coercing columns into stable parquet-friendly types."""
        manifest_to_store = manifest.copy()
        string_columns = [
            "request_path",
            "output_path",
            "manifest_status",
            "api_status",
            "batch_id",
            "input_file_id",
            "output_file_id",
            "error_file_id",
            "last_error",
            "submitted_at",
            "completed_at",
            "updated_at",
        ]
        integer_columns = [
            "chunk_index",
            "request_rows",
            "completed_rows",
            "estimated_input_tokens",
            "attempt_count",
        ]
        for column in string_columns:
            if column in manifest_to_store.columns:
                manifest_to_store[column] = manifest_to_store[column].astype("string")
        path_columns = [column for column in manifest_to_store.columns if column.endswith("_path")]
        for column in path_columns:
            manifest_to_store[column] = manifest_to_store[column].map(
                lambda value: pd.NA if pd.isna(value) else self._to_manifest_path(value)
            ).astype("string")
        for column in integer_columns:
            if column in manifest_to_store.columns:
                manifest_to_store[column] = pd.to_numeric(manifest_to_store[column], errors="coerce").astype("Int64")
        self.write_frame_parquet(manifest_to_store, manifest_path)
        return manifest_path

    def load_manifest(self, manifest_path: Path) -> pd.DataFrame:
        """Load and normalize the manifest from disk."""
        manifest = self.load_frame_parquet(manifest_path)
        string_columns = [
            "request_path",
            "output_path",
            "manifest_status",
            "api_status",
            "batch_id",
            "input_file_id",
            "output_file_id",
            "error_file_id",
            "last_error",
            "submitted_at",
            "completed_at",
            "updated_at",
        ]
        integer_columns = [
            "chunk_index",
            "request_rows",
            "completed_rows",
            "estimated_input_tokens",
            "attempt_count",
        ]
        for column in string_columns:
            if column in manifest.columns:
                manifest[column] = manifest[column].astype("string")
        for column in integer_columns:
            if column in manifest.columns:
                manifest[column] = pd.to_numeric(manifest[column], errors="coerce").astype("Int64")
        path_columns = [column for column in manifest.columns if column.endswith("_path")]
        for column in path_columns:
            manifest[column] = manifest[column].astype("string")
        return manifest.sort_values(["chunk_index"], kind="stable").reset_index(drop=True)

    def update_manifest_status(self, manifest_path: Path, chunk_index: int, **updates) -> pd.DataFrame:
        """Apply targeted field updates to one manifest row and persist the result."""
        manifest = self.load_manifest(manifest_path)
        row_mask = manifest["chunk_index"] == chunk_index
        if not row_mask.any():
            raise KeyError(f"Chunk {chunk_index} is not present in the manifest.")
        for column, value in updates.items():
            manifest.loc[row_mask, column] = value
        manifest.loc[row_mask, "updated_at"] = self._utc_now()
        self.save_manifest(manifest, manifest_path)
        return self.load_manifest(manifest_path)

    def refresh_manifest_statuses(self, manifest_path: Path) -> pd.DataFrame:
        """Refresh all submitted batch rows from the API and persist the latest state."""
        manifest = self.load_manifest(manifest_path)

        for idx, row in manifest.iterrows():
            batch_id = row.get("batch_id")
            if pd.isna(batch_id) or str(batch_id).strip() == "":
                continue
            if str(row.get("manifest_status")) in {"ingested", "failed", "cancelled", "expired"}:
                continue

            batch = self.client.batches.retrieve(str(batch_id))
            api_status = getattr(batch, "status", None)
            manifest.at[idx, "api_status"] = api_status
            manifest.at[idx, "output_file_id"] = getattr(batch, "output_file_id", None) or pd.NA
            manifest.at[idx, "error_file_id"] = getattr(batch, "error_file_id", None) or pd.NA
            manifest.at[idx, "last_error"] = self._format_batch_errors(batch) or manifest.at[idx, "last_error"]
            manifest.at[idx, "updated_at"] = self._utc_now()

            if api_status in self.ACTIVE_BATCH_STATUSES:
                manifest.at[idx, "manifest_status"] = "submitted"
            elif api_status == "completed":
                manifest.at[idx, "manifest_status"] = "completed"
                if pd.isna(manifest.at[idx, "completed_at"]):
                    manifest.at[idx, "completed_at"] = self._utc_now()
            elif api_status in {"failed", "cancelled", "expired"}:
                manifest.at[idx, "manifest_status"] = api_status
                if pd.isna(manifest.at[idx, "completed_at"]):
                    manifest.at[idx, "completed_at"] = self._utc_now()

        self.save_manifest(manifest, manifest_path)
        return self.load_manifest(manifest_path)

    def submit_batch_file(self, request_path: Path, completion_window: str = "24h"):
        """Upload one request file and create a Batch API job."""
        with request_path.open("rb") as handle:
            input_file = self.client.files.create(file=handle, purpose="batch")
        batch = self.client.batches.create(
            input_file_id=input_file.id,
            endpoint="/v1/responses",
            completion_window=completion_window,
        )
        return batch, input_file

    def submit_next_batch_if_ready(self, manifest_path: Path, completion_window: str = "24h") -> tuple[object | None, pd.DataFrame]:
        """Submit the next pending batch only when no earlier batch is still active."""
        manifest = self.refresh_manifest_statuses(manifest_path)

        if manifest["manifest_status"].isin(["failed", "cancelled", "expired"]).any():
            return None, manifest
        if manifest["api_status"].isin(list(self.ACTIVE_BATCH_STATUSES)).any():
            return None, manifest

        pending_mask = manifest["manifest_status"] == "pending"
        if not pending_mask.any():
            return None, manifest

        next_idx = manifest.index[pending_mask][0]
        request_path = self._resolve_manifest_path(str(manifest.at[next_idx, "request_path"]))
        batch, input_file = self.submit_batch_file(request_path, completion_window=completion_window)

        current_attempts = manifest.at[next_idx, "attempt_count"]
        current_attempts = 0 if pd.isna(current_attempts) else int(current_attempts)
        manifest.at[next_idx, "manifest_status"] = "submitted"
        manifest.at[next_idx, "api_status"] = getattr(batch, "status", None)
        manifest.at[next_idx, "batch_id"] = getattr(batch, "id", None)
        manifest.at[next_idx, "input_file_id"] = getattr(input_file, "id", None)
        manifest.at[next_idx, "attempt_count"] = current_attempts + 1
        manifest.at[next_idx, "submitted_at"] = self._utc_now()
        manifest.at[next_idx, "updated_at"] = self._utc_now()
        manifest.at[next_idx, "last_error"] = self._format_batch_errors(batch) or pd.NA
        self.save_manifest(manifest, manifest_path)
        return batch, self.load_manifest(manifest_path)

    def _extract_output_text(self, body: dict) -> str | None:
        """Extract the structured text payload from a batch response body."""
        if body.get("output_text"):
            return body["output_text"]
        for item in body.get("output", []):
            for content in item.get("content", []):
                text = content.get("text")
                if text:
                    return text
        return None

    def parse_batch_output_file(self, path: Path) -> pd.DataFrame:
        """Parse one downloaded batch output JSONL file into normalized result rows."""
        rows = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                payload = json.loads(line)
                response = payload.get("response") or {}
                body = response.get("body") or {}
                output_text = self._extract_output_text(body)
                parsed = json.loads(output_text) if output_text else {}
                usage = body.get("usage") or {}
                custom_id = payload.get("custom_id")
                property_id = parsed.get("pid")
                if property_id is None and custom_id and custom_id.startswith("mine-"):
                    property_id = custom_id.split("mine-", 1)[1]
                rows.append({
                    "property_id": property_id,
                    "llm_opening_year": parsed.get("oy"),
                    "llm_opening_status": parsed.get("os"),
                    "llm_opening_evidence": parsed.get("oe"),
                    "llm_closing_year": parsed.get("cy"),
                    "llm_closing_status": parsed.get("cs"),
                    "llm_closing_evidence": parsed.get("ce"),
                    "api_input_tokens": usage.get("input_tokens"),
                    "api_output_tokens": usage.get("output_tokens"),
                })
        return pd.DataFrame(rows)

    def download_batch_output_file(self, output_file_id: str, path: Path) -> Path:
        """Download one completed batch output file to disk.

        The SDK has exposed response text both as a property and as a callable in
        different releases, so we handle either shape here.
        """
        file_response = self.client.files.content(output_file_id)
        text_payload = file_response.text() if callable(getattr(file_response, "text", None)) else file_response.text
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text_payload, encoding="utf-8")
        return path

    def clean_result_frame(self, frame: pd.DataFrame) -> pd.DataFrame:
        """Normalize result types and deduplicate on property id."""
        cleaned = frame.copy()
        string_columns = [
            "property_id",
            "llm_opening_status",
            "llm_opening_evidence",
            "llm_closing_status",
            "llm_closing_evidence",
        ]
        integer_columns = [
            "llm_opening_year",
            "llm_closing_year",
            "api_input_tokens",
            "api_output_tokens",
        ]
        for column in string_columns:
            if column in cleaned.columns:
                cleaned[column] = cleaned[column].astype("string")
        for column in integer_columns:
            if column in cleaned.columns:
                cleaned[column] = pd.to_numeric(cleaned[column], errors="coerce").astype("Int64")
        if "property_id" in cleaned.columns:
            cleaned = cleaned.sort_values(["property_id"], kind="stable").drop_duplicates(subset=["property_id"], keep="last")
        return cleaned

    def ensure_result_table(self, db_path: Path, table_name: str) -> None:
        """Create the result table if it does not already exist."""
        with duckdb.connect(db_path) as con:
            con.execute(
                f"""
                CREATE TABLE IF NOT EXISTS {table_name} (
                    property_id VARCHAR,
                    llm_opening_year INTEGER,
                    llm_opening_status VARCHAR,
                    llm_opening_evidence VARCHAR,
                    llm_closing_year INTEGER,
                    llm_closing_status VARCHAR,
                    llm_closing_evidence VARCHAR,
                    api_input_tokens INTEGER,
                    api_output_tokens INTEGER
                )
                """
            )

    def upsert_results_table(self, frame: pd.DataFrame, db_path: Path, table_name: str) -> None:
        """Upsert cleaned result rows into the DuckDB result table keyed by property id."""
        cleaned = self.clean_result_frame(frame)
        self.ensure_result_table(db_path, table_name)
        with duckdb.connect(db_path) as con:
            con.register("incoming_results_df", cleaned[self.RESULT_COLUMNS])
            con.execute(f"DELETE FROM {table_name} USING incoming_results_df WHERE {table_name}.property_id = incoming_results_df.property_id")
            con.execute(f"INSERT INTO {table_name} SELECT * FROM incoming_results_df")

    def load_results_table(self, db_path: Path, table_name: str) -> pd.DataFrame:
        """Load the DuckDB result table if present, otherwise return an empty frame."""
        self.ensure_result_table(db_path, table_name)
        with duckdb.connect(db_path, read_only=True) as con:
            return con.execute(f"SELECT * FROM {table_name} ORDER BY property_id").df()

    def ingest_completed_batches(
        self,
        manifest_path: Path,
        db_path: Path,
        table_name: str,
        overwrite_outputs: bool = False,
    ) -> pd.DataFrame:
        """Download, parse, and write completed batch outputs into DuckDB."""
        manifest = self.refresh_manifest_statuses(manifest_path)

        for idx, row in manifest.iterrows():
            if str(row.get("manifest_status")) not in {"completed", "ingested"}:
                continue
            output_file_id = row.get("output_file_id")
            if pd.isna(output_file_id) or str(output_file_id).strip() == "":
                continue

            output_path = self._resolve_manifest_path(str(row["output_path"]))
            if overwrite_outputs or not output_path.exists():
                self.download_batch_output_file(str(output_file_id), output_path)

            result_frame = self.parse_batch_output_file(output_path)
            result_frame = self.clean_result_frame(result_frame)
            self.upsert_results_table(result_frame, db_path=db_path, table_name=table_name)

            manifest.at[idx, "manifest_status"] = "ingested"
            manifest.at[idx, "completed_rows"] = len(result_frame)
            manifest.at[idx, "updated_at"] = self._utc_now()

        self.save_manifest(manifest, manifest_path)
        return self.load_manifest(manifest_path)

    def run_periodic_batch_monitor(
        self,
        manifest_path: Path,
        db_path: Path,
        table_name: str,
        poll_interval_seconds: int = 300,
        completion_window: str = "24h",
        max_cycles: int | None = None,
        overwrite_outputs: bool = False,
    ) -> pd.DataFrame:
        """Periodically refresh, ingest, and advance the batch queue until done.

        Sleeps between polling cycles and stops once every chunk is either
        ingested or terminal, or `max_cycles` is reached (whichever first) --
        `max_cycles=1` makes this a single non-blocking pass, suitable for a
        cron-style repeated invocation of the CLI script instead of one long-
        running process.
        """
        cycles = 0
        while True:
            manifest = self.ingest_completed_batches(
                manifest_path=manifest_path,
                db_path=db_path,
                table_name=table_name,
                overwrite_outputs=overwrite_outputs,
            )
            _, manifest = self.submit_next_batch_if_ready(
                manifest_path=manifest_path,
                completion_window=completion_window,
            )
            manifest = self.refresh_manifest_statuses(manifest_path)

            status_values = set(manifest["manifest_status"].astype(str).tolist())
            active = manifest["api_status"].isin(list(self.ACTIVE_BATCH_STATUSES)).any()
            pending = (manifest["manifest_status"] == "pending").any()
            if status_values.issubset({"ingested", "failed", "cancelled", "expired"}) and not active and not pending:
                return manifest

            cycles += 1
            if max_cycles is not None and cycles >= max_cycles:
                return manifest

            time.sleep(poll_interval_seconds)


def run_imputation(
    db_path: Path = DEFAULT_DB_PATH,
    credentials_path: Optional[Path] = None,
    *,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    csv_dir: Path = DEFAULT_CSV_DIR,
    model: str = "gpt-5.4-nano",
    service_tier: str = "flex",
    full_query_limit: int | None = None,
    max_enqueued_input_tokens: int = 1_800_000,
    max_requests_per_file: int = 50_000,
    fixed_overhead_tokens: int = 700,
    poll_interval_seconds: int = 300,
    max_cycles: int | None = 1,
    overwrite_manifest: bool = False,
) -> pd.DataFrame:
    """End-to-end orchestration: load fused candidates, create/load the batch
    manifest, submit the next batch if the queue is clear, ingest any
    completed batches, export `property_llm_years` to CSV, and return the
    final manifest. `max_cycles=1` (the default) makes one non-blocking pass
    -- suitable for a periodic/cron-style re-invocation of the CLI script;
    pass a larger `max_cycles` (or `None` for unbounded) to block and poll
    until the whole queue drains in one process.

    `db_path` is opened read-only for candidate loading (this function never
    mutates it directly -- writes go through `MineYearBatchEngine`'s own
    `duckdb.connect(db_path)` calls in `upsert_results_table`/
    `ensure_result_table`, each their own short-lived connection).
    """
    if credentials_path is None:
        raise ValueError("credentials_path is required (path to a plain-text OpenAI API key file)")

    db_path = Path(db_path)
    output_dir = Path(output_dir)
    api_key = Path(credentials_path).read_text().strip()
    client = OpenAI(api_key=api_key, timeout=900.0)
    engine = MineYearBatchEngine(client=client, model=model, project_root=db_path.parent, live_service_tier=service_tier)

    with duckdb.connect(str(db_path), read_only=True) as con:
        property_texts = load_fused_property_texts(con)
        properties = load_properties(con)

    candidates, summary = engine.prepare_candidates(property_texts=property_texts, properties=properties)
    logger.info("snl_mining imputation candidates: %s", summary.to_dict(orient="records")[0])
    if full_query_limit is not None:
        candidates = candidates.head(full_query_limit).copy()

    batch_chunks = engine.split_for_batches(
        candidates,
        fixed_overhead_tokens=fixed_overhead_tokens,
        max_enqueued_input_tokens=max_enqueued_input_tokens,
        max_requests_per_batch=max_requests_per_file,
    )
    manifest_path = output_dir / "mine_year_extract_manifest.parquet"
    engine.create_manifest(
        batch_chunks,
        manifest_path=manifest_path,
        request_dir=output_dir / "batch_requests",
        output_dir=output_dir / "batch_outputs",
        fixed_overhead_tokens=fixed_overhead_tokens,
        overwrite=overwrite_manifest,
    )

    manifest = engine.run_periodic_batch_monitor(
        manifest_path=manifest_path,
        db_path=db_path,
        table_name=LLM_RESULT_TABLE,
        poll_interval_seconds=poll_interval_seconds,
        max_cycles=max_cycles,
    )

    with duckdb.connect(str(db_path)) as con:
        export_property_llm_years_csv(con, csv_dir)

    return manifest
