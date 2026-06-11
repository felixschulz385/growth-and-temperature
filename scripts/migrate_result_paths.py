#!/usr/bin/env python3
"""Migrate duckreg result directories to the spatial-extent-aware layout."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from gnt.analysis.config import (  # noqa: E402
    DEFAULT_EXCEL,
    FULL_SAMPLE_SPATIAL_EXTENT,
    RESULTS_DIR,
    normalize_clustering_label,
    normalize_fixed_effects_label,
    normalize_resolution_label,
    normalize_spatial_extent_label,
    normalize_temporal_extent_label,
)
from gnt.analysis.results import get_model_metadata  # noqa: E402

SETTINGS_ANCHOR_KEYS = {
    "HDI": ("migration_hdi_anchor_year", "hdi_anchor_year"),
    "WB": ("migration_wb_anchor_year", "wb_anchor_year"),
}
QUERY_TOKEN_RE = re.compile(r"\b(HDI|WB)_([A-Z]+)(?:_(\d{4}))?\s*(?:==|=)\s*([01])\b")
COUNTRY_FILTER_RE = re.compile(r"\(?\s*country\s+IN\s*\([^)]*\)\s*\)?", re.IGNORECASE)
LEGACY_RESOLUTION_TOKEN_RE = re.compile(r"_(500m|1km|5km|50km)(?=_|$)", re.IGNORECASE)
PARTITIONED_SPATIAL_EXTENT_RE = re.compile(r"^(HDI|WB)_[A-Z_]+_\d{4}$")
LEGACY_PARTITION_SUFFIX_RE = re.compile(r"_(HDI|WB)_[A-Z]+(?:_[A-Z]+)*(?:_\d{4})?$")
LEGACY_MODEL_ALIAS_SUFFIXES = (
    "_twfe",
    "_twyfe",
    "_ufe",
    "_af",
)


def _duckreg_root(base_path: Path) -> Path:
    return base_path / "duckreg"


def _normalize_text(value: Any) -> str:
    if value is None or pd.isna(value):
        return ""
    text = str(value).strip()
    return "" if text == "nan" else text


def _canonicalize_expression(expr: str) -> str:
    text = _normalize_text(expr)
    if not text or text == "0":
        return "0"
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"\(\s+", "(", text)
    text = re.sub(r"\s+\)", ")", text)
    return text.strip()


def _normalize_model_name(model_name: str) -> str:
    text = _normalize_text(model_name)
    text = LEGACY_RESOLUTION_TOKEN_RE.sub("", text)
    text = LEGACY_PARTITION_SUFFIX_RE.sub("", text)
    changed = True
    while changed:
        changed = False
        for suffix in LEGACY_MODEL_ALIAS_SUFFIXES:
            if text.endswith(suffix):
                text = text[: -len(suffix)]
                changed = True
    return text


def _canonicalize_instruments(expr: str) -> str:
    text = _canonicalize_expression(expr)
    if text == "0":
        return text

    text = re.sub(r"\bL\d+\.", "", text)

    old_iv = re.fullmatch(r"(.+)\s+\(([^()]+)\)", text)
    if old_iv and "~" not in text:
        endogenous = _canonicalize_expression(old_iv.group(1))
        instrument = _canonicalize_expression(old_iv.group(2))
        return f"({endogenous} ~ {instrument})"

    return text


def _is_partitioned_spatial_extent(spatial_extent: str) -> bool:
    return bool(PARTITIONED_SPATIAL_EXTENT_RE.fullmatch(_normalize_text(spatial_extent)))


def _parse_settings(
    config_path: Path,
    hdi_anchor_year: Optional[str] = None,
    wb_anchor_year: Optional[str] = None,
) -> Dict[str, str]:
    settings_df = pd.read_excel(config_path, sheet_name="Settings")
    settings: Dict[str, str] = {}
    for _, row in settings_df.iterrows():
        key = _normalize_text(row.get("key"))
        if not key:
            continue
        val = _normalize_text(row.get("value"))
        if val:
            settings[key] = val
    if hdi_anchor_year:
        settings["migration_hdi_anchor_year"] = _normalize_text(hdi_anchor_year)
    if wb_anchor_year:
        settings["migration_wb_anchor_year"] = _normalize_text(wb_anchor_year)
    return settings


def _query_has_country_subset(query: str) -> bool:
    return bool(COUNTRY_FILTER_RE.search(_normalize_text(query)))


def _infer_query_residual(query: str) -> str:
    text = _normalize_text(query)
    if not text:
        return "0"

    text = COUNTRY_FILTER_RE.sub("", text)
    text = re.sub(r"\(?\s*year\s*[<>]=?\s*\d{4}\s*\)?", "", text, flags=re.IGNORECASE)
    text = QUERY_TOKEN_RE.sub("", text)
    text = re.sub(r"\(\s*\)", "", text)
    text = re.sub(r"\s+(AND|OR)\s+(AND|OR)\s+", r" \1 ", text, flags=re.IGNORECASE)
    text = re.sub(r"^\s*(AND|OR)\s+", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+(AND|OR)\s*$", "", text, flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip(" ()")
    return _canonicalize_expression(text)


def _load_model_rows(
    config_path: Path,
    settings: Dict[str, str],
) -> Tuple[Dict[Tuple[str, str, str], List[Dict[str, Any]]], Dict[str, List[Dict[str, Any]]]]:
    models_df = pd.read_excel(config_path, sheet_name="Models")
    index: Dict[Tuple[str, str, str], List[Dict[str, Any]]] = {}
    by_name: Dict[str, List[Dict[str, Any]]] = {}

    for _, row in models_df.iterrows():
        model_name = _normalize_text(row.get("model_name"))
        if not model_name:
            continue

        resolution = normalize_resolution_label(row.get("data_source"))
        query = _normalize_text(row.get("query"))
        key = (
            _canonicalize_expression(row.get("dependent")),
            _canonicalize_expression(row.get("independent")),
            _canonicalize_instruments(row.get("instruments")),
        )
        row_info = {
            "model_name": model_name,
            "spatial_extent": _infer_spatial_extent(query, settings, resolution),
            "has_country_subset": bool(_normalize_text(row.get("subset"))),
            "query_residual": _infer_query_residual(query),
            "fixed_effects": normalize_fixed_effects_label(row.get("fixed_effects")),
            "resolution": resolution,
            "clustering": normalize_clustering_label(row.get("clustering"), resolution),
        }
        index.setdefault(key, []).append(row_info)
        by_name.setdefault(model_name, []).append(row_info)

    return index, by_name


def _parse_formula(formula: str) -> Tuple[str, str, str, str, Optional[str]]:
    parts = [_normalize_text(part) for part in str(formula).split("|")]
    lhs_rhs = parts[0]
    if "~" not in lhs_rhs:
        raise ValueError(f"Malformed formula: {formula!r}")

    dependent, independent = (_canonicalize_expression(x) for x in lhs_rhs.split("~", 1))
    fixed_effects = normalize_fixed_effects_label(parts[1] if len(parts) >= 2 else None)
    instruments = "0"
    clustering: Optional[str] = None

    if len(parts) >= 4:
        instruments = _canonicalize_instruments(parts[2])
        clustering = parts[3]
    elif len(parts) == 3:
        third = parts[2]
        if third in {"subdivision", "country", "ADM2", "Country"}:
            clustering = third
        else:
            instruments = _canonicalize_instruments(third)

    return dependent, independent, instruments, fixed_effects, clustering


def _infer_fixed_effects(formula_fixed_effects: str, legacy_rel: Path) -> str:
    if len(legacy_rel.parts) >= 5:
        legacy_fixed_effects = _normalize_text(legacy_rel.parts[1])
        if legacy_fixed_effects:
            return normalize_fixed_effects_label(legacy_fixed_effects)
    return formula_fixed_effects


def _infer_resolution(data_source: str) -> str:
    raw = _normalize_text(data_source)
    if not raw:
        raise ValueError("Missing data_source in result metadata")
    stem = Path(raw).stem
    stem = stem.lstrip("_")
    raw = raw.lstrip("_")
    return normalize_resolution_label(stem or raw)


def _infer_clustering(
    metadata: Dict[str, Any],
    formula_cluster: Optional[str],
    resolution: str,
    legacy_rel: Path,
) -> str:
    candidate = (
        _normalize_text(metadata.get("clustering"))
        or (legacy_rel.parts[-1] if len(legacy_rel.parts) >= 5 else "")
        or _normalize_text(formula_cluster)
    )
    return normalize_clustering_label(candidate, resolution)


def _infer_anchor_year(settings: Dict[str, str], family: str) -> str:
    for key in SETTINGS_ANCHOR_KEYS[family]:
        if key in settings:
            year = _normalize_text(settings[key])
            if re.fullmatch(r"\d{4}", year):
                return year
            raise ValueError(f"Invalid anchor year for {family}: {key}={year!r}")
    raise ValueError(
        f"Missing {family} anchor year in Settings sheet. "
        f"Expected one of: {', '.join(SETTINGS_ANCHOR_KEYS[family])}. "
        f"You can also pass --{family.lower()}-anchor-year to this script."
    )


def _infer_temporal_extent(query: str, resolution: str) -> str:
    text = _normalize_text(query)
    if not text:
        return normalize_temporal_extent_label(None, resolution)

    lower_bounds = [int(y) for y in re.findall(r"year\s*>=\s*(\d{4})", text)]
    upper_inclusive = [int(y) for y in re.findall(r"year\s*<=\s*(\d{4})", text)]
    upper_exclusive = [int(y) - 1 for y in re.findall(r"year\s*<\s*(\d{4})", text)]

    start_year = max(lower_bounds) if lower_bounds else None
    end_candidates = upper_inclusive + upper_exclusive
    end_year = min(end_candidates) if end_candidates else None

    if start_year is None and end_year is None:
        return normalize_temporal_extent_label(None, resolution)

    default_start, default_end = normalize_temporal_extent_label(None, resolution).split("-")
    if start_year is None:
        start_year = int(default_start)
    if end_year is None:
        end_year = int(default_end)
    return f"{start_year}-{end_year}"


def _default_anchor_year_from_query(query: str, resolution: str) -> str:
    temporal_extent = _infer_temporal_extent(query, resolution)
    start_year = int(temporal_extent.split("-", 1)[0])
    return str(start_year - 1)


def _infer_spatial_extent(query: str, settings: Dict[str, str], resolution: str) -> str:
    text = _normalize_text(query)
    if not text:
        return FULL_SAMPLE_SPATIAL_EXTENT

    matches = list(QUERY_TOKEN_RE.finditer(text))
    if not matches:
        stripped = COUNTRY_FILTER_RE.sub("", text)
        stripped = re.sub(r"\(?\s*year\s*[<>]=?\s*\d{4}\s*\)?", "", stripped, flags=re.IGNORECASE)
        stripped = re.sub(r"\s+(AND|OR)\s+", " ", stripped, flags=re.IGNORECASE).strip(" ()")
        return FULL_SAMPLE_SPATIAL_EXTENT if not stripped else normalize_spatial_extent_label(stripped)

    family = matches[0].group(1)
    buckets: List[str] = []
    explicit_years: List[str] = []
    positive_buckets: List[str] = []
    for match in matches:
        if match.group(1) != family:
            raise ValueError(f"Mixed subset families in query: {query!r}")
        bucket = match.group(2)
        if bucket not in buckets:
            buckets.append(bucket)
        if match.group(4) == "1" and bucket not in positive_buckets:
            positive_buckets.append(bucket)
        if match.group(3):
            explicit_years.append(match.group(3))

    if positive_buckets:
        buckets = positive_buckets

    if explicit_years:
        unique_years = sorted(set(explicit_years))
        if len(unique_years) != 1:
            raise ValueError(f"Mixed subset years in query: {query!r}")
        year = unique_years[0]
    else:
        try:
            year = _infer_anchor_year(settings, family)
        except ValueError:
            year = _default_anchor_year_from_query(query, resolution)

    return normalize_spatial_extent_label(f"{family}_{'_'.join(buckets)}_{year}")


def _spatial_extent_matches(candidate_extent: str, result_extent: str) -> bool:
    candidate = normalize_spatial_extent_label(candidate_extent)
    result = normalize_spatial_extent_label(result_extent)
    if candidate == result:
        return True
    return (
        candidate == FULL_SAMPLE_SPATIAL_EXTENT
        and _is_partitioned_spatial_extent(result)
    )


def _partition_fallback_candidates(
    candidates: List[Dict[str, Any]],
    spatial_extent: str,
    has_country_subset: bool,
    query_residual: str,
) -> List[Dict[str, Any]]:
    if not _is_partitioned_spatial_extent(spatial_extent):
        return []
    return [
        candidate
        for candidate in candidates
        if candidate["spatial_extent"] == FULL_SAMPLE_SPATIAL_EXTENT
        and candidate["has_country_subset"] == has_country_subset
        and candidate["query_residual"] == query_residual
    ]


def _classify_result_json(
    result_json: Path,
    model_index: Dict[Tuple[str, str, str], List[Dict[str, Any]]],
    models_by_name: Dict[str, List[Dict[str, Any]]],
    settings: Dict[str, str],
    legacy_rel: Path,
) -> Tuple[str, str, str, str, str, str]:
    with open(result_json) as fh:
        data = json.load(fh)

    metadata = get_model_metadata(data)
    formula = _normalize_text(metadata.get("formula"))
    dependent, independent, instruments, fixed_effects, formula_cluster = _parse_formula(formula)
    fixed_effects = _infer_fixed_effects(fixed_effects, legacy_rel)
    resolution = _infer_resolution(metadata.get("data_source"))
    clustering = _infer_clustering(metadata, formula_cluster, resolution, legacy_rel)
    match_key = (
        dependent,
        independent,
        instruments,
    )

    query = _normalize_text(metadata.get("query"))
    temporal_extent = _infer_temporal_extent(query, resolution)
    spatial_extent = _infer_spatial_extent(query, settings, resolution)
    has_country_subset = _query_has_country_subset(query)
    query_residual = _infer_query_residual(query)

    legacy_model_name = legacy_rel.parts[0] if legacy_rel.parts else ""
    normalized_legacy = _normalize_model_name(legacy_model_name)

    model_candidates = model_index.get(match_key, [])
    if not model_candidates and normalized_legacy:
        model_candidates = models_by_name.get(normalized_legacy, [])
    filtered_candidates = [
        candidate
        for candidate in model_candidates
        if _spatial_extent_matches(candidate["spatial_extent"], spatial_extent)
        and candidate["has_country_subset"] == has_country_subset
        and candidate["query_residual"] == query_residual
    ]
    if not filtered_candidates and normalized_legacy:
        filtered_candidates = [
            candidate
            for candidate in models_by_name.get(normalized_legacy, [])
            if _spatial_extent_matches(candidate["spatial_extent"], spatial_extent)
            and candidate["has_country_subset"] == has_country_subset
            and candidate["query_residual"] == query_residual
        ]
    if not filtered_candidates:
        filtered_candidates = _partition_fallback_candidates(
            model_candidates,
            spatial_extent,
            has_country_subset,
            query_residual,
        )
    if not filtered_candidates and normalized_legacy:
        filtered_candidates = _partition_fallback_candidates(
            models_by_name.get(normalized_legacy, []),
            spatial_extent,
            has_country_subset,
            query_residual,
        )
    if not filtered_candidates and normalized_legacy and normalized_legacy in models_by_name:
        filtered_candidates = models_by_name[normalized_legacy]
    if not filtered_candidates and legacy_model_name and legacy_model_name in models_by_name:
        filtered_candidates = models_by_name[legacy_model_name]
    if not filtered_candidates:
        raise ValueError(
            f"Expected at least one Models-sheet match for {result_json}, got 0: "
            f"{model_candidates or match_key}"
        )

    ranked_groups = [
        [
            candidate for candidate in filtered_candidates
            if candidate["fixed_effects"] == fixed_effects
            and candidate["resolution"] == resolution
            and candidate["clustering"] == clustering
        ],
        [
            candidate for candidate in filtered_candidates
            if candidate["fixed_effects"] == fixed_effects
        ],
        filtered_candidates,
    ]
    matching_candidates: List[Dict[str, Any]] = []
    for group in ranked_groups:
        if len(group) == 1:
            matching_candidates = group
            break
    if not matching_candidates and normalized_legacy and normalized_legacy in models_by_name:
        named_group = models_by_name[normalized_legacy]
        if len(named_group) == 1:
            matching_candidates = named_group
    if not matching_candidates and legacy_model_name and legacy_model_name in models_by_name:
        named_group = models_by_name[legacy_model_name]
        if len(named_group) == 1:
            matching_candidates = named_group
    if not matching_candidates:
        raise ValueError(
            f"Expected exactly one Models-sheet match for {result_json}, got {len(filtered_candidates)}: "
            f"{filtered_candidates}"
        )

    model_name = matching_candidates[0]["model_name"]
    if normalized_legacy:
        exact_name_matches = [
            candidate for candidate in (matching_candidates or filtered_candidates)
            if candidate["model_name"] == normalized_legacy
        ]
        if len(exact_name_matches) == 1:
            model_name = exact_name_matches[0]["model_name"]

    return (
        model_name,
        fixed_effects,
        resolution,
        temporal_extent,
        spatial_extent,
        clustering,
    )

def _iter_legacy_result_jsons(duckreg_root: Path) -> Iterable[Tuple[Path, Path]]:
    for result_json in sorted(duckreg_root.glob("**/results_*.json")):
        rel = result_json.relative_to(duckreg_root)
        if len(rel.parts) < 7:
            yield result_json, rel.parent


def _target_dir(parts: Tuple[str, str, str, str, str, str]) -> Path:
    model_name, fixed_effects, resolution, temporal_extent, spatial_extent, clustering = parts
    return Path(
        model_name,
        fixed_effects,
        resolution,
        temporal_extent,
        spatial_extent,
        clustering,
    )


def _associated_run_files(result_json: Path) -> List[Path]:
    files = [result_json]
    stem_suffix = result_json.stem.removeprefix("results_")
    txt_path = result_json.with_name(f"results_{stem_suffix}.txt")
    if txt_path.exists():
        files.append(txt_path)
    return files


def _collect_conflicts(source_files: List[Path], target_dir: Path) -> List[Path]:
    return [target_dir / item.name for item in source_files if (target_dir / item.name).exists()]


def _remove_empty_parents(start_dir: Path, stop_dir: Path) -> None:
    current = start_dir
    while current != stop_dir and current.exists():
        try:
            current.rmdir()
        except OSError:
            break
        current = current.parent


def migrate_results(
    base_path: Path,
    config_path: Path,
    apply: bool = False,
    hdi_anchor_year: Optional[str] = None,
    wb_anchor_year: Optional[str] = None,
) -> int:
    duckreg_root = _duckreg_root(base_path)
    if not duckreg_root.exists():
        raise FileNotFoundError(f"DuckReg results directory not found: {duckreg_root}")

    settings = _parse_settings(
        config_path,
        hdi_anchor_year=hdi_anchor_year,
        wb_anchor_year=wb_anchor_year,
    )
    model_index, models_by_name = _load_model_rows(config_path, settings)

    planned_moves: List[Tuple[List[Path], Path, str]] = []
    conflicts: Dict[Path, List[Path]] = {}
    failures: Dict[Path, str] = {}

    for result_json, legacy_rel in _iter_legacy_result_jsons(duckreg_root):
        try:
            parts = _classify_result_json(
                result_json,
                model_index,
                models_by_name,
                settings,
                legacy_rel,
            )
        except Exception as exc:
            failures[result_json] = str(exc)
            continue

        target_rel = _target_dir(parts)
        spatial_extent = parts[4]
        target_dir = duckreg_root / target_rel
        source_files = _associated_run_files(result_json)
        file_conflicts = _collect_conflicts(source_files, target_dir)
        if file_conflicts:
            conflicts[result_json] = file_conflicts
            continue

        planned_moves.append((source_files, target_dir, spatial_extent))

    for source_files, target_dir, spatial_extent in planned_moves:
        print(f"{'MOVE' if apply else 'PLAN'} {source_files[0].parent} -> {target_dir} [{spatial_extent}]")

    if failures:
        print("\nClassification failures:")
        for result_json, message in failures.items():
            print(f"  {result_json}: {message}")

    if conflicts:
        print("\nConflicts:")
        for result_json, paths in conflicts.items():
            print(f"  {result_json}")
            for path in paths:
                print(f"    exists: {path}")
        if apply:
            return 1

    if not apply:
        print(
            f"\nDry run complete: {len(planned_moves)} result move(s) planned, "
            f"{len(conflicts)} conflict(s), {len(failures)} classification failure(s)."
        )
        return 0 if not conflicts and not failures else 1

    moved = 0
    touched_dirs: set[Path] = set()
    for source_files, target_dir, _ in planned_moves:
        target_dir.mkdir(parents=True, exist_ok=True)
        for item in source_files:
            shutil.move(str(item), str(target_dir / item.name))
            touched_dirs.add(item.parent)
        moved += 1
    for parent_dir in sorted(touched_dirs, key=lambda p: len(p.parts), reverse=True):
        _remove_empty_parents(parent_dir, duckreg_root)

    print(f"\nMigration complete: moved {moved} result set(s).")
    return 0 if not failures else 1


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Migrate duckreg result directories to include spatial extent."
    )
    parser.add_argument(
        "--base-path",
        default=str(RESULTS_DIR),
        help="Base analysis output directory (default: output/analysis)",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_EXCEL),
        help="Analysis workbook path used for Settings and Models matching",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Apply filesystem moves. Default is dry-run.",
    )
    parser.add_argument(
        "--hdi-anchor-year",
        help="Override the HDI anchor year used for legacy subset queries without an explicit year.",
    )
    parser.add_argument(
        "--wb-anchor-year",
        help="Override the WB anchor year used for legacy subset queries without an explicit year.",
    )
    return parser


def main() -> int:
    args = build_parser().parse_args()
    return migrate_results(
        Path(args.base_path),
        Path(args.config_path),
        apply=args.apply,
        hdi_anchor_year=args.hdi_anchor_year,
        wb_anchor_year=args.wb_anchor_year,
    )


if __name__ == "__main__":
    raise SystemExit(main())
