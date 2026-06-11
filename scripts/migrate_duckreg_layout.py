#!/usr/bin/env python3
"""Migrate legacy duckreg results into the new variant-aware folder layout.

This version is JSON-driven: it classifies each legacy result from the
``results_*.json`` payload itself rather than consulting ``AnalysisConfig``.

Legacy layout:
    output/analysis/duckreg/<model_name>/results_*.json

New layout:
    output/analysis/duckreg/<model_name>/<FE>/<RES>/<TEX>/<CLUSTER>/results_*.json
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Set, Tuple

DEFAULT_TEMPORAL_EXTENTS = {
    "500m": "2012-2020",
    "1km": "2000-2020",
    "5km": "1991-2020",
    "50km": "2000-2020",
    "ADM2": "1992-2020",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("output/analysis/duckreg"),
        help="Legacy duckreg results directory",
    )
    parser.add_argument(
        "--model",
        action="append",
        default=[],
        help="Only migrate the given model(s); can be passed multiple times",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Replace destination files if they already exist",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move files. Default is dry-run.",
    )
    return parser


def normalize_string(value: Any) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip()
    return text if text and text != "nan" else None


def normalize_model_name(model_name: str) -> str:
    """Drop legacy FE and resolution suffixes from the model folder name."""
    normalized = model_name
    normalized = re.sub(r"_(?:twfe|pooled|ufe|twyfe)$", "", normalized)
    normalized = re.sub(r"_(?:\d+km|\d+m)$", "", normalized)
    return normalized


def infer_resolution(data_source: Optional[str], model_name: str) -> str:
    source = normalize_string(data_source) or ""
    stem = Path(source).stem if source else ""
    raw = stem or source

    mapping = {
        "500m": "500m",
        "1km": "1km",
        "5km": "5km",
        "50km": "50km",
        "adm2_1km": "ADM2",
        "ADM2": "ADM2",
    }
    if raw in mapping:
        return mapping[raw]

    lowered = model_name.lower()
    for token in ("50km", "5km", "1km", "500m"):
        if token in lowered:
            return token
    if "adm" in lowered:
        return "ADM2"

    raise ValueError(f"Could not infer resolution from data_source={data_source!r}")


def infer_fixed_effects(formula: Optional[str], model_name: str) -> str:
    text = normalize_string(formula) or ""
    parts = [part.strip() for part in text.split("|")]
    fe_part = parts[1] if len(parts) >= 2 else ""

    if not fe_part:
        return "NO"

    fe_terms = [term.strip() for term in fe_part.split("+")]
    fe_terms = [term for term in fe_terms if term]
    normalized = " + ".join(fe_terms)

    mapping = {
        "pixel_id": "PX",
        "pixel_id + country*year": "PX+CY",
        "pixel_id + year": "PX+YR",
        "subdivision": "ADM2",
        "GID_2": "ADM2",
        "subdivision + country*year": "ADM2+CY",
        "GID_2 + country*year": "ADM2+CY",
        "subdivision + year": "ADM2+YR",
        "GID_2 + year": "ADM2+YR",
    }
    if normalized in mapping:
        return mapping[normalized]

    lowered = model_name.lower()
    if "_twfe" in lowered:
        return "PX+CY"
    if "_twyfe" in lowered:
        return "PX+YR"
    if "_ufe" in lowered:
        return "PX"
    if "_pooled" in lowered:
        return "NO"

    raise ValueError(f"Could not infer fixed effects from formula={formula!r}")


def infer_temporal_extent(query: Optional[str], resolution: str) -> str:
    text = normalize_string(query)
    if not text:
        return DEFAULT_TEMPORAL_EXTENTS[resolution]

    lower_bounds = [int(y) for y in re.findall(r"year\s*>=\s*(\d{4})", text)]
    upper_inclusive = [int(y) for y in re.findall(r"year\s*<=\s*(\d{4})", text)]
    upper_exclusive = [int(y) - 1 for y in re.findall(r"year\s*<\s*(\d{4})", text)]

    start_year = max(lower_bounds) if lower_bounds else None
    end_candidates = upper_inclusive + upper_exclusive
    end_year = min(end_candidates) if end_candidates else None

    if start_year is None and end_year is None:
        return DEFAULT_TEMPORAL_EXTENTS[resolution]

    if start_year is None:
        start_year = int(DEFAULT_TEMPORAL_EXTENTS[resolution].split("-")[0])
    if end_year is None:
        end_year = int(DEFAULT_TEMPORAL_EXTENTS[resolution].split("-")[1])

    return f"{start_year}-{end_year}"


def infer_clustering(payload: Dict[str, Any], resolution: str) -> str:
    meta = payload.get("analysis_metadata", {})

    explicit = normalize_string(meta.get("clustering"))
    if explicit:
        if explicit in ("Country", "country"):
            return "Country"
        if explicit in ("ADM2", "subdivision"):
            return "ADM2"

    settings = meta.get("settings", {})
    se_method = settings.get("se_method")
    if isinstance(se_method, dict):
        values = {str(v).strip() for v in se_method.values()}
        if "country" in values:
            return "Country"
        if "subdivision" in values:
            return "ADM2"

    if resolution in ("50km", "ADM2"):
        return "Country"
    return "ADM2"


def classify_result(model_name: str, payload: Dict[str, Any]) -> Dict[str, str]:
    meta = payload.get("analysis_metadata", {})
    formula = normalize_string(meta.get("formula"))
    data_source = normalize_string(meta.get("data_source"))
    query = normalize_string(meta.get("query"))

    resolution = infer_resolution(data_source, model_name)
    fixed_effects = infer_fixed_effects(formula, model_name)
    temporal_extent = infer_temporal_extent(query, resolution)
    clustering = infer_clustering(payload, resolution)

    return {
        "model_name": normalize_model_name(model_name),
        "fixed_effects": fixed_effects,
        "resolution": resolution,
        "temporal_extent": temporal_extent,
        "clustering": clustering,
    }


def iter_legacy_model_dirs(results_dir: Path, selected_models: Iterable[str]) -> Iterable[Path]:
    selected = {m.strip() for m in selected_models if m.strip()}
    for child in sorted(results_dir.iterdir()):
        if not child.is_dir():
            continue
        if selected and child.name not in selected:
            continue
        if any(child.glob("results_*.json")):
            yield child


def associated_run_files(json_file: Path) -> List[Path]:
    suffix = json_file.stem.removeprefix("results_")
    files = [json_file]
    txt = json_file.with_name(f"results_{suffix}.txt")
    if txt.exists():
        files.append(txt)
    return files


def associated_shared_files(model_dir: Path) -> List[Path]:
    files: List[Path] = []
    coeffs = model_dir / "coefficients.csv"
    if coeffs.exists():
        files.append(coeffs)
    files.extend(sorted(model_dir.glob("first_stage_*.csv")))
    return files


def destination_dir(results_dir: Path, spec: Dict[str, str]) -> Path:
    return (
        results_dir
        / spec["model_name"]
        / spec["fixed_effects"]
        / spec["resolution"]
        / spec["temporal_extent"]
        / spec["clustering"]
    )


def move_one(src: Path, dst: Path, replace: bool, execute: bool) -> str:
    if dst.exists():
        if not replace:
            return f"skip-exists {src} -> {dst}"
        if execute:
            if dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)

    if execute:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    return f"{'move' if execute else 'plan'} {src} -> {dst}"


def migrate_results(
    results_dir: Path,
    selected_models: List[str],
    replace: bool,
    execute: bool,
) -> int:
    migrated = 0

    for model_dir in iter_legacy_model_dirs(results_dir, selected_models):
        json_files = sorted(model_dir.glob("results_*.json"))
        resolved_targets: List[Path] = []
        target_set: Set[Path] = set()

        for json_file in json_files:
            try:
                with json_file.open() as fh:
                    payload = json.load(fh)
                spec = classify_result(model_dir.name, payload)
            except Exception as exc:
                print(f"skip-error {json_file}: {exc}")
                continue

            target_dir = destination_dir(results_dir, spec)
            resolved_targets.append(target_dir)
            target_set.add(target_dir.resolve())

            if json_file.parent.resolve() == target_dir.resolve():
                print(f"skip-already-migrated {json_file}")
                continue

            for src in associated_run_files(json_file):
                dst = target_dir / src.name
                print(move_one(src, dst, replace=replace, execute=execute))
                migrated += 1

        shared_files = associated_shared_files(model_dir)
        if shared_files:
            if len(target_set) != 1:
                print(
                    f"skip-shared-ambiguous {model_dir}: "
                    f"{len(shared_files)} shared file(s), {len(target_set)} destinations"
                )
            else:
                target_dir = resolved_targets[0]
                for src in shared_files:
                    dst = target_dir / src.name
                    print(move_one(src, dst, replace=replace, execute=execute))
                    migrated += 1

    return migrated


def main() -> int:
    args = build_parser().parse_args()
    results_dir = args.results_dir.resolve()

    print(f"results_dir={results_dir}")
    print(f"mode={'EXECUTE' if args.execute else 'DRY-RUN'}")

    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    migrated = migrate_results(
        results_dir=results_dir,
        selected_models=args.model,
        replace=args.replace,
        execute=args.execute,
    )
    print(f"done files_considered={migrated}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
