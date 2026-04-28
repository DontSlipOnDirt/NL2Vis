#!/usr/bin/env python3
"""
Recompute missing summary metrics in scores.json for one or more result folders.

Usage:
  python results/fill_missing_metrics.py
  python results/fill_missing_metrics.py results/20260425_144805_Qwen2.5-Coder-14B-Instruct_gpt-4o-mini

Edit TARGET_FOLDERS below to control which result folders are processed by default.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

import pandas as pd


# Configure the folders to process here. Relative paths are resolved from the repo root.
TARGET_FOLDERS = [
    Path(r"results/20260424_112106_Qwen2.5-Coder-7B-Instruct_NoVision"),
    Path(r"results/20260421_072403_Qwen2.5-Coder-14B-Instruct-AWQ_NoVision_ablation_no_composer_template"),
    Path(r"results/20260423_043242_Qwen3-Coder-30B-A3B-Instruct-FP8_NoVision"),
    Path(r"results/20260425_072348_Qwen2.5-Coder-14B-Instruct-AWQ_NoVision"),
    Path(r"results/20260413_230818_Qwen2.5-7B-Instruct_gpt-4o-mini"),
    Path(r"results/20260413_123628_Qwen3-Coder-30B-A3B-Instruct-FP8_gpt-4o-mini"),
    Path(r"results/20260412_232504_Qwen2.5-Coder-14B-Instruct-AWQ_gpt-4o-mini"),
    Path(r"results/20260421_030056_Qwen2.5-Coder-14B-Instruct-AWQ_NoVision_ablation_no_processor"),
    Path(r"results/20260426_232824_Qwen2.5-Coder-32B-Instruct-AWQ_NoVision"),
    Path(r"results/20260426_062107_Qwen2.5-Coder-14B-Instruct_gpt-4o-mini"),
    Path(r"results/20260427_161919_Qwen2.5-Coder-32B-Instruct-AWQ_NoVision"),
    Path(r"results/20260418_163315_Qwen2.5-Coder-32B-Instruct-AWQ_gpt-4o-mini"),
    Path(r"results/20260421_082913_Qwen2.5-Coder-14B-Instruct-AWQ_NoVision_ablation_no_proc_no_comp_tmpl"),
    Path(r"results/20260422_054917_Qwen2.5-Coder-32B-Instruct-AWQ_NoVision_ablation_no_composer_template"),
    Path(r"results/20260420_014754_Qwen2.5-Coder-7B-Instruct-AWQ_NoVision"),
    Path(r"results/20260421_160411_Qwen2.5-Coder-32B-Instruct-AWQ_NoVision_ablation_no_processor"),
    Path(r"results/20260420_142325_Qwen2.5-Coder-32B-Instruct-AWQ_gpt-4o-mini"),
    Path(r"results/20260423_010457_Qwen3-Coder-30B-A3B-Instruct-FP8_NoVision_ablation_no_processor"),
    Path(r"results/20260425_024744_Qwen2.5-Coder-14B-Instruct-AWQ_NoVision"),
    Path(r"results/20260421_102609_Qwen3-Coder-30B-A3B-Instruct-FP8_NoVision_ablation_no_processor")
]

DATASET_PATH = Path(r"visEval_dataset/visEval.json")

EXCLUDED_SCORE_COLUMNS = {
    "id",
    "chart",
    "hardness",
    "is_multi_table",
    "text_model",
    "vision_model",
}

PRIMARY_METRICS = [
    "invalid_rate",
    "illegal rate",
    "pass_rate",
    "readability_score",
    "quality_score",
]

RENAMED_METRICS = {
    "total_inference_time": "avg_total_inference_time",
    "inference_count": "avg_inference_count",
}


def normalize_query(query: str) -> str:
    return " ".join(str(query).split())


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_dataset(path: Path) -> dict[str, dict[str, Any]]:
    raw = load_json(path)
    return {str(key): value for key, value in raw.items()}


def build_query_lookup(dataset: dict[str, dict[str, Any]]) -> dict[tuple[str, str], str]:
    lookup: dict[tuple[str, str], str] = {}
    duplicates: list[tuple[str, str, str, str]] = []

    for instance_id, record in dataset.items():
        db_id = str(record.get("db_id", ""))
        for query in record.get("nl_queries", []):
            key = (db_id, normalize_query(query))
            if key in lookup and lookup[key] != instance_id:
                duplicates.append((db_id, key[1], lookup[key], instance_id))
                continue
            lookup[key] = instance_id

    if duplicates:
        print("Warning: duplicate query texts found while building trace lookup.")
        for db_id, query, first_id, second_id in duplicates[:5]:
            print(f"  db_id={db_id} query={query!r} -> {first_id}, {second_id}")

    return lookup


def infer_is_multi_table(instance_id: str, dataset: dict[str, dict[str, Any]]) -> bool:
    record = dataset.get(str(instance_id), {})
    vql = record.get("vis_query", {}).get("VQL", "")
    return "JOIN" in str(vql).upper()


def add_is_multi_table_column(df: pd.DataFrame, dataset: dict[str, dict[str, Any]]) -> pd.DataFrame:
    if "is_multi_table" in df.columns:
        result = df.copy()
        result["is_multi_table"] = result["is_multi_table"].fillna(
            result["id"].astype(str).map(lambda instance_id: infer_is_multi_table(instance_id, dataset))
        )
        return result

    result = df.copy()
    result["is_multi_table"] = result["id"].astype(str).map(
        lambda instance_id: infer_is_multi_table(instance_id, dataset)
    )
    return result


def attach_token_metrics(df: pd.DataFrame, folder: Path, dataset: dict[str, dict[str, Any]]) -> pd.DataFrame:
    trace_path = folder / "api_trace.json"
    if not trace_path.exists():
        return df

    lookup = build_query_lookup(dataset)
    token_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {
            "total_prompt_tokens": 0.0,
            "total_response_tokens": 0.0,
            "total_tokens": 0.0,
            "token_count": 0.0,
        }
    )

    unmapped = 0
    with trace_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line:
                continue
            try:
                entry = json.loads(line)
            except json.JSONDecodeError:
                continue

            db_id = str(entry.get("db_id", ""))
            query = normalize_query(entry.get("query", ""))
            instance_id = lookup.get((db_id, query))
            if instance_id is None:
                unmapped += 1
                continue

            prompt_tokens = float(entry.get("prompt_token", 0) or 0)
            response_tokens = float(entry.get("response_token", 0) or 0)
            token_totals[instance_id]["total_prompt_tokens"] += prompt_tokens
            token_totals[instance_id]["total_response_tokens"] += response_tokens
            token_totals[instance_id]["total_tokens"] += prompt_tokens + response_tokens
            token_totals[instance_id]["token_count"] += 1

    if unmapped:
        print(f"Warning: {unmapped} trace entries could not be mapped to a dataset instance.")

    token_df = pd.DataFrame.from_dict(token_totals, orient="index")
    token_df.index.name = "id"
    token_df = token_df.reset_index()

    if token_df.empty:
        return df

    merged = df.merge(token_df, on="id", how="left", suffixes=("", "_trace"))

    for column in [
        "total_prompt_tokens",
        "total_response_tokens",
        "total_tokens",
        "token_count",
    ]:
        trace_column = f"{column}_trace"
        if trace_column in merged.columns:
            if column in merged.columns:
                merged[column] = merged[column].fillna(merged[trace_column])
                merged = merged.drop(columns=[trace_column])
            else:
                merged = merged.rename(columns={trace_column: column})

    if "avg_prompt_tokens" not in merged.columns:
        merged["avg_prompt_tokens"] = merged["total_prompt_tokens"] / merged["token_count"]
    else:
        merged["avg_prompt_tokens"] = merged["avg_prompt_tokens"].fillna(
            merged["total_prompt_tokens"] / merged["token_count"]
        )

    if "avg_response_tokens" not in merged.columns:
        merged["avg_response_tokens"] = merged["total_response_tokens"] / merged["token_count"]
    else:
        merged["avg_response_tokens"] = merged["avg_response_tokens"].fillna(
            merged["total_response_tokens"] / merged["token_count"]
        )

    if "avg_total_tokens" not in merged.columns:
        merged["avg_total_tokens"] = merged["total_tokens"] / merged["token_count"]
    else:
        merged["avg_total_tokens"] = merged["avg_total_tokens"].fillna(
            merged["total_tokens"] / merged["token_count"]
        )

    return merged


def summarize_scores(df: pd.DataFrame) -> dict[str, Any]:
    summary: dict[str, Any] = {}

    for metric in PRIMARY_METRICS:
        if metric in df.columns:
            summary[metric] = float(df[metric].mean())

    for column in df.columns:
        if column in PRIMARY_METRICS or column in EXCLUDED_SCORE_COLUMNS:
            continue
        if column not in RENAMED_METRICS and not pd.api.types.is_numeric_dtype(df[column]):
            continue

        output_key = RENAMED_METRICS.get(column, column)
        summary[output_key] = float(pd.to_numeric(df[column], errors="coerce").mean())

    for key in ["avg_inference_time", "avg_total_inference_time", "avg_inference_count"]:
        if key in summary:
            summary[key] = round(summary[key], 4)

    for key in ["avg_prompt_tokens", "avg_response_tokens", "avg_total_tokens"]:
        if key in summary:
            summary[key] = round(summary[key], 4)

    if "token_count" in summary:
        summary["token_count"] = round(summary["token_count"], 4)

    return summary


def summarize_subset(df: pd.DataFrame) -> dict[str, Any]:
    summary = summarize_scores(df)
    summary["query_count"] = int(len(df))
    return summary


def enrich_folder(folder: Path, dataset: dict[str, dict[str, Any]]) -> None:
    scores_path = folder / "scores.json"
    detailed_path = folder / "detailed_results.csv"

    if not scores_path.exists():
        print(f"Skipping {folder}: scores.json not found")
        return
    if not detailed_path.exists():
        print(f"Skipping {folder}: detailed_results.csv not found")
        return

    with detailed_path.open("r", encoding="utf-8") as handle:
        detailed_df = pd.read_csv(handle)

    detailed_df = add_is_multi_table_column(detailed_df, dataset)
    detailed_df = attach_token_metrics(detailed_df, folder, dataset)

    if "text_model" in detailed_df.columns:
        detailed_df["text_model"] = detailed_df["text_model"].fillna("")
    if "vision_model" in detailed_df.columns:
        detailed_df["vision_model"] = detailed_df["vision_model"].fillna("")

    overall_scores = summarize_scores(detailed_df)
    single_scores = summarize_subset(detailed_df[~detailed_df["is_multi_table"]])
    multi_scores = summarize_subset(detailed_df[detailed_df["is_multi_table"]])

    with scores_path.open("r", encoding="utf-8") as handle:
        scores_data = json.load(handle)

    scores_data["scores"] = overall_scores
    scores_data["single_table_scores"] = single_scores
    scores_data["multi_table_scores"] = multi_scores

    with scores_path.open("w", encoding="utf-8") as handle:
        json.dump(scores_data, handle, indent=2)

    print(f"Updated {scores_path}")


def main() -> int:
    folders = [Path(arg) for arg in sys.argv[1:]] if len(sys.argv) > 1 else TARGET_FOLDERS

    if not folders:
        print("No folders configured. Edit TARGET_FOLDERS or pass folder paths on the command line.")
        return 1

    if not DATASET_PATH.exists():
        print(f"Dataset file not found: {DATASET_PATH}")
        return 1

    dataset = load_dataset(DATASET_PATH)

    for folder in folders:
        resolved_folder = folder if folder.is_absolute() else Path.cwd() / folder
        enrich_folder(resolved_folder, dataset)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())