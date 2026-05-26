"""
Evaluate PB1-PB4 ablation outputs against prior-art qrels.

This script is intentionally separate from the runtime pipeline. The pipeline
produces candidates/evidence/reports; this script scores retrieval rankings and
summarizes internal grounding proxy metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover - optional for json-only use
    pd = None  # type: ignore


DEFAULT_K_VALUES = [5, 10, 20, 50, 100]
VARIANTS = ["pb1", "pb2", "pb3", "pb4"]


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def canonical_doc_id(value: Any) -> str:
    text = normalize_text(value).upper()
    if not text:
        return ""
    text = text.replace(" ", "-").replace("_", "-")
    text = re.sub(r"[^A-Z0-9-]", "", text)
    text = re.sub(r"-+", "-", text).strip("-")
    # CLEF/IP ids often include a kind code. Keep the family-level id for
    # conservative duplicate matching, e.g. EP-123-A1 -> EP-123.
    match = re.match(r"^([A-Z]{2})-?(\d+)", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return text


def read_table(path: Path) -> List[Dict[str, Any]]:
    suffix = path.suffix.lower()
    if suffix == ".parquet":
        if pd is None:
            raise RuntimeError("Reading parquet qrels requires pandas/pyarrow.")
        return pd.read_parquet(path).to_dict(orient="records")
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            return list(csv.DictReader(f, delimiter=delimiter))
    if suffix == ".txt":
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                parts = line.split()
                if len(parts) >= 4:
                    rows.append({"topic_id": parts[0], "doc_id": parts[2], "relevance": parts[3]})
                elif len(parts) >= 2:
                    rows.append({"topic_id": parts[0], "doc_id": parts[1], "relevance": parts[2] if len(parts) >= 3 else 1})
        return rows
    if suffix == ".jsonl":
        rows = []
        with path.open("r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
        return rows
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(data, list):
            return [row for row in data if isinstance(row, dict)]
        if isinstance(data, dict):
            rows = []
            for topic_id, docs in data.items():
                if isinstance(docs, dict):
                    for doc_id, rel in docs.items():
                        rows.append({"topic_id": topic_id, "doc_id": doc_id, "relevance": rel})
                elif isinstance(docs, list):
                    for item in docs:
                        if isinstance(item, dict):
                            rows.append({"topic_id": topic_id, **item})
                        else:
                            rows.append({"topic_id": topic_id, "doc_id": item, "relevance": 1})
            return rows
    raise RuntimeError(f"Unsupported table format: {path}")


def load_qrels(path: Path) -> Dict[str, Dict[str, float]]:
    rows = read_table(path)
    qrels: Dict[str, Dict[str, float]] = {}
    topic_keys = ["topic_id", "query_id", "qid", "topic", "id"]
    doc_keys = [
        "candidate_canonical_doc_id",
        "candidate_doc_id",
        "doc_id",
        "patent_id",
        "document_id",
        "candidate_id",
    ]
    rel_keys = ["relevance", "rel", "label", "score"]

    for row in rows:
        topic_id = next((normalize_text(row.get(key)) for key in topic_keys if normalize_text(row.get(key))), "")
        doc_id = next((normalize_text(row.get(key)) for key in doc_keys if normalize_text(row.get(key))), "")
        if not topic_id or not doc_id:
            continue
        rel_raw = next((row.get(key) for key in rel_keys if row.get(key) is not None), 1)
        try:
            rel = float(rel_raw)
        except Exception:
            rel = 1.0 if normalize_text(rel_raw).lower() not in {"0", "false", "no"} else 0.0
        if rel <= 0:
            continue
        canonical_topic = canonical_doc_id(topic_id)
        canonical_doc = canonical_doc_id(doc_id)
        qrels.setdefault(topic_id, {})[canonical_doc] = rel
        if canonical_topic and canonical_topic != topic_id:
            qrels.setdefault(canonical_topic, {})[canonical_doc] = rel
    return qrels


def infer_topic_id_from_variants(variants: Dict[str, Any], fallback: str = "query_1") -> str:
    for state in variants.values():
        if not isinstance(state, dict):
            continue
        metadata = state.get("input_metadata") or {}
        if isinstance(metadata, dict):
            for key in ["topic_id", "query_id", "qid", "doc_id", "patent_id"]:
                value = normalize_text(metadata.get(key))
                if value:
                    return value
        for key in ["topic_id", "query_id", "qid"]:
            value = normalize_text(state.get(key))
            if value:
                return value
    return fallback


def load_results(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """Load either one-query or multi-query ablation output.

    Supported shapes:
    - {pb1: state, pb2: state, ...}
    - {topic_id: {pb1: state, pb2: state, ...}, ...}
    - [{topic_id: "...", pb1: state, ...}, ...]
    """

    data = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(data, list):
        out: Dict[str, Dict[str, Dict[str, Any]]] = {}
        for idx, item in enumerate(data, start=1):
            if not isinstance(item, dict):
                continue
            topic_id = normalize_text(item.get("topic_id") or item.get("query_id") or idx)
            out[topic_id] = {variant: item[variant] for variant in VARIANTS if isinstance(item.get(variant), dict)}
        return out

    if not isinstance(data, dict):
        raise RuntimeError("Results JSON must be an object or list.")

    if any(variant in data for variant in VARIANTS):
        variants = {variant: data[variant] for variant in VARIANTS if isinstance(data.get(variant), dict)}
        topic_id = normalize_text(data.get("topic_id") or data.get("query_id")) or infer_topic_id_from_variants(variants)
        return {topic_id: variants}

    out = {}
    for topic_id, value in data.items():
        if isinstance(value, dict):
            out[normalize_text(topic_id)] = {
                variant: value[variant] for variant in VARIANTS if isinstance(value.get(variant), dict)
            }
    return out


def state_rankings(state: Dict[str, Any]) -> Tuple[List[str], List[str]]:
    """Return raw and metric rankings.

    The old retrieval notebook scored the full retrieval ranking after
    publication-kind/canonical deduplication. It did not score the compact
    generation context. Prefer `candidates` over `screened_candidates` here so
    Recall@50/100 is not silently capped by MULTIAGENT_CANDIDATE_TOP_K.
    """

    rows = state.get("candidates") or state.get("screened_candidates") or []
    raw_ranking: List[str] = []
    canonical_ranking: List[str] = []
    if isinstance(rows, list):
        for item in rows:
            if isinstance(item, dict):
                doc_id = item.get("doc_id") or item.get("patent_id") or item.get("id")
            else:
                doc_id = item
            raw_doc_id = normalize_text(doc_id)
            canonical_id = canonical_doc_id(raw_doc_id)
            if raw_doc_id:
                raw_ranking.append(raw_doc_id)
            if canonical_id and canonical_id not in canonical_ranking:
                canonical_ranking.append(canonical_id)
    if canonical_ranking:
        return raw_ranking, canonical_ranking

    analysis = state.get("analysis") or {}
    rows = analysis.get("ranked_prior_art") or []
    if isinstance(rows, list):
        for item in rows:
            if isinstance(item, dict):
                raw_doc_id = normalize_text(item.get("patent_id") or item.get("doc_id"))
                canonical_id = canonical_doc_id(raw_doc_id)
                if raw_doc_id:
                    raw_ranking.append(raw_doc_id)
                if canonical_id and canonical_id not in canonical_ranking:
                    canonical_ranking.append(canonical_id)
    return raw_ranking, canonical_ranking


def state_ranking(state: Dict[str, Any]) -> List[str]:
    return state_rankings(state)[1]


def dcg(relevances: Iterable[float]) -> float:
    total = 0.0
    for idx, rel in enumerate(relevances, start=1):
        total += (2.0 ** float(rel) - 1.0) / math.log2(idx + 1.0)
    return total


def score_ranking(ranking: List[str], relevant: Dict[str, float], k_values: List[int]) -> Dict[str, float]:
    out: Dict[str, float] = {}
    if not relevant:
        for k in k_values:
            out[f"recall@{k}"] = 0.0
            out[f"mrr@{k}"] = 0.0
            out[f"map@{k}"] = 0.0
        return out

    relevant_ids = set(relevant)

    for k in k_values:
        top = ranking[:k]
        hits = set(top) & relevant_ids
        recall = len(hits) / len(relevant_ids) if relevant_ids else 0.0
        reciprocal_rank = 0.0
        for idx, doc_id in enumerate(top, start=1):
            if doc_id in relevant_ids:
                reciprocal_rank = 1.0 / idx
                break
        hit_count = 0
        precision_sum = 0.0
        seen = set()
        for idx, doc_id in enumerate(top, start=1):
            if doc_id in seen:
                continue
            seen.add(doc_id)
            if doc_id in relevant_ids:
                hit_count += 1
                precision_sum += hit_count / float(idx)
        denom = min(len(relevant_ids), int(k))
        average_precision = 0.0 if denom <= 0 else precision_sum / float(denom)
        out[f"recall@{k}"] = recall
        out[f"mrr@{k}"] = reciprocal_rank
        out[f"map@{k}"] = average_precision
    return out


def mean_dict(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 6)
        for key in keys
    }


def evaluate(results: Dict[str, Dict[str, Dict[str, Any]]], qrels: Dict[str, Dict[str, float]], k_values: List[int]) -> Dict[str, Any]:
    per_topic: Dict[str, Dict[str, Any]] = {}
    pooled: Dict[str, List[Dict[str, float]]] = {variant: [] for variant in VARIANTS}
    pooled_counts: Dict[str, List[Dict[str, float]]] = {variant: [] for variant in VARIANTS}

    for topic_id, variants in results.items():
        topic_rels = qrels.get(topic_id) or qrels.get(canonical_doc_id(topic_id)) or {}
        per_topic[topic_id] = {}
        for variant, state in variants.items():
            raw_ranking, ranking = state_rankings(state)
            metrics = score_ranking(ranking, topic_rels, k_values)
            proxy = state.get("proxy_metrics", {}) if isinstance(state, dict) else {}
            per_topic[topic_id][variant] = {
                "ranking_size": len(ranking),
                "num_retrieved": len(raw_ranking),
                "num_retrieved_canonical": len(ranking),
                "num_duplicate_variants_removed": max(0, len(raw_ranking) - len(ranking)),
                "num_relevant": len(topic_rels),
                "retrieval_metrics": {key: round(value, 6) for key, value in metrics.items()},
                "proxy_metrics": proxy,
            }
            pooled.setdefault(variant, []).append(metrics)
            pooled_counts.setdefault(variant, []).append(
                {
                    "avg_relevant_per_topic": float(len(topic_rels)),
                    "avg_retrieved_per_topic": float(len(raw_ranking)),
                    "avg_retrieved_canonical_per_topic": float(len(ranking)),
                    "avg_duplicate_variants_removed": float(max(0, len(raw_ranking) - len(ranking))),
                }
            )

    macro = {}
    for variant, rows in pooled.items():
        if not rows:
            continue
        macro[variant] = {
            **mean_dict(pooled_counts.get(variant, [])),
            **mean_dict(rows),
        }
    return {"macro": macro, "per_topic": per_topic}


def parse_k_values(value: str) -> List[int]:
    if not value:
        return DEFAULT_K_VALUES
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PB1-PB4 ablation outputs.")
    parser.add_argument("--results-json", required=True, help="JSON from --run-ablation or a batch ablation run.")
    parser.add_argument("--qrels", required=True, help="Qrels file: parquet, csv, tsv, json, or jsonl.")
    parser.add_argument("--k", default="5,10,20,50,100", help="Comma-separated cutoffs.")
    parser.add_argument("--output-json", default="", help="Optional output path.")
    args = parser.parse_args()

    results = load_results(Path(args.results_json))
    qrels = load_qrels(Path(args.qrels))
    report = evaluate(results, qrels, parse_k_values(args.k))
    rendered = json.dumps(report, ensure_ascii=False, indent=2)
    print(rendered)
    if args.output_json:
        Path(args.output_json).write_text(rendered, encoding="utf-8")


if __name__ == "__main__":
    main()
