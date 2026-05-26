"""Build a 200-query benchmark doc-id parquet for KNN/PB2 evaluation.

This mirrors the RAG-based notebook selection logic:
- split PAC test topics by whether the query topic has citations,
- select the first N topics from each group after sorting by
  (number of qrel candidate docs, topic_id),
- emit the unique relevant candidate canonical doc ids to index.
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

DEFAULT_INDEXING_PARQUET_DIR = "/kaggle/input/datasets/djnhngocduc/indexing-parquet"
DEFAULT_TOPICS_PARQUET_CANDIDATES = [
    f"{DEFAULT_INDEXING_PARQUET_DIR}/pac_test_topics_clean.parquet",
    f"{DEFAULT_INDEXING_PARQUET_DIR}/pac_test_topics.parquet",
]
DEFAULT_QRELS_PARQUET_CANDIDATES = [
    f"{DEFAULT_INDEXING_PARQUET_DIR}/pac_test_qrels_clean.parquet",
    f"{DEFAULT_INDEXING_PARQUET_DIR}/pac_test_qrels.parquet",
]
DEFAULT_WORK_DIR = "/kaggle/working"


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        import pandas as pd

        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value).strip()


def normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    try:
        import pandas as pd

        if isinstance(value, float) and pd.isna(value):
            return []
    except Exception:
        pass
    if hasattr(value, "tolist") and not isinstance(value, (str, bytes)):
        value = value.tolist()
    if isinstance(value, (list, tuple, set)):
        out: List[str] = []
        seen = set()
        for item in value:
            text = normalize_text(item)
            if text and text not in seen:
                seen.add(text)
                out.append(text)
        return out
    text = normalize_text(value)
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        quoted = re.findall(r"'([^']+)'|\"([^\"]+)\"", text)
        values = [a or b for a, b in quoted]
        if values:
            return [item for item in (normalize_text(v) for v in values) if item]
    return [text] if text else []


def canonical_doc_id(value: Any) -> str:
    text = normalize_text(value).upper().replace(" ", "-").replace("_", "-")
    text = re.sub(r"-+", "-", text).strip("-")
    match = re.match(r"^([A-Z]{2})-?(\d+)(?:-[A-Z][0-9A-Z]*)?$", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return text


def normalize_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    text = normalize_text(value).lower()
    return text in {"1", "true", "yes", "y", "on"}


def topic_has_citations(row: Dict[str, Any]) -> bool:
    try:
        citation_count = int(row.get("num_citations", 0) or 0)
    except Exception:
        citation_count = 0
    return citation_count > 0 or bool(normalize_list(row.get("citations")))


def select_topic_subset(topic_rows: List[Dict[str, Any]], quota: int) -> List[Dict[str, Any]]:
    selected_rows = []
    for topic_row in sorted(topic_rows, key=lambda item: (len(item["candidate_ids"]), item["topic_id"])):
        if len(selected_rows) >= quota:
            break
        if topic_row["candidate_ids"]:
            selected_rows.append(topic_row)
    return selected_rows


def write_parquet(df: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False, compression="zstd")
    except Exception:
        df.to_parquet(path, index=False)


def resolve_existing_path(
    explicit_path: str,
    env_name: str,
    default_candidates: Iterable[str],
    label: str,
) -> Path:
    checked: List[str] = []
    raw_env_path = os.getenv(env_name, "").strip()
    for raw_path in [explicit_path, raw_env_path, *default_candidates]:
        raw_path = normalize_text(raw_path)
        if not raw_path or raw_path in checked:
            continue
        checked.append(raw_path)
        path = Path(raw_path)
        if path.exists():
            return path
    raise FileNotFoundError(
        f"Cannot find {label}. Checked these paths:\n" + "\n".join(f"- {path}" for path in checked)
    )


def default_output_path(filename: str) -> str:
    return str(Path(DEFAULT_WORK_DIR) / filename)


def build_benchmark_doc_ids(
    topics_parquet: Path,
    qrels_parquet: Path,
    output_doc_ids_parquet: Path,
    summary_json: Path,
    target_non_citation_query_count: int,
    target_citation_query_count: int,
    output_topics_parquet: Path | None = None,
    output_qrels_parquet: Path | None = None,
) -> Dict[str, Any]:
    import pandas as pd

    if not topics_parquet.exists():
        raise FileNotFoundError(f"Topics parquet not found: {topics_parquet}")
    if not qrels_parquet.exists():
        raise FileNotFoundError(f"Qrels parquet not found: {qrels_parquet}")

    test = pd.read_parquet(topics_parquet)
    qrels = pd.read_parquet(qrels_parquet)
    if "topic_id" not in test.columns and "query_id" in test.columns:
        test = test.rename(columns={"query_id": "topic_id"})
    if "topic_id" not in qrels.columns and "query_id" in qrels.columns:
        qrels = qrels.rename(columns={"query_id": "topic_id"})
    if "candidate_doc_id" not in qrels.columns:
        for candidate_col in ["doc_id", "document_id", "patent_id"]:
            if candidate_col in qrels.columns:
                qrels = qrels.rename(columns={candidate_col: "candidate_doc_id"})
                break
    if "topic_id" not in test.columns:
        raise ValueError(f"Topics parquet must contain topic_id: {topics_parquet}")
    if "topic_id" not in qrels.columns or "candidate_doc_id" not in qrels.columns:
        raise ValueError(f"Qrels parquet must contain topic_id and candidate_doc_id: {qrels_parquet}")

    test_clean = test.copy()
    if "full_text_word_count" in test_clean.columns:
        test_clean = test_clean[test_clean["full_text_word_count"].fillna(0).astype(int) > 0].copy()

    test_clean["topic_id"] = test_clean["topic_id"].astype(str).str.strip()
    qrels["topic_id"] = qrels["topic_id"].astype(str).str.strip()
    qrels["candidate_doc_id"] = qrels["candidate_doc_id"].astype(str).str.strip()

    if "has_citations" not in test_clean.columns:
        test_clean["has_citations"] = test_clean.apply(lambda row: topic_has_citations(row.to_dict()), axis=1)
    else:
        test_clean["has_citations"] = test_clean["has_citations"].fillna(False).map(normalize_bool)
    test_clean["citation_group"] = test_clean["has_citations"].map(
        {True: "with_citation", False: "without_citation"}
    )

    valid_topics = set(test_clean["topic_id"])
    qrels_clean = qrels[qrels["topic_id"].isin(valid_topics)].copy()
    if "candidate_canonical_doc_id" not in qrels_clean.columns:
        qrels_clean["candidate_canonical_doc_id"] = qrels_clean["candidate_doc_id"].map(canonical_doc_id)
    else:
        qrels_clean["candidate_canonical_doc_id"] = qrels_clean.apply(
            lambda row: canonical_doc_id(row.get("candidate_canonical_doc_id"))
            or canonical_doc_id(row.get("candidate_doc_id")),
            axis=1,
        )
    qrels_clean = qrels_clean[qrels_clean["candidate_canonical_doc_id"] != ""].copy()

    topic_meta_cols = ["topic_id", "has_citations"]
    if "num_citations" in test_clean.columns:
        topic_meta_cols.append("num_citations")
    for column in ["has_citations", "num_citations"]:
        if column in qrels_clean.columns:
            qrels_clean = qrels_clean.drop(columns=[column])
    qrels_clean = qrels_clean.merge(test_clean[topic_meta_cols], on="topic_id", how="left")
    qrels_clean["has_citations"] = qrels_clean["has_citations"].fillna(False).astype(bool)
    if "requires_kind_code_expansion" not in qrels_clean.columns:
        qrels_clean["requires_kind_code_expansion"] = qrels_clean["has_citations"]
    else:
        qrels_clean["requires_kind_code_expansion"] = (
            qrels_clean["requires_kind_code_expansion"].fillna(False).map(normalize_bool)
            | qrels_clean["has_citations"]
        )

    topic_to_candidate_ids = {
        topic_id: sorted(set(group["candidate_canonical_doc_id"]))
        for topic_id, group in qrels_clean.groupby("topic_id", sort=True)
    }

    topic_rows = []
    for row in test_clean.to_dict(orient="records"):
        topic_id = normalize_text(row.get("topic_id"))
        candidate_ids = topic_to_candidate_ids.get(topic_id, [])
        if not candidate_ids:
            continue
        topic_rows.append(
            {
                "topic_id": topic_id,
                "has_citations": bool(row.get("has_citations")),
                "candidate_ids": candidate_ids,
            }
        )

    non_citation_topic_rows = [row for row in topic_rows if not row["has_citations"]]
    citation_topic_rows = [row for row in topic_rows if row["has_citations"]]
    selected_non_citation_rows = select_topic_subset(
        non_citation_topic_rows,
        target_non_citation_query_count,
    )
    selected_citation_rows = select_topic_subset(
        citation_topic_rows,
        target_citation_query_count,
    )

    if len(selected_non_citation_rows) < target_non_citation_query_count:
        raise ValueError(
            f"Only {len(selected_non_citation_rows)} non-citation topics are available. "
            f"Requested {target_non_citation_query_count}."
        )
    if len(selected_citation_rows) < target_citation_query_count:
        raise ValueError(
            f"Only {len(selected_citation_rows)} citation topics are available. "
            f"Requested {target_citation_query_count}."
        )

    selected_topic_rows = selected_non_citation_rows + selected_citation_rows
    selected_topic_ids = [row["topic_id"] for row in selected_topic_rows]
    selected_topic_set = set(selected_topic_ids)
    selected_qrels = qrels_clean[qrels_clean["topic_id"].isin(selected_topic_set)].copy()
    selected_doc_ids = sorted(set(selected_qrels["candidate_canonical_doc_id"]))

    doc_requires_kind_code = (
        selected_qrels.groupby("candidate_canonical_doc_id")["requires_kind_code_expansion"]
        .max()
        .to_dict()
    )
    doc_topic_ids = (
        selected_qrels.groupby("candidate_canonical_doc_id")["topic_id"]
        .apply(lambda values: ";".join(sorted(set(map(str, values)))))
        .to_dict()
    )
    doc_citation_groups = (
        selected_qrels.groupby("candidate_canonical_doc_id")["has_citations"]
        .apply(
            lambda values: ";".join(
                sorted({"with_citation" if bool(value) else "without_citation" for value in values})
            )
        )
        .to_dict()
    )

    index_doc_ids_df = pd.DataFrame({"doc_id": selected_doc_ids})
    index_doc_ids_df["canonical_doc_id"] = index_doc_ids_df["doc_id"]
    index_doc_ids_df["is_qrel_doc"] = True
    index_doc_ids_df["source_type"] = "qrel"
    index_doc_ids_df["requires_kind_code_expansion"] = (
        index_doc_ids_df["canonical_doc_id"].map(doc_requires_kind_code).fillna(False).astype(bool)
    )
    index_doc_ids_df["query_ids"] = index_doc_ids_df["canonical_doc_id"].map(doc_topic_ids).fillna("")
    index_doc_ids_df["citation_groups"] = (
        index_doc_ids_df["canonical_doc_id"].map(doc_citation_groups).fillna("")
    )
    write_parquet(index_doc_ids_df, output_doc_ids_parquet)

    benchmark_test = test_clean[test_clean["topic_id"].isin(selected_topic_set)].copy()
    sort_cols = [col for col in ["has_citations", "topic_id", "candidate_canonical_doc_id", "line_no"] if col in selected_qrels.columns]
    benchmark_qrels = selected_qrels.sort_values(sort_cols).reset_index(drop=True) if sort_cols else selected_qrels
    benchmark_test = benchmark_test.sort_values(["has_citations", "topic_id"]).reset_index(drop=True)

    if output_topics_parquet:
        write_parquet(benchmark_test, output_topics_parquet)
    if output_qrels_parquet:
        write_parquet(benchmark_qrels, output_qrels_parquet)

    summary = {
        "topics_parquet": str(topics_parquet),
        "qrels_parquet": str(qrels_parquet),
        "target_non_citation_query_count": target_non_citation_query_count,
        "target_citation_query_count": target_citation_query_count,
        "target_benchmark_query_count": target_non_citation_query_count + target_citation_query_count,
        "num_selected_topics": int(len(selected_topic_ids)),
        "num_selected_non_citation_topics": int(len(selected_non_citation_rows)),
        "num_selected_citation_topics": int(len(selected_citation_rows)),
        "num_benchmark_topics_rows": int(len(benchmark_test)),
        "num_benchmark_qrels_rows": int(len(benchmark_qrels)),
        "num_benchmark_unique_relevant_docs": int(len(selected_doc_ids)),
        "num_kind_code_expansion_docs": int(index_doc_ids_df["requires_kind_code_expansion"].sum()),
        "benchmark_doc_ids_path": str(output_doc_ids_parquet),
        "benchmark_topics_path": str(output_topics_parquet) if output_topics_parquet else "",
        "benchmark_qrels_path": str(output_qrels_parquet) if output_qrels_parquet else "",
        "sample_selected_non_citation_topic_ids": [row["topic_id"] for row in selected_non_citation_rows[:10]],
        "sample_selected_citation_topic_ids": [row["topic_id"] for row in selected_citation_rows[:10]],
    }
    summary_json.parent.mkdir(parents=True, exist_ok=True)
    summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build benchmark doc ids from PAC topics/qrels.")
    parser.add_argument(
        "--topics-parquet",
        default="",
        help=(
            "PAC test topics parquet. If omitted, uses BENCHMARK_TOPICS_PATH or "
            "the default Kaggle indexing-parquet pac_test_topics_clean/pac_test_topics files."
        ),
    )
    parser.add_argument(
        "--qrels-parquet",
        default="",
        help=(
            "PAC test qrels parquet. If omitted, uses BENCHMARK_QRELS_PATH or "
            "the default Kaggle indexing-parquet pac_test_qrels_clean/pac_test_qrels files."
        ),
    )
    parser.add_argument("--output-doc-ids-parquet", default="")
    parser.add_argument("--summary-json", default="")
    parser.add_argument("--output-topics-parquet", default="")
    parser.add_argument("--output-qrels-parquet", default="")
    parser.add_argument("--target-non-citation-query-count", type=int, default=100)
    parser.add_argument("--target-citation-query-count", type=int, default=100)
    args = parser.parse_args()

    target_query_count = args.target_non_citation_query_count + args.target_citation_query_count
    topics_parquet = resolve_existing_path(
        args.topics_parquet,
        "BENCHMARK_TOPICS_PATH",
        DEFAULT_TOPICS_PARQUET_CANDIDATES,
        "PAC test topics parquet",
    )
    qrels_parquet = resolve_existing_path(
        args.qrels_parquet,
        "BENCHMARK_QRELS_PATH",
        DEFAULT_QRELS_PARQUET_CANDIDATES,
        "PAC test qrels parquet",
    )
    output_doc_ids_parquet = Path(
        args.output_doc_ids_parquet
        or default_output_path(f"benchmark_{target_query_count}_queries_index_doc_ids.parquet")
    )
    summary_json = Path(
        args.summary_json
        or default_output_path(f"benchmark_{target_query_count}_queries_summary.json")
    )
    output_topics_parquet = Path(
        args.output_topics_parquet
        or default_output_path(f"pac_test_topics_benchmark_{target_query_count}.parquet")
    )
    output_qrels_parquet = Path(
        args.output_qrels_parquet
        or default_output_path(f"pac_test_qrels_benchmark_{target_query_count}.parquet")
    )

    summary = build_benchmark_doc_ids(
        topics_parquet=topics_parquet,
        qrels_parquet=qrels_parquet,
        output_doc_ids_parquet=output_doc_ids_parquet,
        summary_json=summary_json,
        target_non_citation_query_count=args.target_non_citation_query_count,
        target_citation_query_count=args.target_citation_query_count,
        output_topics_parquet=output_topics_parquet,
        output_qrels_parquet=output_qrels_parquet,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
