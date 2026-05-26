"""
Run the multi-agent pipeline on the benchmark_200 topic set.

The runner keeps retrieval fixed to Elasticsearch KNN and varies only the
runtime LLM model used by Agent 1 for PB2. It writes per-topic outputs plus
macro retrieval metrics using recall, MRR, and MAP.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

try:
    import pandas as pd
except Exception:  # pragma: no cover - runtime dependency
    pd = None  # type: ignore

try:
    from patent_multiagent_langgraph import (
        DEFAULT_PIPELINE_VARIANT,
        DEFAULT_RETRIEVAL_BACKEND,
        PipelineConfig,
        limit_words,
        normalize_list,
        normalize_pipeline_variant,
        normalize_text,
        run_query_understanding_stage,
        run_retrieval_stage,
        run_pipeline,
        validate_runtime_environment,
    )
    from evaluate_ablation import canonical_doc_id, load_qrels, score_ranking
except ImportError:  # pragma: no cover - package style import
    from .patent_multiagent_langgraph import (
        DEFAULT_PIPELINE_VARIANT,
        DEFAULT_RETRIEVAL_BACKEND,
        PipelineConfig,
        limit_words,
        normalize_list,
        normalize_pipeline_variant,
        normalize_text,
        run_query_understanding_stage,
        run_retrieval_stage,
        run_pipeline,
        validate_runtime_environment,
    )
    from .evaluate_ablation import canonical_doc_id, load_qrels, score_ranking


GROQ_DIVERSE_COMMON_LIMIT_MODEL_IDS = [
    "openai/gpt-oss-120b",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
]
GROQ_GPT_OSS_STRICT_MODEL_IDS = [
    "openai/gpt-oss-120b",
    "openai/gpt-oss-20b",
]
GROQ_GPT_OSS_120B_MODEL_IDS = ["openai/gpt-oss-120b"]
MODEL_PRESETS = {
    "groq-free-diverse-common": GROQ_DIVERSE_COMMON_LIMIT_MODEL_IDS,
    "groq-gpt-oss-strict": GROQ_GPT_OSS_STRICT_MODEL_IDS,
    "groq-gpt-oss-120b": GROQ_GPT_OSS_120B_MODEL_IDS,
    "groq-free": GROQ_DIVERSE_COMMON_LIMIT_MODEL_IDS,
    "groq-free-balanced": GROQ_DIVERSE_COMMON_LIMIT_MODEL_IDS,
    "groq-free-diverse": GROQ_DIVERSE_COMMON_LIMIT_MODEL_IDS,
}
DEFAULT_MODEL_IDS = GROQ_DIVERSE_COMMON_LIMIT_MODEL_IDS
DEFAULT_K_VALUES = [5, 10, 20]
GROQ_FREE_COMMON_SLEEP_SECONDS = 30.0
GROQ_FREE_BALANCED_COMMON_SLEEP_SECONDS = 30.0
GROQ_FREE_8K_COMMON_SLEEP_SECONDS = 20.0
GROQ_FREE_LARGE_COMMON_SLEEP_SECONDS = 20.0
GROQ_FREE_DIVERSE_COMMON_SLEEP_SECONDS = 30.0

TOPIC_ID_KEYS = [
    "topic_id",
    "query_id",
    "qid",
    "topic",
    "id",
    "doc_id",
    "patent_id",
    "query_doc_id",
]

TEXT_KEYS = [
    "query_text",
    "retrieval_text",
    "text",
    "title",
    "abstract",
    "claims",
    "description",
    "full_text",
]

DROP_STATE_KEYS = {
    "candidate_docs",
    "final_report",
}


def env_path(name: str) -> str:
    return normalize_text(os.getenv(name, ""))


def candidate_paths(kind: str) -> List[str]:
    if kind == "topics":
        return [
            env_path("MULTIAGENT_BENCHMARK_TOPICS_PATH"),
            env_path("BENCHMARK_TOPICS_PATH"),
            "/kaggle/input/datasets/djnhngocduc/indexing-parquet/pac_test_topics_benchmark_200.parquet",
            "/kaggle/working/clefip2011_pac_topics/processed/pac_test_topics_benchmark_200.parquet",
            "/kaggle/input/datasets/djnhngocduc/indexing-parquet/pac_test_topics_clean.parquet",
        ]
    return [
        env_path("MULTIAGENT_BENCHMARK_QRELS_PATH"),
        env_path("BENCHMARK_QRELS_PATH"),
        "/kaggle/input/datasets/djnhngocduc/indexing-parquet/pac_test_qrels_benchmark_200.parquet",
        "/kaggle/working/clefip2011_pac_topics/processed/pac_test_qrels_benchmark_200.parquet",
        "/kaggle/input/datasets/djnhngocduc/indexing-parquet/pac_test_qrels_clean.parquet",
    ]


def resolve_existing_path(explicit: str, candidates: List[str], label: str) -> Path:
    if normalize_text(explicit):
        path = Path(explicit)
        if not path.exists():
            raise FileNotFoundError(f"{label} path does not exist: {path}")
        return path
    for item in candidates:
        if not normalize_text(item):
            continue
        path = Path(item)
        if path.exists():
            return path
    checked = "\n".join(f"- {item}" for item in candidates if normalize_text(item))
    raise FileNotFoundError(f"Could not find {label}. Checked:\n{checked}")


def parse_csv(value: str) -> List[str]:
    return [part.strip() for part in value.split(",") if part.strip()]


def resolve_models(value: str, preset: str) -> List[str]:
    explicit = parse_csv(value)
    if explicit:
        selected = explicit
    else:
        key = normalize_text(preset).lower() or "groq-free-diverse-common"
        if key not in MODEL_PRESETS:
            raise ValueError(f"Unknown model preset {preset!r}. Choose one of: {', '.join(sorted(MODEL_PRESETS))}")
        selected = list(MODEL_PRESETS[key])
    return selected


def groq_rate_limit_note(model_preset: str, provider: str) -> str:
    if normalize_text(provider).lower() != "groq":
        return ""
    if normalize_text(model_preset).lower() in MODEL_PRESETS:
        return (
            "Groq benchmark compares GPT-OSS 120B, Qwen3 32B, and Llama 3.3 70B. "
            "Their free-plan limits are not identical, so the run should use the common "
            "conservative denominator: 30 RPM, 1K RPD, and 6K TPM. "
            "The lowest daily token cap is Llama 3.3 70B at 100K TPD, so split full "
            "benchmark runs with MA_LIMIT/resume if the account hits daily quota."
        )
    return ""


def parse_k_values(value: str) -> List[int]:
    items = []
    for part in parse_csv(value):
        number = int(part)
        if number <= 0:
            raise ValueError("Metric cutoffs must be positive.")
        items.append(number)
    return items or DEFAULT_K_VALUES


def jsonable(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return None
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(item) for item in value]
    if pd is not None:
        try:
            missing = pd.isna(value)
            if isinstance(missing, bool) and missing:
                return None
            if hasattr(missing, "item") and getattr(missing, "shape", ()) == ():
                if bool(missing.item()):
                    return None
        except Exception:
            pass
    if hasattr(value, "item"):
        try:
            return jsonable(value.item())
        except Exception:
            pass
    return str(value)


def compact_state(state: Dict[str, Any], save_full_state: bool = False) -> Dict[str, Any]:
    if save_full_state:
        return jsonable(state)
    compact = {}
    for key, value in state.items():
        if key in DROP_STATE_KEYS:
            continue
        compact[key] = value
    analysis = compact.get("analysis")
    if isinstance(analysis, dict) and normalize_text(analysis.get("final_report_markdown")):
        analysis = dict(analysis)
        analysis["final_report_markdown"] = limit_words(analysis["final_report_markdown"], 180)
        compact["analysis"] = analysis
    context = compact.get("retrieval_context")
    if isinstance(context, dict) and normalize_text(context.get("query_text")):
        context = dict(context)
        context["query_text"] = limit_words(context["query_text"], 120)
        compact["retrieval_context"] = context
    return jsonable(compact)


def load_topics(path: Path) -> List[Dict[str, Any]]:
    if path.suffix.lower() != ".parquet":
        raise RuntimeError("Benchmark topics should be a parquet file.")
    if pd is None:
        raise RuntimeError("Reading benchmark parquet requires pandas and pyarrow.")
    return pd.read_parquet(path).to_dict(orient="records")


def topic_id_from_row(row: Dict[str, Any], qrels: Dict[str, Dict[str, float]]) -> str:
    for key in TOPIC_ID_KEYS:
        value = normalize_text(row.get(key))
        if value:
            return value
    for value in row.values():
        text = normalize_text(value)
        if text and (text in qrels or canonical_doc_id(text) in qrels):
            return text
    return ""


def row_has_qrels(topic_id: str, qrels: Dict[str, Dict[str, float]]) -> bool:
    return bool(qrels.get(topic_id) or qrels.get(canonical_doc_id(topic_id)))


def row_text(row: Dict[str, Any], text_keys: Iterable[str]) -> str:
    parts: List[str] = []
    seen = set()
    for key in text_keys:
        value = row.get(key)
        if isinstance(value, (list, tuple)):
            text = " ".join(normalize_text(item) for item in value if normalize_text(item))
        else:
            text = normalize_text(value)
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        parts.append(f"{key}: {text}")
    if not parts:
        for key, value in row.items():
            if key in TOPIC_ID_KEYS:
                continue
            text = normalize_text(value)
            if text and text.lower() not in seen:
                seen.add(text.lower())
                parts.append(f"{key}: {text}")
    return limit_words("\n".join(parts), 1800)


def selected_topics(
    rows: List[Dict[str, Any]],
    qrels: Dict[str, Dict[str, float]],
    text_keys: List[str],
    limit: int,
    allow_missing_qrels: bool,
) -> List[Dict[str, Any]]:
    out = []
    seen = set()
    for row in rows:
        topic_id = topic_id_from_row(row, qrels)
        if not topic_id:
            continue
        if topic_id in seen:
            continue
        if not allow_missing_qrels and not row_has_qrels(topic_id, qrels):
            continue
        text = row_text(row, text_keys)
        if not normalize_text(text):
            continue
        metadata = jsonable(row)
        if isinstance(metadata, dict):
            metadata.setdefault("topic_id", topic_id)
            metadata.setdefault("canonical_topic_id", canonical_doc_id(topic_id))
        out.append({"topic_id": topic_id, "input_text": text, "input_metadata": metadata})
        seen.add(topic_id)
        if limit > 0 and len(out) >= limit:
            break
    return out


def state_ranking_for_metrics(state: Dict[str, Any], source: str) -> Tuple[List[str], List[str]]:
    if source == "final":
        analysis = state.get("analysis") or {}
        rows = analysis.get("ranked_prior_art") if isinstance(analysis, dict) else []
    elif source == "screened":
        rows = state.get("screened_candidates") or []
    else:
        rows = state.get("candidates") or []

    raw: List[str] = []
    canonical: List[str] = []
    if isinstance(rows, list):
        for item in rows:
            if isinstance(item, dict):
                doc_id = item.get("patent_id") or item.get("doc_id") or item.get("id")
            else:
                doc_id = item
            raw_doc_id = normalize_text(doc_id)
            canonical_id = canonical_doc_id(raw_doc_id)
            if raw_doc_id:
                raw.append(raw_doc_id)
            if canonical_id and canonical_id not in canonical:
                canonical.append(canonical_id)
    if canonical:
        return raw, canonical
    if source != "candidates":
        return state_ranking_for_metrics(state, "candidates")
    return raw, canonical


def mean_metrics(rows: List[Dict[str, float]]) -> Dict[str, float]:
    if not rows:
        return {}
    keys = sorted({key for row in rows for key in row})
    return {
        key: round(sum(float(row.get(key, 0.0)) for row in rows) / len(rows), 6)
        for key in keys
    }


def evaluate_model_outputs(
    outputs: Dict[str, Dict[str, Dict[str, Any]]],
    qrels: Dict[str, Dict[str, float]],
    k_values: List[int],
    ranking_source: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    summary_rows: List[Dict[str, Any]] = []
    per_topic_rows: List[Dict[str, Any]] = []

    for model_id, topic_states in outputs.items():
        metric_rows: List[Dict[str, float]] = []
        count_rows: List[Dict[str, float]] = []
        error_count = 0
        for topic_id, state in topic_states.items():
            if not isinstance(state, dict) or state.get("error"):
                error_count += 1
                continue
            topic_rels = qrels.get(topic_id) or qrels.get(canonical_doc_id(topic_id)) or {}
            raw_ranking, ranking = state_ranking_for_metrics(state, ranking_source)
            metrics = score_ranking(ranking, topic_rels, k_values)
            metric_rows.append(metrics)
            count_rows.append(
                {
                    "avg_relevant_per_topic": float(len(topic_rels)),
                    "avg_retrieved_per_topic": float(len(raw_ranking)),
                    "avg_retrieved_canonical_per_topic": float(len(ranking)),
                    "avg_duplicate_variants_removed": float(max(0, len(raw_ranking) - len(ranking))),
                }
            )
            per_topic_rows.append(
                {
                    "model": model_id,
                    "topic_id": topic_id,
                    "num_relevant": len(topic_rels),
                    "num_retrieved": len(raw_ranking),
                    "num_retrieved_canonical": len(ranking),
                    **{key: round(value, 6) for key, value in metrics.items()},
                }
            )
        summary_rows.append(
            {
                "model": model_id,
                "num_topics": len(metric_rows),
                "num_errors": error_count,
                "ranking_source": ranking_source,
                **mean_metrics(count_rows),
                **mean_metrics(metric_rows),
            }
        )

    sort_key = f"map@{max(k_values)}"
    summary_rows.sort(
        key=lambda row: (
            -float(row.get(sort_key, 0.0)),
            -float(row.get(f"mrr@{max(k_values)}", 0.0)),
            -float(row.get(f"recall@{max(k_values)}", 0.0)),
            normalize_text(row.get("model")),
        )
    )
    return summary_rows, per_topic_rows


def save_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), ensure_ascii=False, indent=2), encoding="utf-8")


def load_resume(path: Path) -> Dict[str, Dict[str, Dict[str, Any]]]:
    if not path.exists():
        return {}
    data = json.loads(path.read_text(encoding="utf-8-sig"))
    models = data.get("model_outputs") if isinstance(data, dict) else {}
    return models if isinstance(models, dict) else {}


def is_completed_stage_state(state: Any, stage: str) -> bool:
    if not isinstance(state, dict) or state.get("error"):
        return False
    saved_stage = normalize_text(state.get("stage"))
    return saved_stage == stage


def clear_runtime_memory() -> None:
    try:
        import gc

        gc.collect()
    except Exception:
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def safe_slug(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip())
    return text.strip("_") or "model"


def make_config(args: argparse.Namespace, model_id: str) -> PipelineConfig:
    llm_max_tokens = int(args.max_output_tokens)
    if args.stage == "agent1":
        llm_max_tokens = min(llm_max_tokens, int(os.getenv("MULTIAGENT_AGENT1_MAX_OUTPUT_TOKENS", "900")))
    return PipelineConfig(
        pipeline_variant=normalize_pipeline_variant(args.variant),
        candidate_screen_top_k=args.candidate_top_k,
        evidence_top_docs=args.evidence_top_docs,
        max_iterations=args.max_iterations,
        min_evidence_docs=args.min_evidence_docs,
        llm_provider=args.provider,
        llm_model=model_id,
        llm_temperature=args.temperature,
        llm_max_tokens=llm_max_tokens,
        llm_strict=not args.allow_llm_fallback,
        retrieval_backend=DEFAULT_RETRIEVAL_BACKEND,
        retrieval_strict=not args.allow_retrieval_fallback,
        knn_top_k=args.retrieval_top_k,
        knn_num_candidates=args.knn_num_candidates,
        knn_chunk_fetch_multiplier=args.knn_chunk_fetch_multiplier,
        knn_variant_fetch_multiplier=args.knn_variant_fetch_multiplier,
        knn_max_fetch_size=args.knn_max_fetch_size,
        knn_score_agg=args.knn_score_agg,
        knn_rrf_k=args.knn_rrf_k,
    )


def run_one_topic(args: argparse.Namespace, model_id: str, topic: Dict[str, Any]) -> Dict[str, Any]:
    config = make_config(args, model_id)
    state = run_pipeline(
        topic["input_text"],
        input_metadata=topic["input_metadata"],
        config=config,
        prefer_langgraph=args.prefer_langgraph,
        variant=args.variant,
    )
    return compact_state(state, save_full_state=args.save_full_state)


def run_agent1_topic(args: argparse.Namespace, model_id: str, topic: Dict[str, Any]) -> Dict[str, Any]:
    config = make_config(args, model_id)
    state = run_query_understanding_stage(
        topic["input_text"],
        input_metadata=topic["input_metadata"],
        config=config,
        variant=args.variant,
    )
    return compact_state(state, save_full_state=args.save_full_state)


def run_retrieval_topic(args: argparse.Namespace, model_id: str, agent_state: Dict[str, Any]) -> Dict[str, Any]:
    config = make_config(args, model_id)
    state = run_retrieval_stage(
        agent_state,
        config=config,
        variant=args.variant,
        include_screening=True,
        include_report=True,
    )
    return compact_state(state, save_full_state=args.save_full_state)


def write_tables(out_dir: Path, summary_rows: List[Dict[str, Any]], per_topic_rows: List[Dict[str, Any]]) -> None:
    if pd is None:
        print("[WARN] pandas is not installed; skipped CSV/parquet table outputs.")
        return
    pd.DataFrame(summary_rows).to_csv(out_dir / "model_ablation_summary.csv", index=False)
    pd.DataFrame(per_topic_rows).to_csv(out_dir / "model_ablation_per_topic.csv", index=False)
    try:
        if summary_rows:
            pd.DataFrame(summary_rows).to_parquet(out_dir / "model_ablation_summary.parquet", index=False)
        if per_topic_rows:
            pd.DataFrame(per_topic_rows).to_parquet(out_dir / "model_ablation_per_topic.parquet", index=False)
    except Exception as exc:
        print(f"[WARN] skipped parquet table outputs: {exc}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run benchmark_200 model ablation for the multi-agent pipeline.")
    parser.add_argument("--topics", default="", help="Benchmark topics parquet. Defaults to BENCHMARK_TOPICS_PATH.")
    parser.add_argument("--qrels", default="", help="Benchmark qrels parquet. Defaults to BENCHMARK_QRELS_PATH.")
    parser.add_argument("--output-dir", default="/kaggle/working/multiagent_model_ablation", help="Output directory.")
    parser.add_argument(
        "--models",
        default=os.getenv("MULTIAGENT_BENCHMARK_MODELS", ""),
        help="Comma-separated model IDs. If empty, --model-preset is used.",
    )
    parser.add_argument(
        "--model-preset",
        default=os.getenv("MULTIAGENT_MODEL_PRESET", "groq-free-diverse-common"),
        choices=sorted(MODEL_PRESETS),
        help="Model preset used when --models is empty.",
    )
    parser.add_argument("--provider", default=os.getenv("MULTIAGENT_LLM_PROVIDER", "groq"), help="LLM provider.")
    parser.add_argument("--variant", default=DEFAULT_PIPELINE_VARIANT, choices=["pb1", "pb2", "pb3", "pb4"])
    parser.add_argument("--stage", default="all", choices=["all", "agent1", "retrieval"], help="Run full pipeline, Agent 1 only, or retrieval from saved Agent 1 output.")
    parser.add_argument("--agent1-input", default="", help="Agent 1 JSON to read for --stage retrieval. Defaults to output-dir/model_ablation_agent1_outputs.json.")
    parser.add_argument("--agent1-output", default="", help="Agent 1 JSON to write for --stage agent1. Defaults to output-dir/model_ablation_agent1_outputs.json.")
    parser.add_argument("--k", default="5,10,20", help="Comma-separated metric cutoffs.")
    parser.add_argument("--ranking-source", default="final", choices=["final", "screened", "candidates"])
    parser.add_argument("--limit", type=int, default=0, help="Optional number of topics for smoke tests. 0 means all.")
    parser.add_argument("--text-columns", default=",".join(TEXT_KEYS), help="Comma-separated topic text columns.")
    parser.add_argument("--allow-missing-qrels", action="store_true", help="Do not skip topics without qrels.")
    parser.add_argument("--resume", action="store_true", help="Reuse existing model outputs in output-dir.")
    parser.add_argument("--save-full-state", action="store_true", help="Keep candidate_docs and full reports in JSON.")
    parser.add_argument("--prefer-langgraph", action="store_true", help="Use LangGraph instead of the linear runner.")
    parser.add_argument("--allow-llm-fallback", action="store_true", help="Continue with heuristic fallback if LLM calls fail.")
    parser.add_argument("--continue-on-error", action="store_true", help="Record per-topic errors and continue without heuristic fallback.")
    parser.add_argument("--allow-retrieval-fallback", action="store_true", help="Continue if retrieval fails for a topic.")
    parser.add_argument("--retrieval-top-k", type=int, default=int(os.getenv("MULTIAGENT_BENCHMARK_RETRIEVAL_TOP_K", "300")))
    parser.add_argument("--candidate-top-k", type=int, default=int(os.getenv("MULTIAGENT_CANDIDATE_TOP_K", "20")))
    parser.add_argument("--evidence-top-docs", type=int, default=int(os.getenv("MULTIAGENT_EVIDENCE_TOP_DOCS", "3")))
    parser.add_argument("--max-iterations", type=int, default=int(os.getenv("MULTIAGENT_MAX_ITERATIONS", "1")))
    parser.add_argument("--min-evidence-docs", type=int, default=int(os.getenv("MULTIAGENT_MIN_EVIDENCE_DOCS", "2")))
    parser.add_argument("--knn-num-candidates", type=int, default=int(os.getenv("MULTIAGENT_BENCHMARK_KNN_NUM_CANDIDATES", "1500")))
    parser.add_argument("--knn-chunk-fetch-multiplier", type=int, default=int(os.getenv("MULTIAGENT_KNN_CHUNK_FETCH_MULTIPLIER", "12")))
    parser.add_argument(
        "--knn-variant-fetch-multiplier",
        type=int,
        default=int(os.getenv("MULTIAGENT_KNN_VARIANT_FETCH_MULTIPLIER", os.getenv("VARIANT_FETCH_MULTIPLIER", "3"))),
    )
    parser.add_argument("--knn-max-fetch-size", type=int, default=int(os.getenv("MULTIAGENT_BENCHMARK_KNN_MAX_FETCH_SIZE", "1500")))
    parser.add_argument("--knn-score-agg", default=os.getenv("MULTIAGENT_KNN_SCORE_AGG", "max"), choices=["max", "sum"])
    parser.add_argument("--knn-rrf-k", type=int, default=int(os.getenv("MULTIAGENT_KNN_RRF_K", "60")))
    parser.add_argument("--temperature", type=float, default=float(os.getenv("MULTIAGENT_LLM_TEMPERATURE", "0")))
    parser.add_argument("--max-output-tokens", type=int, default=int(os.getenv("MULTIAGENT_LLM_MAX_OUTPUT_TOKENS", "2500")))
    parser.add_argument(
        "--sleep-seconds",
        type=float,
        default=None,
        help=(
            "Pause after each topic run. Defaults to a Groq-free safe delay "
            "for Agent/API stages; pass 0 to disable."
        ),
    )
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument("--check-runtime", action="store_true", help="Print runtime validation and exit.")
    args = parser.parse_args()

    provider_explicit = any(arg == "--provider" or arg.startswith("--provider=") for arg in sys.argv[1:])
    ranking_source_explicit = any(arg == "--ranking-source" or arg.startswith("--ranking-source=") for arg in sys.argv[1:])
    if not provider_explicit and not parse_csv(args.models) and args.model_preset.startswith("groq"):
        args.provider = "groq"
    if not ranking_source_explicit and args.variant in {"pb1", "pb2"}:
        args.ranking_source = "candidates"
    if args.sleep_seconds is None:
        if args.provider == "groq" and args.stage in {"all", "agent1"}:
            args.sleep_seconds = GROQ_FREE_COMMON_SLEEP_SECONDS
        else:
            args.sleep_seconds = 0.0

    if args.check_runtime:
        check_models = resolve_models(args.models, args.model_preset)
        print(
            json.dumps(
                validate_runtime_environment(make_config(args, check_models[0]), require_langgraph=args.prefer_langgraph),
                ensure_ascii=False,
                indent=2,
            )
        )
        return

    if pd is None:
        raise RuntimeError("Running benchmark model ablation requires pandas and pyarrow.")

    topics_path = resolve_existing_path(args.topics, candidate_paths("topics"), "benchmark topics")
    qrels_path = resolve_existing_path(args.qrels, candidate_paths("qrels"), "benchmark qrels")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    k_values = parse_k_values(args.k)
    models = resolve_models(args.models, args.model_preset)
    if not models:
        raise SystemExit("No models selected.")

    qrels = load_qrels(qrels_path)
    rows = load_topics(topics_path)
    topics = selected_topics(
        rows,
        qrels,
        parse_csv(args.text_columns),
        limit=max(0, args.limit),
        allow_missing_qrels=args.allow_missing_qrels,
    )
    if not topics:
        raise SystemExit("No benchmark topics selected. Check topics/qrels paths and topic id columns.")

    run_config = {
        "topics_path": str(topics_path),
        "qrels_path": str(qrels_path),
        "num_topics_selected": len(topics),
        "models": models,
        "model_preset": args.model_preset,
        "provider": args.provider,
        "variant": args.variant,
        "stage": args.stage,
        "retrieval_backend": DEFAULT_RETRIEVAL_BACKEND,
        "retrieval_top_k": args.retrieval_top_k,
        "knn_num_candidates": args.knn_num_candidates,
        "knn_chunk_fetch_multiplier": args.knn_chunk_fetch_multiplier,
        "knn_variant_fetch_multiplier": args.knn_variant_fetch_multiplier,
        "knn_max_fetch_size": args.knn_max_fetch_size,
        "knn_rrf_k": args.knn_rrf_k,
        "candidate_top_k": args.candidate_top_k,
        "ranking_source": args.ranking_source,
        "k_values": k_values,
        "sleep_seconds": args.sleep_seconds,
        "rate_limit_note": groq_rate_limit_note(args.model_preset, args.provider),
    }

    agent1_path = Path(args.agent1_output) if normalize_text(args.agent1_output) else out_dir / "model_ablation_agent1_outputs.json"
    agent1_input_path = Path(args.agent1_input) if normalize_text(args.agent1_input) else agent1_path
    results_path = out_dir / "model_ablation_outputs.json"

    if args.stage == "agent1":
        model_outputs: Dict[str, Dict[str, Dict[str, Any]]] = load_resume(agent1_path) if args.resume else {}
        for model_id in models:
            model_outputs.setdefault(model_id, {})
        save_json(agent1_path, {"config": run_config, "model_outputs": model_outputs})

        total = len(models) * len(topics)
        completed = sum(
            1
            for model_id in models
            for topic in topics
            if is_completed_stage_state(model_outputs.get(model_id, {}).get(topic["topic_id"]), "agent1")
        )
        print(
            f"[START_AGENT1] topics={len(topics)} models={len(models)} "
            f"total_runs={total} completed={completed} remaining={max(0, total - completed)}",
            flush=True,
        )
        print(f"[DATA] topics={topics_path}", flush=True)
        print(f"[DATA] qrels={qrels_path}", flush=True)
        print(f"[AGENT1_OUT] {agent1_path}", flush=True)

        runs_since_save = 0
        for model_id in models:
            model_done = sum(
                1
                for topic in topics
                if is_completed_stage_state(model_outputs.get(model_id, {}).get(topic["topic_id"]), "agent1")
            )
            print(
                f"[MODEL_START] stage=agent1 model={model_id} completed={model_done}/{len(topics)} "
                f"remaining={max(0, len(topics) - model_done)}",
                flush=True,
            )
            for idx, topic in enumerate(topics, start=1):
                topic_id = topic["topic_id"]
                if args.resume and is_completed_stage_state(model_outputs.get(model_id, {}).get(topic_id), "agent1"):
                    continue
                started = time.time()
                try:
                    state = run_agent1_topic(args, model_id, topic)
                    state["benchmark_topic_id"] = topic_id
                    state["ablation_model"] = model_id
                    state["stage"] = "agent1"
                except Exception as exc:
                    if not (args.allow_llm_fallback or args.continue_on_error):
                        raise
                    state = {
                        "benchmark_topic_id": topic_id,
                        "ablation_model": model_id,
                        "stage": "agent1",
                        "error": str(exc),
                        "input_text": topic.get("input_text", ""),
                        "input_metadata": topic.get("input_metadata", {}),
                    }
                model_outputs.setdefault(model_id, {})[topic_id] = jsonable(state)
                runs_since_save += 1
                completed += 1
                model_done += 1
                elapsed = time.time() - started
                status = "error" if state.get("error") else "ok"
                print(
                    f"[AGENT1] run={completed}/{total} model_run={model_done}/{len(topics)} "
                    f"model={model_id} topic_index={idx}/{len(topics)} id={topic_id} "
                    f"status={status} seconds={elapsed:.1f}",
                    flush=True,
                )
                if runs_since_save >= max(1, args.save_every):
                    save_json(agent1_path, {"config": run_config, "model_outputs": model_outputs})
                    print(f"[SAVE] stage=agent1 completed={completed}/{total} path={agent1_path}", flush=True)
                    runs_since_save = 0
                clear_runtime_memory()
                if args.sleep_seconds > 0:
                    time.sleep(args.sleep_seconds)
            print(
                f"[MODEL_DONE] stage=agent1 model={model_id} completed={model_done}/{len(topics)}",
                flush=True,
            )
        save_json(agent1_path, {"config": run_config, "model_outputs": model_outputs})
        print(f"[SAVED_AGENT1] completed={completed}/{total} path={agent1_path}", flush=True)
        return

    agent1_outputs: Dict[str, Dict[str, Dict[str, Any]]] = {}
    if args.stage == "retrieval":
        if not agent1_input_path.exists():
            raise FileNotFoundError(f"Missing Agent 1 output for --stage retrieval: {agent1_input_path}")
        agent1_outputs = load_resume(agent1_input_path)
        run_config["agent1_input"] = str(agent1_input_path)

    model_outputs: Dict[str, Dict[str, Dict[str, Any]]] = load_resume(results_path) if args.resume else {}
    for model_id in models:
        model_outputs.setdefault(model_id, {})
    save_json(results_path, {"config": run_config, "model_outputs": model_outputs})

    total = len(models) * len(topics)
    completed = sum(
        1
        for model_id in models
        for topic in topics
        if is_completed_stage_state(model_outputs.get(model_id, {}).get(topic["topic_id"]), args.stage)
    )
    print(
        f"[START_{args.stage.upper()}] topics={len(topics)} models={len(models)} "
        f"total_runs={total} completed={completed} remaining={max(0, total - completed)}",
        flush=True,
    )
    print(f"[DATA] topics={topics_path}", flush=True)
    print(f"[DATA] qrels={qrels_path}", flush=True)
    if args.stage == "retrieval":
        print(f"[AGENT1_IN] {agent1_input_path}", flush=True)

    runs_since_save = 0
    for model_id in models:
        model_done = sum(
            1
            for topic in topics
            if is_completed_stage_state(model_outputs.get(model_id, {}).get(topic["topic_id"]), args.stage)
        )
        print(
            f"[MODEL_START] stage={args.stage} model={model_id} completed={model_done}/{len(topics)} "
            f"remaining={max(0, len(topics) - model_done)}",
            flush=True,
        )
        for idx, topic in enumerate(topics, start=1):
            topic_id = topic["topic_id"]
            if args.resume and is_completed_stage_state(model_outputs.get(model_id, {}).get(topic_id), args.stage):
                continue
            started = time.time()
            try:
                if args.stage == "retrieval":
                    agent_state = agent1_outputs.get(model_id, {}).get(topic_id)
                    if not isinstance(agent_state, dict):
                        raise RuntimeError(f"Missing Agent 1 state for model={model_id} topic={topic_id}")
                    if agent_state.get("error"):
                        raise RuntimeError(f"Agent 1 state has error for model={model_id} topic={topic_id}: {agent_state.get('error')}")
                    state = run_retrieval_topic(args, model_id, agent_state)
                    state["stage"] = "retrieval"
                else:
                    state = run_one_topic(args, model_id, topic)
                    state["stage"] = "all"
                state["benchmark_topic_id"] = topic_id
                state["ablation_model"] = model_id
            except Exception as exc:
                if not args.allow_retrieval_fallback and not args.allow_llm_fallback and not args.continue_on_error:
                    raise
                state = {
                    "benchmark_topic_id": topic_id,
                    "ablation_model": model_id,
                    "stage": args.stage,
                    "error": str(exc),
                    "input_metadata": topic.get("input_metadata", {}),
                }
            model_outputs.setdefault(model_id, {})[topic_id] = jsonable(state)
            runs_since_save += 1
            completed += 1
            model_done += 1
            elapsed = time.time() - started
            status = "error" if state.get("error") else "ok"
            print(
                f"[{args.stage.upper()}] run={completed}/{total} model_run={model_done}/{len(topics)} "
                f"model={model_id} topic_index={idx}/{len(topics)} id={topic_id} "
                f"status={status} seconds={elapsed:.1f}",
                flush=True,
            )
            if runs_since_save >= max(1, args.save_every):
                save_json(results_path, {"config": run_config, "model_outputs": model_outputs})
                print(f"[SAVE] stage={args.stage} completed={completed}/{total} path={results_path}", flush=True)
                runs_since_save = 0
            clear_runtime_memory()
            if args.sleep_seconds > 0:
                time.sleep(args.sleep_seconds)
        print(
            f"[MODEL_DONE] stage={args.stage} model={model_id} completed={model_done}/{len(topics)}",
            flush=True,
        )

    summary_rows, per_topic_rows = evaluate_model_outputs(model_outputs, qrels, k_values, args.ranking_source)
    save_json(results_path, {"config": run_config, "model_outputs": model_outputs})
    save_json(out_dir / "model_ablation_summary.json", {"config": run_config, "summary": summary_rows})
    save_json(out_dir / "model_ablation_per_topic.json", {"config": run_config, "per_topic": per_topic_rows})
    write_tables(out_dir, summary_rows, per_topic_rows)

    print("[SUMMARY]", flush=True)
    if pd is not None:
        print(pd.DataFrame(summary_rows).to_string(index=False), flush=True)
    else:
        print(json.dumps(summary_rows, ensure_ascii=False, indent=2), flush=True)
    if summary_rows:
        best = summary_rows[0]
        best_metric = f"map@{max(k_values)}"
        print(f"[BEST_MODEL] {best['model']} {best_metric}={best.get(best_metric, 0.0)}", flush=True)


if __name__ == "__main__":
    main()

