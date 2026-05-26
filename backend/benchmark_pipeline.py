import logging
import os
import sys
import time

import pandas as pd

try:
    from dotenv import load_dotenv

    load_dotenv()
except Exception:
    pass


sys.path.append(os.path.dirname(os.path.abspath(__file__)))

logging.getLogger("app.services").setLevel(logging.WARNING)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)


def _pick_sample_query(df: pd.DataFrame) -> tuple[str, str, str]:
    sample_row = df.iloc[0].to_dict()
    doc_id = sample_row.get("topic_id") or sample_row.get("query_doc_id") or "UNKNOWN_ID"

    for field in ["query_text", "combined_short", "retrieval_text", "claims", "abstract", "title"]:
        value = sample_row.get(field)
        if value and len(str(value)) > 20:
            return str(doc_id), field, str(value)

    raise RuntimeError("No suitable text query found in the first row.")


def benchmark_runner() -> None:
    print("\n" + "=" * 64)
    print("--- MULTI-AGENT PIPELINE LATENCY BENCHMARK ---")
    print("=" * 64 + "\n")

    parquet_path = "pac_test_plan_views.parquet"
    if not os.path.exists(parquet_path):
        raise FileNotFoundError(f"Cannot find parquet file at {parquet_path}")

    print(f"[PREPARE] Loading '{parquet_path}'...")
    df = pd.read_parquet(parquet_path)
    print(f"Parquet loaded. Total rows: {len(df)}, columns: {list(df.columns)[:6]}...")

    doc_id, chosen_field, text_query = _pick_sample_query(df)
    print("\n[SELECTED QUERY]")
    print(f"  - Target Doc ID: {doc_id}")
    print(f"  - Source Field: {chosen_field}")
    print(f"  - Text Length: {len(text_query)} characters")
    print(f"  - Snippet: '{text_query[:120]}...'\n")

    from app.services.multiagent_adapter import (
        multiagent_service,
        run_analysis_stage,
        run_evidence_stage,
        run_output_stage,
        run_query_understanding_stage,
        run_retrieval_stage,
    )

    runtime_check = multiagent_service.validate_runtime()
    if not runtime_check.get("ok"):
        print("[RUNTIME WARNING] validate_runtime_environment reported issues:")
        for issue in runtime_check.get("issues", []):
            print(f"  - {issue}")
        print()

    input_text, metadata, _ = multiagent_service._prepare_input(text_query)
    config = multiagent_service._config("pb4")
    timings: dict[str, float] = {}

    print("-" * 64)
    print("--- STARTING MULTI-AGENT STAGE BENCHMARKS ---")
    print("-" * 64)

    started = time.perf_counter()
    state = run_query_understanding_stage(input_text, input_metadata=metadata, config=config, variant="pb4", verbose=True)
    timings["Agent 1 Query Understanding"] = time.perf_counter() - started
    print(
        "  -> Agent 1 done. "
        f"search_queries={len(state.get('search_queries', []))}, "
        f"claim_elements={len(state.get('claim_elements', []))}"
    )

    started = time.perf_counter()
    state = run_retrieval_stage(state, config=config, variant="pb4", include_screening=True, include_report=False)
    timings["Fixed ES KNN Retrieval"] = time.perf_counter() - started
    print(
        "  -> Retrieval done. "
        f"raw_candidates={len(state.get('candidates', []))}, "
        f"screened={len(state.get('screened_candidates', []))}"
    )

    started = time.perf_counter()
    state = run_evidence_stage(state, config=config, variant="pb4", include_report=False)
    timings["Agent 2 Evidence Extraction"] = time.perf_counter() - started
    print(f"  -> Agent 2 done. evidence_docs={len(state.get('evidence', []))}")

    started = time.perf_counter()
    state = run_analysis_stage(state, config=config, variant="pb4", include_coverage=True)
    timings["Agent 3 Prior-Art Analysis"] = time.perf_counter() - started
    print(
        "  -> Agent 3 done. "
        f"coverage={state.get('coverage', {}).get('confidence', 'unknown')}"
    )

    started = time.perf_counter()
    state = run_output_stage(state, config=config, variant="pb4")
    timings["Output Finalization"] = time.perf_counter() - started
    print(f"  -> Output done. final_report_chars={len(state.get('final_report', ''))}")

    total_seconds = sum(timings.values())
    print("\n" + "=" * 64)
    print("--- BENCHMARK SUMMARY REPORT ---")
    print("=" * 64)
    for index, (name, seconds) in enumerate(timings.items(), start=1):
        pct = (seconds / total_seconds * 100.0) if total_seconds else 0.0
        print(f"{index}. {name:<32} {seconds:8.2f} sec ({pct:5.2f}%)")
    print("-" * 64)
    print(f"TOTAL PIPELINE LATENCY:          {total_seconds:8.2f} sec")
    print("=" * 64 + "\n")


if __name__ == "__main__":
    benchmark_runner()
