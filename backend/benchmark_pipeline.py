import os
import sys
import time
import logging
import pandas as pd
from dotenv import load_dotenv

# Tải cấu hình .env trước khi load app modules
load_dotenv()

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Khóa console log mặc định để không bị in rác lúc benchmark
logging.getLogger("app.services").setLevel(logging.WARNING)
logging.getLogger("elastic_transport").setLevel(logging.WARNING)

def benchmark_runner():
    print("\n" + "="*60)
    print("--- PIPELINE LATENCY BENCHMARK (PRECISE TIMING) ---")
    print("="*60 + "\n")

    # 1. Read Parquet
    parquet_path = "pac_test_plan_views.parquet"
    if not os.path.exists(parquet_path):
        print(f"Error: Cannot find parquet file at {parquet_path}")
        return

    print(f"[PREPARE] Loading '{parquet_path}'...")
    df = pd.read_parquet(parquet_path)
    print(f"Parquet Loaded. Total rows: {len(df)}, Columns: {list(df.columns)[:6]}...")
    
    sample_row = df.iloc[0].to_dict()
    doc_id = sample_row.get("topic_id") or sample_row.get("query_doc_id") or "UNKNOWN_ID"
    
    text_query = ""
    chosen_field = ""
    for field in ["query_text", "combined_short", "claims", "abstract", "title"]:
        val = sample_row.get(field)
        if val and len(str(val)) > 20:
            text_query = str(val)
            chosen_field = field
            break

            
    if not text_query:
        print("Error: No suitable text query found in the first row!")
        return
        
    print(f"\n[SELECTED QUERY]")
    print(f"  - Target Doc ID: {doc_id}")
    print(f"  - Source Field: {chosen_field}")
    print(f"  - Text Length: {len(text_query)} characters")
    print(f"  - Snippet: '{text_query[:100]}...'\n")
    
    print("-" * 60)
    print("--- STARTING REAL-TIME STEP BENCHMARKS ---")
    print("-" * 60)

    # 0. Imports
    print("Initializing Modules...")
    from app.services.analyzer import analyzer_service
    from app.services.profiler import analyze_document_profile
    from app.services.es_client import get_es_client, execute_bm25_search
    from app.services.graph_retriever import DynamicGraphRetriever
    
    # --- STEP 1: NLP PROFILING ---
    print("\nExecuting Step 1 (NLP & Profiling)...")
    t1_start = time.perf_counter()
    
    raw_doc = {
        "title": text_query[:200],
        "claims": text_query,
        "abstract": text_query
    }
    profile = analyze_document_profile(raw_doc)
    
    t1_end = time.perf_counter()
    time_step_1 = (t1_end - t1_start) * 1000.0 # milliseconds
    print(f"  -> STEP 1 DONE. Calculated Strength: '{profile['query_strength']}'")
    
    # --- STEP 2: ELASTIC CLOUD BM25 SEARCH ---
    print("\nExecuting Step 2 (Elastic Cloud Retrieval)...")
    t2_start = time.perf_counter()
    
    es_client = get_es_client()
    filter_conf = profile.get("retrieval_planner", {}).get("filter_config", {})
    base_candidates = execute_bm25_search(es_client, text_query, top_k=150, filter_config=filter_conf)
    
    t2_end = time.perf_counter()
    time_step_2 = (t2_end - t2_start) # seconds
    print(f"  -> STEP 2 DONE. Retrived {len(base_candidates)} candidates.")
    
    # --- STEP 3: DYNAMIC GRAPH RESCORER ---
    print("\nExecuting Step 3 (Dynamic Graph Rescoring)...")
    t3_start = time.perf_counter()
    
    graph_engine = DynamicGraphRetriever(base_candidates)
    rescored_candidates = graph_engine.execute_hybrid_graph_rank_rescore(profile)
    top_5 = rescored_candidates[:5]
    
    t3_end = time.perf_counter()
    time_step_3 = (t3_end - t3_start) * 1000.0 # milliseconds
    print(f"  -> STEP 3 DONE. Dynamic Graph populated. Top 1 ID: '{top_5[0][0]}'")
    
    # --- STEP 4: NVIDIA NIM GENERATION ---
    print("\nExecuting Step 4 (Expert Generation via NVIDIA NIM API)...")
    t4_start = time.perf_counter()
    
    context_rag = analyzer_service._build_retrieval_context(top_5)
    
    SYSTEM_PROMPT = (
        "You are an expert patent analyzer. Generate valid JSON."
    )
    user_content = f"QUERY:\n{text_query[:1000]}\n\nCONTEXT:\n{context_rag[:2000]}\n\nReturn short brief JSON report."
    
    completion = analyzer_service.client.chat.completions.create(
        model="meta/llama-3.1-70b-instruct",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content}
        ],
        temperature=0.1,
        max_tokens=1000,
        stream=False
    )
    
    t4_end = time.perf_counter()
    time_step_4 = (t4_end - t4_start) # seconds
    print(f"  -> STEP 4 DONE. Response received.")

    # =========================================================
    # SUMMARY
    # =========================================================
    print("\n" + "="*60)
    print("--- BENCHMARK SUMMARY REPORT ---")
    print("="*60)
    
    total_seconds = (time_step_1/1000.0) + time_step_2 + (time_step_3/1000.0) + time_step_4
    
    print(f"1. STEP 1 (NLP Profiling):       {time_step_1:7.2f} ms ({(time_step_1/1000.0/total_seconds)*100:5.2f}%)")
    print(f"2. STEP 2 (Cloud ES Search):     {time_step_2:7.2f} sec ({(time_step_2/total_seconds)*100:5.2f}%)")
    print(f"3. STEP 3 (Graph Rescore):       {time_step_3:7.2f} ms ({(time_step_3/1000.0/total_seconds)*100:5.2f}%)")
    print(f"4. STEP 4 (AI Generation):       {time_step_4:7.2f} sec ({(time_step_4/total_seconds)*100:5.2f}%)")
    print("-" * 60)
    print(f"TOTAL PIPELINE LATENCY:          {total_seconds:7.2f} seconds")
    print("="*60 + "\n")


if __name__ == "__main__":
    benchmark_runner()
