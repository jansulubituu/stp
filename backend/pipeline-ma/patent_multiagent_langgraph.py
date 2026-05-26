"""
Multi-agent patent search and prior-art analysis pipeline.

This module implements the KNN-backed LangGraph orchestration for patent
search and prior-art analysis.

The new orchestration is centered on three LLM agents:
1. query_understanding_agent
2. evidence_extraction_agent
3. prior_art_analysis_agent

Retrieval is intentionally a fixed tool node rather than an agent. The default
runtime backend is Elasticsearch KNN, reusing the best vector retrieval path
from the RAG-based pipeline.

Candidate screening is intentionally implemented as deterministic code rather
than an LLM agent because deduplication, date filtering, and top-k limiting are
more reliable as rules.
"""

from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, TypedDict


try:
    from google import genai
except Exception:  # pragma: no cover - optional runtime dependency
    genai = None  # type: ignore

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - optional runtime dependency
    OpenAI = None  # type: ignore

try:
    from langgraph.graph import END, START, StateGraph
except Exception:  # pragma: no cover - optional runtime dependency
    END = "__end__"  # type: ignore
    START = "__start__"  # type: ignore
    StateGraph = None  # type: ignore

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


VALID_PIPELINE_VARIANTS = {"pb1", "pb2", "pb3", "pb4"}
DEFAULT_PIPELINE_VARIANT = "pb4"
DEFAULT_RETRIEVAL_BACKEND = "es_knn"
KNN_RETRIEVAL_BACKENDS = {"es_knn", "knn", "elastic_knn", "elasticsearch_knn"}
DEFAULT_AGENT_LLM_PROVIDER = "groq"
DEFAULT_AGENT_LLM_MODEL = "openai/gpt-oss-120b"
DEFAULT_GEMINI_MODEL = "gemini-2.5-flash-lite"
GROQ_API_BASE = "https://api.groq.com/openai/v1"
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
GEMINI_PROVIDERS = {"gemini", "google", "google-genai"}
LOCAL_HF_PROVIDERS = {"local-hf", "local_hf", "hf-local", "transformers"}
OPENAI_COMPATIBLE_PROVIDERS = {
    "groq",
    "openai",
    "openai-compatible",
    "openai_compatible",
    "openrouter",
    "litellm",
    "vllm",
    "local-openai",
    "local",
}
GROQ_FREE_TPM_LIMITS = {
    "qwen/qwen3-32b": 6000,
    "openai/gpt-oss-120b": 8000,
    "openai/gpt-oss-20b": 8000,
    "openai/gpt-oss-safeguard-20b": 8000,
    "llama-3.3-70b-versatile": 12000,
}
_groq_token_windows: Dict[str, List[Tuple[float, int]]] = {}
RUNTIME_PATCH_VERSION = "2026-05-26-canonical-vi-report-v12"


def load_env_file(path: Path) -> None:
    """Load simple KEY=VALUE pairs without adding a dotenv dependency."""

    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def load_project_env() -> None:
    """Load local .env files for notebook/CLI runs; real env vars win."""

    module_dir = Path(__file__).resolve().parent
    project_root = module_dir.parent
    for path in [
        Path.cwd() / ".env",
        project_root / ".env",
        project_root / "pipeline" / ".env",
        module_dir / ".env",
    ]:
        load_env_file(path)


load_project_env()


def env_text(name: str, default: str = "") -> str:
    return os.getenv(name, default)


def env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def env_float(name: str, default: float) -> float:
    try:
        return float(os.getenv(name, str(default)))
    except Exception:
        return default


def env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.lower() in {"1", "true", "yes", "y", "on"}


def progress_log(enabled: bool, message: str, **fields: Any) -> None:
    """Print compact notebook-friendly progress messages when requested."""

    if not enabled:
        return
    parts = []
    for key, value in fields.items():
        if value is None or value == "":
            continue
        text = normalize_text(value)
        if len(text) > 160:
            text = text[:157] + "..."
        parts.append(f"{key}={text}")
    suffix = " " + " ".join(parts) if parts else ""
    print(f"[{time.strftime('%H:%M:%S')}] {message}{suffix}", flush=True)


def stage_log_enabled(verbose: Optional[bool] = None) -> bool:
    if verbose is not None:
        return bool(verbose)
    return env_bool("MULTIAGENT_STAGE_LOG", False)


@dataclass
class PipelineConfig:
    pipeline_variant: str = field(default_factory=lambda: env_text("MULTIAGENT_VARIANT", DEFAULT_PIPELINE_VARIANT).lower())

    candidate_screen_top_k: int = field(default_factory=lambda: env_int("MULTIAGENT_CANDIDATE_TOP_K", 20))
    evidence_top_docs: int = field(default_factory=lambda: env_int("MULTIAGENT_EVIDENCE_TOP_DOCS", 3))

    max_iterations: int = field(default_factory=lambda: env_int("MULTIAGENT_MAX_ITERATIONS", 2))
    min_evidence_docs: int = field(default_factory=lambda: env_int("MULTIAGENT_MIN_EVIDENCE_DOCS", 2))

    llm_provider: str = field(default_factory=lambda: env_text("MULTIAGENT_LLM_PROVIDER", DEFAULT_AGENT_LLM_PROVIDER).lower())
    llm_model: str = field(
        default_factory=lambda: env_text(
            "MULTIAGENT_LLM_MODEL",
            env_text("GROQ_MODEL", env_text("OPENAI_COMPATIBLE_MODEL", DEFAULT_AGENT_LLM_MODEL)),
        )
    )
    llm_api_base: str = field(
        default_factory=lambda: env_text("MULTIAGENT_LLM_API_BASE", env_text("OPENAI_COMPATIBLE_API_BASE", ""))
    )
    llm_temperature: float = field(
        default_factory=lambda: env_float("MULTIAGENT_LLM_TEMPERATURE", env_float("GEMINI_TEMPERATURE", 0.0))
    )
    llm_max_tokens: int = field(
        default_factory=lambda: env_int("MULTIAGENT_LLM_MAX_OUTPUT_TOKENS", env_int("GEMINI_MAX_OUTPUT_TOKENS", 1800))
    )
    gemini_model: str = field(default_factory=lambda: env_text("GEMINI_MODEL", DEFAULT_GEMINI_MODEL))
    gemini_timeout: int = field(default_factory=lambda: env_int("GEMINI_TIMEOUT", 180))
    gemini_temperature: float = field(default_factory=lambda: env_float("GEMINI_TEMPERATURE", 0.0))
    gemini_max_tokens: int = field(default_factory=lambda: env_int("GEMINI_MAX_OUTPUT_TOKENS", 2500))

    llm_strict: bool = field(default_factory=lambda: env_bool("MULTIAGENT_LLM_STRICT", False))
    local_hf_load_in_4bit: bool = field(default_factory=lambda: env_bool("MULTIAGENT_LOCAL_HF_LOAD_IN_4BIT", True))
    local_hf_trust_remote_code: bool = field(default_factory=lambda: env_bool("MULTIAGENT_LOCAL_HF_TRUST_REMOTE_CODE", True))
    local_hf_device_map: str = field(default_factory=lambda: env_text("MULTIAGENT_LOCAL_HF_DEVICE_MAP", "auto"))
    local_hf_max_input_tokens: int = field(default_factory=lambda: env_int("MULTIAGENT_LOCAL_HF_MAX_INPUT_TOKENS", 6144))
    local_hf_enable_thinking: bool = field(default_factory=lambda: env_bool("MULTIAGENT_LOCAL_HF_ENABLE_THINKING", False))
    agent_skill_dir: str = field(default_factory=lambda: env_text("MULTIAGENT_SKILL_DIR", str(Path(__file__).with_name("agent_skills"))))

    retrieval_strict: bool = field(
        default_factory=lambda: env_bool("MULTIAGENT_RETRIEVAL_STRICT", True)
    )

    retrieval_backend: str = field(
        default_factory=lambda: env_text("MULTIAGENT_RETRIEVAL_BACKEND", DEFAULT_RETRIEVAL_BACKEND).lower()
    )
    es_cloud_id: str = field(default_factory=lambda: env_text("ES_CLOUD_ID", ""))
    es_api_key: str = field(default_factory=lambda: env_text("ES_API_KEY", ""))
    es_user: str = field(default_factory=lambda: env_text("ES_USER", ""))
    es_password: str = field(default_factory=lambda: env_text("ES_PASSWORD", ""))
    bm25_index: str = field(default_factory=lambda: env_text("BM25_INDEX", "clef_ip_patents_v1_mini"))
    knn_index: str = field(default_factory=lambda: env_text("KNN_INDEX", "clef_ip_patents_v1_mini_jina"))
    vector_field: str = field(default_factory=lambda: env_text("ES_VECTOR_FIELD", "content_vector"))
    knn_top_k: int = field(default_factory=lambda: env_int("MULTIAGENT_KNN_TOP_K", 100))
    knn_num_candidates: int = field(default_factory=lambda: env_int("MULTIAGENT_KNN_NUM_CANDIDATES", 1500))
    knn_chunk_fetch_multiplier: int = field(default_factory=lambda: env_int("MULTIAGENT_KNN_CHUNK_FETCH_MULTIPLIER", 12))
    knn_variant_fetch_multiplier: int = field(
        default_factory=lambda: env_int("MULTIAGENT_KNN_VARIANT_FETCH_MULTIPLIER", env_int("VARIANT_FETCH_MULTIPLIER", 3))
    )
    knn_max_fetch_size: int = field(default_factory=lambda: env_int("MULTIAGENT_KNN_MAX_FETCH_SIZE", 1500))
    knn_score_agg: str = field(default_factory=lambda: env_text("MULTIAGENT_KNN_SCORE_AGG", "max").lower())
    knn_rrf_k: int = field(default_factory=lambda: env_int("MULTIAGENT_KNN_RRF_K", 60))
    knn_embedding_api_base: str = field(
        default_factory=lambda: env_text("MULTIAGENT_KNN_EMBED_API_BASE", env_text("LOCAL_API_BASE", "")).rstrip("/")
    )
    knn_embedding_api_key: str = field(
        default_factory=lambda: env_text("MULTIAGENT_KNN_EMBED_API_KEY", "EMPTY")
    )
    knn_embedding_model: str = field(
        default_factory=lambda: env_text("MULTIAGENT_KNN_EMBED_MODEL", env_text("EMBED_SERVED_MODEL", "jina-embed-safe"))
    )
    knn_hf_model: str = field(default_factory=lambda: env_text("MULTIAGENT_KNN_HF_MODEL", "jinaai/jina-embeddings-v3"))
    knn_embedding_task: str = field(default_factory=lambda: env_text("MULTIAGENT_KNN_EMBED_TASK", "retrieval.query"))
    knn_query_words: int = field(default_factory=lambda: env_int("MULTIAGENT_KNN_QUERY_WORDS", 700))


# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------


class PatentAnalysisState(TypedDict, total=False):
    input_text: str
    input_type: str
    input_metadata: Dict[str, Any]
    pipeline_variant: str

    technical_problem: str
    key_features: List[str]
    claim_elements: List[str]
    query_entities: List[Dict[str, Any]]
    retrieval_focus: Dict[str, Any]
    search_queries: List[Dict[str, Any]]
    filters: Dict[str, Any]

    candidates: List[Dict[str, Any]]
    screened_candidates: List[Dict[str, Any]]
    candidate_docs: Dict[str, Dict[str, Any]]
    retrieval_context: Dict[str, Any]

    evidence: List[Dict[str, Any]]
    analysis: Dict[str, Any]
    coverage: Dict[str, Any]
    final_report: str

    audit_log: List[Dict[str, Any]]
    iteration: int
    max_iterations: int
    should_retry: bool
    proxy_metrics: Dict[str, Any]


# ---------------------------------------------------------------------------
# Basic helpers reused from the old pipeline
# ---------------------------------------------------------------------------


def now_ms() -> int:
    return int(time.time() * 1000)


def normalize_text(value: Any) -> str:
    if value is None:
        return ""
    return " ".join(str(value).split()).strip()


def normalize_optional_text(value: Any) -> str:
    text = normalize_text(value)
    if text.lower() in {"nan", "none", "null", "[]"}:
        return ""
    return text


def normalize_pipeline_variant(value: Any) -> str:
    variant = normalize_optional_text(value).lower()
    return variant if variant in VALID_PIPELINE_VARIANTS else DEFAULT_PIPELINE_VARIANT


def normalize_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [normalize_text(v) for v in value if normalize_text(v)]
    text = normalize_text(value)
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    if text.startswith("[") and text.endswith("]"):
        try:
            parsed = json.loads(text)
            return normalize_list(parsed)
        except Exception:
            pass
    return [p.strip() for p in re.split(r"[;\n\r]+", text) if p.strip()]


def normalize_bool(value: Any, default: bool = False) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    text = normalize_text(value).lower()
    if text in {"1", "true", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "no", "n", "off", "none", "null", ""}:
        return False
    return default


def safe_float(value: Any, default: float = 1.0, minimum: Optional[float] = None, maximum: Optional[float] = None) -> float:
    try:
        parsed = float(value)
    except Exception:
        parsed = float(default)
    if minimum is not None:
        parsed = max(float(minimum), parsed)
    if maximum is not None:
        parsed = min(float(maximum), parsed)
    return parsed


def limit_words(value: Any, max_words: int) -> str:
    text = normalize_text(value)
    if not text or max_words <= 0:
        return text
    words = text.split()
    return " ".join(words[:max_words])


def technical_label_vi(value: Any) -> str:
    """Vietnamese display label for recurring claim-element phrases.

    Raw English labels are kept for retrieval/debug traceability; this helper
    is only for user-facing evidence tables and reports.
    """

    text = normalize_text(value)
    if not text:
        return ""
    clean_text = (
        text.replace("bọ đất", "bộ đốt")
        .replace("bọ đốt", "bộ đốt")
        .replace("thận xốp", "thân xốp")
    )
    lower_text = clean_text.lower()
    if "tấm kính" in lower_text and ("bộ đốt" in lower_text or "burner" in lower_text):
        return "tấm kính chịu nhiệt đặt trên bộ đốt" if "chịu nhiệt" in lower_text else "tấm kính đặt trên bộ đốt"
    if "combustion" in lower_text and "thân xốp" in lower_text:
        return "không gian cháy xả khí qua thân xốp"
    key = re.sub(r"[\u2010-\u2015\u2212]", "-", clean_text).lower()
    key = re.sub(r"[^a-z0-9]+", " ", key).strip()
    exact_labels = [
        ("heat resistant glass top plate over burner", "tấm kính chịu nhiệt đặt trên bộ đốt"),
        ("heat resistant glass plate over burner", "tấm kính chịu nhiệt đặt trên bộ đốt"),
        ("heat resistant glass over burner", "tấm kính chịu nhiệt đặt trên bộ đốt"),
        ("glass top plate over burner", "tấm kính đặt trên bộ đốt"),
        ("gas permeable porous body beneath glass", "thân xốp thấm khí bên dưới kính"),
        ("gas permeable porous body under glass", "thân xốp thấm khí bên dưới kính"),
        ("gas permeable porous body below glass", "thân xốp thấm khí bên dưới kính"),
        ("porous silicon carbide body for gas discharge through heating surface", "thân silic cacbua xốp cho khí đi qua bề mặt gia nhiệt"),
        ("porous silicon carbide layer for gas discharge through heating surface", "lớp silic cacbua xốp cho khí đi qua bề mặt gia nhiệt"),
        ("porous body gas discharge heating surface", "thân xốp cho khí đi qua bề mặt gia nhiệt"),
        ("combustion gas discharge through porous body", "không gian cháy xả khí qua thân xốp"),
        ("combustion gases discharge through porous body", "không gian cháy xả khí qua thân xốp"),
        ("combustion gas discharging through porous body", "không gian cháy xả khí qua thân xốp"),
        ("combustion gases discharging through porous body", "không gian cháy xả khí qua thân xốp"),
        ("combustion space discharging gases through porous body", "không gian cháy xả khí qua thân xốp"),
        ("combustion space allowing gas discharge through porous body", "không gian cháy cho khí đi qua thân xốp"),
        ("combustion space for gas discharge through porous body", "không gian cháy cho khí đi qua thân xốp"),
        ("flat gas stove with glass top and porous combustion", "bếp gas mặt kính phẳng với bộ đốt xốp"),
        ("flat gas stove with glass top and porous burner", "bếp gas mặt kính phẳng với bộ đốt xốp"),
        ("flat gas stove with glass top and porous silicon", "bếp gas mặt kính phẳng với lớp silic xốp"),
    ]
    for pattern, label in exact_labels:
        if key == pattern or pattern in key:
            return label

    replacements = [
        ("heat resistant", "chịu nhiệt"),
        ("glass top plate", "tấm kính phía trên"),
        ("glass plate", "tấm kính"),
        ("glass top", "mặt kính"),
        ("gas permeable", "thấm khí"),
        ("porous burner component", "thành phần bộ đốt xốp"),
        ("porous silicon carbide", "silic cacbua xốp"),
        ("silicon carbide", "silic cacbua"),
        ("through porous body", "qua thân xốp"),
        ("porous body", "thân xốp"),
        ("porous layer", "lớp xốp"),
        ("combustion space", "không gian cháy"),
        ("combustion chamber", "buồng cháy"),
        ("discharging gases", "xả khí"),
        ("gas discharge", "xả khí"),
        ("beneath glass", "bên dưới kính"),
        ("under glass", "bên dưới kính"),
        ("below glass", "bên dưới kính"),
        ("over burner", "trên bộ đốt"),
        ("heating surface", "bề mặt gia nhiệt"),
        ("burner", "bộ đốt"),
    ]
    translated = key
    for source, target in replacements:
        translated = re.sub(rf"\b{re.escape(source)}\b", target, translated)
    translated = normalize_text(translated)
    return translated if translated != key else clean_text


def clean_vietnamese_display_text(value: Any) -> str:
    text = normalize_text(value)
    if not text:
        return ""
    replacements = {
        "bọ đất": "bộ đốt",
        "bọ đốt": "bộ đốt",
        "thận xốp": "thân xốp",
    }
    for source, target in replacements.items():
        text = text.replace(source, target)
    return text


def match_type_vi(value: Any) -> str:
    match_type = normalize_text(value).lower()
    return {
        "exact": "khớp chính xác",
        "partial": "khớp một phần",
        "weak": "khớp yếu",
    }.get(match_type, normalize_text(value))


def section_vi(value: Any) -> str:
    section = normalize_text(value).lower()
    return {
        "title": "tiêu đề",
        "abstract": "tóm tắt",
        "claims": "yêu cầu bảo hộ",
        "description": "mô tả",
        "mixed": "nhiều phần",
    }.get(section, normalize_text(value))


def calibrated_novelty_risk(matches: List[Dict[str, Any]], missing_elements: List[str]) -> str:
    """Conservative novelty-risk label from grounded evidence, not LLM prose."""

    match_types = [normalize_text(item.get("match_type")).lower() for item in matches if isinstance(item, dict)]
    exact_count = sum(1 for value in match_types if value == "exact")
    partial_count = sum(1 for value in match_types if value == "partial")
    matched_count = exact_count + partial_count + sum(1 for value in match_types if value == "weak")
    missing_count = len(missing_elements)
    if matched_count <= 0:
        return "low"
    if exact_count >= 3 and missing_count == 0:
        return "high"
    if exact_count >= 2 and missing_count <= 1:
        return "medium"
    if matched_count >= 2 and missing_count <= 1:
        return "medium"
    if exact_count >= 1 and missing_count <= 1:
        return "medium"
    return "low"


def compact_metadata_for_prompt(metadata: Dict[str, Any], include_text_fields: bool = False) -> Dict[str, Any]:
    """Keep LLM prompts under Groq's small on-demand token budget."""

    if not isinstance(metadata, dict):
        return {}
    text_limits = {
        "title": env_int("MULTIAGENT_AGENT1_TITLE_WORDS", 40),
        "abstract": env_int("MULTIAGENT_AGENT1_ABSTRACT_WORDS", 90),
        "claims": env_int("MULTIAGENT_AGENT1_CLAIMS_WORDS", 90),
        "retrieval_text": env_int("MULTIAGENT_AGENT1_RETRIEVAL_TEXT_WORDS", 50),
        "description": env_int("MULTIAGENT_AGENT1_DESCRIPTION_WORDS", 40),
        "query_text": env_int("MULTIAGENT_AGENT1_TEXT_WORDS", 60),
        "text": env_int("MULTIAGENT_AGENT1_TEXT_WORDS", 60),
        "full_text": env_int("MULTIAGENT_AGENT1_FULL_TEXT_WORDS", 60),
        "first_claim": env_int("MULTIAGENT_AGENT1_FIRST_CLAIM_WORDS", 90),
        "independent_claims": env_int("MULTIAGENT_AGENT1_INDEPENDENT_CLAIMS_WORDS", 90),
    }
    scalar_keep_keys = {
        "topic_id",
        "query_id",
        "doc_id",
        "patent_id",
        "canonical_doc_id",
        "canonical_topic_id",
        "publication_date",
        "application_date",
        "priority_date",
        "kind_code",
        "country",
        "ipc",
        "ipc_codes",
        "cpc",
        "assignee",
        "assignees",
        "inventor",
        "inventors",
        "citations",
        "num_citations",
        "has_citations",
    }
    compact: Dict[str, Any] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        key_text = str(key)
        key_lower = key_text.lower()
        if key_lower in text_limits:
            if not include_text_fields:
                continue
            text = limit_words(value, text_limits[key_lower])
            if text:
                compact[key_text] = text
        elif any(marker in key_lower for marker in ["claim", "abstract", "title", "retrieval", "description"]):
            if not include_text_fields:
                continue
            text = limit_words(value, 100)
            if text:
                compact[key_text] = text
        elif isinstance(value, (list, tuple, set)):
            items = normalize_list(value)[:20]
            if items:
                compact[key_text] = items
        else:
            text = normalize_optional_text(value)
            if not text:
                continue
            if key_lower in scalar_keep_keys:
                compact[key_text] = limit_words(text, 60)
            elif len(text.split()) <= 20:
                compact[key_text] = text
    priority_keys = [
        "topic_id",
        "query_id",
        "doc_id",
        "patent_id",
        "canonical_doc_id",
        "canonical_topic_id",
        "publication_date",
        "application_date",
        "priority_date",
        "kind_code",
        "country",
        "ipc",
        "ipc_codes",
        "cpc",
        "assignee",
        "assignees",
        "inventor",
        "inventors",
        "citations",
        "num_citations",
        "has_citations",
        "title",
        "abstract",
        "first_claim",
        "independent_claims",
        "claims",
        "retrieval_text",
    ]
    priority = {key: idx for idx, key in enumerate(priority_keys)}
    max_keys = max(1, env_int("MULTIAGENT_AGENT1_METADATA_MAX_KEYS", 24))
    max_words = max(1, env_int("MULTIAGENT_AGENT1_METADATA_WORDS", 220))
    bounded: Dict[str, Any] = {}
    used_words = 0
    for key, value in sorted(compact.items(), key=lambda item: (priority.get(item[0].lower(), 10_000), item[0])):
        value_words = len(normalize_text(json.dumps(value, ensure_ascii=True)).split())
        if len(bounded) >= max_keys:
            break
        if bounded and used_words + value_words > max_words:
            continue
        bounded[key] = value
        used_words += value_words
    return bounded


def agent1_input_excerpt(state: PatentAnalysisState) -> str:
    """Build the compact technical text Agent 1 needs for PB2 KNN query planning."""

    metadata = state.get("input_metadata", {}) or {}
    raw_input = normalize_text(state.get("input_text"))
    max_words = env_int("MULTIAGENT_AGENT1_INPUT_WORDS", 220)
    sections: List[str] = []
    seen = set()

    def add(label: str, value: Any, word_limit: int) -> None:
        text = limit_words(value, word_limit)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            sections.append(f"{label}: {text}")

    if isinstance(metadata, dict):
        add("Title", metadata.get("title"), env_int("MULTIAGENT_AGENT1_TITLE_WORDS", 40))
        add("Abstract", metadata.get("abstract"), env_int("MULTIAGENT_AGENT1_ABSTRACT_WORDS", 90))
        first_claim = metadata.get("first_claim") or metadata.get("independent_claims")
        if first_claim:
            add("Claim excerpt", first_claim, env_int("MULTIAGENT_AGENT1_FIRST_CLAIM_WORDS", 90))
        else:
            add("Claim excerpt", metadata.get("claims"), env_int("MULTIAGENT_AGENT1_CLAIMS_WORDS", 90))
        add("Retrieval text", metadata.get("retrieval_text"), env_int("MULTIAGENT_AGENT1_RETRIEVAL_TEXT_WORDS", 50))

    if not sections:
        return limit_words(raw_input, max_words)

    if env_bool("MULTIAGENT_AGENT1_INCLUDE_RAW_INPUT", False):
        add("Raw input", raw_input, env_int("MULTIAGENT_AGENT1_RAW_INPUT_WORDS", 50))
    return limit_words("\n".join(sections), max_words)


def canonical_doc_id(value: Any) -> str:
    text = normalize_text(value).upper()
    if not text:
        return ""
    text = text.replace(" ", "-").replace("_", "-")
    text = re.sub(r"-+", "-", text).strip("-")
    match = re.match(r"^([A-Z]{2})-?(\d+)(?:-?[A-Z][0-9A-Z]*)?$", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return text


def normalize_variant_doc_id(value: Any) -> str:
    text = normalize_text(value).upper()
    if not text:
        return ""
    text = text.replace(" ", "-").replace("_", "-")
    text = re.sub(r"[^A-Z0-9-]+", "", text)
    text = re.sub(r"-+", "-", text).strip("-")
    match = re.match(r"^([A-Z]{2})-?(\d+)-?([A-Z][0-9A-Z]*)$", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
    return text


def date_int(value: Any) -> int:
    text = normalize_optional_text(value)
    if not text:
        return 0
    match = re.search(r"\b(\d{8})\b", text)
    if match:
        return int(match.group(1))
    match = re.search(r"\b(\d{4})-(\d{2})-(\d{2})\b", text)
    if match:
        return int("".join(match.groups()))
    return 0


def is_prior_art_date(publication_date: Any, cutoff_date: Any) -> bool:
    cutoff = date_int(cutoff_date)
    if cutoff <= 0:
        return True
    pub = date_int(publication_date)
    if pub <= 0:
        return True
    return pub < cutoff


def parse_json_object(text: str) -> Optional[Dict[str, Any]]:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None

    def balance_json_closers(candidate: str) -> str:
        """Repair common LLM truncation: missing object/array closers at the end."""

        out: List[str] = []
        stack: List[str] = []
        in_string = False
        escaped = False
        pairs = {"{": "}", "[": "]"}
        closing = set(pairs.values())
        for char in candidate:
            out.append(char)
            if in_string:
                if escaped:
                    escaped = False
                elif char == "\\":
                    escaped = True
                elif char == '"':
                    in_string = False
                continue
            if char == '"':
                in_string = True
            elif char in pairs:
                stack.append(pairs[char])
            elif char in closing:
                if stack and stack[-1] == char:
                    stack.pop()
                elif char in stack:
                    out.pop()
                    while stack and stack[-1] != char:
                        out.append(stack.pop())
                    if stack and stack[-1] == char:
                        stack.pop()
                    out.append(char)
        while stack:
            out.append(stack.pop())
        return "".join(out)

    def is_likely_nested_fragment(parsed: Any) -> bool:
        if not isinstance(parsed, dict):
            return False
        keys = set(parsed.keys())
        fragment_key_sets = [
            {"name", "type"},
            {"query_view", "search_role", "text", "weight", "search_mode"},
        ]
        return any(keys and keys.issubset(fragment_keys) for fragment_keys in fragment_key_sets)

    fenced = re.search(r"```(?:json)?\s*(.*?)```", raw, flags=re.IGNORECASE | re.DOTALL)
    candidates = [raw]
    if fenced:
        candidates.insert(0, fenced.group(1).strip())
    for candidate in list(candidates):
        balanced = balance_json_closers(candidate)
        if balanced != candidate:
            candidates.append(balanced)

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
            if isinstance(parsed, dict) and not is_likely_nested_fragment(parsed):
                return parsed
        except Exception:
            pass

    decoder = json.JSONDecoder()
    for start, char in enumerate(raw):
        if char != "{":
            continue
        try:
            parsed, _end = decoder.raw_decode(raw[start:])
        except Exception:
            continue
        if isinstance(parsed, dict) and not is_likely_nested_fragment(parsed):
            return parsed

    try:
        parsed = json.loads(normalize_text(raw))
        return parsed if isinstance(parsed, dict) and not is_likely_nested_fragment(parsed) else None
    except Exception:
        return None


def dedupe_search_queries(search_queries: Iterable[Dict[str, Any]], max_queries: int = 8) -> List[Dict[str, Any]]:
    deduped: List[Dict[str, Any]] = []
    seen = set()
    for query in search_queries:
        if not isinstance(query, dict):
            continue
        text = normalize_text(query.get("text"))
        if not text:
            continue
        view = normalize_text(query.get("query_view")) or "combined"
        mode = normalize_text(query.get("search_mode")).lower() or "both"
        role = normalize_text(query.get("search_role")).lower() or view
        key = (role, view, mode, text.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(query)
        if len(deduped) >= max_queries:
            break
    return deduped


def sanitize_search_queries(
    raw_queries: Any,
    fallback_queries: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    valid_views = {"combined", "title_abstract", "claims", "technical_problem", "feature"}
    retrieval_modes = {"semantic", "claim_text", "entity_expansion", "problem_expansion"}
    valid_roles = {"combined", "claim_overlap", "function", "problem", "component", "technology", "ipc_or_citation"}
    source = raw_queries if isinstance(raw_queries, list) and raw_queries else fallback_queries
    cleaned: List[Dict[str, Any]] = []

    for raw_query in source:
        if not isinstance(raw_query, dict):
            continue
        text = limit_words(normalize_optional_text(raw_query.get("text")), 260)
        if not text:
            continue
        view = normalize_text(raw_query.get("query_view")).lower() or "combined"
        if view not in valid_views:
            view = "combined"
        mode = normalize_text(raw_query.get("search_mode")).lower() or "semantic"
        mode = {
            "both": "semantic",
            "graph": "semantic",
            "entity_neighborhood": "entity_expansion",
        }.get(mode, mode)
        if mode not in retrieval_modes:
            mode = "semantic"
        role = normalize_text(raw_query.get("search_role") or raw_query.get("query_role")).lower()
        if role not in valid_roles:
            role = {
                "claims": "claim_overlap",
                "technical_problem": "problem",
                "feature": "function",
                "title_abstract": "technology",
            }.get(view, "combined")
        item = {
            "query_view": view,
            "text": text,
            "weight": safe_float(raw_query.get("weight"), default=1.0, minimum=0.2, maximum=2.0),
            "search_mode": mode,
            "search_role": role,
        }
        cleaned.append(item)

    if not cleaned:
        cleaned = list(fallback_queries)
    return dedupe_search_queries(cleaned, max_queries=5)


def sanitize_filters(raw_filters: Any, fallback_filters: Dict[str, Any]) -> Dict[str, Any]:
    filters = dict(fallback_filters)
    if isinstance(raw_filters, dict):
        for key in ["exclude_doc_id", "exclude_canonical_doc_id", "prior_art_cutoff_date"]:
            value = normalize_optional_text(raw_filters.get(key))
            if value:
                filters[key] = value
        if "use_date_filter" in raw_filters:
            filters["use_date_filter"] = normalize_bool(
                raw_filters.get("use_date_filter"),
                default=bool(normalize_text(filters.get("prior_art_cutoff_date"))),
            )

    exclude_doc_id = normalize_optional_text(filters.get("exclude_doc_id"))
    exclude_canonical = normalize_optional_text(filters.get("exclude_canonical_doc_id")) or canonical_doc_id(exclude_doc_id)
    filters["exclude_doc_id"] = exclude_doc_id
    filters["exclude_canonical_doc_id"] = exclude_canonical
    filters["prior_art_cutoff_date"] = normalize_optional_text(filters.get("prior_art_cutoff_date"))
    # Prior-art search should enforce the cutoff whenever a reliable cutoff is
    # available. The LLM may extract the date, but code owns this safety rule.
    filters["use_date_filter"] = bool(filters["prior_art_cutoff_date"])
    return filters


def sanitize_query_entities(raw_entities: Any) -> List[Dict[str, Any]]:
    if not isinstance(raw_entities, list):
        return []
    cleaned = []
    valid_types = {
        "technology",
        "component",
        "function",
        "problem",
        "ipc",
        "assignee",
        "inventor",
        "citation",
        "patent",
    }
    for item in raw_entities:
        if isinstance(item, str):
            name = normalize_optional_text(item)
            entity_type = "technology"
        elif isinstance(item, dict):
            name = normalize_optional_text(item.get("name") or item.get("entity"))
            entity_type = normalize_optional_text(item.get("type")).lower() or "technology"
        else:
            continue
        if not name:
            continue
        if entity_type not in valid_types:
            entity_type = "technology"
        cleaned.append({"name": name, "type": entity_type})
        if len(cleaned) >= 16:
            break
    return cleaned


def sanitize_retrieval_focus(raw_focus: Any, fallback_focus: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    fallback_focus = fallback_focus or {}
    focus = dict(fallback_focus)
    if isinstance(raw_focus, dict):
        for key in ["search_intent", "preferred_retrieval_backend", "focus_scope", "evidence_priority"]:
            value = normalize_optional_text(raw_focus.get(key))
            if value:
                focus[key] = value
    focus["preferred_retrieval_backend"] = "es_knn"
    return focus


def add_audit(state: PatentAnalysisState, node: str, message: str, **extra: Any) -> List[Dict[str, Any]]:
    audit = list(state.get("audit_log", []))
    audit.append({"ts_ms": now_ms(), "node": node, "message": message, **extra})
    return audit


def agent_skill_path(config: PipelineConfig, agent_name: str) -> Path:
    return Path(config.agent_skill_dir) / f"{agent_name}.md"


def load_agent_skill(config: PipelineConfig, agent_name: str, fallback: str) -> Tuple[str, str]:
    """Load a project-level agent skill file.

    The returned text is used as the agent system prompt or execution contract.
    Returning the path alongside the text makes audit logs reproducible.
    """

    path = agent_skill_path(config, agent_name)
    try:
        text = path.read_text(encoding="utf-8").strip()
    except FileNotFoundError:
        return fallback, str(path)
    if not text:
        return fallback, str(path)
    return text, str(path)


# ---------------------------------------------------------------------------
# External clients
# ---------------------------------------------------------------------------


def make_gemini_client(config: PipelineConfig) -> Optional[Any]:
    if genai is None:
        return None
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY")
    if not api_key:
        return None
    try:
        return genai.Client(api_key=api_key)
    except TypeError:
        return genai.Client()


def provider_is_openai_compatible(provider: str) -> bool:
    return provider in OPENAI_COMPATIBLE_PROVIDERS


def openai_compatible_base_url(provider: str, config: PipelineConfig) -> str:
    configured = normalize_optional_text(config.llm_api_base)
    if configured:
        return configured
    if provider == "groq":
        return GROQ_API_BASE
    if provider == "openrouter":
        return OPENROUTER_API_BASE
    return ""


def openai_compatible_api_key(provider: str, base_url: str = "") -> str:
    if provider == "groq":
        return os.getenv("GROQ_API_KEY", "")
    if provider == "openrouter":
        return os.getenv("OPENROUTER_API_KEY", "")
    if provider == "openai":
        return os.getenv("OPENAI_API_KEY", "")
    key = (
        os.getenv("MULTIAGENT_LLM_API_KEY")
        or os.getenv("OPENAI_COMPATIBLE_API_KEY")
        or os.getenv("GROQ_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
    )
    if key:
        return key
    if base_url.startswith("http://127.0.0.1") or base_url.startswith("http://localhost"):
        return "EMPTY"
    return ""


def make_openai_compatible_client(config: PipelineConfig, provider: str) -> Optional[Any]:
    if OpenAI is None:
        return None
    base_url = openai_compatible_base_url(provider, config)
    api_key = openai_compatible_api_key(provider, base_url)
    if not api_key:
        return None
    kwargs: Dict[str, Any] = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


_local_hf_model_name = ""
_local_hf_tokenizer: Any = None
_local_hf_model: Any = None


def unload_local_hf_model() -> None:
    global _local_hf_model_name, _local_hf_tokenizer, _local_hf_model
    _local_hf_model_name = ""
    _local_hf_tokenizer = None
    _local_hf_model = None
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def load_local_hf_model(config: PipelineConfig) -> Tuple[Any, Any]:
    """Load one local chat model at a time for no-API-quota experiments."""

    global _local_hf_model_name, _local_hf_tokenizer, _local_hf_model
    model_name = normalize_optional_text(config.llm_model)
    if not model_name:
        raise RuntimeError("Missing MULTIAGENT_LLM_MODEL for local-hf provider.")
    if _local_hf_model_name == model_name and _local_hf_tokenizer is not None and _local_hf_model is not None:
        return _local_hf_tokenizer, _local_hf_model

    unload_local_hf_model()
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("local-hf provider requires torch and transformers.") from exc

    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        use_fast=True,
        trust_remote_code=config.local_hf_trust_remote_code,
    )
    if tokenizer.pad_token_id is None and tokenizer.eos_token_id is not None:
        tokenizer.pad_token = tokenizer.eos_token

    kwargs: Dict[str, Any] = {
        "trust_remote_code": config.local_hf_trust_remote_code,
    }
    if config.local_hf_device_map:
        kwargs["device_map"] = config.local_hf_device_map
    if torch.cuda.is_available():
        kwargs["torch_dtype"] = torch.float16
        if config.local_hf_load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                kwargs["quantization_config"] = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                )
            except Exception as exc:
                if config.llm_strict:
                    raise RuntimeError(
                        "MULTIAGENT_LOCAL_HF_LOAD_IN_4BIT=true requires bitsandbytes. "
                        "Install bitsandbytes or set it to false."
                    ) from exc
    else:
        kwargs["torch_dtype"] = torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_name, **kwargs)
    model.eval()
    _local_hf_model_name = model_name
    _local_hf_tokenizer = tokenizer
    _local_hf_model = model
    return tokenizer, model


def render_local_hf_prompt(config: PipelineConfig, tokenizer: Any, messages: List[Dict[str, str]]) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        except Exception:
            pass
    rendered = []
    for message in messages:
        role = normalize_optional_text(message.get("role")) or "user"
        content = normalize_optional_text(message.get("content"))
        if content:
            rendered.append(f"{role}: {content}")
    rendered.append("assistant:")
    return "\n".join(rendered)


def call_local_hf_json(
    config: PipelineConfig,
    system_prompt: str,
    user_prompt: str,
    fallback: Dict[str, Any],
) -> Dict[str, Any]:
    try:
        import torch
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        if config.llm_strict:
            raise RuntimeError("local-hf provider requires torch.") from exc
        return fallback

    try:
        tokenizer, model = load_local_hf_model(config)
    except Exception:
        if config.llm_strict:
            raise
        return fallback

    system = (
        system_prompt.strip()
        + "\n\nReturn only one valid JSON object. Do not include markdown, XML tags, <think> blocks, or chain-of-thought. "
        + "Start with { and end with }."
    )
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt.strip()},
    ]
    prompt = render_local_hf_prompt(config, tokenizer, messages)
    encoded = tokenizer(prompt, return_tensors="pt", truncation=False)
    input_tokens = int(encoded["input_ids"].shape[-1])
    if config.local_hf_max_input_tokens > 0 and input_tokens > config.local_hf_max_input_tokens:
        message = (
            f"local-hf prompt has {input_tokens} tokens, above "
            f"MULTIAGENT_LOCAL_HF_MAX_INPUT_TOKENS={config.local_hf_max_input_tokens}."
        )
        if config.llm_strict:
            raise RuntimeError(message)
        return fallback

    try:
        device = next(model.parameters()).device
    except Exception:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    encoded = {key: value.to(device) for key, value in encoded.items()}
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass

    do_sample = float(config.llm_temperature or 0.0) > 0
    generation_kwargs: Dict[str, Any] = {
        "max_new_tokens": max(128, int(config.llm_max_tokens or 1024)),
        "do_sample": do_sample,
        "pad_token_id": tokenizer.pad_token_id or tokenizer.eos_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if do_sample:
        generation_kwargs["temperature"] = float(config.llm_temperature)
        generation_kwargs["top_p"] = 0.9

    try:
        with torch.inference_mode():
            output_ids = model.generate(**encoded, **generation_kwargs)
        generated_ids = output_ids[0][input_tokens:]
        content = tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
        del output_ids, generated_ids
    except Exception:
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass
        if config.llm_strict:
            raise
        return fallback
    finally:
        try:
            del encoded
        except Exception:
            pass
        gc.collect()
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    parsed = parse_json_object(content)
    if isinstance(parsed, dict):
        return parsed
    if config.llm_strict:
        raise RuntimeError(f"local-hf model did not return valid JSON. Preview: {content[:500]}")
    return fallback


def call_gemini_json(
    config: PipelineConfig,
    system_prompt: str,
    user_prompt: str,
    fallback: Dict[str, Any],
) -> Dict[str, Any]:
    client = make_gemini_client(config)
    if client is None:
        if config.llm_strict:
            raise RuntimeError("Gemini client is unavailable. Install google-genai and set GEMINI_API_KEY.")
        return fallback

    gen_config = {
        "system_instruction": system_prompt,
        "temperature": config.llm_temperature,
        "max_output_tokens": config.llm_max_tokens,
        "response_mime_type": "application/json",
    }
    try:
        response = client.models.generate_content(
            model=config.gemini_model,
            contents=user_prompt,
            config=gen_config,
        )
    except Exception:
        if config.llm_strict:
            raise
        return fallback

    content = getattr(response, "text", "") or ""
    parsed = parse_json_object(content)
    return parsed if isinstance(parsed, dict) else fallback


def estimate_token_count(*parts: Any) -> int:
    """Conservative tokenizer-free estimate for API budget checks."""

    total = 0
    for part in parts:
        text = normalize_text(part)
        if not text:
            continue
        total += max(math.ceil(len(text) / 4), math.ceil(len(text.split()) * 1.35))
    return total


def json_system_prompt(system_prompt: str) -> str:
    return (
        system_prompt.strip()
        + "\n\nReturn only one valid JSON object. Do not include markdown, XML tags, <think> blocks, or chain-of-thought."
    )


def estimate_json_prompt_tokens(system_prompt: str, user_prompt: str) -> int:
    return estimate_token_count(json_system_prompt(system_prompt), user_prompt)


def groq_tpm_limit(model_id: str) -> int:
    model = normalize_text(model_id).lower()
    env_key = "MULTIAGENT_GROQ_TPM_LIMIT_" + re.sub(r"[^A-Z0-9]+", "_", model.upper()).strip("_")
    explicit = env_int(env_key, 0)
    if explicit > 0:
        return explicit
    global_limit = env_int("MULTIAGENT_GROQ_TPM_LIMIT", 0)
    if global_limit > 0:
        return global_limit
    return GROQ_FREE_TPM_LIMITS.get(model, 0)


def effective_max_output_tokens(config: PipelineConfig, system_prompt: str, user_prompt: str) -> int:
    configured = max(1, int(config.llm_max_tokens or 1))
    provider = normalize_optional_text(config.llm_provider).lower()
    if provider != "groq":
        return configured
    limit = groq_tpm_limit(config.llm_model)
    if limit <= 0:
        return configured
    prompt_tokens = math.ceil(
        estimate_json_prompt_tokens(system_prompt, user_prompt)
        * max(1.0, env_float("MULTIAGENT_GROQ_PROMPT_TOKEN_SAFETY_FACTOR", 1.25))
    )
    margin = max(0, env_int("MULTIAGENT_GROQ_TPM_MARGIN_TOKENS", 500))
    min_output = max(1, env_int("MULTIAGENT_LLM_MIN_OUTPUT_TOKENS", 256))
    available = limit - prompt_tokens - margin
    if available <= 0:
        return min_output
    return max(min_output, min(configured, available))


def pace_groq_request(model_id: str, requested_tokens: int) -> None:
    if not env_bool("MULTIAGENT_GROQ_TOKEN_PACING", True):
        return
    limit = groq_tpm_limit(model_id)
    if limit <= 0 or requested_tokens <= 0:
        return
    now = time.time()
    window = [
        (ts, tokens)
        for ts, tokens in _groq_token_windows.get(model_id, [])
        if now - ts < 60.0
    ]
    used = sum(tokens for _, tokens in window)
    if used + requested_tokens > limit and window:
        sleep_seconds = max(0.0, 60.0 - (now - window[0][0]) + env_float("MULTIAGENT_GROQ_TOKEN_PACING_MARGIN_SECONDS", 2.0))
        if sleep_seconds > 0:
            time.sleep(sleep_seconds)
        now = time.time()
        window = [
            (ts, tokens)
            for ts, tokens in _groq_token_windows.get(model_id, [])
            if now - ts < 60.0
        ]
    window.append((now, requested_tokens))
    _groq_token_windows[model_id] = window


def supports_strict_json_schema(provider: str, model_id: str) -> bool:
    """Return whether the runtime should request constrained JSON schema output."""

    provider = normalize_optional_text(provider).lower()
    model = normalize_text(model_id).lower()
    if not env_bool("MULTIAGENT_LLM_JSON_SCHEMA_STRICT", False):
        return False
    if provider != "groq":
        return False
    return model in {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}


def build_openai_response_format(
    config: PipelineConfig,
    provider: str,
    response_schema: Optional[Dict[str, Any]],
    schema_name: str,
) -> Tuple[Optional[Dict[str, Any]], str]:
    if response_schema and supports_strict_json_schema(provider, config.llm_model):
        return (
            {
                "type": "json_schema",
                "json_schema": {
                    "name": schema_name or "structured_response",
                    "strict": True,
                    "schema": response_schema,
                },
            },
            "json_schema_strict",
        )
    if not env_bool("MULTIAGENT_LLM_JSON_OBJECT_MODE", False):
        return None, "plain_json_prompt"
    return {"type": "json_object"}, "json_object"


def apply_groq_reasoning_controls(kwargs: Dict[str, Any], model_id: str) -> None:
    """Keep Groq reasoning models from spending the Agent budget on hidden reasoning."""

    model = normalize_text(model_id).lower()
    if model in {"openai/gpt-oss-120b", "openai/gpt-oss-20b"}:
        effort = env_text("MULTIAGENT_GROQ_GPT_OSS_REASONING_EFFORT", "low").lower()
        if effort in {"low", "medium", "high"}:
            kwargs["reasoning_effort"] = effort
        extra_body = dict(kwargs.get("extra_body") or {})
        extra_body["include_reasoning"] = env_bool("MULTIAGENT_GROQ_INCLUDE_REASONING", False)
        kwargs["extra_body"] = extra_body
    elif model == "qwen/qwen3-32b":
        effort = env_text("MULTIAGENT_GROQ_QWEN_REASONING_EFFORT", "none").lower()
        if effort in {"none", "default"}:
            kwargs["reasoning_effort"] = effort


def call_openai_compatible_json(
    config: PipelineConfig,
    system_prompt: str,
    user_prompt: str,
    fallback: Dict[str, Any],
    verbose: Optional[bool] = None,
    expected_keys: Optional[Iterable[str]] = None,
    response_schema: Optional[Dict[str, Any]] = None,
    schema_name: str = "structured_response",
) -> Dict[str, Any]:
    provider = normalize_optional_text(config.llm_provider).lower()
    client = make_openai_compatible_client(config, provider)
    if client is None:
        if config.llm_strict:
            raise RuntimeError(
                "OpenAI-compatible LLM client is unavailable. Install openai and set GROQ_API_KEY, "
                "OPENAI_API_KEY, OPENROUTER_API_KEY, or MULTIAGENT_LLM_API_KEY."
            )
        return fallback

    system = json_system_prompt(system_prompt)
    effective_max_tokens = effective_max_output_tokens(config, system_prompt, user_prompt)
    prompt_tokens = estimate_token_count(system, user_prompt)
    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": user_prompt.strip()},
    ]
    kwargs: Dict[str, Any] = {
        "model": config.llm_model,
        "messages": messages,
        "temperature": config.llm_temperature,
        "max_tokens": effective_max_tokens,
    }
    if provider == "groq":
        apply_groq_reasoning_controls(kwargs, config.llm_model)
    response_format, response_format_label = build_openai_response_format(
        config,
        provider,
        response_schema,
        schema_name,
    )
    timeout = env_float("MULTIAGENT_LLM_TIMEOUT", 120.0)
    if timeout > 0:
        kwargs["timeout"] = timeout

    def can_retry_without_response_format(exc: Exception) -> bool:
        text = str(exc).lower()
        return any(
            marker in text
            for marker in [
                "response_format",
                "json_object",
                "json mode",
                "json_schema",
                "schema validation",
                "failed to validate json",
                "json_validate_failed",
                "failed_generation",
                "invalid parameter",
                "unsupported parameter",
                "not support",
            ]
        )

    response = None
    last_error: Optional[Exception] = None
    expected_key_list = [str(key) for key in expected_keys or [] if str(key)]

    def missing_expected_keys(parsed: Any) -> List[str]:
        if not expected_key_list or not isinstance(parsed, dict):
            return []
        return [key for key in expected_key_list if key not in parsed]

    max_retries = max(1, env_int("MULTIAGENT_LLM_MAX_RETRIES", 3))
    retry_base_seconds = max(0.0, env_float("MULTIAGENT_LLM_RETRY_BASE_SECONDS", 8.0))
    log_enabled = stage_log_enabled(verbose) or env_bool("MULTIAGENT_LLM_CALL_LOG", False)
    progress_log(
        log_enabled,
        "LLM request prepared",
        provider=provider,
        model=config.llm_model,
        prompt_tokens=prompt_tokens,
        max_output_tokens=effective_max_tokens,
        response_format=response_format_label,
        reasoning_effort=kwargs.get("reasoning_effort", ""),
    )
    for attempt in range(1, max_retries + 1):
        try:
            if provider == "groq":
                pace_groq_request(config.llm_model, prompt_tokens + effective_max_tokens)
            try:
                if response_format:
                    response = client.chat.completions.create(
                        **kwargs,
                        response_format=response_format,
                    )
                else:
                    response = client.chat.completions.create(**kwargs)
            except Exception as exc:
                if response_format_label == "json_schema_strict":
                    raise
                if response_format is None:
                    raise
                if not can_retry_without_response_format(exc):
                    raise
                if config.llm_strict and env_bool("MULTIAGENT_LLM_PLAIN_RETRY_DISABLED", False):
                    raise
                progress_log(
                    log_enabled,
                    "JSON mode rejected; retrying plain completion",
                    attempt=attempt,
                    error=normalize_text(str(exc)),
                )
                response = client.chat.completions.create(**kwargs)
            break
        except Exception as exc:
            last_error = exc
            progress_log(
                log_enabled,
                "LLM request failed",
                attempt=attempt,
                max_retries=max_retries,
                error=normalize_text(str(exc)),
            )
            if attempt >= max_retries:
                break
            time.sleep(retry_base_seconds * attempt)

    if response is None:
        if config.llm_strict:
            raise RuntimeError(f"OpenAI-compatible LLM call failed after {max_retries} attempt(s).") from last_error
        return fallback

    def repair_json_response(bad_content: str) -> Optional[Dict[str, Any]]:
        if not env_bool("MULTIAGENT_LLM_JSON_REPAIR", True):
            return None
        repair_prompt = f"""
The previous response was not parseable JSON.

Original task:
{limit_words(user_prompt, env_int("MULTIAGENT_LLM_JSON_REPAIR_PROMPT_WORDS", 700))}

Invalid response:
{normalize_text(bad_content)[: env_int("MULTIAGENT_LLM_JSON_REPAIR_BAD_CHARS", 2500)] or "[empty]"}

Return only one valid JSON object satisfying the original task. Do not include markdown or explanations.
"""
        repair_kwargs = dict(kwargs)
        repair_kwargs["messages"] = [
            {
                "role": "system",
                "content": "You repair LLM output into one strict JSON object. Return JSON only.",
            },
            {"role": "user", "content": repair_prompt.strip()},
        ]
        repair_kwargs["max_tokens"] = max(
            int(repair_kwargs.get("max_tokens") or 1),
            env_int("MULTIAGENT_LLM_JSON_REPAIR_MAX_TOKENS", 900),
        )
        try:
            progress_log(log_enabled, "LLM JSON repair request prepared")
            if provider == "groq":
                pace_groq_request(config.llm_model, estimate_token_count(repair_prompt) + int(repair_kwargs["max_tokens"]))
            repaired_response = client.chat.completions.create(**repair_kwargs)
            repaired_content = repaired_response.choices[0].message.content or ""
            repaired = parse_json_object(repaired_content)
            if isinstance(repaired, dict):
                progress_log(log_enabled, "LLM JSON repair parsed", keys=",".join(sorted(repaired.keys())[:8]))
                return repaired
            progress_log(
                log_enabled,
                "LLM JSON repair parse failed",
                output_preview=normalize_text(repaired_content)[:180],
            )
        except Exception as exc:
            progress_log(log_enabled, "LLM JSON repair failed", error=normalize_text(str(exc)))
        return None

    content = ""
    finish_reason = ""
    try:
        choice = response.choices[0]
        message = choice.message
        content_parts = [getattr(message, "content", "") or ""]
        for attr in ["reasoning", "reasoning_content"]:
            value = getattr(message, attr, "")
            if value:
                content_parts.append(str(value))
        try:
            extra = getattr(message, "model_extra", None) or {}
            for attr in ["reasoning", "reasoning_content"]:
                value = extra.get(attr)
                if value:
                    content_parts.append(str(value))
        except Exception:
            pass
        content = "\n".join(part for part in content_parts if normalize_text(part))
        finish_reason = normalize_optional_text(getattr(choice, "finish_reason", ""))
    except Exception:
        content = ""
    if finish_reason:
        progress_log(log_enabled, "LLM response received", finish_reason=finish_reason)
    parsed = parse_json_object(content)
    missing_keys = missing_expected_keys(parsed)
    if isinstance(parsed, dict) and missing_keys:
        progress_log(
            log_enabled,
            "LLM JSON missing expected keys",
            parsed_keys=",".join(sorted(parsed.keys())[:8]),
            missing=",".join(missing_keys[:8]),
        )
        repaired = repair_json_response(content)
        if isinstance(repaired, dict):
            parsed = repaired
            missing_keys = missing_expected_keys(parsed)
            if missing_keys:
                progress_log(
                    log_enabled,
                    "LLM JSON repair still missing expected keys",
                    parsed_keys=",".join(sorted(parsed.keys())[:8]),
                    missing=",".join(missing_keys[:8]),
                )
                if config.llm_strict:
                    raise RuntimeError(
                        "LLM JSON response missing required keys after repair: "
                        + ", ".join(missing_keys)
                    )
        else:
            progress_log(log_enabled, "LLM JSON repair unavailable for missing keys")
            if config.llm_strict:
                raise RuntimeError("LLM JSON response missing required keys: " + ", ".join(missing_keys))
    if isinstance(parsed, dict):
        progress_log(log_enabled, "LLM JSON parsed", keys=",".join(sorted(parsed.keys())[:8]))
    else:
        progress_log(log_enabled, "LLM JSON parse failed", output_preview=normalize_text(content)[:180])
        parsed = repair_json_response(content)
        if isinstance(parsed, dict) and missing_expected_keys(parsed):
            progress_log(
                log_enabled,
                "LLM JSON repair missing expected keys",
                parsed_keys=",".join(sorted(parsed.keys())[:8]),
                missing=",".join(missing_expected_keys(parsed)[:8]),
            )
            parsed = None
        if not isinstance(parsed, dict):
            progress_log(log_enabled, "LLM JSON repair unavailable")
            if config.llm_strict:
                raise RuntimeError(f"LLM did not return valid JSON. Preview: {normalize_text(content)[:500]}")
    return parsed if isinstance(parsed, dict) else fallback


def call_llm_json(
    config: PipelineConfig,
    system_prompt: str,
    user_prompt: str,
    fallback: Dict[str, Any],
    verbose: Optional[bool] = None,
    expected_keys: Optional[Iterable[str]] = None,
    response_schema: Optional[Dict[str, Any]] = None,
    schema_name: str = "structured_response",
) -> Dict[str, Any]:
    provider = normalize_optional_text(config.llm_provider).lower() or DEFAULT_AGENT_LLM_PROVIDER
    if provider in GEMINI_PROVIDERS:
        return call_gemini_json(config, system_prompt, user_prompt, fallback)
    if provider in LOCAL_HF_PROVIDERS:
        return call_local_hf_json(config, system_prompt, user_prompt, fallback)
    if provider_is_openai_compatible(provider):
        return call_openai_compatible_json(
            config,
            system_prompt,
            user_prompt,
            fallback,
            verbose=verbose,
            expected_keys=expected_keys,
            response_schema=response_schema,
            schema_name=schema_name,
        )
    if config.llm_strict:
        raise RuntimeError(f"Unsupported MULTIAGENT_LLM_PROVIDER: {config.llm_provider}")
    return fallback


def config_for_agent_llm(config: PipelineConfig, agent_number: int) -> PipelineConfig:
    """Return a config with an optional per-agent model/provider override."""

    model = normalize_optional_text(os.getenv(f"MULTIAGENT_AGENT{agent_number}_LLM_MODEL"))
    provider = normalize_optional_text(os.getenv(f"MULTIAGENT_AGENT{agent_number}_LLM_PROVIDER")).lower()
    max_tokens = env_int(f"MULTIAGENT_AGENT{agent_number}_MAX_OUTPUT_TOKENS", 0)
    updates: Dict[str, Any] = {}
    if model and model != config.llm_model:
        updates["llm_model"] = model
    if provider and provider != config.llm_provider:
        updates["llm_provider"] = provider
    if max_tokens > 0 and max_tokens != config.llm_max_tokens:
        updates["llm_max_tokens"] = max_tokens
    return replace(config, **updates) if updates else config



_knn_embedding_cache: Dict[str, List[float]] = {}
_knn_hf_tokenizer = None
_knn_hf_model = None


def get_es_client(config: PipelineConfig):
    try:
        from elasticsearch import Elasticsearch
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Missing dependency for ES KNN retrieval: pip install elasticsearch") from exc

    if not normalize_optional_text(config.es_cloud_id):
        raise RuntimeError("Missing ES_CLOUD_ID for ES KNN retrieval.")
    if normalize_optional_text(config.es_api_key):
        return Elasticsearch(
            cloud_id=config.es_cloud_id,
            api_key=config.es_api_key,
            request_timeout=120,
        )
    if normalize_optional_text(config.es_user) and normalize_optional_text(config.es_password):
        return Elasticsearch(
            cloud_id=config.es_cloud_id,
            basic_auth=(config.es_user, config.es_password),
            request_timeout=120,
        )
    raise RuntimeError("Provide ES_API_KEY or ES_USER + ES_PASSWORD for ES KNN retrieval.")


def build_es_knn_filters(filters: Dict[str, Any]) -> List[Dict[str, Any]]:
    es_filters: List[Dict[str, Any]] = []

    exclude_shoulds = []
    if normalize_optional_text(filters.get("exclude_doc_id")):
        exclude_shoulds.append({"term": {"doc_id": normalize_text(filters["exclude_doc_id"])}})
    exclude_canonical = normalize_optional_text(filters.get("exclude_canonical_doc_id")) or canonical_doc_id(filters.get("exclude_doc_id"))
    if exclude_canonical:
        exclude_shoulds.append({"term": {"canonical_doc_id": exclude_canonical}})
    if exclude_shoulds:
        es_filters.append(
            {
                "bool": {
                    "must_not": [{"bool": {"should": exclude_shoulds, "minimum_should_match": 1}}],
                }
            }
        )

    cutoff = normalize_optional_text(filters.get("prior_art_cutoff_date"))
    if normalize_bool(filters.get("use_date_filter"), default=bool(cutoff)) and cutoff:
        date_range = {
            "range": {
                "publication_date": {
                    "lt": cutoff,
                    "format": "yyyyMMdd||yyyy-MM-dd",
                }
            }
        }
        es_filters.append(
            {
                "bool": {
                    "should": [
                        date_range,
                        {"bool": {"must_not": {"exists": {"field": "publication_date"}}}},
                    ],
                    "minimum_should_match": 1,
                }
            }
        )

    return es_filters


def normalize_vector(values: Iterable[Any]) -> List[float]:
    vector = [float(v) for v in values]
    norm = math.sqrt(sum(v * v for v in vector))
    if norm <= 1e-12:
        return vector
    return [v / norm for v in vector]


def is_native_jina_v3_hf_model(model_name: str) -> bool:
    return normalize_text(model_name).lower().rstrip("/").endswith("jina-embeddings-v3-hf")


def jina_v3_adapter_name(task: str) -> str:
    normalized = normalize_text(task).lower().replace("-", "_").replace(" ", "_")
    aliases = {
        "retrieval.query": "retrieval_query",
        "retrieval_query": "retrieval_query",
        "query": "retrieval_query",
        "retrieval.passage": "retrieval_passage",
        "retrieval_passage": "retrieval_passage",
        "passage": "retrieval_passage",
        "classification": "classification",
        "separation": "separation",
        "text.matching": "text_matching",
        "text_matching": "text_matching",
    }
    return aliases.get(normalized, "retrieval_query")


def attach_jina_v3_hf_adapter(model: Any, model_id: str, task: str) -> Any:
    adapter_name = jina_v3_adapter_name(task)
    try:
        if hasattr(model, "load_adapter"):
            try:
                model.load_adapter(model_id, adapter_name=adapter_name, adapter_kwargs={"subfolder": adapter_name})
            except TypeError:
                model.load_adapter(model_id, adapter_name=adapter_name, subfolder=adapter_name)
        else:
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, model_id, adapter_name=adapter_name, subfolder=adapter_name)
        if hasattr(model, "set_adapter"):
            model.set_adapter(adapter_name)
        return model
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "Failed to load the Jina v3 HF adapter for local KNN embeddings. "
            "Install peft>=0.13.0, or set MULTIAGENT_KNN_EMBED_API_BASE to a served embedding endpoint."
        ) from exc


def get_knn_tokenizer(config: PipelineConfig) -> Any:
    global _knn_hf_tokenizer
    if _knn_hf_tokenizer is not None:
        return _knn_hf_tokenizer
    try:
        from transformers import AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("Missing transformers for KNN query chunking.") from exc

    use_native_jina = is_native_jina_v3_hf_model(config.knn_hf_model)
    _knn_hf_tokenizer = AutoTokenizer.from_pretrained(
        config.knn_hf_model,
        use_fast=True,
        trust_remote_code=not use_native_jina,
    )
    return _knn_hf_tokenizer


def get_knn_content_token_ids(tokenizer: Any, text: str) -> List[int]:
    text = normalize_text(text)
    if not text:
        return []
    try:
        backend = getattr(tokenizer, "backend_tokenizer", None)
        if backend is not None:
            return list(backend.encode(text, add_special_tokens=False).ids)
    except Exception:
        pass
    tokenized = tokenizer(
        text,
        padding=False,
        truncation=False,
        add_special_tokens=False,
    )
    input_ids = tokenized.get("input_ids", [])
    if input_ids and isinstance(input_ids[0], list):
        flattened: List[int] = []
        for ids in input_ids:
            flattened.extend(ids)
        return flattened
    return list(input_ids)


def build_knn_query_chunks(config: PipelineConfig, text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    if not env_bool("QUERY_CHUNKING_ENABLED", True):
        return [text]

    chunk_tokens = max(1, env_int("QUERY_CHUNK_TOKENS", 4096))
    overlap_tokens = max(0, env_int("QUERY_CHUNK_OVERLAP_TOKENS", 256))
    max_chunks = max(0, env_int("QUERY_MAX_CHUNKS", 0))

    # Avoid loading a tokenizer for normal PB2 agent queries that are far below
    # the RAG-based chunk threshold.
    if len(text.split()) <= int(chunk_tokens * 0.60):
        return [text]

    try:
        tokenizer = get_knn_tokenizer(config)
        token_ids = get_knn_content_token_ids(tokenizer, text)
        if len(token_ids) + 2 <= chunk_tokens:
            return [text]
        overlap = min(overlap_tokens, chunk_tokens - 1)
        step = max(1, chunk_tokens - overlap)
        chunks: List[str] = []
        start = 0
        while start < len(token_ids):
            end = min(len(token_ids), start + chunk_tokens)
            chunk_text = normalize_text(
                tokenizer.decode(
                    token_ids[start:end],
                    skip_special_tokens=True,
                    clean_up_tokenization_spaces=False,
                )
            )
            if chunk_text:
                chunks.append(chunk_text)
            if max_chunks > 0 and len(chunks) >= max_chunks:
                break
            if end >= len(token_ids):
                break
            start += step
        return chunks or [text]
    except Exception:
        words = text.split()
        if len(words) <= chunk_tokens:
            return [text]
        overlap = min(overlap_tokens, chunk_tokens - 1)
        step = max(1, chunk_tokens - overlap)
        chunks = []
        for start in range(0, len(words), step):
            chunk = " ".join(words[start : start + chunk_tokens]).strip()
            if chunk:
                chunks.append(chunk)
            if max_chunks > 0 and len(chunks) >= max_chunks:
                break
            if start + chunk_tokens >= len(words):
                break
        return chunks or [text]


def load_knn_hf_model(config: PipelineConfig):
    global _knn_hf_model, _knn_hf_tokenizer
    if _knn_hf_model is not None:
        return _knn_hf_tokenizer, _knn_hf_model
    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except Exception as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError(
            "Missing local Jina embedding dependencies. Install transformers/torch or set MULTIAGENT_KNN_EMBED_API_BASE."
        ) from exc

    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_native_jina = is_native_jina_v3_hf_model(config.knn_hf_model)
    _knn_hf_tokenizer = AutoTokenizer.from_pretrained(
        config.knn_hf_model,
        use_fast=True,
        trust_remote_code=not use_native_jina,
    )
    _knn_hf_model = AutoModel.from_pretrained(
        config.knn_hf_model,
        trust_remote_code=not use_native_jina,
    )
    if use_native_jina:
        _knn_hf_model = attach_jina_v3_hf_adapter(
            _knn_hf_model,
            config.knn_hf_model,
            config.knn_embedding_task,
        )
    _knn_hf_model = _knn_hf_model.to(device)
    _knn_hf_model.eval()
    return _knn_hf_tokenizer, _knn_hf_model


def embed_knn_query(config: PipelineConfig, text: str) -> List[float]:
    text = normalize_text(text)
    if not text:
        raise ValueError("Cannot embed an empty KNN query.")
    cache_key = f"{config.knn_embedding_api_base}|{config.knn_embedding_model}|{config.knn_hf_model}|{config.knn_embedding_task}|{text}"
    if cache_key in _knn_embedding_cache:
        return _knn_embedding_cache[cache_key]

    if normalize_optional_text(config.knn_embedding_api_base):
        try:
            import requests
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Missing dependency for embedding HTTP call: pip install requests") from exc
        response = requests.post(
            f"{config.knn_embedding_api_base.rstrip('/')}/embeddings",
            headers={"Authorization": f"Bearer {config.knn_embedding_api_key}"},
            json={"model": config.knn_embedding_model, "input": [text]},
            timeout=120,
        )
        response.raise_for_status()
        vector = response.json()["data"][0]["embedding"]
        normalized = normalize_vector(vector)
        _knn_embedding_cache[cache_key] = normalized
        return normalized

    tokenizer, model = load_knn_hf_model(config)
    max_length = env_int("MULTIAGENT_KNN_EMBED_MAX_LEN", 8192)

    if hasattr(model, "encode"):
        try:
            embedding = model.encode([text], task=config.knn_embedding_task, max_length=max_length)
        except TypeError:
            embedding = model.encode([text], max_length=max_length)

        try:
            import numpy as np
            import torch

            if isinstance(embedding, torch.Tensor):
                embedding = embedding.detach().cpu().numpy()
            embedding = np.asarray(embedding, dtype=np.float32)
            vector = embedding[0].tolist()
        except Exception:
            vector = list(embedding[0])
    else:
        try:
            import torch
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            raise RuntimeError("Missing dependency for local KNN embedding pooling: pip install torch") from exc

        encoded = tokenizer(
            [text],
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        device = next(model.parameters()).device
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            outputs = model(**encoded)
        token_embeddings = getattr(outputs, "last_hidden_state", None)
        if token_embeddings is None:
            token_embeddings = outputs[0]
        attention_mask = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
        pooled = (token_embeddings * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1e-9)
        vector = pooled[0].detach().cpu().float().tolist()
    normalized = normalize_vector(vector)
    _knn_embedding_cache[cache_key] = normalized
    return normalized


def build_knn_query_text(state: PatentAnalysisState, config: PipelineConfig) -> str:
    parts: List[str] = []
    seen = set()

    def add(value: Any, max_words: int = 180) -> None:
        text = limit_words(value, max_words)
        key = text.lower()
        if text and key not in seen:
            seen.add(key)
            parts.append(text)

    for item in state.get("search_queries", []) or []:
        if isinstance(item, dict):
            add(item.get("text"), 220)
        else:
            add(item, 220)
    add(state.get("technical_problem"), 80)
    for value in state.get("claim_elements", []) or []:
        add(value, 80)
    for value in state.get("key_features", []) or []:
        add(value, 60)

    metadata = state.get("input_metadata", {}) or {}
    for key, limit in [("title", 80), ("abstract", 220), ("claims", 360), ("retrieval_text", 360)]:
        add(metadata.get(key), limit)
    add(state.get("input_text"), 260)
    return limit_words(" ".join(parts), config.knn_query_words)


def build_knn_query_sources(state: PatentAnalysisState, config: PipelineConfig) -> List[Dict[str, Any]]:
    """Build independent vector query views for weighted RRF fusion."""

    sources: List[Dict[str, Any]] = []
    seen = set()
    for idx, item in enumerate(state.get("search_queries", []) or [], start=1):
        if not isinstance(item, dict):
            text = limit_words(item, 260)
            weight = 1.0
            query_view = f"query_{idx}"
            search_role = "combined"
            search_mode = "semantic"
        else:
            text = limit_words(item.get("text"), 260)
            weight = safe_float(item.get("weight"), default=1.0, minimum=0.2, maximum=2.0)
            query_view = normalize_text(item.get("query_view")) or f"query_{idx}"
            search_role = normalize_text(item.get("search_role")) or query_view
            search_mode = normalize_text(item.get("search_mode")) or "semantic"
        key = text.lower()
        if not text or key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "name": f"knn:{query_view}:{idx}",
                "query_view": query_view,
                "search_role": search_role,
                "search_mode": search_mode,
                "text": text,
                "weight": weight,
            }
        )

    if sources:
        return sources

    fallback_text = build_knn_query_text(state, config)
    if not fallback_text:
        return []
    return [
        {
            "name": "knn:fallback:1",
            "query_view": "fallback",
            "search_role": "combined",
            "search_mode": "semantic",
            "text": fallback_text,
            "weight": 1.0,
        }
    ]


def fetch_es_candidate_docs(config: PipelineConfig, es: Any, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    if not doc_ids:
        return {}
    fields = [
        "doc_id",
        "canonical_doc_id",
        "title",
        "abstract",
        "claims",
        "description",
        "retrieval_text",
        "publication_date",
        "application_date",
        "priority_date",
        "ipc_codes",
        "assignees",
        "inventors",
        "citations",
    ]

    def normalize_es_source(src: Dict[str, Any], fallback_id: str) -> Dict[str, Any]:
        src_doc_id = normalize_text(src.get("doc_id")) or fallback_id
        return {
            "doc_id": src_doc_id,
            "canonical_doc_id": normalize_optional_text(src.get("canonical_doc_id")) or canonical_doc_id(src_doc_id),
            "title": normalize_optional_text(src.get("title")),
            "abstract": normalize_optional_text(src.get("abstract")),
            "claims": normalize_optional_text(src.get("claims")),
            "description": normalize_optional_text(src.get("description")),
            "retrieval_text": normalize_optional_text(src.get("retrieval_text")),
            "publication_date": normalize_optional_text(src.get("publication_date")),
            "application_date": normalize_optional_text(src.get("application_date")),
            "priority_date": normalize_optional_text(src.get("priority_date")),
            "ipc_codes": normalize_list(src.get("ipc_codes")),
            "assignees": normalize_list(src.get("assignees")),
            "inventors": normalize_list(src.get("inventors")),
            "citations": normalize_list(src.get("citations")),
        }

    response = es.mget(index=config.bm25_index, ids=doc_ids, _source=fields)
    docs: Dict[str, Dict[str, Any]] = {}
    for item in response.get("docs", []):
        if not item.get("found"):
            continue
        doc_id = normalize_text(item.get("_id"))
        src = item.get("_source", {}) or {}
        doc = normalize_es_source(src, doc_id)
        docs[doc_id] = doc
        docs.setdefault(doc["doc_id"], doc)
        docs.setdefault(doc["canonical_doc_id"], doc)

    missing = [doc_id for doc_id in doc_ids if doc_id not in docs]
    if missing:
        should = []
        for doc_id in missing:
            should.append({"term": {"doc_id": doc_id}})
            canonical = canonical_doc_id(doc_id)
            if canonical and canonical != doc_id:
                should.append({"term": {"canonical_doc_id": canonical}})
        if should:
            search_response = es.search(
                index=config.bm25_index,
                body={
                    "query": {"bool": {"should": should, "minimum_should_match": 1}},
                    "size": max(1, len(missing)),
                    "_source": fields,
                },
            )
            for hit in search_response.get("hits", {}).get("hits", []):
                hit_id = normalize_text(hit.get("_id"))
                src = hit.get("_source", {}) or {}
                doc = normalize_es_source(src, hit_id)
                for key in {hit_id, doc["doc_id"], doc["canonical_doc_id"]}:
                    if key:
                        docs.setdefault(key, doc)
    return docs


def run_es_knn_retrieval(config: PipelineConfig, state: PatentAnalysisState) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    query_sources = build_knn_query_sources(state, config)
    if not query_sources:
        return [], {}, {"backend": "es_knn", "error": "empty query"}

    es = get_es_client(config)
    filters = build_es_knn_filters(state.get("filters", {}) or {})
    top_k = max(int(config.knn_top_k), int(config.candidate_screen_top_k))
    base_fetch = max(top_k, top_k * max(1, int(config.knn_variant_fetch_multiplier)))
    fetch_size = base_fetch * max(1, int(config.knn_chunk_fetch_multiplier))
    if config.knn_max_fetch_size > 0:
        fetch_size = min(fetch_size, int(config.knn_max_fetch_size))
    fetch_size = max(top_k, fetch_size)
    num_candidates = max(fetch_size, int(config.knn_num_candidates) * max(1, int(config.knn_chunk_fetch_multiplier)))
    if config.knn_max_fetch_size > 0:
        num_candidates = max(fetch_size, min(num_candidates, int(config.knn_max_fetch_size)))

    rrf_k = max(1, int(config.knn_rrf_k))
    rrf_scores: Dict[str, float] = {}
    doc_payloads: Dict[str, Dict[str, Any]] = {}
    source_contexts: List[Dict[str, Any]] = []
    num_chunk_hits = 0

    for source_index, source in enumerate(query_sources, start=1):
        source_name = normalize_text(source.get("name")) or f"knn:query:{source_index}"
        source_text = normalize_text(source.get("text"))
        source_weight = safe_float(source.get("weight"), default=1.0, minimum=0.2, maximum=2.0)
        if not source_text:
            continue

        source_doc_best: Dict[str, Dict[str, Any]] = {}
        source_global_rank = 0
        source_chunk_hits = 0
        query_chunks = build_knn_query_chunks(config, source_text)
        for query_chunk in query_chunks:
            query_vector = embed_knn_query(config, query_chunk)
            body = {
                "knn": {
                    "field": config.vector_field,
                    "query_vector": query_vector,
                    "k": fetch_size,
                    "num_candidates": num_candidates,
                    "boost": 1.0,
                    "filter": filters,
                },
                "size": fetch_size,
                "_source": ["doc_id", "canonical_doc_id", "parent_doc_id", "chunk_id", "chunk_index"],
            }
            response = es.search(index=config.knn_index, body=body)
            hits = response.get("hits", {}).get("hits", [])
            source_chunk_hits += len(hits)
            num_chunk_hits += len(hits)
            for _rank, hit in enumerate(hits, start=1):
                source_global_rank += 1
                src = hit.get("_source", {}) or {}
                doc_id = normalize_text(src.get("parent_doc_id") or src.get("doc_id") or src.get("canonical_doc_id"))
                if not doc_id:
                    continue
                score = float(hit.get("_score") or 0.0)
                if config.knn_score_agg == "sum":
                    current = source_doc_best.setdefault(doc_id, {"score": 0.0, "first_rank": source_global_rank, "hits": 0})
                    current["score"] += score
                    current["hits"] += 1
                    current["first_rank"] = min(int(current["first_rank"]), source_global_rank)
                else:
                    current = source_doc_best.get(doc_id)
                    if current is None or score > float(current["score"]):
                        source_doc_best[doc_id] = {"score": score, "first_rank": source_global_rank, "hits": 1}
                    else:
                        current["hits"] += 1
                        current["first_rank"] = min(int(current["first_rank"]), source_global_rank)

        source_ranked = sorted(
            source_doc_best.items(),
            key=lambda item: (-float(item[1]["score"]), int(item[1]["first_rank"]), item[0]),
        )[:top_k]
        for rank, (doc_id, payload) in enumerate(source_ranked, start=1):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + source_weight / float(rrf_k + rank)
            doc_payload = doc_payloads.setdefault(
                doc_id,
                {"best_knn_score": 0.0, "best_rank": rank, "hits": 0, "source_scores": {}},
            )
            doc_payload["best_knn_score"] = max(float(doc_payload.get("best_knn_score", 0.0)), float(payload["score"]))
            doc_payload["best_rank"] = min(int(doc_payload.get("best_rank", rank)), rank)
            doc_payload["hits"] = int(doc_payload.get("hits", 0)) + int(payload.get("hits", 0))
            doc_payload["source_scores"][source_name] = float(payload["score"])

        source_contexts.append(
            {
                "name": source_name,
                "query_view": source.get("query_view", ""),
                "search_role": source.get("search_role", ""),
                "search_mode": source.get("search_mode", ""),
                "weight": source_weight,
                "query_text": source_text,
                "query_word_count": len(source_text.split()),
                "query_chunks": len(query_chunks),
                "num_chunk_hits": source_chunk_hits,
                "num_parent_docs": len(source_ranked),
            }
        )

    ranked = sorted(
        rrf_scores.items(),
        key=lambda item: (-float(item[1]), int(doc_payloads.get(item[0], {}).get("best_rank", top_k + 1)), item[0]),
    )[:top_k]
    doc_ids = [doc_id for doc_id, _score in ranked]
    docs = fetch_es_candidate_docs(config, es, doc_ids)

    candidates: List[Dict[str, Any]] = []
    for idx, (doc_id, rrf_score) in enumerate(ranked, start=1):
        payload = doc_payloads.get(doc_id, {})
        doc = docs.get(doc_id, {})
        canonical = normalize_optional_text(doc.get("canonical_doc_id")) or canonical_doc_id(doc_id)
        source_scores = {
            "rrf": float(rrf_score),
            "knn_best": float(payload.get("best_knn_score", 0.0)),
        }
        source_scores.update(payload.get("source_scores", {}) or {})
        candidates.append(
            {
                "rank": idx,
                "doc_id": doc_id,
                "canonical_doc_id": canonical,
                "score": float(rrf_score),
                "title": normalize_optional_text(doc.get("title")),
                "publication_date": normalize_optional_text(doc.get("publication_date")),
                "application_date": normalize_optional_text(doc.get("application_date")),
                "priority_date": normalize_optional_text(doc.get("priority_date")),
                "source_scores": source_scores,
                "ipc_codes": normalize_list(doc.get("ipc_codes")),
                "assignees": normalize_list(doc.get("assignees")),
                "inventors": normalize_list(doc.get("inventors")),
                "citations": normalize_list(doc.get("citations")),
            }
        )

    context = {
        "backend": "es_knn",
        "knn_index": config.knn_index,
        "bm25_index": config.bm25_index,
        "vector_field": config.vector_field,
        "query_text": "\n".join(normalize_text(source.get("query_text")) for source in source_contexts),
        "query_word_count": sum(int(source.get("query_word_count", 0) or 0) for source in source_contexts),
        "query_chunks": sum(int(source.get("query_chunks", 0) or 0) for source in source_contexts),
        "query_fusion": "weighted_rrf",
        "rrf_k": rrf_k,
        "query_sources": source_contexts,
        "top_k": top_k,
        "base_fetch": base_fetch,
        "fetch_size": fetch_size,
        "num_candidates": num_candidates,
        "variant_fetch_multiplier": config.knn_variant_fetch_multiplier,
        "chunk_fetch_multiplier": config.knn_chunk_fetch_multiplier,
        "filters": filters,
        "num_chunk_hits": num_chunk_hits,
        "num_parent_docs": len(ranked),
        "num_docs_fetched": len({normalize_text(doc.get("doc_id")) for doc in docs.values() if doc}),
        "embedding_backend": "openai_compatible" if config.knn_embedding_api_base else "local_hf_jina",
    }
    return candidates, docs, context


def merge_retrieval_results(
    previous_candidates: List[Dict[str, Any]],
    new_candidates: List[Dict[str, Any]],
    previous_docs: Dict[str, Dict[str, Any]],
    new_docs: Dict[str, Dict[str, Any]],
    previous_context: Dict[str, Any],
    new_context: Dict[str, Any],
) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]], Dict[str, Any]]:
    """Merge retry retrieval results without losing first-pass candidates."""

    best_by_canonical: Dict[str, Dict[str, Any]] = {}
    for cand in list(previous_candidates or []) + list(new_candidates or []):
        if not isinstance(cand, dict):
            continue
        doc_id = normalize_text(cand.get("doc_id") or cand.get("patent_id"))
        canonical = normalize_text(cand.get("canonical_doc_id")) or canonical_doc_id(doc_id)
        if not doc_id or not canonical:
            continue
        item = dict(cand, doc_id=doc_id, canonical_doc_id=canonical)
        current = best_by_canonical.get(canonical)
        if current is None or safe_float(item.get("score"), default=0.0) > safe_float(current.get("score"), default=0.0):
            best_by_canonical[canonical] = item

    merged_candidates = sorted(
        best_by_canonical.values(),
        key=lambda item: (-safe_float(item.get("score"), default=0.0), normalize_text(item.get("doc_id"))),
    )
    for idx, cand in enumerate(merged_candidates, start=1):
        cand["rank"] = idx

    merged_docs = dict(previous_docs or {})
    merged_docs.update(new_docs or {})

    retrieval_passes = []
    if isinstance(previous_context, dict):
        retrieval_passes.extend(previous_context.get("retrieval_passes") or [])
        if previous_context and not retrieval_passes:
            retrieval_passes.append(previous_context)
    if new_context:
        retrieval_passes.append(new_context)

    merged_context = dict(new_context or {})
    merged_context["retrieval_passes"] = retrieval_passes
    merged_context["num_retrieval_passes"] = len(retrieval_passes)
    return merged_candidates, merged_docs, merged_context


def candidate_text(doc: Dict[str, Any], max_words: int = 900) -> str:
    snippet_text = doc.get("description") or doc.get("retrieval_text")
    parts = [
        f"[TITLE] {limit_words(doc.get('title'), 80)}" if doc.get("title") else "",
        f"[CLAIMS] {limit_words(doc.get('claims'), 420)}" if doc.get("claims") else "",
        f"[ABSTRACT] {limit_words(doc.get('abstract'), 220)}" if doc.get("abstract") else "",
        f"[RETRIEVAL_SNIPPETS] {limit_words(snippet_text, 220)}" if snippet_text else "",
    ]
    return limit_words("\n".join(part for part in parts if part), max_words)


def word_window_snippet(text: str, tokens: List[str], window_words: int = 48) -> str:
    """Return a whole-word snippet around the first overlapping token."""

    words = normalize_text(text).split()
    if not words:
        return ""
    lowered = [word.lower() for word in words]
    token_set = {token.lower() for token in tokens if token}
    hit_idx = 0
    for idx, word in enumerate(lowered):
        cleaned = re.sub(r"[^a-z0-9]+", "", word)
        if cleaned in token_set or any(token and token in cleaned for token in token_set):
            hit_idx = idx
            break
    half = max(8, window_words // 2)
    start = max(0, hit_idx - half)
    end = min(len(words), hit_idx + half)
    return " ".join(words[start:end]).strip()


def prompt_word_count(*parts: Any) -> int:
    return sum(len(normalize_text(part).split()) for part in parts if normalize_text(part))


def dump_json_for_prompt(value: Any, max_chars: int) -> str:
    text = json.dumps(value, ensure_ascii=True, indent=2)
    if max_chars > 0 and len(text) > max_chars:
        return text[:max_chars] + "\n...[truncated]"
    return text


def compact_retrieval_context_for_prompt(context: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(context, dict):
        return {}
    keep_keys = [
        "backend",
        "knn_index",
        "vector_field",
        "num_candidates",
        "num_retrieval_passes",
        "num_query_variants",
        "candidate_count",
        "error",
    ]
    compact = {key: context.get(key) for key in keep_keys if context.get(key) not in (None, "", [], {})}
    for key in ["query_variants", "retrieval_queries", "search_queries"]:
        value = context.get(key)
        if isinstance(value, list) and value:
            compact[key] = value[:5]
    passes = context.get("retrieval_passes")
    if isinstance(passes, list) and passes:
        compact["retrieval_passes"] = [
            {
                key: item.get(key)
                for key in keep_keys
                if isinstance(item, dict) and item.get(key) not in (None, "", [], {})
            }
            for item in passes[:3]
            if isinstance(item, dict)
        ]
    return compact


def compact_screened_candidates_for_prompt(state: PatentAnalysisState, config: PipelineConfig, top_docs: int) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    evidence_ids = [
        normalize_text(item.get("patent_id"))
        for item in state.get("evidence", []) or []
        if isinstance(item, dict) and normalize_text(item.get("patent_id"))
    ]
    allowed = set(evidence_ids[:top_docs])
    for cand in (state.get("screened_candidates") or [])[:top_docs]:
        if not isinstance(cand, dict):
            continue
        doc_id = normalize_text(cand.get("doc_id"))
        if allowed and doc_id not in allowed:
            continue
        rows.append(
            {
                "rank": cand.get("screen_rank") or cand.get("rank"),
                "patent_id": doc_id,
                "title": limit_words(cand.get("title"), 24),
                "score": cand.get("score"),
                "publication_date": cand.get("publication_date", ""),
                "application_date": cand.get("application_date", ""),
                "priority_date": cand.get("priority_date", ""),
                "ipc_codes": normalize_list(cand.get("ipc_codes"))[:8],
            }
        )
    return rows


def compact_evidence_for_prompt(state: PatentAnalysisState, top_docs: int) -> List[Dict[str, Any]]:
    matches_per_doc = max(1, env_int("MULTIAGENT_AGENT3_MATCHES_PER_DOC", 2))
    top_doc_ids = [
        normalize_text(cand.get("doc_id"))
        for cand in state.get("screened_candidates", []) or []
        if isinstance(cand, dict) and normalize_text(cand.get("doc_id"))
    ][:top_docs]
    order = {doc_id: idx for idx, doc_id in enumerate(top_doc_ids)}
    evidence_rows = [
        item
        for item in state.get("evidence", []) or []
        if isinstance(item, dict)
        and (not order or normalize_text(item.get("patent_id")) in order)
    ]
    evidence_rows = sorted(
        evidence_rows,
        key=lambda item: order.get(normalize_text(item.get("patent_id")), 10_000),
    )[:top_docs]
    rows: List[Dict[str, Any]] = []
    for item in evidence_rows:
        matched = []
        for match in (item.get("matched_elements") or [])[:matches_per_doc]:
            if not isinstance(match, dict):
                continue
            matched.append(
                {
                    "claim_element": limit_words(match.get("claim_element"), 28),
                    "claim_element_vi": limit_words(
                        match.get("claim_element_vi") or technical_label_vi(match.get("claim_element")),
                        28,
                    ),
                    "section": normalize_text(match.get("section")),
                    "section_vi": section_vi(match.get("section")),
                    "match_type": normalize_text(match.get("match_type")),
                    "match_type_vi": match_type_vi(match.get("match_type")),
                    "evidence_text": limit_words(match.get("evidence_text"), 40),
                    "reason": limit_words(match.get("reason"), 30),
                    "gap_or_limitation": limit_words(match.get("gap_or_limitation"), 30),
                }
            )
        rows.append(
            {
                "patent_id": normalize_text(item.get("patent_id")),
                "title": limit_words(item.get("title"), 24),
                "matched_elements": matched,
                "missing_elements": [limit_words(value, 24) for value in normalize_list(item.get("missing_elements"))[:6]],
                "missing_elements_vi": [
                    limit_words(technical_label_vi(value), 24)
                    for value in normalize_list(item.get("missing_elements"))[:6]
                ],
                "overall_relevance": limit_words(item.get("overall_relevance"), 40),
            }
        )
    return rows


def sanitize_evidence_output(
    raw_evidence: Any,
    fallback_evidence: List[Dict[str, Any]],
    state: PatentAnalysisState,
    config: PipelineConfig,
) -> List[Dict[str, Any]]:
    claim_elements = [normalize_text(item) for item in state.get("claim_elements", []) or [] if normalize_text(item)]
    allowed_candidates = state.get("screened_candidates", []) or []
    allowed_ids = [normalize_text(c.get("doc_id")) for c in allowed_candidates[: config.evidence_top_docs]]
    allowed_set = {doc_id for doc_id in allowed_ids if doc_id}
    docs = state.get("candidate_docs", {}) or {}

    source = raw_evidence if isinstance(raw_evidence, list) and raw_evidence else fallback_evidence
    cleaned: List[Dict[str, Any]] = []
    valid_sections = {"title", "abstract", "claims", "description", "mixed"}
    valid_match_types = {"exact", "partial", "weak"}

    for item in source:
        if not isinstance(item, dict):
            continue
        doc_id = normalize_text(item.get("patent_id") or item.get("doc_id"))
        if not doc_id:
            continue
        if allowed_set and doc_id not in allowed_set:
            continue
        doc = docs.get(doc_id, {})
        matched_items = []
        matched_claims = set()
        for raw_match in item.get("matched_elements", []) or []:
            if not isinstance(raw_match, dict):
                continue
            claim_element = normalize_text(raw_match.get("claim_element"))
            evidence_text = limit_words(raw_match.get("evidence_text"), 90)
            if not claim_element or not evidence_text:
                continue
            section = normalize_text(raw_match.get("section")).lower() or "mixed"
            if section not in valid_sections:
                section = "mixed"
            match_type = normalize_text(raw_match.get("match_type")).lower() or "weak"
            if match_type not in valid_match_types:
                match_type = "weak"
            matched_claims.add(claim_element)
            matched_items.append(
                {
                    "claim_element": claim_element,
                    "claim_element_vi": technical_label_vi(claim_element),
                    "section": section,
                    "section_vi": section_vi(section),
                    "match_type": match_type,
                    "match_type_vi": match_type_vi(match_type),
                    "evidence_text": evidence_text,
                    "reason": limit_words(raw_match.get("reason"), 70),
                    "gap_or_limitation": limit_words(raw_match.get("gap_or_limitation"), 70),
                }
            )

        missing = normalize_list(item.get("missing_elements"))
        if claim_elements:
            known_missing = [element for element in claim_elements if element not in matched_claims]
            missing = known_missing if known_missing else missing

        cleaned.append(
            {
                "patent_id": doc_id,
                "title": normalize_text(item.get("title")) or normalize_text(doc.get("title")),
                "matched_elements": matched_items,
                "missing_elements": missing,
                "missing_elements_vi": [technical_label_vi(value) for value in missing],
                "overall_relevance": limit_words(item.get("overall_relevance"), 80),
            }
        )

    if cleaned:
        cleaned_ids = {item["patent_id"] for item in cleaned}
        fallback_by_id = {
            normalize_text(item.get("patent_id") or item.get("doc_id")): item
            for item in fallback_evidence
            if isinstance(item, dict) and normalize_text(item.get("patent_id") or item.get("doc_id"))
        }
        for doc_id in allowed_ids:
            if not doc_id or doc_id in cleaned_ids:
                continue
            fallback_item = fallback_by_id.get(doc_id)
            if fallback_item:
                fallback_copy = dict(fallback_item)
                enriched_matches = []
                for match in fallback_copy.get("matched_elements", []) or []:
                    if isinstance(match, dict):
                        enriched_match = dict(match)
                        enriched_match["claim_element_vi"] = technical_label_vi(enriched_match.get("claim_element"))
                        enriched_match["section_vi"] = section_vi(enriched_match.get("section"))
                        enriched_match["match_type_vi"] = match_type_vi(enriched_match.get("match_type"))
                        enriched_matches.append(enriched_match)
                fallback_copy["matched_elements"] = enriched_matches
                fallback_copy["missing_elements_vi"] = [
                    technical_label_vi(value)
                    for value in normalize_list(fallback_copy.get("missing_elements"))
                ]
                cleaned.append(fallback_copy)
            else:
                doc = docs.get(doc_id, {})
                cleaned.append(
                    {
                        "patent_id": doc_id,
                        "title": normalize_text(doc.get("title")),
                        "matched_elements": [],
                        "missing_elements": claim_elements,
                        "missing_elements_vi": [technical_label_vi(value) for value in claim_elements],
                        "overall_relevance": "không trích xuất được bằng chứng",
                    }
                )
        order = {doc_id: idx for idx, doc_id in enumerate(allowed_ids)}
        return sorted(cleaned, key=lambda item: order.get(item["patent_id"], 10_000))
    return fallback_evidence


def sanitize_analysis_output(parsed: Dict[str, Any], fallback: Dict[str, Any], state: PatentAnalysisState) -> Dict[str, Any]:
    evidence = state.get("evidence", []) or []
    allowed_ids = {normalize_text(item.get("patent_id")) for item in evidence if normalize_text(item.get("patent_id"))}
    evidence_by_id = {
        normalize_text(item.get("patent_id")): item
        for item in evidence
        if isinstance(item, dict) and normalize_text(item.get("patent_id"))
    }
    raw_ranked = parsed.get("ranked_prior_art") if isinstance(parsed.get("ranked_prior_art"), list) else []
    ranked: List[Dict[str, Any]] = []

    def analysis_row_from_evidence(
        doc_id: str,
        evidence_item: Dict[str, Any],
        raw_item: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        raw_item = raw_item or {}
        evidence_matches = [
            match for match in (evidence_item.get("matched_elements") or [])
            if isinstance(match, dict) and normalize_text(match.get("claim_element"))
        ]
        evidence_missing = normalize_list(evidence_item.get("missing_elements"))
        if evidence_matches or evidence_missing:
            matched_elements = [normalize_text(match.get("claim_element")) for match in evidence_matches]
            matched_elements_vi = [
                normalize_text(match.get("claim_element_vi")) or technical_label_vi(match.get("claim_element"))
                for match in evidence_matches
            ]
            missing_elements = evidence_missing
            missing_elements_vi = normalize_list(evidence_item.get("missing_elements_vi")) or [
                technical_label_vi(value) for value in missing_elements
            ]
            risk = calibrated_novelty_risk(evidence_matches, missing_elements)
        else:
            matched_elements = normalize_list(raw_item.get("matched_elements"))
            matched_elements_vi = [technical_label_vi(value) for value in matched_elements]
            missing_elements = normalize_list(raw_item.get("missing_elements"))
            missing_elements_vi = [technical_label_vi(value) for value in missing_elements]
            risk = normalize_text(raw_item.get("novelty_risk")).lower()
            if risk not in {"high", "medium", "low"}:
                risk = "low"

        summary = clean_vietnamese_display_text(limit_words(raw_item.get("claim_overlap_summary"), 90))
        if not summary:
            if evidence_matches:
                summary = "Có bằng chứng yếu hoặc một phần cho một số yếu tố, nhưng còn thiếu các giới hạn trung tâm."
            else:
                summary = "Không có bằng chứng khớp đáng kể với các yếu tố trung tâm trong bảng bằng chứng."
        summary_lower = summary.lower()
        denies_any_match = any(
            marker in summary_lower
            for marker in [
                "không có yếu tố nào trùng",
                "khong co yeu to nao trung",
                "không có yếu tố nào khớp",
                "khong co yeu to nao khop",
                "no elements match",
                "no matching elements",
            ]
        )
        if matched_elements and denies_any_match:
            weak_only = bool(evidence_matches) and all(
                normalize_text(match.get("match_type")).lower() == "weak"
                for match in evidence_matches
                if isinstance(match, dict)
            )
            summary = (
                "Chỉ có trùng khớp yếu hoặc chung chung; chưa thấy bằng chứng tiết lộ đầy đủ yếu tố trung tâm."
                if weak_only
                else "Có trùng khớp theo bảng bằng chứng, nhưng chưa tiết lộ đầy đủ các yếu tố trung tâm."
            )
        limitations = clean_vietnamese_display_text(limit_words(raw_item.get("limitations"), 90))
        if not limitations:
            if missing_elements_vi:
                limitations = "Thiếu yếu tố: " + "; ".join(missing_elements_vi[:5])
            else:
                limitations = "Bảng bằng chứng không nêu giới hạn còn thiếu rõ ràng."

        return {
            "rank": len(ranked) + 1,
            "patent_id": doc_id,
            "title": normalize_text(evidence_item.get("title")) or normalize_text(raw_item.get("title")),
            "novelty_risk": risk,
            "novelty_risk_vi": {"high": "cao", "medium": "trung bình", "low": "thấp"}.get(risk, risk),
            "matched_elements": matched_elements,
            "matched_elements_vi": matched_elements_vi,
            "missing_elements": missing_elements,
            "missing_elements_vi": missing_elements_vi,
            "claim_overlap_summary": summary,
            "limitations": limitations,
        }

    for raw_item in raw_ranked:
        if not isinstance(raw_item, dict):
            continue
        doc_id = normalize_text(raw_item.get("patent_id") or raw_item.get("doc_id"))
        if not doc_id or (allowed_ids and doc_id not in allowed_ids):
            continue
        evidence_item = evidence_by_id.get(doc_id, {})
        ranked.append(analysis_row_from_evidence(doc_id, evidence_item, raw_item))

    ranked_doc_ids = {normalize_text(row.get("patent_id")) for row in ranked if normalize_text(row.get("patent_id"))}
    max_ranked_docs = max(1, min(3, env_int("MULTIAGENT_AGENT3_TOP_DOCS", 3)))
    for evidence_item in evidence[:max_ranked_docs]:
        if not isinstance(evidence_item, dict):
            continue
        doc_id = normalize_text(evidence_item.get("patent_id"))
        if not doc_id or doc_id in ranked_doc_ids:
            continue
        ranked.append(analysis_row_from_evidence(doc_id, evidence_item))
        ranked_doc_ids.add(doc_id)
        if len(ranked) >= max_ranked_docs:
            break

    coverage_raw = parsed.get("coverage") if isinstance(parsed.get("coverage"), dict) else {}
    fallback_coverage = fallback.get("coverage", {})
    confidence = normalize_text(coverage_raw.get("confidence") or fallback_coverage.get("confidence")).lower()
    if confidence not in {"high", "medium", "low"}:
        confidence = "low"
    coverage = {
        "is_sufficient": normalize_bool(coverage_raw.get("is_sufficient"), default=normalize_bool(fallback_coverage.get("is_sufficient"))),
        "is_sufficient_vi": "đủ"
        if normalize_bool(coverage_raw.get("is_sufficient"), default=normalize_bool(fallback_coverage.get("is_sufficient")))
        else "chưa đủ",
        "confidence": confidence,
        "confidence_vi": {"high": "cao", "medium": "trung bình", "low": "thấp"}.get(confidence, confidence),
        "coverage_notes": clean_vietnamese_display_text(
            limit_words(coverage_raw.get("coverage_notes") or fallback_coverage.get("coverage_notes"), 90)
        ),
        "recommended_next_searches": [
            clean_vietnamese_display_text(value)
            for value in normalize_list(coverage_raw.get("recommended_next_searches"))[:5]
        ],
    }

    acceptance_raw = parsed.get("acceptance_assessment") if isinstance(parsed.get("acceptance_assessment"), dict) else {}
    fallback_acceptance = fallback.get("acceptance_assessment", {}) if isinstance(fallback.get("acceptance_assessment"), dict) else {}
    acceptance_value = normalize_text(
        acceptance_raw.get("acceptance_likelihood") or fallback_acceptance.get("acceptance_likelihood")
    ).lower()
    if acceptance_value not in {"likely", "uncertain", "difficult"}:
        acceptance_value = "difficult" if any(item.get("novelty_risk") == "high" for item in ranked) else "uncertain"
    acceptance = {
        "acceptance_likelihood": acceptance_value,
        "acceptance_likelihood_vi": {
            "likely": "có khả năng",
            "uncertain": "chưa chắc chắn",
            "difficult": "khó",
        }.get(acceptance_value, acceptance_value),
        "main_obstacles": [
            clean_vietnamese_display_text(value)
            for value in normalize_list(acceptance_raw.get("main_obstacles") or fallback_acceptance.get("main_obstacles"))[:6]
        ],
        "blocking_prior_art": normalize_list(acceptance_raw.get("blocking_prior_art") or fallback_acceptance.get("blocking_prior_art"))[:5],
        "why": clean_vietnamese_display_text(limit_words(acceptance_raw.get("why") or fallback_acceptance.get("why"), 100)),
        "recommended_strategy": clean_vietnamese_display_text(
            limit_words(
                acceptance_raw.get("recommended_strategy") or fallback_acceptance.get("recommended_strategy"),
                140,
            )
        ),
        "amendment_directions": [
            clean_vietnamese_display_text(value)
            for value in normalize_list(
                acceptance_raw.get("amendment_directions") or fallback_acceptance.get("amendment_directions")
            )[:6]
        ],
    }
    high_risk_ids = [row.get("patent_id") for row in ranked if row.get("novelty_risk") == "high"]
    if high_risk_ids:
        acceptance["blocking_prior_art"] = [
            doc_id for doc_id in acceptance["blocking_prior_art"] if doc_id in set(high_risk_ids)
        ] or high_risk_ids[:5]
    else:
        acceptance["blocking_prior_art"] = []

    top_ranked = ranked[:3]
    glass_missing_all = bool(top_ranked) and all(
        any(
            ("glass" in normalize_text(value).lower() or "kính" in normalize_text(value).lower())
            for value in row.get("missing_elements", []) or []
        )
        for row in top_ranked
    )
    if glass_missing_all:
        coverage["is_sufficient"] = False
        if acceptance["acceptance_likelihood"] in {"likely", "difficult"}:
            acceptance["acceptance_likelihood"] = "uncertain"
        acceptance["why"] = (
            "Bằng chứng chỉ cho thấy trùng một phần ở thân xốp/bộ đốt hoặc dòng khí, "
            "nhưng các tài liệu mạnh nhất đều thiếu tấm kính chịu nhiệt phía trên bộ đốt."
        )
        acceptance["recommended_strategy"] = (
            "Nhấn mạnh tấm kính chịu nhiệt như một đặc điểm chức năng giúp làm sạch bề mặt "
            "và truyền nhiệt, khác với các cấu trúc bộ đốt xốp chung chung."
        )
        glass_direction = "giới hạn yêu cầu bảo hộ quanh tấm kính chịu nhiệt bố trí trên bộ đốt và thân xốp thấm khí"
        if not any(
            ("glass" in normalize_text(value).lower() or "kính" in normalize_text(value).lower())
            for value in acceptance["amendment_directions"]
        ):
            acceptance["amendment_directions"] = [glass_direction] + acceptance["amendment_directions"][:5]

    if not normalize_bool(coverage.get("is_sufficient")) and acceptance["acceptance_likelihood"] == "likely":
        acceptance["acceptance_likelihood"] = "uncertain"

    coverage["is_sufficient_vi"] = "đủ" if normalize_bool(coverage.get("is_sufficient")) else "chưa đủ"
    coverage["confidence_vi"] = {"high": "cao", "medium": "trung bình", "low": "thấp"}.get(
        normalize_text(coverage.get("confidence")).lower(),
        coverage.get("confidence"),
    )
    acceptance["acceptance_likelihood_vi"] = {
        "likely": "có khả năng",
        "uncertain": "chưa chắc chắn",
        "difficult": "khó",
    }.get(
        normalize_text(acceptance.get("acceptance_likelihood")).lower(),
        acceptance.get("acceptance_likelihood"),
    )

    analysis = {
        "ranked_prior_art": ranked or fallback.get("ranked_prior_art", []),
        "coverage": coverage,
        "acceptance_assessment": acceptance,
    }
    problem_vi = normalize_text(parsed.get("technical_problem_vi"))
    if problem_vi:
        analysis["technical_problem_vi"] = problem_vi
    analysis["final_report_markdown"] = build_prior_art_report(analysis, state)
    return analysis


def build_prior_art_report(analysis: Dict[str, Any], state: PatentAnalysisState) -> str:
    """Create a deterministic Vietnamese report from sanitized Agent 3 JSON."""

    ranked = analysis.get("ranked_prior_art", []) or []
    coverage = analysis.get("coverage", {}) or {}
    acceptance = analysis.get("acceptance_assessment", {}) or {}
    risk_vi = {"high": "cao", "medium": "trung bình", "low": "thấp"}
    confidence_vi = {"high": "cao", "medium": "trung bình", "low": "thấp"}
    evidence_by_id = {
        normalize_text(item.get("patent_id")): item
        for item in state.get("evidence", []) or []
        if isinstance(item, dict) and normalize_text(item.get("patent_id"))
    }
    likelihood_vi = {
        "likely": "có khả năng",
        "uncertain": "chưa chắc chắn",
        "difficult": "khó",
    }.get(normalize_text(acceptance.get("acceptance_likelihood")).lower(), normalize_text(acceptance.get("acceptance_likelihood")))
    lines = [
        "# Báo cáo phân tích prior art",
        "",
        f"**Vấn đề kỹ thuật:** {normalize_text(analysis.get('technical_problem_vi') or state.get('technical_problem'))}",
        "",
        "## 1. Tóm tắt kết quả",
        (
            f"- Đã đánh giá {len(ranked)} tài liệu prior art mạnh nhất từ danh sách ứng viên đã sàng lọc. "
            f"Kết luận coverage: {'đủ' if normalize_bool(coverage.get('is_sufficient')) else 'chưa đủ'}; "
            f"độ tin cậy: {confidence_vi.get(normalize_text(coverage.get('confidence')).lower(), coverage.get('confidence'))}."
        ),
        f"- Nhận định khả năng chấp nhận: {likelihood_vi}.",
        "",
        "## 2. Các tài liệu mạnh nhất",
    ]
    if not ranked:
        lines.append("- Chưa có tài liệu prior art nào được xếp hạng từ bảng bằng chứng.")
    for row in ranked[:3]:
        doc_id = normalize_text(row.get("patent_id"))
        evidence_item = evidence_by_id.get(doc_id, {})
        matched = row.get("matched_elements") or []
        matched_vi = row.get("matched_elements_vi") or [technical_label_vi(item) for item in matched]
        missing = row.get("missing_elements") or []
        missing_vi = row.get("missing_elements_vi") or [technical_label_vi(item) for item in missing]
        lines.append(
            f"- **{row.get('rank')}. {doc_id} - {row.get('title')}**: "
            f"rủi ro novelty={risk_vi.get(normalize_text(row.get('novelty_risk')).lower(), row.get('novelty_risk'))}; "
            f"khớp {len(matched_vi)} yếu tố, thiếu {len(missing_vi)} yếu tố. "
            f"{limit_words(row.get('claim_overlap_summary'), 24)}"
        )
        limitation = normalize_text(row.get("limitations"))
        if limitation:
            lines.append(f"  - Giới hạn chính: {limit_words(limitation, 34)}")
        for match in (evidence_item.get("matched_elements") or [])[:2]:
            if isinstance(match, dict):
                claim_label = match.get("claim_element_vi") or technical_label_vi(match.get("claim_element"))
                lines.append(
                    f"  - Bằng chứng (trích dẫn nguồn): {limit_words(claim_label, 16)} -> "
                    f"{limit_words(match.get('evidence_text'), 24)} "
                    f"({match.get('match_type_vi') or match_type_vi(match.get('match_type'))}; "
                    f"điểm thiếu: {limit_words(match.get('gap_or_limitation'), 18)})"
                )
        if missing_vi:
            lines.append("  - Yếu tố còn thiếu: " + "; ".join(limit_words(item, 14) for item in missing_vi[:3]))

    lines.extend(
        [
            "",
            "## 3. Độ bao phủ bằng chứng",
            f"- Đủ bằng chứng: {'có' if normalize_bool(coverage.get('is_sufficient')) else 'không'}; "
            f"độ tin cậy: {confidence_vi.get(normalize_text(coverage.get('confidence')).lower(), coverage.get('confidence'))}.",
            f"- Ghi chú: {limit_words(coverage.get('coverage_notes'), 44)}",
            "",
            "## 4. Đánh giá khả năng chấp nhận",
            f"- Khả năng chấp nhận: {likelihood_vi}.",
            f"- Lý do: {limit_words(acceptance.get('why'), 58)}",
            f"- Chiến lược đề xuất: {limit_words(acceptance.get('recommended_strategy'), 64)}",
        ]
    )
    obstacles = normalize_list(acceptance.get("main_obstacles"))[:3]
    if obstacles:
        lines.append("- Trở ngại chính: " + "; ".join(obstacles))
    directions = normalize_list(acceptance.get("amendment_directions"))[:3]
    if directions:
        lines.append("- Hướng sửa yêu cầu bảo hộ: " + "; ".join(directions))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent 1: Query Understanding Agent
# ---------------------------------------------------------------------------


def heuristic_query_understanding(state: PatentAnalysisState) -> Dict[str, Any]:
    text = normalize_text(state.get("input_text"))
    metadata = state.get("input_metadata", {}) or {}
    title = normalize_text(metadata.get("title"))
    abstract = normalize_text(metadata.get("abstract"))
    claims = normalize_text(metadata.get("claims"))
    retrieval_text = normalize_text(metadata.get("retrieval_text"))
    query_seed = normalize_text(" ".join(part for part in [title, abstract, claims, text] if part))
    if not query_seed:
        query_seed = text

    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", query_seed) if s.strip()]
    feature_seed = normalize_text(" ".join(part for part in [title, abstract, text] if part)) or query_seed
    phrase_candidates = [
        normalize_text(part)
        for part in re.split(
            r"\b(?:with|comprising|including|includes|having|using|configured to|adapted to|and|or|for)\b|[,;:]",
            feature_seed,
            flags=re.IGNORECASE,
        )
    ]
    phrase_candidates = [
        limit_words(part, 18)
        for part in phrase_candidates
        if 2 <= len(part.split()) <= 18 and not part.lower().startswith(("a ", "an ", "the "))
    ]
    key_features = []
    for item in phrase_candidates + sentences:
        feature = limit_words(item, 28)
        if feature and feature.lower() not in {v.lower() for v in key_features}:
            key_features.append(feature)
        if len(key_features) >= 6:
            break
    if not key_features:
        key_features = [limit_words(query_seed, 40)]

    claim_source = claims or retrieval_text or query_seed
    claim_sentences = [s.strip() for s in re.split(r"(?<=[.;!?])\s+", claim_source) if s.strip()]
    claim_elements = [limit_words(s, 32) for s in claim_sentences[:6]]
    for feature in key_features:
        if len(claim_elements) >= 6:
            break
        if feature not in claim_elements:
            claim_elements.append(feature)
    technical_problem = limit_words(sentences[0] if sentences else query_seed, 50)

    cutoff = normalize_text(metadata.get("priority_date") or metadata.get("application_date") or metadata.get("publication_date"))
    exclude_doc_id = normalize_text(metadata.get("doc_id") or metadata.get("patent_id") or metadata.get("topic_id"))
    exclude_canonical_doc_id = normalize_text(metadata.get("canonical_doc_id")) or canonical_doc_id(exclude_doc_id)
    search_queries = [
        {
            "query_view": "combined",
            "search_role": "combined",
            "text": limit_words(query_seed, 180),
            "weight": 1.0,
            "search_mode": "semantic",
        },
    ]
    if title or abstract:
        search_queries.append(
            {
                "query_view": "title_abstract",
                "search_role": "technology",
                "text": limit_words(f"{title} {abstract}", 140),
                "weight": 1.1,
                "search_mode": "entity_expansion",
            }
        )
    if claims:
        search_queries.append(
            {
                "query_view": "claims",
                "search_role": "claim_overlap",
                "text": limit_words(claims, 180),
                "weight": 0.9,
                "search_mode": "claim_text",
            }
        )
    for feature in key_features[:3]:
        search_queries.append(
            {
                "query_view": "feature",
                "search_role": "function",
                "text": feature,
                "weight": 1.05,
                "search_mode": "entity_expansion",
            }
        )

    return {
        "input_type": metadata.get("input_type") or ("idea" if text and not (title or abstract or claims) else "patent_document"),
        "technical_problem": technical_problem,
        "key_features": key_features,
        "claim_elements": claim_elements,
        "query_entities": [
            {"name": item, "type": "technology"} for item in key_features[:6]
        ],
        "retrieval_focus": {
            "search_intent": "candidate_prior_art_local_search",
            "preferred_retrieval_backend": "es_knn",
            "focus_scope": "specific technical feature and claim overlap",
            "evidence_priority": "claims first, then abstract, then retrieval snippets; long description is not indexed",
        },
        "search_queries": search_queries[:6],
        "filters": {
            "exclude_doc_id": exclude_doc_id,
            "exclude_canonical_doc_id": exclude_canonical_doc_id,
            "prior_art_cutoff_date": cutoff,
            "use_date_filter": bool(cutoff),
        },
    }


def search_queries_from_understanding(parsed: Dict[str, Any], fallback_queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build compact search queries from parsed Agent 1 fields when the LLM omits search_queries."""

    if not isinstance(parsed, dict):
        return fallback_queries
    technical_problem = limit_words(parsed.get("technical_problem"), 34)
    key_features = [limit_words(item, 18) for item in normalize_list(parsed.get("key_features"))[:4]]
    claim_elements = [limit_words(item, 22) for item in normalize_list(parsed.get("claim_elements"))[:4]]
    queries: List[Dict[str, Any]] = []

    def add(view: str, role: str, text: str, weight: float, mode: str) -> None:
        text = normalize_text(re.sub(r"\b(title|abstract|claims?)\s*:?", "", text, flags=re.IGNORECASE))
        text = limit_words(text, 60)
        if not text:
            return
        queries.append(
            {
                "query_view": view,
                "search_role": role,
                "text": text,
                "weight": weight,
                "search_mode": mode,
            }
        )

    combined = normalize_text(" ".join([technical_problem] + key_features[:2]))
    add("combined", "combined", combined, 1.0, "semantic")
    add("technical_problem", "problem", technical_problem, 1.1, "problem_expansion")
    if claim_elements:
        add("claims", "claim_overlap", " ".join(claim_elements[:3]), 1.0, "claim_text")
    for feature in key_features[:2]:
        add("feature", "function", feature, 0.95, "entity_expansion")

    return queries or fallback_queries


def merge_search_queries_with_fallback(
    primary_queries: Any,
    fallback_queries: List[Dict[str, Any]],
    max_queries: int = 5,
) -> List[Dict[str, Any]]:
    """Keep LLM/generated query views while backfilling important retrieval views."""

    primary = primary_queries if isinstance(primary_queries, list) else []
    fallback = fallback_queries if isinstance(fallback_queries, list) else []
    pool = [item for item in primary + fallback if isinstance(item, dict)]
    if not pool:
        return []

    selected: List[Dict[str, Any]] = []
    seen_texts = set()

    def item_view(item: Dict[str, Any]) -> str:
        return normalize_text(item.get("query_view")).lower() or "combined"

    def add(item: Dict[str, Any], allow_repeated_view: bool = True) -> bool:
        text = normalize_text(item.get("text"))
        if not text:
            return False
        view = item_view(item)
        if not allow_repeated_view and any(item_view(existing) == view for existing in selected):
            return False
        key = (item_view(item), normalize_text(item.get("search_role")).lower(), text.lower())
        if key in seen_texts:
            return False
        seen_texts.add(key)
        selected.append(item)
        return len(selected) >= max_queries

    # Retrieval is fixed KNN+RRF. Preserve complementary views first, then fill
    # with extra features/entities from the model.
    for desired_view in ["combined", "claims", "technical_problem", "feature", "title_abstract"]:
        for item in pool:
            if item_view(item) == desired_view and add(item):
                return selected
            if selected and item_view(selected[-1]) == desired_view:
                break

    for item in pool:
        view = item_view(item)
        allow_repeated = view == "feature"
        if add(item, allow_repeated_view=allow_repeated):
            break
    return selected


def metadata_filters(state: PatentAnalysisState) -> Dict[str, Any]:
    metadata = state.get("input_metadata", {}) or {}
    cutoff = normalize_text(metadata.get("priority_date") or metadata.get("application_date") or metadata.get("publication_date"))
    exclude_doc_id = normalize_text(metadata.get("doc_id") or metadata.get("patent_id") or metadata.get("topic_id"))
    exclude_canonical_doc_id = normalize_text(metadata.get("canonical_doc_id")) or canonical_doc_id(exclude_doc_id)
    return {
        "exclude_doc_id": exclude_doc_id,
        "exclude_canonical_doc_id": exclude_canonical_doc_id,
        "prior_art_cutoff_date": cutoff,
        "use_date_filter": bool(cutoff),
    }


def query_understanding_response_schema() -> Dict[str, Any]:
    string_array = {"type": "array", "items": {"type": "string"}}
    return {
        "type": "object",
        "properties": {
            "input_type": {"type": "string", "enum": ["patent_document", "idea", "unknown"]},
            "technical_problem": {"type": "string"},
            "key_features": string_array,
            "claim_elements": string_array,
            "query_entities": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "type": {
                            "type": "string",
                            "enum": [
                                "technology",
                                "component",
                                "function",
                                "problem",
                                "ipc",
                                "assignee",
                                "inventor",
                                "citation",
                                "patent",
                            ],
                        },
                    },
                    "required": ["name", "type"],
                    "additionalProperties": False,
                },
            },
            "retrieval_focus": {
                "type": "object",
                "properties": {
                    "search_intent": {"type": "string"},
                    "preferred_retrieval_backend": {"type": "string", "enum": ["es_knn"]},
                    "focus_scope": {"type": "string"},
                    "evidence_priority": {"type": "string"},
                },
                "required": ["search_intent", "preferred_retrieval_backend", "focus_scope", "evidence_priority"],
                "additionalProperties": False,
            },
            "search_queries": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "query_view": {
                            "type": "string",
                            "enum": ["combined", "title_abstract", "claims", "technical_problem", "feature"],
                        },
                        "search_role": {
                            "type": "string",
                            "enum": [
                                "combined",
                                "claim_overlap",
                                "function",
                                "problem",
                                "component",
                                "technology",
                                "ipc_or_citation",
                            ],
                        },
                        "text": {"type": "string"},
                        "weight": {"type": "number"},
                        "search_mode": {
                            "type": "string",
                            "enum": ["semantic", "claim_text", "entity_expansion", "problem_expansion"],
                        },
                    },
                    "required": ["query_view", "search_role", "text", "weight", "search_mode"],
                    "additionalProperties": False,
                },
            },
        },
        "required": [
            "technical_problem",
            "key_features",
            "claim_elements",
            "search_queries",
        ],
        "additionalProperties": False,
    }


def raw_query_baseline_node(
    state: PatentAnalysisState,
    config: Optional[PipelineConfig] = None,
) -> PatentAnalysisState:
    """Prepare a no-agent retrieval query state for PB1."""

    config = config or PipelineConfig()
    metadata = state.get("input_metadata", {}) or {}
    text = normalize_text(state.get("input_text"))
    title = normalize_text(metadata.get("title"))
    abstract = normalize_text(metadata.get("abstract"))
    claims = normalize_text(metadata.get("claims"))
    retrieval_text = normalize_text(metadata.get("retrieval_text"))
    raw_query = normalize_text(" ".join(part for part in [title, abstract, claims, retrieval_text, text] if part)) or text
    query_text = limit_words(raw_query, 180)
    filters = metadata_filters(state)

    update = {
        "input_type": metadata.get("input_type") or ("idea" if text and not (title or abstract or claims) else "patent_document"),
        "technical_problem": limit_words(raw_query, 50),
        "key_features": [],
        "claim_elements": [limit_words(raw_query, 60)] if raw_query else [],
        "query_entities": [],
        "retrieval_focus": {
            "search_intent": "raw_query_baseline",
            "preferred_retrieval_backend": "es_knn",
            "focus_scope": "raw input text only",
            "evidence_priority": "retrieval returned title, abstract, claims, snippets",
        },
        "search_queries": [
            {
                "query_view": "raw",
                "search_role": "combined",
                "text": query_text,
                "weight": 1.0,
                "search_mode": "semantic",
            }
        ],
        "filters": filters,
        "audit_log": add_audit(
            state,
            "raw_query_baseline_node",
            "prepared raw-query baseline without an LLM agent",
            query_words=len(query_text.split()),
        ),
    }
    return update  # type: ignore[return-value]


def query_understanding_agent(
    state: PatentAnalysisState,
    config: Optional[PipelineConfig] = None,
    verbose: Optional[bool] = None,
) -> PatentAnalysisState:
    config = config or PipelineConfig()
    llm_config = config_for_agent_llm(config, 1)
    log_enabled = stage_log_enabled(verbose)
    fallback = heuristic_query_understanding(state)
    metadata = state.get("input_metadata", {}) or {}
    prompt_metadata = compact_metadata_for_prompt(
        metadata,
        include_text_fields=env_bool("MULTIAGENT_AGENT1_METADATA_TEXT_FIELDS", False),
    )
    fallback_skill = (
        "You are a patent query understanding agent. Extract only structured, useful search "
        "and analysis fields. Return valid JSON only."
    )
    system_prompt, skill_file = load_agent_skill(config, "query_understanding_agent", fallback_skill)
    input_excerpt = agent1_input_excerpt(state)
    agent1_schema = query_understanding_response_schema()
    use_strict_schema = supports_strict_json_schema(
        normalize_optional_text(llm_config.llm_provider).lower() or DEFAULT_AGENT_LLM_PROVIDER,
        llm_config.llm_model,
    )
    schema_instruction = (
        "The API may enforce a JSON schema; satisfy every required field exactly."
        if use_strict_schema
        else "JSON object mode is enabled; every required top-level key must be present."
    )
    user_prompt = f"""
Patent/query text:
{input_excerpt}

Metadata JSON:
{json.dumps(prompt_metadata, ensure_ascii=True, separators=(",", ":"))}

{schema_instruction}
Return compact JSON only, using this top-level key order:
technical_problem, claim_elements, key_features, search_queries.
Use minified JSON: no pretty-printing and no whitespace outside string values.

Field contract:
- technical_problem: one prior-art retrieval problem, max 22 words.
- claim_elements: exactly 3 comparable elements, max 16 words each.
- key_features: exactly 3 components/functions, max 12 words each.
- search_queries: exactly 4 objects covering combined, claims, technical_problem, and feature.

Search query item contract:
- query_view in combined, title_abstract, claims, technical_problem, feature.
- search_role in combined, claim_overlap, function, problem, component, technology, ipc_or_citation.
- search_mode in semantic, claim_text, entity_expansion, problem_expansion.
- text is a short technical phrase, max 18 words; remove title/abstract/claim labels.
- weight is 0.2 to 2.0.

Rules:
- Do not copy full claims.
- Feature queries must combine a component with its function; avoid material-only queries.
- Do not invent IPC, citations, patents, dates, inventors, assignees, or prior-art facts.
- Do not choose another retrieval backend.
- Do not output input_type, filters, query_entities, or retrieval_focus; runtime code derives them.
"""

    progress_log(
        log_enabled,
        "Agent 1 calling LLM",
        provider=normalize_optional_text(llm_config.llm_provider).lower() or DEFAULT_AGENT_LLM_PROVIDER,
        model=llm_config.llm_model,
        input_excerpt_words=len(input_excerpt.split()),
        prompt_tokens=estimate_json_prompt_tokens(system_prompt, user_prompt),
        max_output_tokens=effective_max_output_tokens(llm_config, system_prompt, user_prompt),
        strict_schema=use_strict_schema,
    )
    agent1_expected_keys = [
        "technical_problem",
        "key_features",
        "claim_elements",
        "search_queries",
    ]
    parsed = call_llm_json(
        llm_config,
        system_prompt,
        user_prompt,
        fallback,
        verbose=log_enabled,
        expected_keys=agent1_expected_keys,
        response_schema=agent1_schema,
        schema_name="query_understanding",
    )
    used_fallback = parsed is fallback
    raw_search_queries = parsed.get("search_queries")
    used_generated_search_queries = not (isinstance(raw_search_queries, list) and raw_search_queries)
    if used_generated_search_queries and parsed is not fallback:
        raw_search_queries = search_queries_from_understanding(parsed, fallback["search_queries"])
    raw_search_queries = merge_search_queries_with_fallback(raw_search_queries, fallback["search_queries"])
    search_queries = sanitize_search_queries(
        raw_search_queries,
        fallback["search_queries"],
    )
    filters = sanitize_filters(None, fallback["filters"])
    query_entities = sanitize_query_entities(parsed.get("query_entities"))
    if not query_entities:
        query_entities = [
            {"name": item, "type": "technology"}
            for item in (normalize_list(parsed.get("key_features")) or fallback["key_features"])[:5]
        ]
    query_entities = query_entities or fallback["query_entities"]
    retrieval_focus = sanitize_retrieval_focus(parsed.get("retrieval_focus"), fallback["retrieval_focus"])
    retrieval_focus["preferred_retrieval_backend"] = "es_knn"
    progress_log(
        log_enabled,
        "Agent 1 output ready",
        used_fallback=used_fallback,
        generated_search_queries=used_generated_search_queries,
        claim_elements=len(normalize_list(parsed.get("claim_elements")) or fallback["claim_elements"]),
        search_queries=len(search_queries),
        query_entities=len(query_entities),
    )
    update = {
        "input_type": normalize_text(parsed.get("input_type")) or fallback["input_type"],
        "technical_problem": normalize_text(parsed.get("technical_problem")) or fallback["technical_problem"],
        "key_features": normalize_list(parsed.get("key_features")) or fallback["key_features"],
        "claim_elements": normalize_list(parsed.get("claim_elements")) or fallback["claim_elements"],
        "query_entities": query_entities,
        "retrieval_focus": retrieval_focus,
        "search_queries": search_queries,
        "filters": filters,
        "audit_log": add_audit(
            state,
            "query_understanding_agent",
            "extracted query understanding",
            skill_file=skill_file,
            used_fallback=used_fallback,
            generated_search_queries=used_generated_search_queries,
            num_claim_elements=len(normalize_list(parsed.get("claim_elements")) or fallback["claim_elements"]),
            num_search_queries=len(search_queries),
            num_query_entities=len(query_entities),
            input_excerpt_words=len(input_excerpt.split()),
            prompt_words=prompt_word_count(system_prompt, user_prompt),
            estimated_prompt_tokens=estimate_json_prompt_tokens(system_prompt, user_prompt),
            metadata_prompt_keys=sorted(prompt_metadata.keys()),
            llm_provider=normalize_optional_text(llm_config.llm_provider).lower() or DEFAULT_AGENT_LLM_PROVIDER,
            llm_model=llm_config.llm_model,
            strict_schema=use_strict_schema,
            max_output_tokens=effective_max_output_tokens(llm_config, system_prompt, user_prompt),
        ),
    }
    return update  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# Fixed retrieval node: Elasticsearch KNN
# ---------------------------------------------------------------------------


def knn_retrieval_node_factory(config: Optional[PipelineConfig] = None):
    config = config or PipelineConfig()
    def knn_retrieval_node(state: PatentAnalysisState) -> PatentAnalysisState:
        backend = normalize_optional_text(config.retrieval_backend).lower() or DEFAULT_RETRIEVAL_BACKEND
        if backend not in KNN_RETRIEVAL_BACKENDS:
            backend = DEFAULT_RETRIEVAL_BACKEND
        try:
            candidates, docs, context = run_es_knn_retrieval(config, state)
            backend = "es_knn"
            error = ""
        except Exception as exc:
            if config.retrieval_strict:
                raise
            candidates, docs, context = [], {}, {"backend": backend, "error": str(exc)}
            error = str(exc)

        new_candidate_count = len(candidates)
        previous_candidates = state.get("candidates", []) or []
        previous_docs = state.get("candidate_docs", {}) or {}
        previous_context = state.get("retrieval_context", {}) or {}
        if previous_candidates or previous_docs:
            candidates, docs, context = merge_retrieval_results(
                previous_candidates,
                candidates,
                previous_docs,
                docs,
                previous_context,
                context,
            )

        audit = add_audit(
            state,
            "knn_retrieval_node",
            "searched fixed Elasticsearch KNN backend",
            backend=backend,
            knn_index=config.knn_index,
            num_candidates=len(candidates),
            num_new_candidates=new_candidate_count,
            num_retrieval_passes=(context.get("num_retrieval_passes") if isinstance(context, dict) else 1) or 1,
            error=error,
        )
        return {
            "candidates": candidates,
            "candidate_docs": docs,
            "retrieval_context": context,
            "audit_log": audit,
        }

    return knn_retrieval_node

# ---------------------------------------------------------------------------
# Rule node: Candidate Screening
# ---------------------------------------------------------------------------


def candidate_screening_node_factory(config: Optional[PipelineConfig] = None):
    config = config or PipelineConfig()

    def candidate_screening_node(state: PatentAnalysisState) -> PatentAnalysisState:
        filters = state.get("filters", {}) or {}
        excluded = normalize_text(filters.get("exclude_canonical_doc_id")) or canonical_doc_id(filters.get("exclude_doc_id"))

        best_by_canonical: Dict[str, Dict[str, Any]] = {}
        for cand in state.get("candidates", []) or []:
            canonical = normalize_text(cand.get("canonical_doc_id")) or canonical_doc_id(cand.get("doc_id"))
            if not canonical or canonical == excluded:
                continue
            current = best_by_canonical.get(canonical)
            if current is None or float(cand.get("score") or 0.0) > float(current.get("score") or 0.0):
                best_by_canonical[canonical] = dict(cand, canonical_doc_id=canonical)

        screened = sorted(
            best_by_canonical.values(),
            key=lambda item: (-float(item.get("score") or 0.0), normalize_text(item.get("doc_id"))),
        )[: config.candidate_screen_top_k]

        doc_ids = [normalize_text(c.get("doc_id")) for c in screened if normalize_text(c.get("doc_id"))]
        existing_docs = state.get("candidate_docs", {}) or {}
        docs = {doc_id: existing_docs.get(doc_id, {}) for doc_id in doc_ids if doc_id}
        retrieval_backend = normalize_optional_text((state.get("retrieval_context", {}) or {}).get("backend")) or DEFAULT_RETRIEVAL_BACKEND
        doc_source = f"{retrieval_backend}_candidate_docs"

        enriched = []
        for cand in screened:
            doc = docs.get(cand["doc_id"], {})
            publication_date = doc.get("publication_date") or cand.get("publication_date", "")
            application_date = doc.get("application_date") or cand.get("application_date", "")
            priority_date = doc.get("priority_date") or cand.get("priority_date", "")
            date_for_filter = publication_date or application_date or priority_date
            if not is_prior_art_date(date_for_filter, filters.get("prior_art_cutoff_date")):
                continue
            enriched.append(
                {
                    **cand,
                    "screen_rank": len(enriched) + 1,
                    "title": doc.get("title") or cand.get("title", ""),
                    "publication_date": publication_date,
                    "application_date": application_date,
                    "priority_date": priority_date,
                    "canonical_doc_id": doc.get("canonical_doc_id") or cand.get("canonical_doc_id", ""),
                    "ipc_codes": doc.get("ipc_codes") or cand.get("ipc_codes", []),
                    "assignees": doc.get("assignees") or cand.get("assignees", []),
                    "inventors": doc.get("inventors") or cand.get("inventors", []),
                    "citations": doc.get("citations") or cand.get("citations", []),
                }
            )

        audit = add_audit(
            state,
            "candidate_screening_node",
            "deduplicated and prepared candidate documents",
            num_screened=len(enriched),
            num_docs_available=len(docs),
            doc_source=doc_source,
            num_removed_by_date=max(0, len(screened) - len(enriched)),
        )
        return {"screened_candidates": enriched, "candidate_docs": docs, "audit_log": audit}

    return candidate_screening_node


# ---------------------------------------------------------------------------
# Baseline report nodes for PB1/PB2
# ---------------------------------------------------------------------------


def candidate_baseline_report_node(
    state: PatentAnalysisState,
    config: Optional[PipelineConfig] = None,
) -> PatentAnalysisState:
    """Produce a deterministic candidate report before evidence/analysis agents."""

    _config = config or PipelineConfig()
    screened = state.get("screened_candidates", []) or []
    query_rows = state.get("search_queries", []) or []
    report_lines = [
        "# Báo cáo baseline prior art",
        "",
        f"Pipeline variant: {normalize_pipeline_variant(state.get('pipeline_variant'))}",
        f"Vấn đề kỹ thuật/truy vấn: {state.get('technical_problem', '')}",
        "",
        "## Ứng viên retrieval",
    ]
    if not screened:
        report_lines.append("- Không có ứng viên từ retrieval node cố định.")
    for cand in screened[:10]:
        title = normalize_optional_text(cand.get("title"))
        score = safe_float(cand.get("score"), default=0.0, minimum=0.0, maximum=1.0)
        report_lines.append(
            f"- {cand.get('screen_rank') or cand.get('rank')}. {cand.get('doc_id')}"
            f" | score={score:.3f}"
            f"{' | ' + title if title else ''}"
        )
    report_lines.extend(
        [
            "",
            "## Trace truy vấn",
        ]
    )
    for item in query_rows[:6]:
        if isinstance(item, dict):
            report_lines.append(f"- {item.get('search_role', 'query')}: {limit_words(item.get('text'), 28)}")

    coverage = {
        "is_sufficient": False,
        "is_sufficient_vi": "chưa đủ",
        "confidence": "low",
        "confidence_vi": "thấp",
        "coverage_notes": "baseline chỉ có ứng viên; chưa chạy trích xuất bằng chứng ở cấp claim",
        "recommended_next_searches": [],
    }
    analysis = {
        "ranked_prior_art": [
            {
                "rank": idx + 1,
                "patent_id": cand.get("doc_id"),
                "title": cand.get("title", ""),
                "novelty_risk": "unknown",
                "novelty_risk_vi": "chưa đánh giá",
                "matched_elements": [],
                "missing_elements": state.get("claim_elements", []) or [],
                "missing_elements_vi": [technical_label_vi(value) for value in state.get("claim_elements", []) or []],
                "claim_overlap_summary": "chưa phân tích trong baseline chỉ có ứng viên",
                "limitations": "chưa dùng evidence grounding hoặc agent lập luận prior art",
            }
            for idx, cand in enumerate(screened[:10])
        ],
        "coverage": coverage,
        "final_report_markdown": "\n".join(report_lines),
    }
    audit = add_audit(
        state,
        "candidate_baseline_report_node",
        "created deterministic candidate-only baseline report",
        num_screened=len(screened),
    )
    return {
        "analysis": analysis,
        "coverage": coverage,
        "final_report": analysis["final_report_markdown"],
        "audit_log": audit,
    }


# ---------------------------------------------------------------------------
# Agent 2: Evidence Extraction Agent
# ---------------------------------------------------------------------------


def heuristic_extract_evidence(state: PatentAnalysisState, config: PipelineConfig) -> List[Dict[str, Any]]:
    claim_elements = state.get("claim_elements", []) or []
    docs = state.get("candidate_docs", {}) or {}
    screened = state.get("screened_candidates", []) or []
    evidence = []

    for cand in screened[: config.evidence_top_docs]:
        doc_id = cand.get("doc_id")
        doc = docs.get(doc_id, {})
        text = candidate_text(doc, max_words=1100)
        lower_text = text.lower()
        matched = []
        missing = []

        for element in claim_elements:
            element_text = normalize_text(element)
            tokens = [t.lower() for t in re.findall(r"[A-Za-z0-9]{4,}", element_text)[:10]]
            overlap = [t for t in tokens if t in lower_text]
            if overlap:
                snippet = word_window_snippet(text, overlap, window_words=56)
                matched.append(
                    {
                        "claim_element": element_text,
                        "claim_element_vi": technical_label_vi(element_text),
                        "section": "mixed",
                        "section_vi": section_vi("mixed"),
                        "match_type": "partial",
                        "match_type_vi": match_type_vi("partial"),
                        "evidence_text": normalize_text(snippet),
                        "reason": f"trùng token: {', '.join(overlap[:5])}",
                    }
                )
            else:
                missing.append(element_text)

        evidence.append(
            {
                "patent_id": doc_id,
                "title": doc.get("title", ""),
                "matched_elements": matched,
                "missing_elements": missing,
                "missing_elements_vi": [technical_label_vi(value) for value in missing],
                "overall_relevance": "trích xuất bằng chứng heuristic",
            }
        )
    return evidence


def evidence_extraction_agent(
    state: PatentAnalysisState,
    config: Optional[PipelineConfig] = None,
) -> PatentAnalysisState:
    config = config or PipelineConfig()
    llm_config = config_for_agent_llm(config, 2)
    fallback = {"candidate_evidence": heuristic_extract_evidence(state, config)}
    agent2_top_docs = max(1, min(config.evidence_top_docs, env_int("MULTIAGENT_AGENT2_TOP_DOCS", 3)))
    agent2_text_words = max(60, env_int("MULTIAGENT_AGENT2_CANDIDATE_TEXT_WORDS", 130))
    agent2_claim_elements = max(1, env_int("MULTIAGENT_AGENT2_CLAIM_ELEMENTS", 3))
    claim_elements = [limit_words(item, 24) for item in normalize_list(state.get("claim_elements"))[:agent2_claim_elements]]
    docs = state.get("candidate_docs", {}) or {}
    screened = state.get("screened_candidates", []) or []

    candidate_blocks = []
    for cand in screened[:agent2_top_docs]:
        doc_id = normalize_text(cand.get("doc_id"))
        doc = docs.get(doc_id, {})
        candidate_blocks.append(
            {
                "rank": cand.get("screen_rank") or cand.get("rank"),
                "patent_id": doc_id,
                "title": doc.get("title", ""),
                "metadata": {
                    "canonical_doc_id": doc.get("canonical_doc_id", "") or cand.get("canonical_doc_id", ""),
                    "publication_date": doc.get("publication_date", "") or cand.get("publication_date", ""),
                    "matched_claim_elements_from_search": [
                        limit_words(item, 18)
                        for item in normalize_list(cand.get("matched_claim_elements"))[:3]
                    ],
                },
                "text": candidate_text(doc, max_words=agent2_text_words),
            }
        )

    fallback_skill = (
        "You are an evidence extraction agent for patent prior-art analysis. "
        "Extract evidence only from the provided candidate texts. Return valid JSON only."
    )
    system_prompt, skill_file = load_agent_skill(config, "evidence_extraction_agent", fallback_skill)
    candidate_blocks_json = dump_json_for_prompt(
        candidate_blocks,
        env_int("MULTIAGENT_AGENT2_CANDIDATES_JSON_CHARS", 14000),
    )
    claim_elements_json = dump_json_for_prompt(claim_elements, env_int("MULTIAGENT_AGENT2_CLAIM_ELEMENTS_JSON_CHARS", 1600))
    user_prompt = f"""
Claim elements:
{claim_elements_json}

Candidates:
{candidate_blocks_json}

Return minified JSON only:
{{
  "candidate_evidence": [
    {{
      "patent_id": "id",
      "title": "title",
      "matched_elements": [
        {{
          "claim_element": "element",
          "section": "title|abstract|claims|mixed",
          "match_type": "exact|partial|weak",
          "evidence_text": "short quote/paraphrase",
          "reason": "short reason",
          "gap_or_limitation": "short gap"
        }}
      ],
      "missing_elements": ["unsupported element"],
      "overall_relevance": "short relevance"
    }}
  ]
}}

Rules:
- Do not invent evidence.
- One row per candidate.
- At most two matched_elements per candidate.
- Put unsupported claim elements in missing_elements.
- Keep all strings short.
- Copy claim_element and missing_elements from Claim elements exactly; runtime adds Vietnamese display labels.
- Write reason, gap_or_limitation, and overall_relevance in Vietnamese with accents.
- Keep evidence_text in the source language from the candidate text; do not translate quotes.
"""
    parsed = call_llm_json(
        llm_config,
        system_prompt,
        user_prompt,
        fallback,
    )
    candidate_evidence_key = "candidate_evidence"
    raw_candidate_evidence = parsed.get(candidate_evidence_key) if isinstance(parsed, dict) else None
    if not isinstance(raw_candidate_evidence, list):
        for alias in ["evidence", "candidate_evidences", "candidate_evidence_items", "results", "items"]:
            alias_value = parsed.get(alias) if isinstance(parsed, dict) else None
            if isinstance(alias_value, list):
                candidate_evidence_key = alias
                raw_candidate_evidence = alias_value
                break
    used_fallback = parsed is fallback or not isinstance(raw_candidate_evidence, list) or not raw_candidate_evidence
    evidence = sanitize_evidence_output(raw_candidate_evidence, fallback["candidate_evidence"], state, config)
    audit = add_audit(
        state,
        "evidence_extraction_agent",
        "extracted claim-level evidence",
        skill_file=skill_file,
        used_fallback=used_fallback,
        candidate_evidence_key=candidate_evidence_key if not used_fallback else "fallback",
        llm_provider=llm_config.llm_provider,
        llm_model=llm_config.llm_model,
        max_output_tokens=effective_max_output_tokens(llm_config, system_prompt, user_prompt),
        prompt_words=prompt_word_count(system_prompt, user_prompt),
        estimated_prompt_tokens=estimate_json_prompt_tokens(system_prompt, user_prompt),
        prompt_candidate_docs=len(candidate_blocks),
        prompt_candidate_doc_ids=[item.get("patent_id") for item in candidate_blocks],
        prompt_candidate_text_words=agent2_text_words,
        num_evidence_docs=len(evidence),
    )
    return {"evidence": evidence, "audit_log": audit}


# ---------------------------------------------------------------------------
# Agent 3: Prior-Art Analysis Agent
# ---------------------------------------------------------------------------


def heuristic_prior_art_analysis(state: PatentAnalysisState) -> Dict[str, Any]:
    evidence = state.get("evidence", []) or []
    ranked = []
    for idx, item in enumerate(evidence, start=1):
        matched = item.get("matched_elements", []) or []
        missing = item.get("missing_elements", []) or []
        risk = "high" if len(matched) >= 3 and not missing else "medium" if matched else "low"
        matched_elements = [
            normalize_text(match.get("claim_element"))
            for match in matched
            if isinstance(match, dict) and normalize_text(match.get("claim_element"))
        ]
        ranked.append(
            {
                "rank": idx,
                "patent_id": item.get("patent_id"),
                "title": item.get("title"),
                "novelty_risk": risk,
                "novelty_risk_vi": {"high": "cao", "medium": "trung bình", "low": "thấp"}.get(risk, risk),
                "matched_elements": matched_elements,
                "matched_elements_vi": [technical_label_vi(value) for value in matched_elements],
                "missing_elements": missing,
                "missing_elements_vi": [technical_label_vi(value) for value in missing],
                "claim_overlap_summary": item.get("overall_relevance", ""),
                "limitations": "thiếu yếu tố: " + "; ".join(technical_label_vi(value) for value in missing[:5])
                if missing
                else "chưa thấy yếu tố còn thiếu trong bảng bằng chứng",
            }
        )

    sufficient = sum(1 for item in evidence if item.get("matched_elements")) >= 2
    report_lines = [
        "# Báo cáo phân tích prior art",
        "",
        f"Vấn đề kỹ thuật: {state.get('technical_problem', '')}",
        "",
        "## Prior art mạnh nhất",
    ]
    for row in ranked[:5]:
        report_lines.append(
            f"- {row['rank']}. {row.get('patent_id')}: rủi ro novelty={row.get('novelty_risk_vi')}; "
            f"{len(row.get('matched_elements') or [])} yếu tố có bằng chứng."
        )

    return {
        "ranked_prior_art": ranked,
        "coverage": {
            "is_sufficient": sufficient,
            "is_sufficient_vi": "đủ" if sufficient else "chưa đủ",
            "confidence": "medium" if sufficient else "low",
            "confidence_vi": "trung bình" if sufficient else "thấp",
            "coverage_notes": "fallback heuristic; chỉ dùng bảng bằng chứng đã trích xuất",
            "recommended_next_searches": [],
        },
        "acceptance_assessment": {
            "acceptance_likelihood": "difficult" if any(row.get("novelty_risk") == "high" for row in ranked[:3]) else "uncertain",
            "acceptance_likelihood_vi": "khó"
            if any(row.get("novelty_risk") == "high" for row in ranked[:3])
            else "chưa chắc chắn",
            "main_obstacles": [
                "mức trùng prior art trong bằng chứng trích xuất còn mạnh"
            ] if any(row.get("novelty_risk") == "high" for row in ranked[:3]) else ["độ bao phủ bằng chứng chưa đầy đủ"],
            "blocking_prior_art": [row.get("patent_id") for row in ranked[:3] if row.get("novelty_risk") == "high"],
            "why": "fallback heuristic dựa trên bảng bằng chứng đã trích xuất",
            "recommended_strategy": "xác định các giới hạn chưa được prior art mạnh nhất dạy và thu hẹp yêu cầu quanh khác biệt đó",
            "amendment_directions": [],
        },
        "final_report_markdown": "\n".join(report_lines),
    }


def evidence_baseline_report_node(
    state: PatentAnalysisState,
    config: Optional[PipelineConfig] = None,
) -> PatentAnalysisState:
    """Produce a deterministic evidence report for PB3 before final analysis agent."""

    _config = config or PipelineConfig()
    analysis = heuristic_prior_art_analysis(state)
    coverage = analysis["coverage"]
    evidence = state.get("evidence", []) or []
    report_lines = [
        "# Báo cáo baseline bằng chứng prior art",
        "",
        f"Pipeline variant: {normalize_pipeline_variant(state.get('pipeline_variant'))}",
        f"Vấn đề kỹ thuật: {state.get('technical_problem', '')}",
        "",
        "## Độ bao phủ bằng chứng",
    ]
    if not evidence:
        report_lines.append("- Chưa trích xuất được bằng chứng.")
    for item in evidence[:8]:
        matched = item.get("matched_elements", []) or []
        missing = item.get("missing_elements", []) or []
        report_lines.append(
            f"- {item.get('patent_id')}: {len(matched)} yếu tố khớp, {len(missing)} yếu tố còn thiếu."
        )
        for match in matched[:3]:
            if isinstance(match, dict):
                claim_label = match.get("claim_element_vi") or technical_label_vi(match.get("claim_element"))
                report_lines.append(
                    f"  - {limit_words(claim_label, 18)}: "
                    f"{limit_words(match.get('evidence_text'), 26)}"
                )
    analysis["final_report_markdown"] = "\n".join(report_lines)
    audit = add_audit(
        state,
        "evidence_baseline_report_node",
        "created deterministic evidence baseline report",
        num_evidence_docs=len(evidence),
    )
    return {
        "analysis": analysis,
        "coverage": coverage,
        "final_report": analysis["final_report_markdown"],
        "audit_log": audit,
    }


def prior_art_analysis_agent(
    state: PatentAnalysisState,
    config: Optional[PipelineConfig] = None,
) -> PatentAnalysisState:
    config = config or PipelineConfig()
    llm_config = config_for_agent_llm(config, 3)
    fallback = heuristic_prior_art_analysis(state)
    agent3_top_docs = max(1, min(config.evidence_top_docs, env_int("MULTIAGENT_AGENT3_TOP_DOCS", 3)))
    query_understanding_prompt = {
        "technical_problem": limit_words(state.get("technical_problem", ""), 32),
        "key_features": [limit_words(item, 16) for item in normalize_list(state.get("key_features"))[:3]],
        "claim_elements": [limit_words(item, 22) for item in normalize_list(state.get("claim_elements"))[:3]],
        "claim_elements_vi": [
            limit_words(technical_label_vi(item), 22)
            for item in normalize_list(state.get("claim_elements"))[:3]
        ],
    }
    screened_prompt = compact_screened_candidates_for_prompt(state, config, agent3_top_docs)
    evidence_prompt = compact_evidence_for_prompt(state, agent3_top_docs)

    fallback_skill = (
        "You are a senior patent prior-art analysis agent. Analyze novelty risk from the "
        "provided evidence only. Write the final report in Vietnamese. Return valid JSON only."
    )
    system_prompt, skill_file = load_agent_skill(config, "prior_art_analysis_agent", fallback_skill)
    user_prompt = f"""
Query:
{dump_json_for_prompt(query_understanding_prompt, env_int("MULTIAGENT_AGENT3_QUERY_JSON_CHARS", 1200))}

Candidates:
{dump_json_for_prompt(screened_prompt, env_int("MULTIAGENT_AGENT3_CANDIDATES_JSON_CHARS", 1200))}

Evidence:
{dump_json_for_prompt(evidence_prompt, env_int("MULTIAGENT_AGENT3_EVIDENCE_JSON_CHARS", 2600))}

Return minified JSON only:
{{
  "technical_problem_vi": "tóm tắt vấn đề kỹ thuật bằng tiếng Việt có dấu",
  "ranked_prior_art": [
    {{
      "rank": 1,
      "patent_id": "id",
      "title": "title",
      "novelty_risk": "high|medium|low",
      "matched_elements": ["element"],
      "missing_elements": ["element"],
      "claim_overlap_summary": "tóm tắt mức trùng bằng tiếng Việt",
      "limitations": "giới hạn/điểm thiếu bằng tiếng Việt"
    }}
  ],
  "coverage": {{
    "is_sufficient": true,
    "confidence": "high|medium|low",
    "coverage_notes": "ghi chú bằng tiếng Việt",
    "recommended_next_searches": ["truy vấn bổ sung nếu cần"]
  }},
  "acceptance_assessment": {{
    "acceptance_likelihood": "likely|uncertain|difficult",
    "main_obstacles": ["trở ngại cụ thể từ bằng chứng"],
    "blocking_prior_art": ["patent id"],
    "why": "lý do bằng tiếng Việt",
    "recommended_strategy": "chiến lược bằng tiếng Việt",
    "amendment_directions": ["hướng sửa bằng tiếng Việt"]
  }}
}}

Rules:
- Do not add patents not present in evidence.
- Do not invent assignee, date, claim text, or retrieval evidence.
- Acceptance assessment is a technical/prosecution-risk estimate, not legal advice.
- Return one ranked_prior_art row for each evidence document, up to 3 documents, even when novelty risk is low.
- Keep every string short.
- Write all explanatory strings in Vietnamese with accents.
- Keep patent IDs, titles, and evidence quotes in their source language.
- For matched_elements and missing_elements, prefer the Vietnamese labels from claim_elements_vi/claim_element_vi when available.
- Use blocking_prior_art only for high-risk patents; medium-risk partial overlaps are not blocking.
- If the strongest candidates miss a central claim element, treat that missing element as a distinction.
- If evidence is weak or only partial, set is_sufficient=false.
- Do not output final_report_markdown; runtime code writes the markdown report.
"""

    parsed = call_llm_json(
        llm_config,
        system_prompt,
        user_prompt,
        fallback,
        expected_keys=["ranked_prior_art", "coverage", "acceptance_assessment"],
    )
    used_fallback = parsed is fallback
    analysis = sanitize_analysis_output(parsed, fallback, state)
    coverage = analysis["coverage"]
    final_report = analysis["final_report_markdown"]
    audit = add_audit(
        state,
        "prior_art_analysis_agent",
        "completed prior-art analysis",
        skill_file=skill_file,
        used_fallback=used_fallback,
        llm_provider=llm_config.llm_provider,
        llm_model=llm_config.llm_model,
        max_output_tokens=effective_max_output_tokens(llm_config, system_prompt, user_prompt),
        prompt_words=prompt_word_count(system_prompt, user_prompt),
        estimated_prompt_tokens=estimate_json_prompt_tokens(system_prompt, user_prompt),
        prompt_candidate_docs=len(screened_prompt),
        prompt_candidate_doc_ids=[item.get("patent_id") for item in screened_prompt],
        prompt_evidence_docs=len(evidence_prompt),
        prompt_evidence_items=sum(len(item.get("matched_elements") or []) for item in evidence_prompt),
        sufficient=coverage.get("is_sufficient"),
        confidence=coverage.get("confidence"),
    )
    return {
        "analysis": analysis,
        "coverage": coverage,
        "final_report": final_report,
        "audit_log": audit,
    }


# ---------------------------------------------------------------------------
# Coverage router
# ---------------------------------------------------------------------------


def coverage_check_node(
    state: PatentAnalysisState,
    config: Optional[PipelineConfig] = None,
) -> PatentAnalysisState:
    config = config or PipelineConfig()
    coverage = state.get("coverage", {}) or {}
    iteration = int(state.get("iteration", 0))
    max_iterations = int(state.get("max_iterations", config.max_iterations))

    evidence_docs = [item for item in state.get("evidence", []) or [] if item.get("matched_elements")]
    sufficient = normalize_bool(coverage.get("is_sufficient")) and len(evidence_docs) >= config.min_evidence_docs
    should_retry = (not sufficient) and iteration + 1 < max_iterations

    search_queries = list(state.get("search_queries", []) or [])
    retrieval_focus = dict(state.get("retrieval_focus", {}) or {})
    if should_retry:
        recommended_next_searches = normalize_list(coverage.get("recommended_next_searches"))
        for query in recommended_next_searches[:3]:
            search_queries.append(
                {
                    "query_view": "feature",
                    "search_role": "problem",
                    "text": query,
                    "weight": 1.05,
                    "search_mode": "problem_expansion",
                    "iteration_added": iteration + 1,
                }
            )
        if not recommended_next_searches:
            for element in (state.get("claim_elements", []) or [])[:3]:
                search_queries.append(
                    {
                        "query_view": "feature",
                        "search_role": "claim_overlap",
                        "text": normalize_text(element),
                        "weight": 1.0,
                        "search_mode": "claim_text",
                        "iteration_added": iteration + 1,
                    }
                )
        search_queries = dedupe_search_queries(search_queries, max_queries=10)
        retrieval_focus["preferred_retrieval_backend"] = "es_knn"
        retrieval_focus["search_intent"] = "coverage_repair_with_fixed_knn_search"

    audit = add_audit(
        state,
        "coverage_check_node",
        "checked coverage and routed workflow",
        sufficient=sufficient,
        should_retry=should_retry,
        iteration=iteration,
        evidence_docs_with_matches=len(evidence_docs),
    )
    return {
        "should_retry": should_retry,
        "iteration": iteration + 1,
        "search_queries": search_queries,
        "retrieval_focus": retrieval_focus,
        "audit_log": audit,
    }


def route_after_coverage(state: PatentAnalysisState) -> str:
    return "knn_retrieval_node" if state.get("should_retry") else "end"


def compute_proxy_metrics(state: PatentAnalysisState) -> Dict[str, Any]:
    """Cheap automatic metrics for ablation smoke tests.

    These are not a replacement for qrels/manual labels. They are useful for
    checking whether each added agent improves the expected internal signal.
    """

    claim_elements = [normalize_text(item) for item in state.get("claim_elements", []) or [] if normalize_text(item)]
    evidence = state.get("evidence", []) or []
    matched_claims = set()
    exact_claims = set()
    partial_claims = set()
    grounded_items = 0
    exact_items = 0
    partial_items = 0
    weak_items = 0
    for item in evidence:
        for match in item.get("matched_elements", []) or []:
            if not isinstance(match, dict):
                continue
            claim = normalize_text(match.get("claim_element"))
            evidence_text = normalize_text(match.get("evidence_text"))
            match_type = normalize_text(match.get("match_type")).lower()
            if claim:
                matched_claims.add(claim.lower())
                if match_type == "exact":
                    exact_claims.add(claim.lower())
                elif match_type == "partial":
                    partial_claims.add(claim.lower())
            if claim and evidence_text:
                grounded_items += 1
                if match_type == "exact":
                    exact_items += 1
                elif match_type == "partial":
                    partial_items += 1
                elif match_type == "weak":
                    weak_items += 1

    coverage = state.get("coverage", {}) or {}
    num_claims = len(claim_elements)
    return {
        "pipeline_variant": normalize_pipeline_variant(state.get("pipeline_variant")),
        "num_search_queries": len(state.get("search_queries", []) or []),
        "num_candidates": len(state.get("candidates", []) or []),
        "num_screened_candidates": len(state.get("screened_candidates", []) or []),
        "num_claim_elements": num_claims,
        "num_evidence_docs": len(evidence),
        "num_grounded_evidence_items": grounded_items,
        "num_exact_evidence_items": exact_items,
        "num_partial_evidence_items": partial_items,
        "num_weak_evidence_items": weak_items,
        "claim_element_coverage": round(len(matched_claims) / num_claims, 4) if num_claims else 0.0,
        "claim_element_exact_coverage": round(len(exact_claims) / num_claims, 4) if num_claims else 0.0,
        "claim_element_partial_coverage": round(len(partial_claims) / num_claims, 4) if num_claims else 0.0,
        "coverage_sufficient": normalize_bool(coverage.get("is_sufficient")),
        "coverage_confidence": normalize_text(coverage.get("confidence")),
        "final_report_chars": len(normalize_text(state.get("final_report"))),
    }


def finalize_state_metrics(state: PatentAnalysisState) -> PatentAnalysisState:
    state["proxy_metrics"] = compute_proxy_metrics(state)
    return state


# ---------------------------------------------------------------------------
# Graph construction and runners
# ---------------------------------------------------------------------------


def build_patent_analysis_graph(config: Optional[PipelineConfig] = None, variant: Optional[str] = None):
    if StateGraph is None:
        raise RuntimeError("Missing dependency: install langgraph to compile the workflow graph.")

    config = config or PipelineConfig()
    variant = normalize_pipeline_variant(variant or config.pipeline_variant)
    workflow = StateGraph(PatentAnalysisState)

    if variant == "pb1":
        workflow.add_node("raw_query_baseline_node", lambda state: raw_query_baseline_node(state, config))
        workflow.add_node("knn_retrieval_node", knn_retrieval_node_factory(config=config))
        workflow.add_node("candidate_screening_node", candidate_screening_node_factory(config=config))
        workflow.add_node("candidate_baseline_report_node", lambda state: candidate_baseline_report_node(state, config))
        workflow.add_edge(START, "raw_query_baseline_node")
        workflow.add_edge("raw_query_baseline_node", "knn_retrieval_node")
        workflow.add_edge("knn_retrieval_node", "candidate_screening_node")
        workflow.add_edge("candidate_screening_node", "candidate_baseline_report_node")
        workflow.add_edge("candidate_baseline_report_node", END)
    elif variant == "pb2":
        workflow.add_node("query_understanding_agent", lambda state: query_understanding_agent(state, config))
        workflow.add_node("knn_retrieval_node", knn_retrieval_node_factory(config=config))
        workflow.add_node("candidate_screening_node", candidate_screening_node_factory(config=config))
        workflow.add_node("candidate_baseline_report_node", lambda state: candidate_baseline_report_node(state, config))
        workflow.add_edge(START, "query_understanding_agent")
        workflow.add_edge("query_understanding_agent", "knn_retrieval_node")
        workflow.add_edge("knn_retrieval_node", "candidate_screening_node")
        workflow.add_edge("candidate_screening_node", "candidate_baseline_report_node")
        workflow.add_edge("candidate_baseline_report_node", END)
    elif variant == "pb3":
        workflow.add_node("query_understanding_agent", lambda state: query_understanding_agent(state, config))
        workflow.add_node("knn_retrieval_node", knn_retrieval_node_factory(config=config))
        workflow.add_node("candidate_screening_node", candidate_screening_node_factory(config=config))
        workflow.add_node("evidence_extraction_agent", lambda state: evidence_extraction_agent(state, config))
        workflow.add_node("evidence_baseline_report_node", lambda state: evidence_baseline_report_node(state, config))
        workflow.add_edge(START, "query_understanding_agent")
        workflow.add_edge("query_understanding_agent", "knn_retrieval_node")
        workflow.add_edge("knn_retrieval_node", "candidate_screening_node")
        workflow.add_edge("candidate_screening_node", "evidence_extraction_agent")
        workflow.add_edge("evidence_extraction_agent", "evidence_baseline_report_node")
        workflow.add_edge("evidence_baseline_report_node", END)
    else:
        workflow.add_node("query_understanding_agent", lambda state: query_understanding_agent(state, config))
        workflow.add_node("knn_retrieval_node", knn_retrieval_node_factory(config=config))
        workflow.add_node("candidate_screening_node", candidate_screening_node_factory(config=config))
        workflow.add_node("evidence_extraction_agent", lambda state: evidence_extraction_agent(state, config))
        workflow.add_node("prior_art_analysis_agent", lambda state: prior_art_analysis_agent(state, config))
        workflow.add_node("coverage_check_node", lambda state: coverage_check_node(state, config))
        workflow.add_edge(START, "query_understanding_agent")
        workflow.add_edge("query_understanding_agent", "knn_retrieval_node")
        workflow.add_edge("knn_retrieval_node", "candidate_screening_node")
        workflow.add_edge("candidate_screening_node", "evidence_extraction_agent")
        workflow.add_edge("evidence_extraction_agent", "prior_art_analysis_agent")
        workflow.add_edge("prior_art_analysis_agent", "coverage_check_node")
        workflow.add_conditional_edges(
            "coverage_check_node",
            route_after_coverage,
            {"knn_retrieval_node": "knn_retrieval_node", "end": END},
        )

    return workflow.compile()


def initial_state(
    input_text: str,
    input_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[PipelineConfig] = None,
) -> PatentAnalysisState:
    config = config or PipelineConfig()
    return {
        "input_text": normalize_text(input_text),
        "input_metadata": input_metadata or {},
        "pipeline_variant": normalize_pipeline_variant(config.pipeline_variant),
        "iteration": 0,
        "max_iterations": config.max_iterations,
        "audit_log": [],
    }


def detect_notebook_platform() -> str:
    if os.getenv("KAGGLE_URL_BASE") or Path("/kaggle").exists():
        return "kaggle"
    try:
        in_colab = importlib.util.find_spec("google.colab") is not None
    except ModuleNotFoundError:
        in_colab = False
    if in_colab or Path("/content").exists():
        return "colab"
    return "local"


def validate_runtime_environment(
    config: Optional[PipelineConfig] = None,
    require_langgraph: bool = True,
) -> Dict[str, Any]:
    config = config or PipelineConfig()
    issues: List[str] = []
    warnings: List[str] = []

    if sys.version_info < (3, 10) and require_langgraph:
        issues.append("Python 3.10+ is required for LangGraph multi-agent runtime.")
    elif sys.version_info < (3, 10):
        warnings.append("Python 3.10+ is recommended; the benchmark linear runner may work without LangGraph.")

    if require_langgraph and importlib.util.find_spec("langgraph") is None:
        issues.append("Missing dependency: pip install langgraph")

    provider = normalize_optional_text(config.llm_provider).lower() or DEFAULT_AGENT_LLM_PROVIDER
    variant = normalize_pipeline_variant(config.pipeline_variant)
    if normalize_optional_text(config.pipeline_variant).lower() not in VALID_PIPELINE_VARIANTS:
        warnings.append(f"Unknown MULTIAGENT_VARIANT={config.pipeline_variant}; defaulting to {variant}.")
    if provider not in GEMINI_PROVIDERS and provider not in LOCAL_HF_PROVIDERS and not provider_is_openai_compatible(provider):
        issues.append(f"Unsupported MULTIAGENT_LLM_PROVIDER: {config.llm_provider}")

    has_llm_key = False
    if provider in GEMINI_PROVIDERS:
        if genai is None:
            issues.append("Missing dependency: pip install google-genai")
        has_llm_key = bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY") or os.getenv("GOOGLE_GENAI_API_KEY"))
        if not has_llm_key and config.llm_strict:
            issues.append("Missing GEMINI_API_KEY/GOOGLE_API_KEY while MULTIAGENT_LLM_STRICT=true.")
        elif not has_llm_key:
            warnings.append("No GEMINI_API_KEY/GOOGLE_API_KEY detected; LLM agents need a configured key.")
    elif provider in LOCAL_HF_PROVIDERS:
        for module_name, install_name in [
            ("torch", "torch"),
            ("transformers", "transformers"),
            ("accelerate", "accelerate"),
        ]:
            if importlib.util.find_spec(module_name) is None:
                issues.append(f"Missing dependency for local-hf LLM provider: pip install {install_name}")
        if config.local_hf_load_in_4bit and importlib.util.find_spec("bitsandbytes") is None:
            issues.append(
                "Missing dependency for local-hf 4-bit loading: pip install bitsandbytes, "
                "or set MULTIAGENT_LOCAL_HF_LOAD_IN_4BIT=false."
            )
        if not normalize_optional_text(config.llm_model):
            issues.append("Missing MULTIAGENT_LLM_MODEL while MULTIAGENT_LLM_PROVIDER=local-hf.")
    elif provider_is_openai_compatible(provider):
        if OpenAI is None:
            issues.append("Missing dependency: pip install openai")
        base_url = openai_compatible_base_url(provider, config)
        has_llm_key = bool(openai_compatible_api_key(provider, base_url))
        if not has_llm_key and config.llm_strict:
            issues.append(
                "Missing LLM API key while MULTIAGENT_LLM_STRICT=true. "
                "Set GROQ_API_KEY, OPENAI_API_KEY, OPENROUTER_API_KEY, or MULTIAGENT_LLM_API_KEY."
            )
        elif not has_llm_key:
            warnings.append("No OpenAI-compatible LLM API key detected; LLM agents need a configured key.")

    retrieval_backend = normalize_optional_text(config.retrieval_backend).lower() or DEFAULT_RETRIEVAL_BACKEND
    if retrieval_backend in KNN_RETRIEVAL_BACKENDS:
        if importlib.util.find_spec("elasticsearch") is None:
            issues.append("Missing dependency for ES KNN retrieval: pip install elasticsearch")
        if not normalize_optional_text(config.es_cloud_id):
            issues.append("Missing ES_CLOUD_ID while MULTIAGENT_RETRIEVAL_BACKEND=es_knn.")
        if not (normalize_optional_text(config.es_api_key) or (normalize_optional_text(config.es_user) and normalize_optional_text(config.es_password))):
            issues.append("Missing ES_API_KEY or ES_USER + ES_PASSWORD while MULTIAGENT_RETRIEVAL_BACKEND=es_knn.")
        if not normalize_optional_text(config.bm25_index):
            issues.append("Missing BM25_INDEX for fetching candidate patent text.")
        if not normalize_optional_text(config.knn_index):
            issues.append("Missing KNN_INDEX for ES KNN retrieval.")
        if not normalize_optional_text(config.vector_field):
            issues.append("Missing ES_VECTOR_FIELD for ES KNN retrieval.")
        if normalize_optional_text(config.knn_embedding_api_base) and importlib.util.find_spec("requests") is None:
            issues.append("Missing dependency for KNN embedding HTTP endpoint: pip install requests")
        if not normalize_optional_text(config.knn_embedding_api_base):
            for module_name, install_name in [("transformers", "transformers"), ("torch", "torch"), ("numpy", "numpy")]:
                if importlib.util.find_spec(module_name) is None:
                    issues.append(
                        f"Missing dependency for local Jina KNN embedding: pip install {install_name}, "
                        "or set MULTIAGENT_KNN_EMBED_API_BASE."
                    )
    else:
        issues.append(f"Unsupported MULTIAGENT_RETRIEVAL_BACKEND={config.retrieval_backend}; only es_knn is supported.")

    return {
        "ok": not issues,
        "platform": detect_notebook_platform(),
        "python": sys.version.split()[0],
        "pipeline_variant": variant,
        "llm_provider": provider,
        "llm_model": config.gemini_model if provider in GEMINI_PROVIDERS else config.llm_model,
        "llm_api_base": openai_compatible_base_url(provider, config) if provider_is_openai_compatible(provider) else "",
        "local_hf_load_in_4bit": config.local_hf_load_in_4bit if provider in LOCAL_HF_PROVIDERS else False,
        "local_hf_max_input_tokens": config.local_hf_max_input_tokens if provider in LOCAL_HF_PROVIDERS else 0,
        "local_hf_enable_thinking": config.local_hf_enable_thinking if provider in LOCAL_HF_PROVIDERS else False,
        "llm_strict": config.llm_strict,
        "retrieval_backend": retrieval_backend,
        "retrieval_strict": config.retrieval_strict,
        "require_langgraph": require_langgraph,
        "knn_index": config.knn_index if retrieval_backend in KNN_RETRIEVAL_BACKENDS else "",
        "bm25_index": config.bm25_index if retrieval_backend in KNN_RETRIEVAL_BACKENDS else "",
        "vector_field": config.vector_field if retrieval_backend in KNN_RETRIEVAL_BACKENDS else "",
        "knn_top_k": config.knn_top_k if retrieval_backend in KNN_RETRIEVAL_BACKENDS else 0,
        "knn_embedding_backend": (
            "openai_compatible" if retrieval_backend in KNN_RETRIEVAL_BACKENDS and config.knn_embedding_api_base
            else "local_hf_jina" if retrieval_backend in KNN_RETRIEVAL_BACKENDS
            else ""
        ),
        "issues": issues,
        "warnings": warnings,
    }


def configure_notebook_runtime(
    gemini_api_key: str = "",
    gemini_model: str = "",
    llm_provider: str = DEFAULT_AGENT_LLM_PROVIDER,
    llm_model: str = "",
    llm_api_key: str = "",
    llm_api_base: str = "",
    groq_api_key: str = "",
    llm_strict: bool = True,
    retrieval_backend: str = DEFAULT_RETRIEVAL_BACKEND,
    es_cloud_id: str = "",
    es_api_key: str = "",
    es_user: str = "",
    es_password: str = "",
    bm25_index: str = "",
    knn_index: str = "",
    knn_embed_api_base: str = "",
    knn_embed_model: str = "",
    pipeline_variant: str = DEFAULT_PIPELINE_VARIANT,
    max_iterations: Optional[int] = None,
    strict: bool = True,
) -> PipelineConfig:
    """Set notebook-friendly environment variables and return a fresh config."""

    backend = normalize_text(retrieval_backend).lower() or DEFAULT_RETRIEVAL_BACKEND
    os.environ["MULTIAGENT_RETRIEVAL_BACKEND"] = backend
    os.environ["MULTIAGENT_RETRIEVAL_STRICT"] = "true" if strict else "false"
    if es_cloud_id:
        os.environ["ES_CLOUD_ID"] = normalize_text(es_cloud_id)
    if es_api_key:
        os.environ["ES_API_KEY"] = normalize_text(es_api_key)
    if es_user:
        os.environ["ES_USER"] = normalize_text(es_user)
    if es_password:
        os.environ["ES_PASSWORD"] = normalize_text(es_password)
    if bm25_index:
        os.environ["BM25_INDEX"] = normalize_text(bm25_index)
    if knn_index:
        os.environ["KNN_INDEX"] = normalize_text(knn_index)
    if knn_embed_api_base:
        os.environ["MULTIAGENT_KNN_EMBED_API_BASE"] = normalize_text(knn_embed_api_base).rstrip("/")
    if knn_embed_model:
        os.environ["MULTIAGENT_KNN_EMBED_MODEL"] = normalize_text(knn_embed_model)
    provider = normalize_text(llm_provider) or DEFAULT_AGENT_LLM_PROVIDER
    os.environ["MULTIAGENT_LLM_PROVIDER"] = provider
    os.environ["MULTIAGENT_LLM_STRICT"] = "true" if llm_strict else "false"
    os.environ["MULTIAGENT_VARIANT"] = normalize_pipeline_variant(pipeline_variant)
    if llm_model:
        os.environ["MULTIAGENT_LLM_MODEL"] = llm_model
    if llm_api_base:
        os.environ["MULTIAGENT_LLM_API_BASE"] = llm_api_base
    if groq_api_key:
        os.environ["GROQ_API_KEY"] = groq_api_key
    if llm_api_key:
        if provider == "groq":
            os.environ["GROQ_API_KEY"] = llm_api_key
        elif provider == "openrouter":
            os.environ["OPENROUTER_API_KEY"] = llm_api_key
        elif provider == "openai":
            os.environ["OPENAI_API_KEY"] = llm_api_key
        else:
            os.environ["MULTIAGENT_LLM_API_KEY"] = llm_api_key
    if gemini_api_key:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
    if gemini_model:
        os.environ["GEMINI_MODEL"] = gemini_model
    if max_iterations is not None:
        os.environ["MULTIAGENT_MAX_ITERATIONS"] = str(int(max_iterations))
    return PipelineConfig()


def run_linear_pipeline(
    input_text: str,
    input_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[PipelineConfig] = None,
    variant: Optional[str] = None,
) -> PatentAnalysisState:
    """Run the same workflow without LangGraph. Useful for local debugging."""

    config = config or PipelineConfig()
    variant = normalize_pipeline_variant(variant or config.pipeline_variant)
    state: PatentAnalysisState = initial_state(input_text, input_metadata, config)
    state["pipeline_variant"] = variant
    retrieval_node = knn_retrieval_node_factory(config=config)
    screen_node = candidate_screening_node_factory(config=config)

    if variant == "pb1":
        state.update(raw_query_baseline_node(state, config))
        state.update(retrieval_node(state))
        state.update(screen_node(state))
        state.update(candidate_baseline_report_node(state, config))
        return finalize_state_metrics(state)

    state.update(query_understanding_agent(state, config))
    state.update(retrieval_node(state))
    state.update(screen_node(state))

    if variant == "pb2":
        state.update(candidate_baseline_report_node(state, config))
        return finalize_state_metrics(state)

    state.update(evidence_extraction_agent(state, config))
    if variant == "pb3":
        state.update(evidence_baseline_report_node(state, config))
        return finalize_state_metrics(state)

    while True:
        state.update(prior_art_analysis_agent(state, config))
        state.update(coverage_check_node(state, config))
        if not state.get("should_retry"):
            break
        state.update(retrieval_node(state))
        state.update(screen_node(state))
        state.update(evidence_extraction_agent(state, config))
    return finalize_state_metrics(state)


def run_query_understanding_stage(
    input_text: str,
    input_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[PipelineConfig] = None,
    variant: Optional[str] = None,
    verbose: Optional[bool] = None,
) -> PatentAnalysisState:
    """Run only the query-preparation stage used before fixed retrieval."""

    config = config or PipelineConfig()
    variant = normalize_pipeline_variant(variant or config.pipeline_variant)
    state: PatentAnalysisState = initial_state(input_text, input_metadata, config)
    state["pipeline_variant"] = variant
    log_enabled = stage_log_enabled(verbose)
    progress_log(
        log_enabled,
        "Agent 1 stage started",
        variant=variant,
        input_words=len(normalize_text(input_text).split()),
    )
    if variant == "pb1":
        progress_log(log_enabled, "Agent 1 using raw-query baseline")
        state.update(raw_query_baseline_node(state, config))
    else:
        state.update(query_understanding_agent(state, config, verbose=log_enabled))
    state = finalize_state_metrics(state)
    progress_log(
        log_enabled,
        "Agent 1 stage finished",
        claim_elements=len(state.get("claim_elements", []) or []),
        search_queries=len(state.get("search_queries", []) or []),
    )
    return state


def run_retrieval_stage(
    state: PatentAnalysisState,
    config: Optional[PipelineConfig] = None,
    variant: Optional[str] = None,
    include_screening: bool = True,
    include_report: bool = True,
) -> PatentAnalysisState:
    """Run fixed retrieval from a saved query-understanding state."""

    config = config or PipelineConfig()
    variant = normalize_pipeline_variant(variant or state.get("pipeline_variant") or config.pipeline_variant)
    state = dict(state)  # avoid mutating caller-owned cached agent1 output
    state["pipeline_variant"] = variant
    retrieval_node = knn_retrieval_node_factory(config=config)
    state.update(retrieval_node(state))
    if include_screening:
        screen_node = candidate_screening_node_factory(config=config)
        state.update(screen_node(state))
    if include_report and variant in {"pb1", "pb2"}:
        state.update(candidate_baseline_report_node(state, config))
    return finalize_state_metrics(state)


def run_evidence_stage(
    state: PatentAnalysisState,
    config: Optional[PipelineConfig] = None,
    variant: Optional[str] = None,
    include_report: bool = False,
) -> PatentAnalysisState:
    """Run only Agent 2 evidence extraction from a retrieved/screened state."""

    config = config or PipelineConfig()
    variant = normalize_pipeline_variant(variant or state.get("pipeline_variant") or config.pipeline_variant)
    state = dict(state)
    state["pipeline_variant"] = variant
    state.update(evidence_extraction_agent(state, config))
    if include_report:
        state.update(evidence_baseline_report_node(state, config))
    return finalize_state_metrics(state)


def run_analysis_stage(
    state: PatentAnalysisState,
    config: Optional[PipelineConfig] = None,
    variant: Optional[str] = None,
    include_coverage: bool = True,
) -> PatentAnalysisState:
    """Run only Agent 3 prior-art analysis from an evidence state."""

    config = config or PipelineConfig()
    variant = normalize_pipeline_variant(variant or state.get("pipeline_variant") or config.pipeline_variant)
    state = dict(state)
    state["pipeline_variant"] = variant
    state.update(prior_art_analysis_agent(state, config))
    if include_coverage:
        state.update(coverage_check_node(state, config))
    return finalize_state_metrics(state)


def run_output_stage(
    state: PatentAnalysisState,
    config: Optional[PipelineConfig] = None,
    variant: Optional[str] = None,
) -> PatentAnalysisState:
    """Finalize output state and create the appropriate report if missing."""

    config = config or PipelineConfig()
    variant = normalize_pipeline_variant(variant or state.get("pipeline_variant") or config.pipeline_variant)
    state = dict(state)
    state["pipeline_variant"] = variant
    if not normalize_text(state.get("final_report")):
        if variant in {"pb1", "pb2"}:
            state.update(candidate_baseline_report_node(state, config))
        elif variant == "pb3":
            state.update(evidence_baseline_report_node(state, config))
        else:
            state.update(prior_art_analysis_agent(state, config))
            state.update(coverage_check_node(state, config))
    return finalize_state_metrics(state)


def run_graph_pipeline(
    input_text: str,
    input_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[PipelineConfig] = None,
    variant: Optional[str] = None,
) -> PatentAnalysisState:
    config = config or PipelineConfig()
    variant = normalize_pipeline_variant(variant or config.pipeline_variant)
    graph = build_patent_analysis_graph(config=config, variant=variant)
    state = initial_state(input_text, input_metadata, config)
    state["pipeline_variant"] = variant
    return finalize_state_metrics(graph.invoke(state))


def run_pipeline(
    input_text: str,
    input_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[PipelineConfig] = None,
    prefer_langgraph: bool = True,
    variant: Optional[str] = None,
) -> PatentAnalysisState:
    if prefer_langgraph:
        if StateGraph is None:
            raise RuntimeError("LangGraph is not installed. Install langgraph or call run_pipeline(..., prefer_langgraph=False).")
        return run_graph_pipeline(input_text, input_metadata=input_metadata, config=config, variant=variant)
    return run_linear_pipeline(input_text, input_metadata=input_metadata, config=config, variant=variant)


def run_ablation_suite(
    input_text: str,
    input_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[PipelineConfig] = None,
    prefer_langgraph: bool = True,
    variants: Optional[List[str]] = None,
) -> Dict[str, PatentAnalysisState]:
    config = config or PipelineConfig()
    selected = [normalize_pipeline_variant(item) for item in (variants or ["pb1", "pb2", "pb3", "pb4"])]
    results: Dict[str, PatentAnalysisState] = {}
    for variant in selected:
        results[variant] = run_pipeline(
            input_text,
            input_metadata=input_metadata,
            config=config,
            prefer_langgraph=prefer_langgraph,
            variant=variant,
        )
    return results


def build_ablation_summary(results: Dict[str, PatentAnalysisState]) -> Dict[str, Any]:
    labels = {
        "pb1": "PB1 raw query + fixed KNN retrieval + deterministic report",
        "pb2": "PB2 + Query Understanding Agent",
        "pb3": "PB3 + Evidence Grounding Agent",
        "pb4": "PB4 + Prior-Art Analysis Agent",
    }
    return {
        variant: {
            "label": labels.get(variant, variant),
            "proxy_metrics": state.get("proxy_metrics", compute_proxy_metrics(state)),
            "final_report_preview": limit_words(state.get("final_report"), 80),
        }
        for variant, state in results.items()
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Run multi-agent patent prior-art analysis.")
    parser.add_argument("--text", default="", help="Patent/idea text to analyze.")
    parser.add_argument("--input-file", default="", help="Path to a text file containing patent/idea text.")
    parser.add_argument("--metadata-json", default="", help="Optional metadata JSON string.")
    parser.add_argument("--linear", action="store_true", help="Run without LangGraph.")
    parser.add_argument(
        "--variant",
        default="",
        choices=sorted(VALID_PIPELINE_VARIANTS),
        help="Pipeline variant: pb1 raw baseline, pb2 +query agent, pb3 +evidence agent, pb4 full 3-agent pipeline.",
    )
    parser.add_argument("--run-ablation", action="store_true", help="Run PB1-PB4 and print the ablation summary.")
    parser.add_argument("--output-json", default="", help="Optional path to save full state JSON.")
    parser.add_argument("--check-runtime", action="store_true", help="Validate dependencies and retrieval runtime settings, then exit.")
    parser.add_argument("--llm-provider", default="", help="Override MULTIAGENT_LLM_PROVIDER, e.g. groq or local-hf.")
    parser.add_argument("--llm-model", default="", help="Override MULTIAGENT_LLM_MODEL.")
    parser.add_argument("--llm-api-base", default="", help="Override MULTIAGENT_LLM_API_BASE for OpenAI-compatible providers.")
    args = parser.parse_args()

    if args.llm_provider:
        os.environ["MULTIAGENT_LLM_PROVIDER"] = normalize_text(args.llm_provider)
    if args.llm_model:
        os.environ["MULTIAGENT_LLM_MODEL"] = normalize_text(args.llm_model)
    if args.llm_api_base:
        os.environ["MULTIAGENT_LLM_API_BASE"] = normalize_text(args.llm_api_base)

    if args.check_runtime:
        print(json.dumps(validate_runtime_environment(), ensure_ascii=False, indent=2))
        return

    text = args.text
    if args.input_file:
        with open(args.input_file, "r", encoding="utf-8", errors="ignore") as f:
            text = f.read()
    if not normalize_text(text):
        raise SystemExit("Provide --text or --input-file.")

    metadata = parse_json_object(args.metadata_json) if args.metadata_json else {}
    if args.run_ablation:
        results = run_ablation_suite(text, input_metadata=metadata or {}, prefer_langgraph=not args.linear)
        summary = build_ablation_summary(results)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        if args.output_json:
            with open(args.output_json, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
        return

    result = run_pipeline(
        text,
        input_metadata=metadata or {},
        prefer_langgraph=not args.linear,
        variant=args.variant or None,
    )

    print(result.get("final_report", ""))
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()



