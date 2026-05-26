import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

from app.schemas.analysis import AgentStep, AgentTrace, AnalysisResult
from app.services.nlp_utils import clean_xml_and_extract


logger = logging.getLogger(__name__)

BACKEND_DIR = Path(__file__).resolve().parents[2]
PIPELINE_MA_DIR = BACKEND_DIR / "pipeline-ma"
if str(PIPELINE_MA_DIR) not in sys.path:
    sys.path.insert(0, str(PIPELINE_MA_DIR))

from patent_multiagent_langgraph import (  # noqa: E402
    PipelineConfig,
    canonical_doc_id as ma_canonical_doc_id,
    normalize_text as ma_normalize_text,
    run_analysis_stage,
    run_evidence_stage,
    run_output_stage,
    run_pipeline,
    run_query_understanding_stage,
    run_retrieval_stage,
    validate_runtime_environment,
)


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.getenv(name, str(default)))
    except Exception:
        return default


def _as_list(value: Any) -> List[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [ma_normalize_text(item) for item in value if ma_normalize_text(item)]
    text = ma_normalize_text(value)
    return [text] if text else []


def _best_date(item: Dict[str, Any]) -> str:
    return (
        ma_normalize_text(item.get("publication_date"))
        or ma_normalize_text(item.get("application_date"))
        or ma_normalize_text(item.get("priority_date"))
        or "N/A"
    )


class MultiAgentAnalyzerService:
    """Adapter that preserves the old FastAPI service contract over pipeline-ma."""

    def _config(self, variant: str, *, search: bool = False, selected_count: int = 0) -> PipelineConfig:
        default_candidate_top_k = 100 if search else max(20, selected_count)
        candidate_top_k = _env_int(
            "MULTIAGENT_SEARCH_TOP_K" if search else "MULTIAGENT_CANDIDATE_TOP_K",
            default_candidate_top_k,
        )
        evidence_top_docs = (
            min(selected_count, 5)
            if selected_count > 0
            else _env_int("MULTIAGENT_EVIDENCE_TOP_DOCS", 3)
        )
        max_iterations = _env_int(
            "MULTIAGENT_WEB_MAX_ITERATIONS",
            _env_int("MULTIAGENT_MAX_ITERATIONS", 1),
        )
        return PipelineConfig(
            pipeline_variant=variant,
            candidate_screen_top_k=max(candidate_top_k, selected_count),
            evidence_top_docs=max(1, min(evidence_top_docs, 5)),
            max_iterations=max(1, max_iterations),
        )

    def validate_runtime(self) -> Dict[str, Any]:
        return validate_runtime_environment(self._config("pb4"), require_langgraph=False)

    def _prepare_input(self, query: str) -> Tuple[str, Dict[str, Any], Dict[str, Any]]:
        extracted = clean_xml_and_extract(query)
        input_text = ma_normalize_text(extracted.get("full_text")) or ma_normalize_text(query)
        metadata = {
            key: value
            for key, value in extracted.items()
            if key != "full_text" and value not in (None, "")
        }
        if metadata.get("doc_id") == "N/A":
            metadata.pop("doc_id", None)
        metadata.setdefault(
            "input_type",
            "patent_document"
            if any(ma_normalize_text(metadata.get(key)) for key in ["title", "abstract", "claims"])
            else "idea",
        )
        metadata.setdefault("retrieval_text", input_text)
        return input_text, metadata, extracted

    def search_candidates(self, query: str) -> Dict[str, Any]:
        input_text, metadata, _ = self._prepare_input(query)
        config = self._config("pb2", search=True)
        topic = input_text[:90] + "..." if len(input_text) > 90 else input_text

        try:
            logger.info("--- Starting multi-agent search for: %s ---", topic)
            state = run_query_understanding_stage(
                input_text,
                input_metadata=metadata,
                config=config,
                variant="pb2",
            )
            state = run_retrieval_stage(
                state,
                config=config,
                variant="pb2",
                include_screening=True,
                include_report=False,
            )
            return {
                "candidates": self._search_response_from_state(state),
                "agent_trace": self._agent_trace_from_state(state),
            }
        except Exception as exc:
            logger.exception("Multi-agent search failed")
            return {"candidates": [], "agent_trace": None}

    def analyze(self, query: str) -> AnalysisResult:
        input_text, metadata, _ = self._prepare_input(query)
        config = self._config("pb4")
        try:
            logger.info("--- Starting full multi-agent analysis ---")
            state = run_pipeline(
                input_text,
                input_metadata=metadata,
                config=config,
                prefer_langgraph=False,
                variant="pb4",
            )
            return self._analysis_result_from_state(state, input_text)
        except Exception as exc:
            logger.exception("Full multi-agent analysis failed")
            return self._error_result(input_text, exc)

    def analyze_selected(
        self,
        query: str,
        selected_candidates: List[dict],
        on_progress: Callable[[str, str, AgentTrace | None], None] | None = None,
    ) -> AnalysisResult:
        input_text, metadata, _ = self._prepare_input(query)
        selected_candidates = selected_candidates or []
        config = self._config("pb4", selected_count=len(selected_candidates))

        def notify(agent: str, status: str, state: Dict[str, Any] | None = None) -> None:
            if on_progress:
                on_progress(agent, status, self._agent_trace_from_state(state) if state else None)

        try:
            logger.info("--- Starting multi-agent analysis for selected candidates ---")
            notify("query_understanding", "running")
            state = run_query_understanding_stage(
                input_text,
                input_metadata=metadata,
                config=config,
                variant="pb4",
            )
            notify("query_understanding", "completed", state)
            notify("retrieval", "running", state)
            candidate_rows, docs = self._selected_candidates_to_state(selected_candidates)
            audit_log = list(state.get("audit_log", []) or [])
            audit_log.append(
                {
                    "ts_ms": int(time.time() * 1000),
                    "node": "frontend_selected_candidates",
                    "message": "used user-selected candidates instead of retrieval output",
                    "num_selected": len(candidate_rows),
                }
            )
            state.update(
                {
                    "candidates": candidate_rows,
                    "screened_candidates": candidate_rows,
                    "candidate_docs": docs,
                    "retrieval_context": {
                        "backend": "frontend_selected",
                        "num_parent_docs": len(candidate_rows),
                    },
                    "audit_log": audit_log,
                }
            )
            notify("retrieval", "completed", state)
            notify("evidence_extraction", "running", state)
            state = run_evidence_stage(state, config=config, variant="pb4", include_report=False)
            notify("evidence_extraction", "completed", state)
            notify("prior_art_analysis", "running", state)
            state = run_analysis_stage(state, config=config, variant="pb4", include_coverage=True)
            notify("prior_art_analysis", "completed", state)
            notify("coverage_check", "completed", state)
            state = run_output_stage(state, config=config, variant="pb4")
            return self._analysis_result_from_state(state, input_text)
        except Exception as exc:
            logger.exception("Selected-candidate multi-agent analysis failed")
            notify("pipeline", "failed")
            return self._error_result(input_text, exc)

    def _search_response_from_state(self, state: Dict[str, Any]) -> List[dict]:
        docs = state.get("candidate_docs", {}) or {}
        rows = state.get("screened_candidates", []) or state.get("candidates", []) or []
        results: List[dict] = []

        for row in rows:
            if not isinstance(row, dict):
                continue
            doc_id = ma_normalize_text(row.get("doc_id") or row.get("patent_id") or row.get("id"))
            if not doc_id:
                continue
            doc = docs.get(doc_id) or docs.get(ma_canonical_doc_id(doc_id)) or {}
            merged = {**row, **doc}
            score = row.get("score", 0.0)
            try:
                score_value = float(score)
            except Exception:
                score_value = 0.0
            results.append(
                {
                    "id": doc_id,
                    "title": ma_normalize_text(merged.get("title")) or "N/A",
                    "title_vi": "",
                    "abstract": ma_normalize_text(merged.get("abstract")) or "N/A",
                    "claims": ma_normalize_text(merged.get("claims")) or "",
                    "description": ma_normalize_text(merged.get("description") or merged.get("retrieval_text")) or "",
                    "assignees": _as_list(merged.get("assignees")),
                    "inventors": _as_list(merged.get("inventors")),
                    "ipc_codes": _as_list(merged.get("ipc_codes")),
                    "citations": _as_list(merged.get("citations")),
                    "publication_date": ma_normalize_text(merged.get("publication_date")) or _best_date(merged),
                    "application_date": ma_normalize_text(merged.get("application_date")) or "N/A",
                    "priority_date": ma_normalize_text(merged.get("priority_date")) or "N/A",
                    "score": score_value,
                }
            )
        return results

    def _selected_candidates_to_state(self, selected_candidates: List[dict]) -> Tuple[List[Dict[str, Any]], Dict[str, Dict[str, Any]]]:
        rows: List[Dict[str, Any]] = []
        docs: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(selected_candidates, start=1):
            doc_id = ma_normalize_text(item.get("id") or item.get("doc_id") or item.get("patent_id"))
            if not doc_id:
                continue
            canonical = ma_canonical_doc_id(doc_id)
            doc = {
                "doc_id": doc_id,
                "canonical_doc_id": canonical,
                "title": ma_normalize_text(item.get("title")),
                "abstract": ma_normalize_text(item.get("abstract")),
                "claims": ma_normalize_text(item.get("claims")),
                "description": ma_normalize_text(item.get("description")),
                "retrieval_text": ma_normalize_text(
                    " ".join(
                        part
                        for part in [
                            item.get("title"),
                            item.get("abstract"),
                            item.get("claims"),
                            item.get("description"),
                        ]
                        if ma_normalize_text(part)
                    )
                ),
                "publication_date": ma_normalize_text(item.get("publication_date")),
                "application_date": ma_normalize_text(item.get("application_date")),
                "priority_date": ma_normalize_text(item.get("priority_date")),
                "ipc_codes": _as_list(item.get("ipc_codes")),
                "assignees": _as_list(item.get("assignees")),
                "inventors": _as_list(item.get("inventors")),
                "citations": _as_list(item.get("citations")),
            }
            try:
                score = float(item.get("score") or 0.0)
            except Exception:
                score = 0.0
            row = {
                "rank": rank,
                "screen_rank": rank,
                "doc_id": doc_id,
                "canonical_doc_id": canonical,
                "score": score,
                "title": doc["title"],
                "publication_date": doc["publication_date"],
                "application_date": doc["application_date"],
                "priority_date": doc["priority_date"],
                "ipc_codes": doc["ipc_codes"],
                "assignees": doc["assignees"],
                "inventors": doc["inventors"],
                "citations": doc["citations"],
            }
            rows.append(row)
            docs[doc_id] = doc
            docs.setdefault(canonical, doc)

        return rows, docs

    def _analysis_result_from_state(self, state: Dict[str, Any], input_text: str) -> AnalysisResult:
        analysis = state.get("analysis", {}) or {}
        ranked = analysis.get("ranked_prior_art", []) or []
        coverage = analysis.get("coverage", {}) or state.get("coverage", {}) or {}
        acceptance = analysis.get("acceptance_assessment", {}) or {}

        topic = input_text[:90] + "..." if len(input_text) > 90 else input_text
        summary = (
            ma_normalize_text(acceptance.get("why"))
            or ma_normalize_text(analysis.get("technical_problem_vi"))
            or f"Multi-agent prior-art analysis completed for: {topic}"
        )
        if len(summary) > 180:
            summary = summary[:177] + "..."

        key_points: List[str] = []
        for item in ranked[:5]:
            if not isinstance(item, dict):
                continue
            doc_id = ma_normalize_text(item.get("patent_id") or item.get("doc_id"))
            risk = ma_normalize_text(item.get("novelty_risk_vi") or item.get("novelty_risk"))
            overlap = ma_normalize_text(item.get("claim_overlap_summary") or item.get("limitations"))
            if doc_id:
                key_points.append(f"{doc_id}: {risk or 'unknown'} - {overlap}"[:240])
        if not key_points:
            metrics = state.get("proxy_metrics", {}) or {}
            key_points = [
                f"Screened candidates: {metrics.get('num_screened_candidates', 0)}",
                f"Evidence documents: {metrics.get('num_evidence_docs', 0)}",
            ]

        suggestions = _as_list(coverage.get("recommended_next_searches"))
        if not suggestions:
            suggestions = _as_list(acceptance.get("amendment_directions"))
        if not suggestions and ma_normalize_text(acceptance.get("recommended_strategy")):
            suggestions = [ma_normalize_text(acceptance.get("recommended_strategy"))]

        report = str(state.get("final_report") or analysis.get("final_report_markdown") or "").strip()
        if not report:
            report = "No final report was produced by the multi-agent pipeline."
        return AnalysisResult(
            summary=summary,
            key_points=key_points[:5],
            analysis=report,
            suggestions=suggestions[:3],
            agent_trace=self._agent_trace_from_state(state),
        )

    def _agent_trace_from_state(self, state: Dict[str, Any]) -> AgentTrace:
        steps: List[AgentStep] = []
        claim_elements = _as_list(state.get("claim_elements_vi")) or _as_list(state.get("claim_elements"))
        search_queries = [
            {
                "query_view": ma_normalize_text(item.get("query_view")),
                "text": ma_normalize_text(item.get("text")),
            }
            for item in (state.get("search_queries", []) or [])
            if isinstance(item, dict) and ma_normalize_text(item.get("text"))
        ]
        if state.get("technical_problem") or claim_elements or search_queries:
            steps.append(
                AgentStep(
                    agent="query_understanding",
                    label="Agent 1 - Hiểu truy vấn",
                    summary=f"Đã tách {len(claim_elements)} yếu tố claim và sinh {len(search_queries)} truy vấn tìm kiếm.",
                    details={
                        "technical_problem": ma_normalize_text(state.get("technical_problem_vi"))
                        or ma_normalize_text(state.get("technical_problem")),
                        "key_features": _as_list(state.get("key_features_vi")) or _as_list(state.get("key_features")),
                        "claim_elements": claim_elements,
                        "search_queries": search_queries,
                    },
                )
            )

        retrieval = state.get("retrieval_context", {}) or {}
        candidates = state.get("screened_candidates", []) or state.get("candidates", []) or []
        if retrieval or candidates:
            steps.append(
                AgentStep(
                    agent="retrieval",
                    label="Retrieval - Tìm prior art",
                    summary=f"Đã đưa ra {len(candidates)} tài liệu ứng viên.",
                    details={
                        "backend": ma_normalize_text(retrieval.get("backend")),
                        "embedding_backend": ma_normalize_text(retrieval.get("embedding_backend")),
                        "num_parent_docs": retrieval.get("num_parent_docs", len(candidates)),
                        "num_screened_candidates": len(candidates),
                        "top_document_ids": [
                            ma_normalize_text(item.get("doc_id"))
                            for item in candidates[:5]
                            if isinstance(item, dict) and ma_normalize_text(item.get("doc_id"))
                        ],
                    },
                )
            )

        evidence = state.get("evidence", []) or []
        if evidence:
            steps.append(
                AgentStep(
                    agent="evidence_extraction",
                    label="Agent 2 - Trích xuất bằng chứng",
                    summary=f"Đã đối chiếu bằng chứng trên {len(evidence)} tài liệu.",
                    details={"evidence": evidence},
                )
            )

        analysis = state.get("analysis", {}) or {}
        ranked = analysis.get("ranked_prior_art", []) or []
        if analysis:
            top_risk = ma_normalize_text(ranked[0].get("novelty_risk_vi") or ranked[0].get("novelty_risk")) if ranked else "N/A"
            steps.append(
                AgentStep(
                    agent="prior_art_analysis",
                    label="Agent 3 - Đánh giá prior art",
                    summary=f"Đã xếp hạng {len(ranked)} tài liệu; rủi ro mạnh nhất: {top_risk}.",
                    details={
                        "ranked_prior_art": ranked,
                        "acceptance_assessment": analysis.get("acceptance_assessment", {}),
                    },
                )
            )

        coverage = state.get("coverage", {}) or analysis.get("coverage", {}) or {}
        if coverage:
            sufficient = bool(coverage.get("is_sufficient"))
            steps.append(
                AgentStep(
                    agent="coverage_check",
                    label="Kiểm tra độ phủ",
                    summary="Bằng chứng đủ để kết luận." if sufficient else "Bằng chứng chưa phủ đầy đủ các yếu tố claim.",
                    details=coverage,
                )
            )

        return AgentTrace(
            variant=ma_normalize_text(state.get("pipeline_variant")) or "pb4",
            steps=steps,
            metrics=state.get("proxy_metrics", {}) or {},
        )

    def _error_result(self, input_text: str, exc: Exception) -> AnalysisResult:
        topic = input_text[:90] + "..." if len(input_text) > 90 else input_text
        return AnalysisResult(
            summary=f"Multi-agent pipeline failed: {topic}",
            key_points=[str(exc)],
            analysis=(
                "### Multi-agent pipeline error\n\n"
                f"{type(exc).__name__}: {exc}\n\n"
                "Check ES_CLOUD_ID, ES_API_KEY, GROQ_API_KEY, and the embedding backend settings."
            ),
            suggestions=[
                "Verify backend/.env multi-agent settings.",
                "Run the runtime validation before retrying.",
            ],
        )


multiagent_service = MultiAgentAnalyzerService()
