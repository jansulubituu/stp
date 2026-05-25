import json
import logging
import re
from typing import Any

from google import genai
from google.genai import types

from app.core.config import settings

logger = logging.getLogger(__name__)

IPC_PATTERN = re.compile(r"\b[A-HY]\d{2}[A-Z]\s*\d{1,4}/\d{1,6}\b", re.IGNORECASE)
YEAR_CUTOFF_PATTERN = re.compile(
    r"(?:before|prior to|trước|truoc|trước năm|truoc nam)\s+(?:năm\s*)?(19\d{2}|20\d{2})",
    re.IGNORECASE,
)


def _clean_text(text: str) -> str:
    return " ".join(str(text or "").split()).strip()


def _extract_json_object(text: str) -> dict[str, Any]:
    if not text:
        return {}
    text = re.sub(r"```(?:json)?", "", text).replace("```", "").strip()
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if not match:
        return {}
    try:
        value = json.loads(match.group(0))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _rule_hints(query: str) -> dict[str, Any]:
    ipc_codes = [m.group(0).upper().replace(" ", "") for m in IPC_PATTERN.finditer(query)]
    cutoff = ""
    year_match = YEAR_CUTOFF_PATTERN.search(query)
    if year_match:
        cutoff = f"{year_match.group(1)}0101"
    return {
        "ipc_codes": sorted(set(ipc_codes)),
        "date_cutoff": cutoff,
    }


def _fallback_normalized_query(query: str, hints: dict[str, Any] | None = None) -> dict[str, Any]:
    clean_query = _clean_text(query)
    hints = hints or _rule_hints(clean_query)
    return {
        "intent": "prior_art_search",
        "language": "unknown",
        "technical_problem": clean_query,
        "key_features": [],
        "must_have_features": [],
        "optional_features": [],
        "expanded_terms": [],
        "ipc_codes": hints.get("ipc_codes", []),
        "assignees": [],
        "date_cutoff": hints.get("date_cutoff", ""),
        "embedding_text": (
            "[TECHNICAL QUERY]\n"
            f"{clean_query}\n\n"
            "[SEARCH INTENT]\n"
            "Find prior art and similar patent documents based on technical similarity."
        ),
        "confidence": "low",
        "normalizer": "fallback",
    }


def _sanitize_normalized_query(data: dict[str, Any], original_query: str, hints: dict[str, Any]) -> dict[str, Any]:
    fallback = _fallback_normalized_query(original_query, hints)

    def list_of_strings(name: str) -> list[str]:
        value = data.get(name, [])
        if not isinstance(value, list):
            return []
        return [_clean_text(item) for item in value if _clean_text(item)]

    ipc_codes = hints.get("ipc_codes", [])
    date_cutoff = hints.get("date_cutoff", "")
    embedding_text = _clean_text(data.get("embedding_text", ""))
    if not embedding_text:
        parts = [
            "[TECHNICAL PROBLEM]",
            _clean_text(data.get("technical_problem", "")),
            "[KEY FEATURES]",
            "; ".join(list_of_strings("key_features")),
            "[EXPANDED TERMS]",
            "; ".join(list_of_strings("expanded_terms")),
        ]
        embedding_text = "\n".join(part for part in parts if part)

    normalized = {
        "intent": _clean_text(data.get("intent", "")) or fallback["intent"],
        "language": _clean_text(data.get("language", "")) or fallback["language"],
        "technical_problem": _clean_text(data.get("technical_problem", "")) or fallback["technical_problem"],
        "key_features": list_of_strings("key_features"),
        "must_have_features": list_of_strings("must_have_features"),
        "optional_features": list_of_strings("optional_features"),
        "expanded_terms": list_of_strings("expanded_terms"),
        "ipc_codes": ipc_codes,
        "assignees": [],
        "date_cutoff": date_cutoff,
        "embedding_text": embedding_text or fallback["embedding_text"],
        "confidence": _clean_text(data.get("confidence", "")) or "medium",
        "normalizer": "gemini",
    }

    if normalized["confidence"] not in {"high", "medium", "low"}:
        normalized["confidence"] = "medium"
    return normalized


def normalize_natural_query(query: str) -> dict[str, Any]:
    clean_query = _clean_text(query)
    hints = _rule_hints(clean_query)
    if not clean_query:
        return _fallback_normalized_query(clean_query, hints)

    if not settings.gemini_api_key:
        logger.info("GEMINI_API_KEY is empty; using fallback query normalization.")
        return _fallback_normalized_query(clean_query, hints)

    prompt = f"""
You normalize natural-language patent search queries for vector retrieval.

Return JSON only with this schema:
{{
  "intent": "prior_art_search",
  "language": "vi|en|unknown",
  "technical_problem": "",
  "key_features": [],
  "must_have_features": [],
  "optional_features": [],
  "expanded_terms": [],
  "ipc_codes": [],
  "assignees": [],
  "date_cutoff": "",
  "embedding_text": "",
  "confidence": "high|medium|low"
}}

Rules:
- Preserve the user's technical meaning.
- If the query is Vietnamese, rewrite technical content in English patent-search language.
- Do not invent patent IDs, assignees, IPC codes, or dates.
- Only include IPC/date/assignee if explicitly present in the original query or rule hints.
- Keep embedding_text concise, patent-like, and under 700 words.
- Focus on technical features, structure, mechanism, material, and use case.

Original query:
{clean_query}

Rule-extracted hints:
IPC codes: {hints.get("ipc_codes", [])}
Date cutoff: {hints.get("date_cutoff", "")}
""".strip()

    try:
        client = genai.Client(api_key=settings.gemini_api_key)
        config = types.GenerateContentConfig(
            thinking_config=types.ThinkingConfig(thinking_level="MINIMAL"),
        )
        response = client.models.generate_content(
            model=settings.gemini_model,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=prompt)])],
            config=config,
        )

        parsed = _extract_json_object(response.text or "")
        if not parsed:
            logger.warning("Gemini query normalizer returned non-JSON output; using fallback.")
            return _fallback_normalized_query(clean_query, hints)
        return _sanitize_normalized_query(parsed, clean_query, hints)
    except Exception as exc:
        logger.warning("Gemini query normalization failed: %s", exc)
        return _fallback_normalized_query(clean_query, hints)
