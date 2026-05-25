import logging
from collections import OrderedDict

from openai import OpenAI

from app.core.config import settings

logger = logging.getLogger(__name__)

_TITLE_CACHE: OrderedDict[str, str] = OrderedDict()
_MAX_CACHE_SIZE = 2048
_client: OpenAI | None = None


def _cache_get(title: str) -> str | None:
    value = _TITLE_CACHE.get(title)
    if value is not None:
        _TITLE_CACHE.move_to_end(title)
    return value


def _cache_set(title: str, translation: str) -> None:
    _TITLE_CACHE[title] = translation
    _TITLE_CACHE.move_to_end(title)
    while len(_TITLE_CACHE) > _MAX_CACHE_SIZE:
        _TITLE_CACHE.popitem(last=False)


def _get_client() -> OpenAI | None:
    global _client
    if _client is None and settings.groq_api_key:
        _client = OpenAI(
            api_key=settings.groq_api_key,
            base_url="https://api.groq.com/openai/v1",
            timeout=20,
        )
    return _client


def translate_title_to_vi(title: str) -> str:
    title = " ".join(str(title or "").split()).strip()
    if not title or not settings.translate_titles:
        return ""

    cached = _cache_get(title)
    if cached is not None:
        return cached

    client = _get_client()
    if client is None:
        return ""

    try:
        response = client.chat.completions.create(
            model=settings.groq_model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You translate patent titles from English, French, or German into Vietnamese. "
                        "Return only the translated title. Keep technical wording concise and accurate."
                    ),
                },
                {"role": "user", "content": f"Translate this patent title to Vietnamese:\n{title}"},
            ],
            temperature=0,
            max_tokens=96,
        )
        translated = " ".join((response.choices[0].message.content or "").split()).strip()
        if translated:
            _cache_set(title, translated)
        return translated
    except Exception as exc:
        logger.warning("Groq title translation failed: %s", exc)
        return ""


def translate_titles_to_vi(titles: list[str]) -> dict[str, str]:
    return {title: translate_title_to_vi(title) for title in titles if str(title or "").strip()}
