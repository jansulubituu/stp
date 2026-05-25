import httpx
import logging
from typing import List
from app.core.config import settings

logger = logging.getLogger(__name__)

JINA_EMBEDDING_URL = "https://api.jina.ai/v1/embeddings"

def get_jina_query_embedding(text: str) -> List[float]:
    """
    Calls the Jina AI Embedding API to get the query vector for a text string.
    Uses 'jina-embeddings-v3' with 'retrieval.query' task to match notebook settings.
    """
    if not settings.jina_api_key:
        logger.warning("JINA_API_KEY is empty! Skipping kNN Vector Search.")
        return []

    if not text or not text.strip():
        return []

    # Clean whitespaces to reduce tokens
    clean_text = " ".join(text.split()).strip()
    
    # Limit length to safe length to prevent API issues
    # Jina 3 supports 8k, so we truncate to reasonable ~15,000 characters
    safe_text = clean_text[:15000]

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {settings.jina_api_key}"
    }

    payload = {
        "model": "jina-embeddings-v3",
        "task": "retrieval.query",
        "dimensions": 1024,
        "embedding_type": "float",
        "input": [safe_text]
    }

    try:
        logger.info("Calling Jina Embeddings API...")
        with httpx.Client(timeout=30.0) as client:
            response = client.post(JINA_EMBEDDING_URL, json=payload, headers=headers)
            response.raise_for_status()
            data = response.json()
            
            embedding = data.get("data", [{}])[0].get("embedding", [])
            if embedding:
                logger.info(f"Successfully received embedding vector of size {len(embedding)}")
            return embedding

    except Exception as e:
        logger.error(f"Error calling Jina Embeddings API: {e}")
        return []
