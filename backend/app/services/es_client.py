import os
import re
import logging
from typing import Dict, Any, List, Tuple
from elasticsearch import Elasticsearch

from app.core.config import settings
from app.services.nlp_utils import normalize_text

logger = logging.getLogger(__name__)

# =========================================================
# CONFIGURATIONS (FAITHFUL TO NOTEBOOK CELL 7)
# =========================================================

BM25_FIELDS = [
    "title^3.0",
    "abstract^2.0",
    "retrieval_text^2.0",
    "claims^1.2",
    "description^0.3",
]

BM25_TIE_BREAKER = 0.1
RETRIEVAL_POOL_TOP_K = 300

# Stopwords phục vụ lọc terms BM25 chuẩn notebook
BM25_STOPWORDS = {
    "the", "and", "or", "for", "with", "from", "that", "this", "these", "those", "into", "onto", "wherein",
    "thereof", "therein", "thereby", "therefor", "said", "which", "when", "then", "than", "such", "have", "has",
    "having", "comprising", "comprises", "comprised", "including", "includes", "included", "using", "used", "use",
    "method", "apparatus", "device", "system", "means", "portion", "member", "plurality", "first", "second", "third",
    "one", "more", "least", "between", "within", "without", "about", "above", "below", "each", "other", "same",
    "can", "may", "are", "was", "were", "been", "being", "their", "its", "his", "her", "our", "your", "not",
}

# =========================================================
# ELASTIC CLOUD CLIENT INITIALIZATION
# =========================================================

def get_es_client() -> Elasticsearch:
    """
    Kết nối tới cụm Elastic Cloud sử dụng Cloud ID và API Key từ Settings (.env).
    """
    if not settings.es_cloud_id or not settings.es_api_key:
        raise ValueError(
            "Lỗi cấu hình: ES_CLOUD_ID và ES_API_KEY đang bị trống trong file .env! "
            "Vui lòng kiểm tra lại để kết nối Cloud."
        )
        
    return Elasticsearch(
        cloud_id=settings.es_cloud_id,
        api_key=settings.es_api_key,
        request_timeout=120
    )


# =========================================================
# ES FILTERS & BOOST BUILDERS
# =========================================================

def build_es_filters(filter_config: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Tạo bộ lọc ngày bảo hộ (Prior art cutoff) trước khi search.
    """
    filters = []
    
    before_date = filter_config.get("before_date")
    if before_date:
        filters.append({
            "range": {
                "publication_date": {
                    "lt": before_date,
                    "format": "yyyyMMdd||yyyy-MM-dd"
                }
            }
        })
        
    return filters


def bm25_tokenize(text: str) -> List[str]:
    """Tách từ khóa làm sạch để lọc cụm từ có ý nghĩa giống notebook."""
    return [
        tok.lower()
        for tok in re.findall(r"[A-Za-z][A-Za-z0-9\-]{2,}", str(text or ""))
        if tok.lower() not in BM25_STOPWORDS and not tok.isdigit()
    ]


# =========================================================
# SPARSE SEARCH (BM25) EXECUTOR
# =========================================================

def execute_bm25_search(
    es: Elasticsearch,
    query_text: str,
    top_k: int = 100,
    filter_config: Dict[str, Any] = None
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Thực thi truy vấn Sparse Search (BM25) multi_match chuẩn xác lên Elastic Cloud.
    Trả về danh sách tuple: (doc_id, score, source_metadata)
    """
    if not query_text:
        return []
        
    filters = build_es_filters(filter_config or {})
    
    # Thiết lập truy vấn multi_match theo trọng số chuẩn của Notebook
    bool_query = {
        "must": [
            {
                "multi_match": {
                    "query": query_text,
                    "fields": BM25_FIELDS,
                    "type": "most_fields",
                    "tie_breaker": BM25_TIE_BREAKER,
                    "operator": "or"
                }
            }
        ]
    }
    
    if filters:
        bool_query["filter"] = filters

    body = {
        "query": {
            "bool": bool_query
        },
        "size": max(top_k, RETRIEVAL_POOL_TOP_K),
        "_source": [
            "doc_id", "canonical_doc_id", "title", "abstract", "claims", "description",
            "ipc_codes", "assignees", "inventors", "citations", 
            "date", "publication_date", "application_date", "priority_date"
        ]
    }

    try:
        resp = es.search(index=settings.bm25_index, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        
        results = []
        seen_docs = set()
        
        for hit in hits:
            src = hit.get("_source", {}) or {}
            doc_id = src.get("doc_id") or hit.get("_id")
            
            if doc_id in seen_docs:
                continue
                
            seen_docs.add(doc_id)
            score = float(hit.get("_score") or 0.0)
            
            # Lưu metadata kèm theo để phục vụ bước đồ thị Graph tiếp theo
            results.append((doc_id, score, src))
            
            if len(results) >= top_k:
                break
                
        return results
        
    except Exception as e:
        logger.error(f"Lỗi khi truy vấn BM25 từ Cloud: {e}")
        return []


def fetch_document_metadata_by_ids(es: Elasticsearch, doc_ids: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Lấy nhanh thông tin nguồn (metadata) của danh sách document ids để build đồ thị động.
    """
    if not doc_ids:
        return {}
        
    try:
        resp = es.mget(
            index=settings.bm25_index,
            ids=[str(d) for d in doc_ids],
            _source=[
                "doc_id", "canonical_doc_id", "title", "abstract", "claims", "description",
                "ipc_codes", "assignees", "inventors", "citations", 
                "date", "publication_date", "application_date", "priority_date"
            ]
        )
        
        sources = {}
        for doc in resp.get("docs", []):
            if doc.get("found"):
                sources[str(doc.get("_id"))] = doc.get("_source", {}) or {}
        return sources
        
    except Exception as e:
        logger.error(f"Lỗi khi MGet documents từ Cloud: {e}")
        return {}


# =========================================================
# DENSE SEARCH (kNN Vector) EXECUTOR
# =========================================================

def execute_knn_search(
    es: Elasticsearch,
    query_vector: List[float],
    top_k: int = 100,
    filter_config: Dict[str, Any] = None
) -> List[Tuple[str, float, Dict[str, Any]]]:
    """
    Thực thi Dense Search (kNN Vector) lên Elastic Cloud trên KNN index.
    Tổng hợp các chunk-level hits về parent document ID.
    Trả về: [(doc_id, score, source_metadata), ...]
    """
    if not query_vector:
        return []
        
    filters = build_es_filters(filter_config or {})
    
    # Notebook concept: kNN chunk multiplier (lấy nhiều chunk hơn để gộp lại)
    fetch_size = max(top_k * 8, 800) 
    
    body = {
        "knn": {
            "field": "content_vector",
            "query_vector": query_vector,
            "k": fetch_size,
            "num_candidates": min(fetch_size * 2, 2000),
            "boost": 1.0
        },
        "size": fetch_size,
        "_source": ["parent_doc_id", "doc_id", "canonical_doc_id", "priority_date", "application_date", "publication_date"]
    }
    
    if filters:
        body["knn"]["filter"] = filters

    try:
        logger.info(f"Executing ES kNN Search on index: {settings.knn_index}...")
        resp = es.search(index=settings.knn_index, body=body)
        hits = resp.get("hits", {}).get("hits", [])
        
        doc_best = {}
        for hit in hits:
            src = hit.get("_source", {}) or {}
            doc_id = src.get("parent_doc_id") or src.get("doc_id") or hit.get("_id")
            if not doc_id:
                continue
            
            score = float(hit.get("_score") or 0.0)
            # Lấy Max score như trong Notebook (KNN_SCORE_AGG='max')
            if doc_id not in doc_best or score > doc_best[doc_id]["score"]:
                doc_best[doc_id] = {
                    "score": score,
                    "src": src 
                }
                
        # Sắp xếp và lấy top_k
        ranked = sorted(doc_best.items(), key=lambda item: item[1]["score"], reverse=True)[:top_k]
        
        # Giai đoạn nạp metadata đầy đủ
        top_doc_ids = [str(d) for d, _ in ranked]
        if not top_doc_ids:
            return []
            
        full_sources = fetch_document_metadata_by_ids(es, top_doc_ids)
        
        results = []
        for doc_id, payload in ranked:
            doc_str = str(doc_id)
            final_src = full_sources.get(doc_str) or payload["src"]
            if "doc_id" not in final_src:
                final_src["doc_id"] = doc_str
            results.append((doc_str, payload["score"], final_src))
            
        logger.info(f"kNN search returned {len(results)} aggregated documents")
        return results
        
    except Exception as e:
        logger.error(f"Lỗi khi truy vấn kNN từ Cloud: {e}")
        return []

