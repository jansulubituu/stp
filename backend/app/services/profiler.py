from typing import Any, Dict, List
from app.services.nlp_utils import (
    normalize_text,
    normalize_list,
    word_count,
    normalize_date_yyyymmdd,
    normalize_ipc_list,
    parse_ipc_code,
    clean_party_name
)

# =========================================================
# QUERY PROFILER & QUALITY SCORING
# =========================================================

def pick_prior_art_cutoff(row: Dict[str, Any]) -> str:
    """
    Xác định ngày cắt (Cutoff Date) cho Prior Art.
    Ưu tiên: priority_date > application_date > publication_date.
    """
    priority_date = normalize_date_yyyymmdd(row.get("priority_date"))
    application_date = normalize_date_yyyymmdd(row.get("application_date"))
    publication_date = normalize_date_yyyymmdd(row.get("publication_date"))

    if priority_date:
        return priority_date
    if application_date:
        return application_date
    if publication_date:
        return publication_date

    return ""


def classify_query_strength(stats: Dict[str, int]) -> str:
    """
    Đánh giá chất lượng và độ dài của văn bản sáng chế đầu vào.
    Dựa trên thuật toán chuẩn từ Cell 3 của Notebook.
    """
    title_wc = stats.get("title_word_count", 0) or 0
    abstract_wc = stats.get("abstract_word_count", 0) or 0
    claims_wc = stats.get("claims_word_count", 0) or 0
    desc_wc = stats.get("description_word_count", 0) or 0

    # "strong": Title có + Abstract >= 80 từ + Claims >= 300 từ
    if title_wc > 0 and abstract_wc >= 80 and claims_wc >= 300:
        return "strong"

    # "medium": Title có + Abstract >= 40 từ + Claims >= 100 từ
    if title_wc > 0 and abstract_wc >= 40 and claims_wc >= 100:
        return "medium"

    # "weak": Bất kỳ trường nào có nội dung nhưng quá ngắn
    if title_wc > 0 or abstract_wc > 0 or claims_wc > 0 or desc_wc > 0:
        return "weak"

    return "empty"


def analyze_document_profile(doc: Dict[str, Any]) -> Dict[str, Any]:
    """
    Phân tích chi tiết tài liệu sáng chế đầu vào và đo lường các trường dữ liệu.
    Đây chính là lõi xử lý "trước bước retrieval" (bước số 2 & 3).
    """
    title = normalize_text(doc.get("title"))
    abstract = normalize_text(doc.get("abstract"))
    claims = normalize_text(doc.get("claims"))
    description = normalize_text(doc.get("description"))

    # 1. Đếm từ
    stats = {
        "title_word_count": word_count(title),
        "abstract_word_count": word_count(abstract),
        "claims_word_count": word_count(claims),
        "description_word_count": word_count(description),
    }

    # 2. Phân loại mức độ mạnh/yếu
    query_strength = classify_query_strength(stats)

    # 3. Làm sạch Metadata
    ipc_raw = normalize_list(doc.get("ipc_codes"))
    ipc_norm = normalize_ipc_list(ipc_raw)
    main_ipc = ipc_norm[0] if ipc_norm else ""
    main_ipc_parts = parse_ipc_code(main_ipc) if main_ipc else {}

    assignees_raw = normalize_list(doc.get("assignees"))
    assignees_clean = [clean_party_name(a) for a in assignees_raw if clean_party_name(a)]

    inventors_raw = normalize_list(doc.get("inventors"))
    inventors_clean = [clean_party_name(i) for i in inventors_raw if clean_party_name(i)]

    # 4. Lọc ngày bảo hộ
    cutoff_date = pick_prior_art_cutoff(doc)

    # 5. Xây dựng Boost & Filter Config Json cho Retrieval Planner
    # Đây là phần chuẩn bị config trước khi search
    filter_config = {}
    if cutoff_date:
        filter_config["before_date"] = cutoff_date

    boost_config = {
        "has_ipc_boost": bool(ipc_norm),
        "has_assignee_boost": bool(assignees_clean),
        "query_strength": query_strength
    }

    return {
        "query_strength": query_strength,
        "word_counts": stats,
        "prior_art_cutoff_date": cutoff_date,
        
        "ipc_analysis": {
            "codes_norm": ipc_norm,
            "main_ipc": main_ipc,
            "section": main_ipc_parts.get("ipc_section", ""),
            "class": main_ipc_parts.get("ipc_class", ""),
            "subclass": main_ipc_parts.get("ipc_subclass", ""),
        },
        
        "party_analysis": {
            "assignees": assignees_clean,
            "inventors": inventors_clean,
        },
        
        "retrieval_planner": {
            "filter_config": filter_config,
            "boost_config": boost_config,
            "use_bm25": True,
            "use_knn": query_strength != "weak"  # Nếu query quá yếu/ngắn, hạn chế phụ thuộc vector giống Notebook
        }
    }
