import re
from typing import Any, Dict, List, Optional

# =========================================================
# CORE TEXT UTILS
# =========================================================

def normalize_text(x: Any) -> str:
    """Chuẩn hóa văn bản thô, loại bỏ khoảng trắng thừa."""
    if x is None:
        return ""
    # Dùng str() để xử lý cả số hoặc kiểu dữ liệu khác giống Notebook
    s = str(x)
    if s.lower() in ("nan", "none", "<na>"):
        return ""
    return re.sub(r"\s+", " ", s).strip()


def normalize_list(x: Any) -> List[str]:
    """Chuẩn hóa danh sách các chuỗi từ input hỗn hợp."""
    if x is None:
        return []

    if isinstance(x, str):
        # Nếu là chuỗi JSON list hoặc chuỗi đơn
        s = normalize_text(x)
        if s.startswith("[") and s.endswith("]"):
            # Cố gắng bóc các phần tử nếu stringified list
            items = re.findall(r"['\"](.*?)['\"]", s)
            if items:
                x = items
            else:
                x = [s]
        else:
            return [s] if s else []

    try:
        # Hỗ trợ numpy ndarray, pandas Series nếu có
        if hasattr(x, "tolist"):
            x = x.tolist()
    except Exception:
        pass

    if isinstance(x, (list, tuple)):
        out = []
        seen = set()
        for item in x:
            s = normalize_text(item)
            if s and s not in seen:
                seen.add(s)
                out.append(s)
        return out

    s = normalize_text(x)
    return [s] if s else []


def safe_join(parts: List[str]) -> str:
    """Gộp các mảng văn bản an toàn."""
    parts = [normalize_text(p) for p in parts if normalize_text(p)]
    return " ".join(parts).strip()


def truncate_words(text: Any, max_words: Optional[int]) -> str:
    """Cắt ngắn văn bản theo số lượng từ tối đa."""
    text = normalize_text(text)
    if not text:
        return ""

    if max_words is None or int(max_words) <= 0:
        return text

    words = text.split()
    if len(words) <= max_words:
        return text

    return " ".join(words[:max_words])


def word_count(text: Any) -> int:
    """Đếm số từ của chuỗi."""
    text = normalize_text(text)
    if not text:
        return 0
    return len(text.split())


# =========================================================
# PATENT XML DETECTOR & EXTRACTOR
# =========================================================

def clean_xml_and_extract(raw_text: str) -> Dict[str, Any]:
    """
    Tự động phát hiện nếu đầu vào là XML và bóc tách nội dung các trường bằng lxml.
    Bám sát logic của parse_pac_topic_xml trong Notebook.
    Nếu không phải XML, trả về chuỗi nguyên bản.
    """
    s = normalize_text(raw_text)
    default_res = {
        "doc_id": "N/A", "title": "", "abstract": "", "claims": s, "description": "", 
        "full_text": s, "ipc_codes": [], "assignees": [], "inventors": [], "citations": [],
        "publication_date": "", "application_date": "", "priority_date": "",
        "country": "", "lang": "", "kind": ""
    }

    if not ("<" in s and ">" in s):
        return default_res

    try:
        from lxml import etree
        parser = etree.XMLParser(recover=True, huge_tree=True)
        root = etree.fromstring(raw_text.encode('utf-8', errors='ignore'), parser=parser)
        
        def local_name(tag):
            if tag is None or not isinstance(tag, str): return ""
            return tag.split("}", 1)[1] if "}" in tag else tag

        def iter_real_elements(root):
            for elem in root.iter():
                if isinstance(getattr(elem, "tag", None), str):
                    yield elem

        def get_text(elem):
            try:
                parts = [normalize_text(c) for c in elem.itertext() if normalize_text(c)]
                return " ".join(parts).strip()
            except Exception:
                return ""

        def extract_date(container):
            for elem in container.iter():
                if local_name(elem.tag).lower() == "date":
                    text = get_text(elem)
                    m = re.search(r"\b(\d{8})\b", text)
                    if m: return m.group(1)
            return ""

        title_tags = {"invention-title", "title", "b540", "ti"}
        abstract_tags = {"abstract", "abst", "sdoab"}
        claims_tags = {"claims", "claim", "clms"}
        description_tags = {"description", "desc", "detdesc", "txt"}
        ipc_tags = {"classification-ipc", "ipc", "main-classification", "further-classification", "classification-ipcr", "classifications-ipcr", "b511", "b512", "b510"}
        assignee_tags = {"applicant", "assignee", "applicants", "assignees", "orgname", "b711", "b721", "b731"}
        inventor_tags = {"inventor", "inventors", "b721"}
        citation_tags = {"citation", "patcit", "nplcit"}

        doc_id = root.attrib.get("id", "")
        country = root.attrib.get("country", "")
        lang = root.attrib.get("lang", "")
        kind = root.attrib.get("kind", "")
        
        # Try to get date-publ from root
        pub_date = root.attrib.get("date-publ", "")
        
        title, app_date, pri_date = "", "", ""
        abstract_parts, claims_parts, description_parts = [], [], []
        ipc_codes, assignees, inventors, citations = [], [], [], []
        
        for elem in iter_real_elements(root):
            tag_lower = local_name(elem.tag).lower()
            if not tag_lower: continue
            
            if not country and tag_lower == "country":
                text = get_text(elem)
                if text: country = text
            if not lang and tag_lower == "lang":
                text = get_text(elem)
                if text: lang = text
            if not kind and tag_lower == "kind":
                text = get_text(elem)
                if text: kind = text
                
            if tag_lower in title_tags and not title:
                text = get_text(elem)
                if text: title = text
                
            if tag_lower in abstract_tags:
                text = get_text(elem)
                if text: abstract_parts.append(text)
                
            if tag_lower in claims_tags:
                text = get_text(elem)
                if text: claims_parts.append(text)
                
            if tag_lower in description_tags:
                text = get_text(elem)
                if text: description_parts.append(text)
                
            if (tag_lower == "publication-reference" or tag_lower == "b140") and not pub_date:
                pub_date = extract_date(elem)
                
            if (tag_lower == "application-reference" or tag_lower == "b220") and not app_date:
                app_date = extract_date(elem)
                
            if (tag_lower == "priority-claim" or tag_lower == "b320") and not pri_date:
                pri_date = extract_date(elem)
                
            if tag_lower in ipc_tags:
                text = get_text(elem)
                if text:
                    for p in re.split(r"[;\n\r]+", text):
                        if p.strip(): ipc_codes.append(p.strip())
                        
            if tag_lower in assignee_tags:
                text = get_text(elem)
                if text and len(text) >= 2: assignees.append(text)
                
            if tag_lower in inventor_tags:
                text = get_text(elem)
                if text and len(text) >= 2: inventors.append(text)
                
            if tag_lower in citation_tags:
                text = get_text(elem)
                if text and len(text) >= 2: citations.append(text)

        if not doc_id:
            doc_id_match = re.search(r'<patent-document[^>]*id="([^"]+)"', raw_text, re.IGNORECASE)
            doc_id = doc_id_match.group(1) if doc_id_match else "N/A"

        abstract = " ".join(abstract_parts).strip()
        claims = " ".join(claims_parts).strip()
        description = " ".join(description_parts).strip()
        
        full_text = " ".join([p for p in [title, abstract, claims, description] if p]).strip()
        if not full_text:
            full_text = re.sub(r"<[^>]+>", " ", raw_text)
            full_text = normalize_text(full_text)
            return {"doc_id": doc_id, "title": "", "abstract": "", "claims": full_text, "description": "", 
                    "full_text": full_text, "ipc_codes": [], "assignees": [], "inventors": [], "citations": [],
                    "publication_date": "", "application_date": "", "priority_date": "",
                    "country": "", "lang": "", "kind": ""}

        def unique_list(seq):
            seen = set()
            return [x for x in seq if x not in seen and not seen.add(x)]

        return {
            "doc_id": doc_id,
            "title": title,
            "abstract": abstract,
            "claims": claims,
            "description": description,
            "full_text": full_text,
            "ipc_codes": unique_list(ipc_codes),
            "assignees": unique_list(assignees),
            "inventors": unique_list(inventors),
            "citations": unique_list(citations),
            "publication_date": pub_date,
            "application_date": app_date,
            "priority_date": pri_date,
            "country": country,
            "lang": lang,
            "kind": kind
        }

    except Exception:
        # Fallback to simple regex if lxml fails
        doc_id_match = re.search(r'<patent-document[^>]*id="([^"]+)"', raw_text, re.IGNORECASE)
        doc_id = doc_id_match.group(1) if doc_id_match else "N/A"

        title_match = re.search(r"<B542[^>]*>(.*?)</B542>", raw_text, re.DOTALL | re.IGNORECASE)
        if not title_match:
            title_match = re.search(r"<title[^>]*>(.*?)</title>", raw_text, re.DOTALL | re.IGNORECASE)
        title = normalize_text(title_match.group(1)) if title_match else ""

        abstract_match = re.search(r"<abstract[^>]*>(.*?)</abstract>", raw_text, re.DOTALL | re.IGNORECASE)
        abstract = normalize_text(re.sub(r"<[^>]+>", " ", abstract_match.group(1))) if abstract_match else ""

        claims_match = re.search(r"<claims[^>]*>(.*?)</claims>", raw_text, re.DOTALL | re.IGNORECASE)
        claims = normalize_text(re.sub(r"<[^>]+>", " ", claims_match.group(1))) if claims_match else ""

        full_text = " ".join([p for p in [title, abstract, claims] if p]).strip()
        if not full_text:
            full_text = normalize_text(re.sub(r"<[^>]+>", " ", raw_text))
            
        res = dict(default_res)
        res.update({"doc_id": doc_id, "title": title, "abstract": abstract, "claims": claims, "full_text": full_text or s})
        return res




# =========================================================
# PARTY & METADATA UTILS
# =========================================================

def clean_party_name(name: Any) -> str:
    """Làm sạch tên Inventor/Assignee (nhà sáng chế/công ty)."""
    s = normalize_text(name).lower()

    # Bỏ hậu tố pháp nhân phổ biến (LTD, AG, Corp...)
    s = re.sub(r"\b(ltd|limited|inc|corp|corporation|gmbh|ag|sa|nv|llc|co)\b\.?", " ", s)
    # Giữ lại ký tự latin mở rộng chuẩn Notebook
    s = re.sub(r"[^a-z0-9äöüßéèàçñ\s\-&]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    return s


def normalize_date_yyyymmdd(x: Any) -> str:
    """Chuẩn hóa định dạng ngày về YYYYMMDD."""
    s = normalize_text(x)
    if not s:
        return ""

    if re.fullmatch(r"\d{8}", s):
        return s

    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", s):
        return s.replace("-", "")

    # Tìm cụm 8 chữ số liên tục
    m = re.search(r"\b(\d{8})\b", s)
    if m:
        return m.group(1)

    return ""


# =========================================================
# IPC REGEX & PARSING (THE HEART OF CELL 4)
# =========================================================

# Bắt mã IPC chính quy dạng: G06F 17/30, A61K 31/715
IPC_PATTERN = re.compile(
    r"\b([A-HY])\s*([0-9]{2})\s*([A-Z])\s*([0-9]{1,4})\s*/\s*([0-9]{1,6})\b"
)


def format_ipc_group(section: str, cls_num: str, subclass_letter: str, main_group: str, subgroup: str) -> Dict[str, str]:
    """
    Định dạng cấu trúc phân loại IPC chuẩn mực.
    Giữ nguyên các zero đầu quan trọng ở subgroup và loại bỏ ở main_group giống Notebook.
    """
    section = section.upper()
    cls_num = cls_num.zfill(2)
    subclass_letter = subclass_letter.upper()

    try:
        main_group_norm = str(int(main_group))
    except Exception:
        main_group_norm = main_group.lstrip("0") or "0"

    subgroup = str(subgroup)
    if subgroup.isdigit():
        subgroup_norm = subgroup.zfill(2)
    else:
        subgroup_norm = subgroup

    ipc_class = f"{section}{cls_num}"
    ipc_subclass = f"{section}{cls_num}{subclass_letter}"
    ipc_group = f"{ipc_subclass} {main_group_norm}/{subgroup_norm}"
    ipc_main_group = f"{ipc_subclass} {main_group_norm}"

    return {
        "ipc_normalized": ipc_group,
        "ipc_section": section,
        "ipc_class": ipc_class,
        "ipc_subclass": ipc_subclass,
        "ipc_group": ipc_group,
        "ipc_main_group": ipc_main_group,
    }


def extract_ipc_candidates(text: str) -> List[str]:
    """Trích xuất danh sách mã IPC chuẩn từ chuỗi văn bản thô bất kỳ."""
    text = normalize_text(text).upper()
    out = []

    for m in IPC_PATTERN.finditer(text):
        parsed = format_ipc_group(
            section=m.group(1),
            cls_num=m.group(2),
            subclass_letter=m.group(3),
            main_group=m.group(4),
            subgroup=m.group(5),
        )
        out.append(parsed["ipc_normalized"])

    # fallback cho định dạng tương đối chuẩn (VD: A61K 6/02)
    if not out:
        fallback = re.findall(r"\b([A-HY][0-9]{2}[A-Z]\s+[0-9]{1,4}/[0-9]{1,6})\b", text)
        for code in fallback:
            parsed = parse_ipc_code(code)
            if parsed["ipc_normalized"]:
                out.append(parsed["ipc_normalized"])

    # Loại trùng lặp
    seen = set()
    uniq = []
    for code in out:
        code = normalize_text(code.upper())
        if code and code not in seen:
            seen.add(code)
            uniq.append(code)

    return uniq


def parse_ipc_code(code: str) -> Dict[str, str]:
    """Phân rã một chuỗi mã IPC thành các cấp bậc thành phần."""
    code = normalize_text(code).upper()

    m = IPC_PATTERN.search(code)

    if not m:
        candidates = extract_ipc_candidates(code)
        if candidates:
            code = candidates[0]
            m = IPC_PATTERN.search(code)

    if not m:
        return {
            "ipc_normalized": code,
            "ipc_section": "",
            "ipc_class": "",
            "ipc_subclass": "",
            "ipc_group": "",
            "ipc_main_group": "",
        }

    return format_ipc_group(
        section=m.group(1),
        cls_num=m.group(2),
        subclass_letter=m.group(3),
        main_group=m.group(4),
        subgroup=m.group(5),
    )


def normalize_ipc_list(ipc_codes: Any) -> List[str]:
    """Chuẩn hóa và trích lọc toàn bộ mã IPC từ một list đầu vào thô."""
    raw_list = normalize_list(ipc_codes)
    found = []

    for item in raw_list:
        candidates = extract_ipc_candidates(item)
        if candidates:
            found.extend(candidates)
        else:
            parsed = parse_ipc_code(item)
            if parsed["ipc_normalized"]:
                found.append(parsed["ipc_normalized"])

    seen = set()
    out = []
    for code in found:
        code = normalize_text(code.upper())
        if code and code not in seen:
            seen.add(code)
            out.append(code)

    return out
