import math
from collections import defaultdict
from typing import Dict, Any, List, Tuple, Set

from app.services.nlp_utils import (
    normalize_text, 
    normalize_list, 
    normalize_ipc_list, 
    clean_party_name
)

# =========================================================
# CONSTANTS & WEIGHTS (MATCHING NOTEBOOK DEFAULTS)
# =========================================================

GRAPH_RESCORER_PRESERVE_TOP = 5    # Giữ cứng Top 5 ban đầu không đổi (Hoặc cấu hình động)
GRAPH_RESCORER_EXPAND_TOP_K = 300
GRAPH_SEED_TOP_K = 50

# BỘ HẰNG SỐ CHIẾN LƯỢC ĐÃ ĐƯỢC ĐIỀU CHỈNH THEO YÊU CẦU CỦA BẠN
# (hybrid_graph_rank_rescore_g030_e015_gate010)
GRAPH_WEIGHT = 0.30
EVIDENCE_WEIGHT = 0.15
EVIDENCE_GATE_FLOOR = 0.10
NEW_DOC_WEIGHT = 0.015

# Mức độ phổ biến tối đa để lọc nhiễu (Graph Degree Thresholds)
GRAPH_MAX_CITATION_DEGREE = 40
GRAPH_MAX_ASSIGNEE_DEGREE = 150
GRAPH_MAX_INVENTOR_DEGREE = 80


# =========================================================
# MATHEMATICAL UTILITIES
# =========================================================

def graph_rank_weight(rank: int) -> float:
    """Trọng số giảm dần theo thứ hạng (chuẩn notebook: log decay)."""
    return 1.0 / math.log2(max(2, rank) + 1.0)


def graph_degree_weight(degree: int) -> float:
    """Giảm trọng số của các node quá phổ biến (như công ty khổng lồ)."""
    if degree <= 0:
        return 0.0
    return 1.0 / math.sqrt(float(degree))


def canonical_doc_id(doc_id: str) -> str:
    """Lấy ID rút gọn để gom nhóm (VD: EP-0001858-B1 -> EP-0001858)."""
    text = str(doc_id or "").strip().upper().replace(" ", "-").replace("_", "-")
    import re
    match = re.match(r"^([A-Z]{2})-?(\d+)(?:-[A-Z][0-9A-Z]*)?$", text)
    if match:
        return f"{match.group(1)}-{match.group(2)}"
    return text


# =========================================================
# DYNAMIC ONLINE GRAPH RETRIEVER
# =========================================================

class DynamicGraphRetriever:
    def __init__(self, base_candidates: List[Tuple[str, float, Dict[str, Any]]]):
        """
        Khởi tạo và TỰ ĐỘNG DỰNG ĐỒ THỊ liên kết nóng trên RAM từ danh sách ứng viên ES.
        """
        self.base_candidates = base_candidates
        
        # Cấu trúc dữ liệu ma trận đồ thị kề (Adjacency Matrix)
        self.citation_adj = defaultdict(set)  # Forward: doc -> cited docs
        self.citation_rev = defaultdict(set)  # Reverse: cited doc -> citing docs
        
        self.ipc_adj = defaultdict(set)       # Forward: doc -> ipcs
        self.ipc_rev = defaultdict(set)       # Reverse: ipc -> docs
        
        self.assignee_adj = defaultdict(set)  # Forward: doc -> assignees
        self.assignee_rev = defaultdict(set)  # Reverse: assignee -> docs
        
        self.inventor_adj = defaultdict(set)  # Forward: doc -> inventors
        self.inventor_rev = defaultdict(set)  # Reverse: inventor -> docs
        
        self._build_graph_dynamically()

    def _build_graph_dynamically(self):
        """Quét Metadata của tất cả ứng viên để thiết lập mối quan hệ."""
        for doc_id, _, src in self.base_candidates:
            doc_id = str(doc_id)
            can_id = canonical_doc_id(doc_id)
            
            # 1. Mapping Mã ngành công nghệ IPC
            ipcs = normalize_ipc_list(src.get("ipc_codes", []))
            for ipc in ipcs:
                self.ipc_adj[doc_id].add(ipc)
                self.ipc_adj[can_id].add(ipc)
                self.ipc_rev[ipc].add(doc_id)
                
            # 2. Mapping Người sở hữu (Assignees)
            assignees = [clean_party_name(a) for a in normalize_list(src.get("assignees", [])) if clean_party_name(a)]
            for ass in assignees:
                self.assignee_adj[doc_id].add(ass)
                self.assignee_adj[can_id].add(ass)
                self.assignee_rev[ass].add(doc_id)
                
            # 3. Mapping Tác giả (Inventors)
            inventors = [clean_party_name(i) for i in normalize_list(src.get("inventors", [])) if clean_party_name(i)]
            for inv in inventors:
                self.inventor_adj[doc_id].add(inv)
                self.inventor_adj[can_id].add(inv)
                self.inventor_rev[inv].add(doc_id)
                
            # 4. Mapping Trích dẫn (Citations)
            citations = [canonical_doc_id(c) for c in normalize_list(src.get("citations", [])) if canonical_doc_id(c)]
            for cite in citations:
                self.citation_adj[doc_id].add(cite)
                self.citation_adj[can_id].add(cite)
                self.citation_rev[cite].add(doc_id)

    def _calculate_candidate_path_evidence_score(
        self, 
        doc_id: str, 
        query_profile: Dict[str, Any]
    ) -> float:
        """
        Tính điểm minh chứng cạnh (Evidence Score) cho ứng viên.
        Được mô phỏng trung thực 100% từ hàm `_candidate_path_evidence_score` của Notebook.
        """
        doc_id = str(doc_id)
        can_id = canonical_doc_id(doc_id)
        score = 0.0
        
        # Lấy thông tin đối sánh từ Query đầu vào
        ipc_analysis = query_profile.get("ipc_analysis", {})
        query_ipcs = set(ipc_analysis.get("codes_norm", []))
        
        party_analysis = query_profile.get("party_analysis", {})
        query_assignees = set(party_analysis.get("assignees", []))
        query_inventors = set(party_analysis.get("inventors", []))
        
        # 1. Chia sẻ mã công nghệ IPC (Shared IPC)
        candidate_ipcs = self.ipc_adj.get(doc_id, set()) | self.ipc_adj.get(can_id, set())
        shared_ipcs = candidate_ipcs & query_ipcs
        if shared_ipcs:
            ipc_score = 0.0
            for ipc in shared_ipcs:
                # Notebook weight factor: 0.90 * degree_weight
                degree = len(self.ipc_rev.get(ipc, set()))
                ipc_score += 0.90 * graph_degree_weight(degree)
            score += min(1.20, ipc_score)
            
        # 2. Trùng lặp công ty sở hữu (Same Assignee)
        candidate_assignees = self.assignee_adj.get(doc_id, set()) | self.assignee_adj.get(can_id, set())
        shared_assignees = candidate_assignees & query_assignees
        if shared_assignees:
            assignee_score = 0.0
            for assignee in shared_assignees:
                degree = len(self.assignee_rev.get(assignee, set()))
                if 0 < degree <= GRAPH_MAX_ASSIGNEE_DEGREE:
                    assignee_score += 0.45 * graph_degree_weight(degree)
            score += min(0.75, assignee_score)
            
        # 3. Trùng lặp tác giả (Same Inventor)
        candidate_inventors = self.inventor_adj.get(doc_id, set()) | self.inventor_adj.get(can_id, set())
        shared_inventors = candidate_inventors & query_inventors
        if shared_inventors:
            inventor_score = 0.0
            for inventor in shared_inventors:
                degree = len(self.inventor_rev.get(inventor, set()))
                if 0 < degree <= GRAPH_MAX_INVENTOR_DEGREE:
                    inventor_score += 0.55 * graph_degree_weight(degree)
            score += min(0.80, inventor_score)
            
        # Chuẩn hóa đưa về thang điểm 0.0 -> 1.0 chuẩn Notebook
        # (Notebook formula: min(1.0, score / 4.0))
        return min(1.0, float(score) / 4.0)

    def rank_hybrid_expansion(self) -> List[Tuple[str, float]]:
        """
        Hàm Lan truyền Trọng số Đồ thị (Graph Score Propagation).
        Sử dụng các ứng viên BM25 top đầu làm 'hạt giống' (Seeds) và lan tỏa điểm số.
        """
        scores = defaultdict(float)
        seed_candidates = self.base_candidates[:GRAPH_SEED_TOP_K]
        
        for rank, (seed_doc_id, _, _) in enumerate(seed_candidates, start=1):
            seed_doc_id = str(seed_doc_id)
            seed_can_id = canonical_doc_id(seed_doc_id)
            seed_weight = graph_rank_weight(rank)
            
            seed_keys = {seed_doc_id, seed_can_id}
            for key in seed_keys:
                # A. Lan truyền qua Trích dẫn xuôi (Forward Citations) -> Trọng số cực cao
                for cited_doc in self.citation_adj.get(key, set()):
                    scores[cited_doc] += 1.00 * seed_weight
                    
                # B. Lan truyền qua Trích dẫn ngược (Reverse Citations)
                rev_citations = self.citation_rev.get(key, set())
                deg = len(rev_citations)
                if 0 < deg <= GRAPH_MAX_CITATION_DEGREE:
                    for citing_doc in rev_citations:
                        scores[citing_doc] += 0.18 * seed_weight * graph_degree_weight(deg)
                        
                # C. Lan truyền qua mã công nghệ IPC chung
                for ipc in self.ipc_adj.get(key, set()):
                    for neighbor in self.ipc_rev.get(ipc, set()):
                        scores[neighbor] += 0.08 * seed_weight
                        
                # D. Lan truyền qua Assignee chung
                for assignee in self.assignee_adj.get(key, set()):
                    for neighbor in self.assignee_rev.get(assignee, set()):
                        scores[neighbor] += 0.05 * seed_weight
                        
                # E. Lan truyền qua Inventor chung
                for inventor in self.inventor_adj.get(key, set()):
                    for neighbor in self.inventor_rev.get(inventor, set()):
                        scores[neighbor] += 0.06 * seed_weight
                        
        # Sắp xếp kết quả lan truyền
        ranked = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
        return [(doc_id, float(score)) for doc_id, score in ranked[:GRAPH_RESCORER_EXPAND_TOP_K]]

    def execute_hybrid_graph_rank_rescore(
        self, 
        query_profile: Dict[str, Any]
    ) -> List[Tuple[str, float, Dict[str, Any]]]:
        """
        LÕI THUẬT TOÁN CHIẾN LƯỢC: hybrid_graph_rank_rescore_g030_e015_gate010
        Thực hiện tái chấm điểm tổ hợp giữa Base Rank Score, Graph Rank Score và Evidence Gate.
        """
        if not self.base_candidates:
            return []
            
        # 1. Bước A: Thực hiện lan truyền điểm đồ thị (Graph Expansion)
        graph_ranked = self.rank_hybrid_expansion()
        
        # Lập map để tìm nhanh thứ hạng và metadata
        base_lookup = {str(d): {"rank": r, "score": s, "src": m} for r, (d, s, m) in enumerate(self.base_candidates, start=1)}
        graph_lookup = {str(d): {"rank": r, "score": s} for r, (d, s) in enumerate(graph_ranked, start=1)}
        
        # Tập hợp toàn bộ ứng viên tiềm năng
        candidate_docs = set(base_lookup.keys()) | set(graph_lookup.keys())
        
        # 2. Bước B: Giữ nguyên Top bảo tồn (Preserve Top)
        final_results = []
        preserved_ids = set()
        
        # Trích ra các Top bảo tồn ban đầu (thường giữ Top 5 của BM25 không cho rớt hạng)
        for doc_id, score, src in self.base_candidates[:GRAPH_RESCORER_PRESERVE_TOP]:
            preserved_ids.add(str(doc_id))
            # Cộng base bonus siêu lớn để giữ đỉnh bảng giống Notebook (+100.0)
            final_results.append((str(doc_id), score + 100.0, src))
            
        # 3. Bước C: Áp dụng toán tử Rescorer thần thánh cho các ứng viên còn lại
        scored_rows = []
        for doc_id in candidate_docs:
            if doc_id in preserved_ids:
                continue
                
            base_item = base_lookup.get(doc_id)
            graph_item = graph_lookup.get(doc_id)
            
            # Lấy metadata nguồn dự phòng
            src_meta = (base_item or {}).get("src", {})
            
            # Đổi thứ hạng (rank) sang điểm số (rank score) thông qua log decay
            base_rank_score = 0.0 if not base_item else graph_rank_weight(base_item["rank"])
            graph_rank_score = 0.0 if not graph_item else graph_rank_weight(graph_item["rank"])
            
            # Tính toán điểm minh chứng cạnh (Evidence) và áp dụng Gate Logic
            evidence_score = self._calculate_candidate_path_evidence_score(doc_id, query_profile)
            evidence_gate = max(EVIDENCE_GATE_FLOOR, evidence_score)
            
            # Điểm thưởng cho tài liệu mới chỉ được đồ thị tìm ra
            new_doc_bonus = NEW_DOC_WEIGHT if not base_item and graph_item else 0.0
            
            # --- CÔNG THỨC TOÁN HỌC CHỦ ĐẠO CỦA NOTEBOOK ---
            final_score = (
                base_rank_score
                + GRAPH_WEIGHT * graph_rank_score * evidence_gate
                + EVIDENCE_WEIGHT * evidence_score
                + new_doc_bonus
            )
            
            scored_rows.append((doc_id, float(final_score), src_meta))
            
        # Sắp xếp tất cả các dòng sau tái cấu trúc theo điểm giảm dần
        scored_rows.sort(key=lambda x: -x[1])
        
        # Ghép nối danh sách bảo tồn và danh sách tái chấm điểm
        final_results.extend(scored_rows)
        
        # Trả về Top 300 ứng viên ưu tú nhất
        return final_results[:300]
