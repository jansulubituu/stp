"use client";

import { FormEvent, useEffect, useMemo, useRef, useState } from "react";
import { Bot, History, Menu, Plus, Send, Trash2, User, Sparkles, X, CheckSquare, Square, ChevronLeft, ChevronRight, Search, ChevronDown, ChevronUp, Lightbulb, BookOpen, Scale, FileText, Eye } from "lucide-react";
import ReactMarkdown from "react-markdown";

import { searchCandidates, analyzeSelectedQuery, deleteHistoryItem, fetchHistory } from "@/lib/api";
import type { AnalysisRecord, SearchCandidate } from "@/lib/types";

type ChatMessage =
  | { role: "user"; content: string }
  | { role: "assistant"; record: AnalysisRecord };

const examples = [
  "Đánh giá tính mới (novelty) của giải pháp nhận diện cử chỉ dựa trên đồ thị xương",
  "So sánh phạm vi bảo hộ (claim scope) các sáng chế về xử lý ngôn ngữ lớn của OpenAI và Google",
  "Phân tích khả năng bảo hộ sáng chế (patentability) của công nghệ tối ưu sạc pin xe điện bằng AI",
];

/** Tạo tiêu đề sạch từ query */
function cleanQueryTitle(raw: string): string {
  const isXml = /<[^>]+>/.test(raw);

  if (isXml) {
    const titleMatch = raw.match(/<title>([^<]+)<\/title>/i) || raw.match(/<B541>en<\/B541>\s*<B542>([^<]+)<\/B542>/i);
    if (titleMatch?.[1]?.trim()) {
      const docMatch = raw.match(/doc-number="([^"]+)"/);
      const prefix = docMatch ? `EP${docMatch[1]} – ` : "";
      const title = `${prefix}${titleMatch[1].trim()}`;
      return title.length > 100 ? title.slice(0, 97) + "..." : title;
    }

    const abstractMatch = raw.match(/<abstract[^>]*>([\s\S]*?)<\/abstract>/i);
    if (abstractMatch?.[1]) {
      const abstractClean = abstractMatch[1].replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
      if (abstractClean.length > 5) {
        return abstractClean.length > 100 ? abstractClean.slice(0, 97) + "..." : abstractClean;
      }
    }

    const idMatch = raw.match(/id="([^"]+)"/);
    if (idMatch?.[1]) {
      return `Truy vấn patent: ${idMatch[1]}`;
    }

    return "Truy vấn XML patent";
  }

  const trimmed = raw.replace(/\s+/g, " ").trim();
  if (trimmed.length > 120) {
    return trimmed.slice(0, 117) + "...";
  }
  return trimmed || "Truy vấn mới";
}

/** Hiển thị tiêu đề (tạm chỉ bản gốc; bật lại title_vi khi cần) */
function renderPatentTitles(candidate: SearchCandidate) {
  const titleEn = candidate.title?.trim() || "N/A";
  // const titleVi = candidate.title_vi?.trim();

  return (
    <div style={{ fontWeight: "500", lineHeight: "1.45" }}>{titleEn}</div>
  );
  /* Tạm tắt tiêu đề tiếng Việt
  return (
    <div>
      <div style={{ fontWeight: "500", lineHeight: "1.45" }}>{titleEn}</div>
      {titleVi && titleVi !== titleEn && (
        <div
          style={{
            marginTop: "6px",
            fontSize: "12.5px",
            lineHeight: "1.45",
            color: "var(--muted)",
            fontStyle: "italic",
          }}
        >
          {titleVi}
        </div>
      )}
    </div>
  );
  */
}

/** Format mảng tên Assignee / Inventor thành chuỗi đẹp mắt (Title Case, bỏ lặp) */
function formatPartyNames(names: string[] | undefined): string {
  if (!names || names.length === 0) return "N/A";
  
  // Lọc trùng và chuẩn hóa
  const uniqueNames = Array.from(new Set(names.map(name => {
    let clean = name.replace(/[\[\]]/g, "").trim(); // Xóa ngoặc vuông nếu có
    // Xóa một số hậu tố doanh nghiệp cơ bản ở đuôi (nếu muốn, hoặc chỉ Title Case)
    clean = clean.toLowerCase().replace(/\b(ltd|limited|inc|corp|corporation|gmbh|ag|sa|nv|llc|co)\b\.?/g, "").trim();
    // Chuyển thành Title Case
    return clean.replace(/\b\w/g, c => c.toUpperCase());
  })));
  
  return uniqueNames.filter(Boolean).join(", ") || "N/A";
}

/** Định dạng ngày YYYYMMDD hoặc YYYY-MM-DD thành DD/MM/YYYY */
function formatDate(dateStr: string | undefined | null): string {
  if (!dateStr || dateStr === "N/A") return "N/A";
  
  const clean = dateStr.trim();
  
  // Case 1: YYYYMMDD (e.g. 19921209)
  if (/^\d{8}$/.test(clean)) {
    const yyyy = clean.substring(0, 4);
    const mm = clean.substring(4, 6);
    const dd = clean.substring(6, 8);
    return `${dd}/${mm}/${yyyy}`;
  }
  
  // Case 2: YYYY-MM-DD (e.g. 1992-12-09)
  if (/^\d{4}-\d{2}-\d{2}$/.test(clean)) {
    const parts = clean.split("-");
    return `${parts[2]}/${parts[1]}/${parts[0]}`;
  }

  // Case 3: Try to parse using Javascript Date
  try {
    const date = new Date(clean);
    if (!isNaN(date.getTime())) {
      const dd = String(date.getDate()).padStart(2, '0');
      const mm = String(date.getMonth() + 1).padStart(2, '0');
      const yyyy = date.getFullYear();
      return `${dd}/${mm}/${yyyy}`;
    }
  } catch (e) {
    // Ignore and fallback
  }

  return clean;
}

/** Tách và format Claims thành danh sách dễ đọc */
function renderClaims(claimsText: string | undefined) {
  if (!claimsText || claimsText.trim() === "" || claimsText === "N/A") {
    return <span style={{ color: "var(--muted)" }}>Không có dữ liệu claims</span>;
  }
  
  // Chỉ tách theo newline
  let parts = claimsText.split(/\n+/).filter(p => p.trim() !== "");

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: "10px", marginTop: "8px", maxHeight: "300px", overflowY: "auto", paddingRight: "8px" }}>
      {parts.map((p, idx) => {
        const isMainClaim = /^\d+\.\s/.test(p.trim());
        return (
          <div key={idx} style={{ 
            padding: "10px 14px", 
            background: "rgba(255,255,255,0.03)", 
            borderRadius: "8px", 
            borderLeft: isMainClaim ? "3px solid var(--accent-start)" : "3px solid rgba(255,255,255,0.1)",
            lineHeight: "1.5",
            fontSize: "13px"
          }}>
            {p.trim()}
          </div>
        );
      })}
    </div>
  );
}

/** Định dạng tin nhắn của User để hiển thị bảng thông tin đẹp mắt nếu là XML form input */
function renderUserMessage(content: string) {
  if (content.trim().startsWith("<patent-document>")) {
    const titleMatch = content.match(/<title>([\s\S]*?)<\/title>/i);
    const abstractMatch = content.match(/<abstract>([\s\S]*?)<\/abstract>/i);
    const claimsMatch = content.match(/<claims>([\s\S]*?)<\/claims>/i);
    const descriptionMatch = content.match(/<description>([\s\S]*?)<\/description>/i);
    const ipcMatch = content.match(/<classification-ipc>([\s\S]*?)<\/classification-ipc>/i);

    const title = titleMatch ? titleMatch[1].trim() : "";
    const abstract = abstractMatch ? abstractMatch[1].trim() : "";
    const claims = claimsMatch ? claimsMatch[1].trim() : "";
    const description = descriptionMatch ? descriptionMatch[1].trim() : "";
    const ipc = ipcMatch ? ipcMatch[1].trim() : "";

    return (
      <div style={{ 
        background: "var(--panel)", 
        padding: "20px", 
        borderRadius: "16px", 
        border: "1px solid var(--border)",
        fontSize: "14px",
        maxWidth: "720px",
        width: "100%",
        color: "var(--text)",
        textAlign: "left",
        boxShadow: "var(--shadow)"
      }}>
        <div style={{ 
          borderBottom: "1px solid var(--border)", 
          paddingBottom: "12px", 
          marginBottom: "16px", 
          fontWeight: "700", 
          color: "var(--accent-start)", 
          fontSize: "13px", 
          letterSpacing: "0.6px",
          display: "flex",
          alignItems: "center",
          gap: "8px",
          textTransform: "uppercase"
        }}>
          <FileText size={16} /> THÔNG TIN TRUY VẤN ĐÃ NHẬP
        </div>
        <table style={{ width: "100%", borderCollapse: "separate", borderSpacing: "0 14px", marginTop: "-10px" }}>
          <tbody>
            {title && (
              <tr>
                <td style={{ padding: "0", width: "120px", verticalAlign: "top", color: "var(--muted)", fontWeight: "600" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <Lightbulb size={16} className="text-amber-500" color="#eab308" /> Tiêu đề:
                  </div>
                </td>
                <td style={{ padding: "0 0 0 12px", fontWeight: "700", color: "var(--text)", fontSize: "15px", lineHeight: "1.5" }}>{title}</td>
              </tr>
            )}
            {abstract && (
              <tr>
                <td style={{ padding: "0", verticalAlign: "top", color: "var(--muted)", fontWeight: "600" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <BookOpen size={16} color="#2563eb" /> Tóm tắt:
                  </div>
                </td>
                <td style={{ padding: "0 0 0 12px", color: "#334155", lineHeight: "1.7", textAlign: "justify" }}>{abstract}</td>
              </tr>
            )}
            {claims && (
              <tr>
                <td style={{ padding: "0", verticalAlign: "top", color: "var(--muted)", fontWeight: "600" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <Scale size={16} color="#059669" /> Yêu cầu:
                  </div>
                </td>
                <td style={{ padding: "0 0 0 12px", color: "#334155", whiteSpace: "pre-wrap", lineHeight: "1.7", textAlign: "justify" }}>{claims}</td>
              </tr>
            )}
            {description && (
              <tr>
                <td style={{ padding: "0", verticalAlign: "top", color: "var(--muted)", fontWeight: "600" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <FileText size={16} color="#4f46e5" /> Mô tả:
                  </div>
                </td>
                <td style={{ padding: "0 0 0 12px", color: "#334155", opacity: 0.95, lineHeight: "1.7", textAlign: "justify" }}>
                  <div style={{ display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>{description}</div>
                </td>
              </tr>
            )}
            {ipc && (
              <tr>
                <td style={{ padding: "0", verticalAlign: "top", color: "var(--muted)", fontWeight: "600" }}>
                  <div style={{ display: "flex", alignItems: "center", gap: "6px" }}>
                    <Sparkles size={16} color="#d946ef" /> Mã IPC:
                  </div>
                </td>
                <td style={{ padding: "0 0 0 12px", color: "var(--accent-start)", fontFamily: "monospace", fontWeight: "700", fontSize: "14.5px" }}>{ipc}</td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    );
  }
  return <p style={{ whiteSpace: "pre-wrap" }}>{content}</p>;
}

export default function Home() {
  const [history, setHistory] = useState<AnalysisRecord[]>([]);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [query, setQuery] = useState("");
  const [inputMode, setInputMode] = useState<"text" | "form">("form");
  const [formData, setFormData] = useState({
    title: "", abstract: "", claims: "", description: "", ipc: "",
  });
  
  // Workflow States
  const [workflowStep, setWorkflowStep] = useState<"input" | "searching" | "selecting" | "analyzing" | "result">("input");
  const [currentQueryText, setCurrentQueryText] = useState("");
  const [searchResults, setSearchResults] = useState<SearchCandidate[]>([]);
  const [selectedDocs, setSelectedDocs] = useState<SearchCandidate[]>([]);
  const [viewingDoc, setViewingDoc] = useState<SearchCandidate | null>(null);
  const [activeTab, setActiveTab] = useState<"abstract" | "claims" | "description" | "citations">("abstract");
  
  // Pagination States
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(10);

  const [composerExpanded, setComposerExpanded] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [sidebarOpen, setSidebarOpen] = useState(false);
  const bottomRef = useRef<HTMLDivElement>(null);

  const hasMessages = messages.length > 0;

  useEffect(() => {
    fetchHistory()
      .then(setHistory)
      .catch(() => setError("Không tải được lịch sử tra cứu."));
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, workflowStep]);

  const activeTitle = useMemo(() => {
    const firstUserMessage = messages.find((message) => message.role === "user");
    if (!firstUserMessage) return "Tra cứu mới";
    return cleanQueryTitle(firstUserMessage.content);
  }, [messages]);

  async function handleSearch(overrideValue?: string) {
    let finalQuery = overrideValue || query;
    if (!overrideValue && inputMode === "form") {
      finalQuery = `<patent-document>\n  <title>${formData.title}</title>\n  <abstract>${formData.abstract}</abstract>\n  <claims>${formData.claims}</claims>\n  <description>${formData.description}</description>\n  <classification-ipc>${formData.ipc}</classification-ipc>\n</patent-document>`;
    }
    
    const trimmed = finalQuery.trim();
    if (!trimmed || workflowStep === "searching" || workflowStep === "analyzing") {
      return;
    }

    setError(null);
    setWorkflowStep("searching");
    setCurrentQueryText(trimmed);
    setMessages((current) => [...current, { role: "user", content: trimmed }]);
    
    try {
      const res = await searchCandidates(trimmed);
      setSearchResults(res.candidates || []);
      setWorkflowStep("selecting");
      setSelectedDocs([]);
      setCurrentPage(1);
    } catch {
      setError("Tìm kiếm thất bại. Vui lòng kiểm tra lại dịch vụ Backend.");
      setWorkflowStep("input");
    } finally {
      setQuery("");
      setFormData({ title: "", abstract: "", claims: "", description: "", ipc: "" });
      setSidebarOpen(false);
    }
  }

  async function handleAnalyzeSelected() {
    if (selectedDocs.length === 0) return;
    
    setWorkflowStep("analyzing");
    setError(null);
    
    try {
      const result = await analyzeSelectedQuery(currentQueryText, selectedDocs);
      setMessages((current) => [...current, { role: "assistant", record: result }]);
      setHistory((current) => [result, ...current.filter((item) => item.id !== result.id)]);
      setWorkflowStep("result");
      // Giữ lại searchResults và selectedDocs để có thể "Quay lại" và chọn tiếp doc khác
    } catch {
      setError("Phân tích AI thất bại. Vui lòng thử lại.");
      setWorkflowStep("selecting");
    }
  }

  function handleSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    void handleSearch();
  }

  function openHistory(record: AnalysisRecord) {
    setMessages([
      { role: "user", content: record.query },
      { role: "assistant", record },
    ]);
    setError(null);
    setSidebarOpen(false);
    setWorkflowStep("result");
  }

  async function removeHistory(recordId: number) {
    try {
      await deleteHistoryItem(recordId);
      setHistory((current) => current.filter((item) => item.id !== recordId));
    } catch {
      setError("Không xóa được mục lịch sử.");
    }
  }

  // --- Pagination Logic ---
  const totalPages = Math.ceil(searchResults.length / itemsPerPage);
  const paginatedResults = useMemo(() => {
    const start = (currentPage - 1) * itemsPerPage;
    return searchResults.slice(start, start + itemsPerPage);
  }, [searchResults, currentPage, itemsPerPage]);

  const toggleSelection = (candidate: SearchCandidate) => {
    const isSelected = selectedDocs.some(c => c.id === candidate.id);
    if (isSelected) {
      setSelectedDocs(selectedDocs.filter(c => c.id !== candidate.id));
    } else {
      if (selectedDocs.length >= 5) return;
      setSelectedDocs([...selectedDocs, candidate]);
    }
  };

  return (
    <main className="shell">
      {sidebarOpen && (
        <div className="sidebarOverlay" onClick={() => setSidebarOpen(false)} />
      )}

      <aside className={`sidebar ${sidebarOpen ? "sidebarVisible" : ""}`}>
        <div className="sidebarTop">
          <div className="brand">
            <div className="brandMark" style={{ overflow: "hidden", background: "#ffffff", padding: "2px", border: "1px solid var(--border)" }}>
              <img src="/logo.png" alt="PatSight" style={{ width: "100%", height: "100%", objectFit: "cover", borderRadius: "calc(var(--radius-md) - 2px)" }} />
            </div>
            <div><strong>PatSight</strong><span>Patent Intelligence</span></div>
          </div>
          <button className="closeSidebarBtn" onClick={() => setSidebarOpen(false)}>
            <X size={18} />
          </button>
        </div>
        <button className="newButton" onClick={() => { setMessages([]); setWorkflowStep("input"); setSidebarOpen(false); }}>
          <Plus size={16} /> Tra cứu mới
        </button>
        <div className="historyHeader"><History size={14} /><span>Lịch sử nghiên cứu</span></div>
        <div className="historyList">
          {history.map((item) => (
            <div className="historyItem" key={item.id}>
              <button onClick={() => openHistory(item)}>
                <span>{cleanQueryTitle(item.query)}</span>
                <small>{new Date(item.created_at).toLocaleDateString("vi-VN")} {new Date(item.created_at).toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit" })}</small>
              </button>
              <button className="iconButton" onClick={() => void removeHistory(item.id)}>
                <Trash2 size={14} />
              </button>
            </div>
          ))}
        </div>
      </aside>

      <section className="chatPanel">
        <header className="topbar">
          <button className="menuButton" onClick={() => setSidebarOpen(true)}><Menu size={20} /></button>
          <div className="topbarContent">
            <span>Phiên làm việc</span>
            <h1>{activeTitle}</h1>
          </div>
        </header>

        <div className="conversation" style={{ paddingBottom: workflowStep === "selecting" ? "120px" : "180px" }}>
          {!hasMessages && workflowStep === "input" && (
            <section className="emptyState">
              <div className="emptyIcon" style={{ 
                width: "80px", 
                height: "80px", 
                borderRadius: "24px", 
                background: "rgba(79, 70, 229, 0.04)", 
                border: "1px solid rgba(79, 70, 229, 0.12)", 
                display: "flex", 
                alignItems: "center", 
                justifyContent: "center",
                marginBottom: "24px",
                boxShadow: "none"
              }}>
                <Sparkles size={44} color="var(--accent-start)" strokeWidth={1.2} />
              </div>
              <h2>Tôi có thể trợ giúp gì về sáng chế?</h2>
              <p>Nhập mã bằng sáng chế, mô tả ý tưởng hoặc yêu cầu phân tích để tôi trích xuất dữ liệu, tìm đối chứng và đánh giá.</p>
              <div className="promptGrid">
                {examples.map((example) => (
                  <button key={example} onClick={() => void handleSearch(example)} type="button">{example}</button>
                ))}
              </div>
            </section>
          )}

          {messages.map((message, index) =>
            message.role === "user" ? (
              <article className="message userMessage" key={`user-${index}`}>
                <div className="avatar"><User size={16} /></div>
                {renderUserMessage(message.content)}
              </article>
            ) : (
              <article className="message assistantMessage" key={`assistant-${message.record.id}`}>
                <div className="avatar"><Bot size={16} /></div>
                <div className="analysisResult" style={{ width: "100%" }}>
                  {index === messages.length - 1 && workflowStep === "result" && searchResults.length > 0 && (
                    <div style={{ marginBottom: "16px", borderBottom: "1px solid var(--border)", paddingBottom: "12px" }}>
                      <button 
                        onClick={() => setWorkflowStep("selecting")}
                        style={{
                          display: "inline-flex",
                          alignItems: "center",
                          gap: "6px",
                          background: "rgba(79, 70, 229, 0.05)",
                          border: "1px solid rgba(79, 70, 229, 0.2)",
                          color: "var(--accent-start)",
                          padding: "8px 14px",
                          borderRadius: "8px",
                          fontSize: "13px",
                          cursor: "pointer",
                          fontWeight: "500",
                          transition: "all 0.2s"
                        }}
                        onMouseEnter={(e) => { e.currentTarget.style.background = "rgba(79, 70, 229, 0.12)"; }}
                        onMouseLeave={(e) => { e.currentTarget.style.background = "rgba(79, 70, 229, 0.05)"; }}
                      >
                        <ChevronLeft size={16} /> ← Quay lại danh sách ứng viên để chọn & phân tích tài liệu khác
                      </button>
                    </div>
                  )}
                  <div className="markdown-body"><ReactMarkdown>{message.record.analysis}</ReactMarkdown></div>
                </div>
              </article>
            )
          )}

          {workflowStep === "searching" && (
            <article className="message assistantMessage">
              <div className="avatar"><Search size={16} /></div>
              <div className="thinkingContainer">
                <span className="thinkingText">Đang tìm kiếm...</span>
                <div className="dotGroup"><div className="dot" /><div className="dot" /><div className="dot" /></div>
              </div>
            </article>
          )}

          {workflowStep === "selecting" && (
            <article className="message assistantMessage" style={{ maxWidth: "100%" }}>
              <div className="avatar"><Bot size={16} /></div>
              <div className="selectionContainer" style={{ width: "100%" }}>
                <h3 style={{ margin: "0 0 16px 0", color: "var(--text)" }}>Vui lòng chọn các tài liệu muốn phân tích sâu (Tối đa 5)</h3>
                
                <div className="tableControls" style={{ display: "flex", justifyContent: "space-between", marginBottom: "12px", alignItems: "center" }}>
                  <div className="itemsPerPage">
                    <span style={{ color: "var(--muted)", fontSize: "13px", marginRight: "8px" }}>Hiển thị:</span>
                    <select 
                      value={itemsPerPage} 
                      onChange={(e) => { setItemsPerPage(Number(e.target.value)); setCurrentPage(1); }}
                      style={{ background: "var(--panel)", color: "var(--text)", border: "1px solid var(--border)", borderRadius: "4px", padding: "4px 8px" }}
                    >
                      <option value={5}>5</option>
                      <option value={10}>10</option>
                      <option value={20}>20</option>
                      <option value={50}>50</option>
                      <option value={100}>100</option>
                    </select>
                  </div>
                  <div className="pagination" style={{ display: "flex", gap: "8px", alignItems: "center" }}>
                    <button disabled={currentPage === 1} onClick={() => setCurrentPage(p => p - 1)} className="iconButton"><ChevronLeft size={16} /></button>
                    <span style={{ color: "var(--muted)", fontSize: "13px" }}>Trang {currentPage} / {totalPages || 1}</span>
                    <button disabled={currentPage >= totalPages} onClick={() => setCurrentPage(p => p + 1)} className="iconButton"><ChevronRight size={16} /></button>
                  </div>
                </div>

                <div className="tableWrapper" style={{ overflowX: "auto", background: "var(--panel)", borderRadius: "8px", border: "1px solid var(--border)", boxShadow: "var(--shadow-sm)" }}>
                  <table style={{ width: "100%", borderCollapse: "collapse", fontSize: "14px", textAlign: "left" }}>
                    <thead style={{ background: "var(--sidebar)", borderBottom: "1px solid var(--border)" }}>
                      <tr>
                        <th style={{ padding: "12px", width: "40px" }}></th>
                        <th style={{ padding: "12px", color: "var(--muted)", fontWeight: "500" }}>ID</th>
                        <th style={{ padding: "12px", color: "var(--muted)", fontWeight: "500" }}>Tiêu đề</th>
                        <th style={{ padding: "12px", color: "var(--muted)", fontWeight: "500" }}>Assignee</th>
                        <th style={{ padding: "12px", color: "var(--muted)", fontWeight: "500" }}>Năm</th>
                        <th style={{ padding: "12px", color: "var(--muted)", fontWeight: "500", textAlign: "right" }}>Thao tác</th>
                      </tr>
                    </thead>
                    <tbody>
                      {paginatedResults.map((c) => {
                        const isSelected = selectedDocs.some(doc => doc.id === c.id);
                        const disabled = !isSelected && selectedDocs.length >= 5;
                        return (
                          <tr key={c.id} 
                              onClick={(e) => {
                                // Nếu click vào cột chứa button thì bỏ qua
                                if ((e.target as HTMLElement).closest('.view-btn')) return;
                                if (!disabled) toggleSelection(c);
                              }}
                              style={{ 
                                borderBottom: "1px solid var(--border)", 
                                cursor: disabled ? "not-allowed" : "pointer",
                                opacity: disabled ? 0.5 : 1,
                                background: isSelected ? "rgba(79, 70, 229, 0.05)" : "transparent",
                                transition: "background 0.2s"
                              }}>
                            <td style={{ padding: "12px" }}>
                              {isSelected ? <CheckSquare size={18} color="var(--accent-start)" /> : <Square size={18} color="var(--muted)" />}
                            </td>
                            <td style={{ padding: "12px", color: "var(--accent-start)" }}>{c.id}</td>
                            <td style={{ padding: "12px", color: "var(--text)" }}>
                              <div style={{ display: "-webkit-box", WebkitLineClamp: 3, WebkitBoxOrient: "vertical", overflow: "hidden" }}>
                                {renderPatentTitles(c)}
                              </div>
                            </td>
                            <td style={{ padding: "12px", color: "var(--muted)" }}>
                              <div style={{ maxWidth: "180px", whiteSpace: "nowrap", overflow: "hidden", textOverflow: "ellipsis" }} title={formatPartyNames(c.assignees)}>
                                {formatPartyNames(c.assignees)}
                              </div>
                            </td>
                            <td style={{ padding: "12px", color: "var(--muted)" }}>
                              {c.publication_date && c.publication_date !== "N/A" ? c.publication_date.substring(0, 4) : "N/A"}
                            </td>
                            <td style={{ padding: "12px", textAlign: "right" }} className="view-btn">
                              <button 
                                onClick={(e) => { e.stopPropagation(); setViewingDoc(c); setActiveTab("abstract"); }}
                                style={{ background: "var(--sidebar)", border: "1px solid var(--border)", color: "var(--text)", padding: "4px 10px", borderRadius: "6px", fontSize: "12px", fontWeight: "500", cursor: "pointer", transition: "all 0.2s" }}
                                onMouseEnter={(e) => { e.currentTarget.style.background = "var(--panel)"; }}
                                onMouseLeave={(e) => { e.currentTarget.style.background = "var(--sidebar)"; }}
                              >
                                Xem
                              </button>
                            </td>
                          </tr>
                        );
                      })}
                    </tbody>
                  </table>
                </div>

                <div className="selectionActions" style={{ marginTop: "16px", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                  <div style={{ color: "var(--text)", fontSize: "14px", fontWeight: "500" }}>Đã chọn: <strong style={{ color: "var(--accent-start)" }}>{selectedDocs.length}/5</strong> tài liệu</div>
                  <button 
                    onClick={handleAnalyzeSelected}
                    disabled={selectedDocs.length === 0}
                    style={{ background: selectedDocs.length > 0 ? "var(--accent-gradient)" : "#e2e8f0", color: selectedDocs.length > 0 ? "#fff" : "var(--muted)", padding: "10px 24px", borderRadius: "8px", fontWeight: "600", transition: "all 0.2s", cursor: selectedDocs.length > 0 ? "pointer" : "not-allowed", boxShadow: selectedDocs.length > 0 ? "var(--glow)" : "none" }}
                  >
                    Tiến hành Phân tích AI
                  </button>
                </div>
              </div>
            </article>
          )}

          {workflowStep === "analyzing" && (
            <article className="message assistantMessage">
              <div className="avatar"><Bot size={16} /></div>
              <div className="thinkingContainer">
                <span className="thinkingText">NVIDIA NIM đang đọc kỹ {selectedDocs.length} tài liệu và sinh báo cáo...</span>
                <div className="dotGroup"><div className="dot" /><div className="dot" /><div className="dot" /></div>
              </div>
            </article>
          )}

          {error && <div className="errorBanner">{error}</div>}
          <div ref={bottomRef} />
        </div>

        {viewingDoc && (
          <div className="modal-backdrop" style={{ position: "fixed", top: 0, left: 0, right: 0, bottom: 0, zIndex: 9999, display: "flex", alignItems: "center", justifyContent: "center", padding: "20px" }}>
            <div className="premium-modal" style={{ width: "100%", maxWidth: "800px", maxHeight: "90vh", borderRadius: "16px", display: "flex", flexDirection: "column" }}>
              <div style={{ padding: "18px 24px", borderBottom: "1px solid var(--border)", display: "flex", justifyContent: "space-between", alignItems: "center" }}>
                <h3 style={{ margin: 0, color: "var(--text)", fontSize: "17px", fontWeight: "700" }}>Chi tiết: {viewingDoc.id}</h3>
                <button onClick={() => setViewingDoc(null)} style={{ background: "transparent", color: "var(--muted)", cursor: "pointer" }}><X size={20} /></button>
              </div>
              <div style={{ padding: "24px", overflowY: "auto", fontSize: "14.5px", lineHeight: "1.6" }}>
                <div style={{ marginBottom: "20px" }}>
                  <strong style={{ color: "var(--accent-start)", display: "block", marginBottom: "8px" }}>Tiêu đề:</strong>
                  {renderPatentTitles(viewingDoc)}
                </div>

                {/* Timeline thời gian sáng chế */}
                <div style={{ marginBottom: "24px", background: "rgba(0,0,0,0.01)", padding: "16px", borderRadius: "12px", border: "1px solid var(--border)" }}>
                  <strong style={{ color: "var(--muted)", fontSize: "12px", textTransform: "uppercase", letterSpacing: "0.05em", display: "block", marginBottom: "12px" }}>
                    Dòng thời gian sáng chế
                  </strong>
                  <div className="timeline-container">
                    <div className="timeline-line" />
                    <div className={`timeline-step ${viewingDoc.application_date && viewingDoc.application_date !== "N/A" ? "active" : ""}`}>
                      <div className="timeline-dot">1</div>
                      <span style={{ fontSize: "11px", color: "var(--muted)", marginTop: "6px", fontWeight: "600" }}>Ngày nộp đơn (Application)</span>
                      <strong style={{ fontSize: "13px", marginTop: "2px" }}>
                        {formatDate(viewingDoc.application_date)}
                      </strong>
                    </div>
                    <div className={`timeline-step ${viewingDoc.publication_date && viewingDoc.publication_date !== "N/A" ? "active" : ""}`}>
                      <div className="timeline-dot">2</div>
                      <span style={{ fontSize: "11px", color: "var(--muted)", marginTop: "6px", fontWeight: "600" }}>Ngày công bố (Publication)</span>
                      <strong style={{ fontSize: "13px", marginTop: "2px" }}>
                        {formatDate(viewingDoc.publication_date)}
                      </strong>
                    </div>
                  </div>
                </div>

                {/* Metadata Owner & Inventors */}
                <div style={{ marginBottom: "24px", display: "grid", gridTemplateColumns: "1fr 1fr", gap: "16px" }}>
                  <div style={{ background: "rgba(0,0,0,0.01)", padding: "14px", borderRadius: "10px", border: "1px solid var(--border)" }}>
                    <div style={{ color: "var(--muted)", fontSize: "12px", fontWeight: "600", display: "flex", alignItems: "center", gap: "6px", marginBottom: "6px" }}>
                      <Bot size={14} color="var(--accent-start)" /> Người nộp đơn / Chủ sở hữu (Assignee)
                    </div>
                    <strong style={{ fontSize: "13.5px", color: "var(--text)" }}>
                      {formatPartyNames(viewingDoc.assignees)}
                    </strong>
                  </div>
                  <div style={{ background: "rgba(0,0,0,0.01)", padding: "14px", borderRadius: "10px", border: "1px solid var(--border)" }}>
                    <div style={{ color: "var(--muted)", fontSize: "12px", fontWeight: "600", display: "flex", alignItems: "center", gap: "6px", marginBottom: "6px" }}>
                      <User size={14} color="#8b5cf6" /> Nhà sáng chế / Tác giả (Inventor)
                    </div>
                    <strong style={{ fontSize: "13.5px", color: "var(--text)" }}>
                      {formatPartyNames(viewingDoc.inventors)}
                    </strong>
                  </div>
                </div>

                {/* IPC Classification */}
                <div style={{ marginBottom: "24px" }}>
                  <div style={{ color: "var(--muted)", fontSize: "12px", fontWeight: "600", display: "flex", alignItems: "center", gap: "6px", marginBottom: "8px" }}>
                    <Sparkles size={14} color="#eab308" /> Mã công nghệ (IPC Classification)
                  </div>
                  <div style={{ display: "flex", gap: "6px", flexWrap: "wrap" }}>
                    {viewingDoc.ipc_codes && viewingDoc.ipc_codes.length > 0 ? (
                      viewingDoc.ipc_codes.map((ipc) => (
                        <span key={ipc} style={{ background: "rgba(234, 179, 8, 0.08)", border: "1px solid rgba(234, 179, 8, 0.2)", color: "#b45309", padding: "4px 10px", borderRadius: "6px", fontSize: "12px", fontFamily: "monospace", fontWeight: "700" }}>
                          {ipc}
                        </span>
                      ))
                    ) : (
                      <span style={{ color: "var(--muted)", fontSize: "13px" }}>N/A</span>
                    )}
                  </div>
                </div>

                {/* Tabs Panel */}
                <div style={{ display: "flex", flexDirection: "column", border: "1px solid var(--border)", borderRadius: "12px", overflow: "hidden" }}>
                  <div className="tab-header" style={{ background: "var(--sidebar)", borderBottom: "1px solid var(--border)", margin: 0, padding: "0 12px" }}>
                    <button type="button" className={`tab-btn ${activeTab === "abstract" ? "active" : ""}`} onClick={() => setActiveTab("abstract")}>
                      Tóm tắt (Abstract)
                    </button>
                    <button type="button" className={`tab-btn ${activeTab === "claims" ? "active" : ""}`} onClick={() => setActiveTab("claims")}>
                      Yêu cầu bảo hộ (Claims)
                    </button>
                    <button type="button" className={`tab-btn ${activeTab === "description" ? "active" : ""}`} onClick={() => setActiveTab("description")}>
                      Mô tả chi tiết
                    </button>
                    <button type="button" className={`tab-btn ${activeTab === "citations" ? "active" : ""}`} onClick={() => setActiveTab("citations")}>
                      Tài liệu trích dẫn
                    </button>
                  </div>
                  <div style={{ padding: "20px", maxHeight: "380px", overflowY: "auto", background: "#ffffff", color: "var(--text)" }}>
                    {activeTab === "abstract" && (
                      <div style={{ fontSize: "14px", lineHeight: "1.7", color: "#334155", textAlign: "justify" }}>
                        {viewingDoc.abstract || <span style={{ color: "var(--muted)" }}>Không có dữ liệu abstract</span>}
                      </div>
                    )}
                    {activeTab === "claims" && (
                      <div>
                        {renderClaims(viewingDoc.claims)}
                      </div>
                    )}
                    {activeTab === "description" && (
                      <div style={{ fontSize: "14px", lineHeight: "1.7", color: "#334155", whiteSpace: "pre-wrap", textAlign: "justify" }}>
                        {viewingDoc.description || <span style={{ color: "var(--muted)" }}>Không có dữ liệu mô tả chi tiết</span>}
                      </div>
                    )}
                    {activeTab === "citations" && (
                      <div>
                        <span style={{ fontSize: "13px", color: "var(--muted)", display: "block", marginBottom: "12px" }}>
                          Danh sách bằng sáng chế đối chứng liên quan:
                        </span>
                        <div style={{ display: "flex", gap: "8px", flexWrap: "wrap" }}>
                          {viewingDoc.citations && viewingDoc.citations.length > 0 ? (
                            viewingDoc.citations.map((cit) => (
                              <button
                                key={cit}
                                type="button"
                                className="citation-badge"
                                onClick={() => {
                                  void handleSearch(cit);
                                  setViewingDoc(null);
                                }}
                              >
                                {cit}
                              </button>
                            ))
                          ) : (
                            <span style={{ color: "var(--muted)", fontSize: "13px" }}>Không có tài liệu trích dẫn đối chứng</span>
                          )}
                        </div>
                      </div>
                    )}
                  </div>
                </div>
              </div>
              <div style={{ padding: "16px 20px", borderTop: "1px solid var(--border)", display: "flex", justifyContent: "flex-end" }}>
                <button 
                  onClick={() => {
                    const isSelected = selectedDocs.some(d => d.id === viewingDoc.id);
                    if (!isSelected && selectedDocs.length < 5) toggleSelection(viewingDoc);
                    else if (isSelected) toggleSelection(viewingDoc);
                    setViewingDoc(null);
                  }}
                  style={{ background: selectedDocs.some(d => d.id === viewingDoc.id) ? "var(--danger)" : "var(--accent-gradient)", color: "#fff", padding: "10px 20px", borderRadius: "8px", fontWeight: "600", cursor: "pointer", boxShadow: "var(--shadow-sm)" }}
                >
                  {selectedDocs.some(d => d.id === viewingDoc.id) ? "Bỏ chọn" : "Chọn tài liệu này"}
                </button>
              </div>
            </div>
          </div>
        )}

        {/* Floating Composer */}
        {(workflowStep === "input" || workflowStep === "result") && (
          <div className="composer">
            <div className="composerHeader" style={{ alignItems: "center" }}>
              <button type="button" onClick={() => setInputMode("form")} className={inputMode === "form" ? "active" : ""}>Nhập theo trường</button>
              <button type="button" onClick={() => setInputMode("text")} className={inputMode === "text" ? "active" : ""}>Mô tả ý tưởng sáng chế</button>
              
              <button 
                type="button" 
                onClick={() => setComposerExpanded(!composerExpanded)} 
                style={{ 
                  marginLeft: "auto", 
                  background: "rgba(255,255,255,0.05)", 
                  border: "1px solid var(--border)",
                  color: "var(--muted)", 
                  display: "flex", 
                  alignItems: "center", 
                  gap: "6px",
                  fontSize: "12px",
                  padding: "6px 12px",
                  borderRadius: "20px",
                  cursor: "pointer",
                  transition: "all 0.2s"
                }}
                onMouseEnter={(e) => { e.currentTarget.style.color = "#fff"; e.currentTarget.style.background = "rgba(255,255,255,0.1)"; }}
                onMouseLeave={(e) => { e.currentTarget.style.color = "var(--muted)"; e.currentTarget.style.background = "rgba(255,255,255,0.05)"; }}
              >
                {composerExpanded ? <ChevronDown size={14} /> : <ChevronUp size={14} />}
                {composerExpanded ? "Thu gọn bộ gõ" : "Mở rộng bộ gõ"}
              </button>
            </div>
            {composerExpanded && (
              <form className={`composerForm ${inputMode === "form" ? "formMode" : ""}`} onSubmit={handleSubmit}>
                {inputMode === "text" ? (
                  <textarea
                    onChange={(event) => setQuery(event.target.value)}
                    onKeyDown={(event) => {
                      if (event.key === "Enter" && !event.shiftKey) {
                        event.preventDefault();
                        void handleSearch();
                      }
                    }}
                    placeholder="Nhập mô tả ý tưởng sáng chế..."
                    rows={1}
                    value={query}
                  />
                ) : (
                  <div className="structuredForm">
                    <input type="text" placeholder="Tiêu đề (Title) *" value={formData.title} onChange={e => setFormData({...formData, title: e.target.value})} required />
                    <textarea placeholder="Tóm tắt (Abstract) - tóm lược nội dung" value={formData.abstract} onChange={e => setFormData({...formData, abstract: e.target.value})} rows={2} />
                    <textarea placeholder="Yêu cầu bảo hộ (Claims) - phạm vi độc quyền" value={formData.claims} onChange={e => setFormData({...formData, claims: e.target.value})} rows={3} />
                    <textarea placeholder="Mô tả chi tiết (Description)" value={formData.description} onChange={e => setFormData({...formData, description: e.target.value})} rows={2} />
                    <input type="text" placeholder="Mã phân loại IPC (vd: A61K 31/00)" value={formData.ipc} onChange={e => setFormData({...formData, ipc: e.target.value})} />
                  </div>
                )}
                
                <button
                  aria-label="Gửi yêu cầu"
                  disabled={inputMode === "text" ? !query.trim() : !formData.title.trim()}
                  type="submit"
                >
                  <Send size={18} />
                </button>
              </form>
            )}
          </div>
        )}
      </section>
    </main>
  );
}
