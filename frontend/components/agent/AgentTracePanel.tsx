import { CheckCircle2, Database, FileSearch, Scale, SearchCheck } from "lucide-react";

import type { AgentStep, AgentTrace } from "@/lib/types";

type Props = {
  trace?: AgentTrace | null;
  compact?: boolean;
};

function textList(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter((item): item is string => typeof item === "string" && item.length > 0)
    : [];
}

function objectList(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value)
    ? value.filter((item): item is Record<string, unknown> => typeof item === "object" && item !== null)
    : [];
}

function detailText(value: unknown): string {
  return typeof value === "string" && value.trim() ? value : "";
}

function backendLabel(value: unknown): string {
  const labels: Record<string, string> = {
    es_knn: "Tìm kiếm vector Elasticsearch (KNN)",
    frontend_selected: "Tài liệu do người dùng chọn",
  };
  const text = detailText(value);
  return labels[text] || text || "N/A";
}

function embeddingLabel(value: unknown): string {
  const labels: Record<string, string> = {
    openai_compatible: "Jina API",
    local_hf_jina: "Mô hình Jina cục bộ",
  };
  const text = detailText(value);
  return labels[text] || text || "N/A";
}

function stepLabel(step: AgentStep): string {
  const labels: Record<string, string> = {
    query_understanding: "Agent 1 - Hiểu truy vấn",
    retrieval: "Truy xuất - Tìm tài liệu đối chứng",
    evidence_extraction: "Agent 2 - Trích xuất bằng chứng",
    prior_art_analysis: "Agent 3 - Đánh giá tài liệu đối chứng",
    coverage_check: "Kiểm tra độ phủ",
  };
  return labels[step.agent] || step.label;
}

function StepIcon({ agent }: { agent: string }) {
  if (agent === "query_understanding") return <FileSearch size={17} />;
  if (agent === "retrieval") return <Database size={17} />;
  if (agent === "evidence_extraction") return <SearchCheck size={17} />;
  return <Scale size={17} />;
}

function StepDetails({ step, compact }: { step: AgentStep; compact: boolean }) {
  const details = step.details;
  const claimElements = textList(details.claim_elements);
  const queries = objectList(details.search_queries);
  const evidence = objectList(details.evidence);
  const ranked = objectList(details.ranked_prior_art);
  const documentIds = textList(details.top_document_ids);
  const problem = detailText(details.technical_problem);

  if (step.agent === "query_understanding") {
    return (
      <>
        {problem && <p className="agentDetailText">{problem}</p>}
        {claimElements.length > 0 && (
          <div className="agentChipList">
            {claimElements.map((item) => <span key={item}>{item}</span>)}
          </div>
        )}
        {!compact && queries.length > 0 && (
          <p className="agentDetailText">Đã tạo {queries.length} truy vấn kỹ thuật để tìm tài liệu tương đồng.</p>
        )}
      </>
    );
  }

  // if (step.agent === "retrieval") {
  //   return (
  //     <div className="agentMetricRow">
  //       <span>Nguồn tìm kiếm: {backendLabel(details.backend)}</span>
  //       <span>Mô hình nhúng: {embeddingLabel(details.embedding_backend)}</span>
  //       <span>Ứng viên: {String(details.num_screened_candidates ?? 0)}</span>
  //       {!compact && documentIds.length > 0 && <span>Tài liệu hàng đầu: {documentIds.join(", ")}</span>}
  //     </div>
  //   );
  // }

  if (step.agent === "evidence_extraction" && evidence.length > 0) {
    const validEvidence = evidence.filter((item) => objectList(item.matched_elements).length > 0);
    if (validEvidence.length === 0) return null;

    return (
      <div className="agentEvidenceGrid">
        {validEvidence.map((item) => {
          const matches = objectList(item.matched_elements);
          return (
            <details className="agentEvidenceCard" key={detailText(item.patent_id)} open>
              <summary className="agentEvidenceHeader">
                <strong>{detailText(item.patent_id)}</strong>
                <div className="agentEvidenceBadge">
                  <span className="badgeMatch">{matches.length} khớp</span>
                </div>
              </summary>
              <div className="agentEvidenceBody">
                {matches.map((match, index) => (
                  <div className="agentEvidenceMatchItem" key={index}>
                    <div className="matchLabel">{detailText(match.claim_element_vi) || detailText(match.claim_element)}</div>
                    <div className="matchText">{detailText(match.evidence_text)}</div>
                  </div>
                ))}
              </div>
            </details>
          );
        })}
      </div>
    );
  }

  if (step.agent === "prior_art_analysis" && ranked.length > 0) {
    return (
      <div className="agentRiskGrid">
        {ranked.map((item) => {
          const riskRaw = detailText(item.novelty_risk_vi) || detailText(item.novelty_risk) || "N/A";
          const riskLower = riskRaw.toLowerCase();
          let riskLevel = "low";
          let riskLabel = "🟢 Low Risk";
          if (riskLower.includes("cao") || riskLower.includes("high") || riskLower.includes("mạnh")) {
            riskLevel = "high";
            riskLabel = "🔴 High Risk";
          } else if (riskLower.includes("trung bình") || riskLower.includes("medium")) {
            riskLevel = "medium";
            riskLabel = "🟡 Medium Risk";
          }

          const synthesis = detailText(item.claim_overlap_summary) || detailText(item.limitations) || detailText(item.reasoning) || detailText(item.overlap);

          return (
            <div className={`riskCard risk-${riskLevel}`} key={detailText(item.patent_id)}>
              <div className="riskCardHeader">
                <strong>{detailText(item.patent_id)}</strong>
                <span className="riskBadge">{riskLabel}</span>
              </div>
              {synthesis && (
                <div className="riskSynthesis">
                  <p><strong>AI Synthesis:</strong> {synthesis}</p>
                </div>
              )}
            </div>
          );
        })}
      </div>
    );
  }

  return null;
}

export default function AgentTracePanel({ trace, compact = false }: Props) {
  if (!trace || trace.steps.length === 0) return null;

  return (
    <section className={`agentTrace ${compact ? "agentTraceCompact" : ""}`}>
      <div className="agentTraceHeader">
        <h3>Luồng đa tác tử</h3>
      </div>
      <div className="agentSteps">
        {trace.steps.map((step) => (
          <article className="agentStep" key={step.agent}>
            <div className="agentStepHeading">
              <span className="agentStepIcon"><StepIcon agent={step.agent} /></span>
              <div>
                <strong>{stepLabel(step)}</strong>
                <p>{step.summary}</p>
              </div>
              <CheckCircle2 className="agentStepDone" size={18} />
            </div>
            <StepDetails step={step} compact={compact} />
          </article>
        ))}
      </div>
    </section>
  );
}
