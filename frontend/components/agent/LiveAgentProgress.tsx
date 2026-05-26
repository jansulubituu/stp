import { BrainCircuit, Check, Database, FileSearch, Microscope, Orbit, Scale, Sparkles } from "lucide-react";

import type { AgentStep, AgentTrace } from "@/lib/types";

export type LiveAgentStatus = "pending" | "running" | "completed" | "failed";

type Props = {
  statuses: Record<string, LiveAgentStatus>;
  trace?: AgentTrace | null;
  selectedCount: number;
};

const stages = [
  {
    id: "query_understanding",
    title: "Agent 1",
    name: "Hiểu truy vấn",
    wait: "Chuẩn bị đọc truy vấn",
    running: "Đang phân rã vấn đề kỹ thuật và claim elements",
    icon: BrainCircuit,
  },
  {
    id: "retrieval",
    title: "Ngữ cảnh",
    name: "Nạp tài liệu ứng viên",
    wait: "Chờ Agent 1 hoàn tất",
    running: "Đang đóng gói tài liệu bạn đã chọn",
    icon: Database,
  },
  {
    id: "evidence_extraction",
    title: "Agent 2",
    name: "Trích xuất bằng chứng",
    wait: "Chờ tài liệu sẵn sàng",
    running: "Đang đối chiếu từng yếu tố với bằng chứng",
    icon: Microscope,
  },
  {
    id: "prior_art_analysis",
    title: "Agent 3",
    name: "Phân tích tài liệu đối chứng",
    wait: "Chờ bảng bằng chứng",
    running: "Đang xếp hạng rủi ro tính mới và lập kết luận",
    icon: Scale,
  },
  {
    id: "coverage_check",
    title: "Kiểm định",
    name: "Kiểm tra độ phủ",
    wait: "Chờ kết luận phân tích",
    running: "Đang kiểm tra độ phủ bằng chứng",
    icon: FileSearch,
  },
] as const;

function completedStep(trace: AgentTrace | null | undefined, agent: string): AgentStep | undefined {
  return trace?.steps.find((step) => step.agent === agent);
}

export default function LiveAgentProgress({ statuses, trace, selectedCount }: Props) {
  return (
    <div className="liveAgentConsole">
      <div className="liveAgentBackdrop" />
      <header className="liveAgentHeader">
        <div className="liveAgentOrb"><Orbit size={30} /></div>
        <div>
          <span>Điều phối đa tác tử</span>
          <h3>Đang phân tích {selectedCount} tài liệu đối chứng</h3>
          {/* <p>Các tác tử đang chuyển giao dữ liệu theo thời gian thực</p> */}
        </div>
        <div className="liveSignal"><i /> TRỰC TIẾP</div>
      </header>

      <div className="liveAgentRail">
        {stages.map((stage, index) => {
          const status = statuses[stage.id] ?? "pending";
          const StepIcon = stage.icon;
          const completed = completedStep(trace, stage.id);
          return (
            <article className={`liveAgentNode liveAgentNode-${status}`} key={stage.id}>
              <div className="liveAgentLink">{index < stages.length - 1 && <span />}</div>
              <div className="liveAgentIcon">
                {status === "completed" ? <Check size={20} /> : <StepIcon size={20} />}
              </div>
              <div className="liveAgentCopy">
                <small>{stage.title}</small>
                <strong>{stage.name}</strong>
                <p>{status === "running" ? stage.running : completed?.summary || stage.wait}</p>
              </div>
              {status === "running" && <div className="liveAgentPulse"><b /><b /><b /></div>}
            </article>
          );
        })}
      </div>

      {/* <footer className="liveAgentFooter">
        <Sparkles size={15} />
        <span>{trace ? `Đã nhận ${trace.steps.length} mốc xử lý từ quy trình ${trace.variant.toUpperCase()}` : "Đang thiết lập kênh sự kiện an toàn..."}</span>
      </footer> */}
    </div>
  );
}
