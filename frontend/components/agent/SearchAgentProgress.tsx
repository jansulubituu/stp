"use client";

import { useEffect, useMemo, useState } from "react";
import { BrainCircuit, Database, FileSearch, Sparkles } from "lucide-react";

type Props = {
  query: string;
};

const phases = [
  {
    title: "Đọc đầu vào",
    message: "Nhận diện bối cảnh kỹ thuật và phạm vi tra cứu",
    icon: FileSearch,
  },
  {
    title: "Tách yếu tố claim",
    message: "Phân rã cấu trúc chức năng và thành phần cốt lõi",
    icon: BrainCircuit,
  },
  {
    title: "Sinh truy vấn tìm kiếm",
    message: "Chuẩn hóa cụm kỹ thuật cho truy xuất tương đồng",
    icon: Sparkles,
  },
  {
    title: "Truy xuất tài liệu",
    message: "Đối chiếu vector với kho tài liệu đối chứng",
    icon: Database,
  },
] as const;

function displayTerms(query: string): string[] {
  const text = query.replace(/<[^>]+>/g, " ").replace(/\s+/g, " ").trim();
  const terms = text.match(/[\p{L}\p{N}-]{4,}/gu) ?? [];
  return [...new Set(terms)].slice(0, 6);
}

export default function SearchAgentProgress({ query }: Props) {
  const [phaseIndex, setPhaseIndex] = useState(0);
  const terms = useMemo(() => displayTerms(query), [query]);

  useEffect(() => {
    setPhaseIndex(0);
    const interval = window.setInterval(() => {
      setPhaseIndex((current) => Math.min(current + 1, phases.length - 1));
    }, 540);
    return () => window.clearInterval(interval);
  }, [query]);

  return (
    <section className="searchAgentProgress">
      <div className="searchAgentGlow" />
      <header>
        <div className="searchAgentCore"><BrainCircuit size={25} /></div>
        <div>
          <span>AGENT 1 ĐANG XỬ LÝ</span>
          <h3>Phân rã truy vấn trước khi tìm tài liệu</h3>
          <p>Kết quả sẽ xuất hiện sau khi hoàn tất các bước chuẩn hóa và truy xuất.</p>
        </div>
      </header>

      <div className="searchPhaseRail">
        {phases.map((phase, index) => {
          const Icon = phase.icon;
          const status = index < phaseIndex ? "completed" : index === phaseIndex ? "running" : "pending";
          return (
            <div className={`searchPhase searchPhase-${status}`} key={phase.title}>
              <Icon size={17} />
              <div>
                <strong>{phase.title}</strong>
                <small>{phase.message}</small>
              </div>
              {status === "running" && <i />}
            </div>
          );
        })}
      </div>


    </section>
  );
}
