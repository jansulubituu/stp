export type AgentStep = {
  agent: string;
  label: string;
  status: "pending" | "running" | "completed" | "failed";
  summary: string;
  details: Record<string, unknown>;
};

export type AgentTrace = {
  variant: string;
  steps: AgentStep[];
  metrics: Record<string, unknown>;
};

export type AnalysisRunResponse = {
  run_id: string;
  status: string;
};

export type AnalysisStageEvent = {
  agent: string;
  status: AgentStep["status"];
  trace?: AgentTrace | null;
};

export type AnalysisRecord = {
  id: number;
  query: string;
  summary: string;
  key_points: string[];
  analysis: string;
  suggestions: string[];
  created_at: string;
  agent_trace?: AgentTrace | null;
};

export type SearchCandidate = {
  id: string;
  title: string;
  title_vi?: string;
  abstract: string;
  claims: string;
  description: string;
  assignees: string[];
  inventors: string[];
  ipc_codes: string[];
  citations: string[];
  publication_date: string;
  application_date?: string;
  priority_date?: string;
  score: number;
};

export type SearchResponse = {
  candidates: SearchCandidate[];
  agent_trace?: AgentTrace | null;
};
