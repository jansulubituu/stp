export type AnalysisRecord = {
  id: number;
  query: string;
  summary: string;
  key_points: string[];
  analysis: string;
  suggestions: string[];
  created_at: string;
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
};
