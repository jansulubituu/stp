import type { AnalysisRecord, SearchCandidate, SearchResponse } from "@/lib/types";

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!response.ok) {
    throw new Error(`API request failed: ${response.status}`);
  }

  if (response.status === 204) {
    return undefined as T;
  }

  return response.json() as Promise<T>;
}

export function analyzeQuery(query: string): Promise<AnalysisRecord> {
  return request<AnalysisRecord>("/api/analyze", {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}

export function searchCandidates(query: string): Promise<SearchResponse> {
  return request<SearchResponse>("/api/search", {
    method: "POST",
    body: JSON.stringify({ query }),
  });
}

export function analyzeSelectedQuery(query: string, selected_candidates: SearchCandidate[]): Promise<AnalysisRecord> {
  return request<AnalysisRecord>("/api/analyze-selected", {
    method: "POST",
    body: JSON.stringify({ query, selected_candidates }),
  });
}

export function fetchHistory(): Promise<AnalysisRecord[]> {
  return request<AnalysisRecord[]>("/api/history");
}

export function deleteHistoryItem(recordId: number): Promise<void> {
  return request<void>(`/api/history/${recordId}`, {
    method: "DELETE",
  });
}
