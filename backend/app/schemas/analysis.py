from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class AnalyzeRequest(BaseModel):
    query: str = Field(min_length=2, max_length=5000)

class SearchCandidate(BaseModel):
    id: str
    title: str
    title_vi: str = ""
    abstract: str
    claims: str = ""
    description: str = ""
    assignees: list[str]
    inventors: list[str]
    ipc_codes: list[str] = []
    citations: list[str] = []
    publication_date: str
    application_date: str = ""
    priority_date: str = ""
    score: float


class AgentStep(BaseModel):
    agent: str
    label: str
    status: str = "completed"
    summary: str
    details: dict[str, Any] = Field(default_factory=dict)


class AgentTrace(BaseModel):
    variant: str
    steps: list[AgentStep] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)


class SearchResponse(BaseModel):
    candidates: list[SearchCandidate]
    agent_trace: AgentTrace | None = None

class AnalyzeSelectedRequest(BaseModel):
    query: str = Field(min_length=2, max_length=5000)
    selected_candidates: list[SearchCandidate]


class AnalysisRunResponse(BaseModel):
    run_id: str
    status: str = "queued"


class AnalysisResult(BaseModel):
    summary: str
    key_points: list[str]
    analysis: str
    suggestions: list[str]
    agent_trace: AgentTrace | None = None


class AnalysisResponse(AnalysisResult):
    id: int
    query: str
    created_at: datetime

    model_config = {"from_attributes": True}
