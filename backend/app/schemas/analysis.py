from datetime import datetime

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

class SearchResponse(BaseModel):
    candidates: list[SearchCandidate]

class AnalyzeSelectedRequest(BaseModel):
    query: str = Field(min_length=2, max_length=5000)
    selected_candidates: list[SearchCandidate]

class AnalysisResult(BaseModel):
    summary: str
    key_points: list[str]
    analysis: str
    suggestions: list[str]


class AnalysisResponse(AnalysisResult):
    id: int
    query: str
    created_at: datetime

    model_config = {"from_attributes": True}
