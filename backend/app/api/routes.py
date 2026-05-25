from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import get_db
from app.models.analysis import AnalysisRecord
from app.schemas.analysis import AnalysisResponse, AnalyzeRequest, SearchResponse, AnalyzeSelectedRequest
from app.services.analyzer import analyzer_service

router = APIRouter()

@router.post("/search", response_model=SearchResponse)
def search_candidates(payload: AnalyzeRequest):
    candidates = analyzer_service.search_candidates(payload.query)
    return SearchResponse(candidates=candidates)

@router.post("/analyze-selected", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
def analyze_selected_docs(payload: AnalyzeSelectedRequest, db: Session = Depends(get_db)) -> AnalysisRecord:
    # Chuyển đổi Pydantic model thành dict để truyền vào service
    candidates_dict = [c.model_dump() for c in payload.selected_candidates]
    result = analyzer_service.analyze_selected(payload.query, candidates_dict)
    record = AnalysisRecord(
        query=payload.query,
        summary=result.summary,
        key_points=result.key_points,
        analysis=result.analysis,
        suggestions=result.suggestions,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record

@router.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
def analyze(payload: AnalyzeRequest, db: Session = Depends(get_db)) -> AnalysisRecord:
    result = analyzer_service.analyze(payload.query)
    record = AnalysisRecord(
        query=payload.query,
        summary=result.summary,
        key_points=result.key_points,
        analysis=result.analysis,
        suggestions=result.suggestions,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


@router.get("/history", response_model=list[AnalysisResponse])
def list_history(db: Session = Depends(get_db)) -> list[AnalysisRecord]:
    statement = select(AnalysisRecord).order_by(AnalysisRecord.created_at.desc())
    return list(db.scalars(statement))


@router.get("/history/{record_id}", response_model=AnalysisResponse)
def get_history_item(record_id: int, db: Session = Depends(get_db)) -> AnalysisRecord:
    record = db.get(AnalysisRecord, record_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    return record


@router.delete("/history/{record_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_history_item(record_id: int, db: Session = Depends(get_db)) -> None:
    record = db.get(AnalysisRecord, record_id)
    if record is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis not found")
    db.delete(record)
    db.commit()
