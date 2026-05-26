import threading

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.responses import StreamingResponse
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.db.session import SessionLocal, get_db
from app.models.analysis import AnalysisRecord
from app.schemas.analysis import AnalysisResponse, AnalysisRunResponse, AnalyzeRequest, SearchResponse, AnalyzeSelectedRequest
from app.services.analysis_runs import analysis_runs
from app.services.multiagent_adapter import multiagent_service as analyzer_service

router = APIRouter()


def _new_record(query: str, result) -> AnalysisRecord:
    return AnalysisRecord(
        query=query,
        summary=result.summary,
        key_points=result.key_points,
        analysis=result.analysis,
        suggestions=result.suggestions,
        agent_trace=result.agent_trace.model_dump() if result.agent_trace else None,
    )

@router.post("/search", response_model=SearchResponse)
def search_candidates(payload: AnalyzeRequest):
    result = analyzer_service.search_candidates(payload.query)
    return SearchResponse(**result)

@router.post("/analyze-selected", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
def analyze_selected_docs(payload: AnalyzeSelectedRequest, db: Session = Depends(get_db)) -> AnalysisRecord:
    # Chuyển đổi Pydantic model thành dict để truyền vào service
    candidates_dict = [c.model_dump() for c in payload.selected_candidates]
    result = analyzer_service.analyze_selected(payload.query, candidates_dict)
    record = _new_record(payload.query, result)
    db.add(record)
    db.commit()
    db.refresh(record)
    return record


def _run_selected_analysis(run_id: str, payload: AnalyzeSelectedRequest) -> None:
    candidates_dict = [candidate.model_dump() for candidate in payload.selected_candidates]

    def on_progress(agent: str, stage_status: str, trace) -> None:
        analysis_runs.publish(
            run_id,
            "stage",
            {
                "agent": agent,
                "status": stage_status,
                "trace": trace.model_dump(mode="json") if trace else None,
            },
        )

    try:
        result = analyzer_service.analyze_selected(payload.query, candidates_dict, on_progress=on_progress)
        with SessionLocal() as db:
            record = _new_record(payload.query, result)
            db.add(record)
            db.commit()
            db.refresh(record)
            response = AnalysisResponse.model_validate(record).model_dump(mode="json")
        analysis_runs.publish(run_id, "result", {"record": response}, complete=True)
    except Exception as exc:
        analysis_runs.publish(run_id, "error", {"message": str(exc)}, complete=True)


@router.post("/analyze-selected/runs", response_model=AnalysisRunResponse, status_code=status.HTTP_202_ACCEPTED)
def start_selected_analysis(payload: AnalyzeSelectedRequest) -> AnalysisRunResponse:
    run_id = analysis_runs.create()
    analysis_runs.publish(run_id, "queued", {"status": "queued"})
    threading.Thread(target=_run_selected_analysis, args=(run_id, payload), daemon=True).start()
    return AnalysisRunResponse(run_id=run_id)


@router.get("/analysis-runs/{run_id}/events")
def stream_analysis_events(run_id: str) -> StreamingResponse:
    try:
        analysis_runs.get(run_id)
    except KeyError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Analysis run not found") from exc
    return StreamingResponse(
        analysis_runs.events(run_id),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )

@router.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_201_CREATED)
def analyze(payload: AnalyzeRequest, db: Session = Depends(get_db)) -> AnalysisRecord:
    result = analyzer_service.analyze(payload.query)
    record = _new_record(payload.query, result)
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
