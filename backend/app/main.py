from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.core.config import settings
from app.db.session import Base, engine, ensure_analysis_record_columns

Base.metadata.create_all(bind=engine)
ensure_analysis_record_columns()

app = FastAPI(title="Research Analysis API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_origin_regex=r"https?://.*",  # Cho phép tất cả các nguồn (bao gồm ngrok và localhost các cổng khác nhau) cho môi trường Dev
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router, prefix="/api")


@app.get("/health")
def health_check() -> dict[str, str]:
    return {"status": "ok"}
