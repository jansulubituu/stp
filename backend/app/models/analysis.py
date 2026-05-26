from datetime import datetime

from sqlalchemy import DateTime, Integer, String, Text, func, JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.db.session import Base


class AnalysisRecord(Base):
    __tablename__ = "analysis_records"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, index=True)
    query: Mapped[str] = mapped_column(Text, nullable=False)
    summary: Mapped[str] = mapped_column(Text, nullable=False)
    key_points: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    analysis: Mapped[str] = mapped_column(Text, nullable=False)
    suggestions: Mapped[list[str]] = mapped_column(JSON, nullable=False)
    agent_trace: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
