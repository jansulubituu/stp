from collections.abc import Generator

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.core.config import settings


class Base(DeclarativeBase):
    pass


connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, pool_pre_ping=True, connect_args=connect_args)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


def ensure_analysis_record_columns() -> None:
    """Add nullable display-only fields for databases created before agent traces."""

    columns = {column["name"] for column in inspect(engine).get_columns("analysis_records")}
    if "agent_trace" not in columns:
        with engine.begin() as connection:
            connection.execute(text("ALTER TABLE analysis_records ADD COLUMN agent_trace JSON"))


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
