from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    database_url: str = "postgresql+psycopg://analysis_user:analysis_password@localhost:5432/analysis_app"
    cors_origins: list[str] = ["http://localhost:3000"]
    
    ai_key: str = ""
    nvidia_base_url: str = "https://integrate.api.nvidia.com/v1"
    nvidia_model: str = "abacusai/dracarys-llama-3.1-70b-instruct"
    
    es_cloud_id: str = ""
    es_api_key: str = ""
    bm25_index: str = "clef_ip_patents_v1_mini"
    knn_index: str = "clef_ip_patents_v1_mini_jina"
    jina_api_key: str = ""
    gemini_api_key: str = ""
    gemini_model: str = "gemini-3.1-flash-lite"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    translate_titles: bool = True

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: str | list[str]) -> list[str]:
        if isinstance(value, str):
            return [origin.strip() for origin in value.split(",") if origin.strip()]
        return value


settings = Settings()
