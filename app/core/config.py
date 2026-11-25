from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "DocuMind API"
    app_env: str = "development"
    port: int = 8000
    host: str = "0.0.0.0"
    allowed_origins: str = "http://localhost:3000,http://127.0.0.1:3000,https://documine-tzlh.vercel.app"
    frontend_url: str = ""

    # Provider-Flexible AI Gateway & Official OpenAI Configuration (ADR-006, ADR-007, ADR-008, ADR-017)
    ai_default_provider: str = "openai"
    openai_api_key: str = ""
    openai_embedding_model: str = "text-embedding-3-large"
    openai_embedding_dimensions: int = 1536
    openai_model_fast: str = "gpt-6-luna"
    openai_model_balanced: str = "gpt-6.1-terra"
    openai_model_deep: str = "gpt-6.1-terra"
    openai_tts_model: str = "tts-1-hd"
    openai_tts_default_voice: str = "marin"
    openai_store_completions: bool = False
    openai_timeout_seconds: float = 35.0

    # Deprecated legacy fields kept only so unrefactored legacy modules do not raise AttributeError
    gemini_api_key: str = ""

    # Supabase Configuration (PostgreSQL 16 + pgvector + Auth + Private Storage)
    supabase_url: str = ""
    supabase_service_role_key: str = ""
    supabase_service_key: str = ""
    supabase_anon_key: str = ""
    supabase_storage_bucket: str = "documents"

    # Redis & Durable Background Queue (arq - ADR-014)
    redis_url: str = ""
    enable_arq_worker: bool = True

    # Operational Quotas & Safety Limits (DOC-28: HD-001..HD-003 configurable defaults)
    max_upload_size_mb: int = 50
    max_vision_ocr_pages: int = 100
    default_workspace_storage_limit_mb: int = 1024
    default_monthly_ai_budget_usd: float = 5.0
    require_email_verification: bool = False
    max_access_token_ttl_seconds: int = 86400

    # Local fallback upload directory
    upload_dir: str = "./uploads"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def effective_supabase_service_key(self) -> str:
        """Return whichever service role key variable is populated in .env."""
        return (self.supabase_service_role_key or self.supabase_service_key or "").strip()

    @property
    def default_model(self) -> str:
        return self.openai_model_balanced

    @property
    def lite_model(self) -> str:
        return self.openai_model_fast

    @property
    def live_model(self) -> str:
        return self.openai_model_fast

    @property
    def cors_origins(self) -> List[str]:
        origins = [
            origin.strip().rstrip("/")
            for origin in self.allowed_origins.split(",")
            if origin.strip()
        ]
        if self.frontend_url.strip():
            fe = self.frontend_url.strip().rstrip("/")
            if fe not in origins:
                origins.append(fe)
        return origins


settings = Settings()
