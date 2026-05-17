from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    # LLM provider selection: "groq" or "anthropic"
    LLM_PROVIDER: str = "groq"

    # Groq settings
    GROQ_API_KEY: str = ""
    GROQ_MODEL: str = "llama-3.3-70b-versatile"

    # Anthropic settings
    ANTHROPIC_API_KEY: str = ""
    ANTHROPIC_MODEL: str = "claude-sonnet-4-6"

    # Embedding model (runs locally, no API key needed)
    EMBED_MODEL: str = "all-MiniLM-L6-v2"

    # Paths
    FAISS_STORE_PATH: str = "faiss_store"
    DATA_PATH: str = "data"

    # API server
    HOST: str = "0.0.0.0"
    PORT: int = 8000

    class Config:
        env_file = ".env"
        extra = "ignore"


@lru_cache
def get_settings() -> Settings:
    return Settings()
