from pydantic_settings import BaseSettings, SettingsConfigDict
from typing import Optional


class Settings(BaseSettings):
    """Application settings using Pydantic BaseSettings."""

    # OpenAI
    OPENAI_API_KEY: Optional[str] = None

    # Anthropic
    ANTHROPIC_API_KEY: Optional[str] = None

    # Google
    GOOGLE_API_KEY: Optional[str] = None

    # OpenRouter
    OPENROUTER_API_KEY: Optional[str] = None

    # LangSmith
    LANGSMITH_API_KEY: Optional[str] = None
    LANGSMITH_TRACING: bool = False
    LANGSMITH_PROJECT: str = "langchain-v1-examples"

    # Tavily
    TAVILY_API_KEY: Optional[str] = None

    # Daytona
    DAYTONA_API_KEY: Optional[str] = None

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = Settings()
