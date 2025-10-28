"""Configuration management for MemoryMCP."""

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # Database
    db_path: str = "./data/memory.db"

    # Embeddings
    embed_model: str = "paraphrase-multilingual-mpnet-base-v2"
    embed_device: str = "cpu"

    # API
    api_host: str = "0.0.0.0"
    api_port: int = 8080
    api_key: str | None = None

    # Search
    default_search_limit: int = 5
    similarity_threshold: float = 0.7

    # Performance
    max_text_length: int = 10000
    batch_size: int = 32

    # Logging
    log_level: str = "info"

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    def get_db_dir(self) -> Path:
        """Get database directory path."""
        return Path(self.db_path).parent

    def ensure_db_dir(self) -> None:
        """Create database directory if it doesn't exist."""
        db_dir = self.get_db_dir()
        db_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
settings = Settings()
