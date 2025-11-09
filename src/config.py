"""Configuration management"""

from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppConfig(BaseSettings):
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

    def fetch_db_directory(self) -> Path:
        """Get database directory path."""
        return Path(self.db_path).parent

    def validate_db_directory_exists(self) -> None:
        """Create database directory if it doesn't exist."""
        db_dir = self.fetch_db_directory()
        db_dir.mkdir(parents=True, exist_ok=True)


# Global settings instance
app_config = AppConfig()