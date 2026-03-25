"""PuzzleMind configuration management.

Centralises every tuneable setting as a Pydantic BaseSettings model so
values can be overridden via environment variables or a .env file.
"""

from pathlib import Path
from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    """Application-wide configuration.

    All fields have sensible defaults for local development.
    Override via environment variables or a .env file.
    """

    # ── Server ──────────────────────────────────────────────
    HOST: str = "127.0.0.1"
    PORT: int = 8008
    DEBUG: bool = False

    # ── Paths ───────────────────────────────────────────────
    DATA_DIR: Path = Path("data")
    MILVUS_DB_PATH: str = "data/puzzlemind_milvus.db"
    KNOWLEDGE_DIR: Path = Path("data/knowledge")

    # ── Anthropic / Claude ──────────────────────────────────
    ANTHROPIC_API_KEY: str = ""
    LLM_SONNET_MODEL: str = "claude-sonnet-4-5-20250929"
    LLM_HAIKU_MODEL: str = "claude-haiku-4-5-20251001"
    LLM_MAX_TOKENS: int = 4096
    LLM_TEMPERATURE: float = 0.7

    # ── OpenAI / GPT-4o ─────────────────────────────────────
    OPENAI_API_KEY: str = ""
    GPT4O_MODEL: str = "gpt-4o"
    GPT4O_MAX_TOKENS: int = 4096

    # ── Ollama (local — Qwen 3 8B) ──────────────────────────
    # Ollama exposes an OpenAI-compatible API; we use the
    # openai Python client pointed at this base URL.
    OLLAMA_BASE_URL: str = "http://localhost:11434/v1"
    OLLAMA_MODEL: str = "qwen3:8b"
    OLLAMA_MAX_TOKENS: int = 2048

    # ── Rate Limiting ───────────────────────────────────────
    RATE_LIMIT_RPM: int = 10
    DAILY_INPUT_TOKEN_BUDGET: int = 50_000
    DAILY_OUTPUT_TOKEN_BUDGET: int = 10_000

    # ── Face Recognition ────────────────────────────────────
    FACE_RECOGNITION_MODEL: str = "GhostFaceNet"
    FACE_DETECTOR: str = "mediapipe"
    FACE_HIGH_CONFIDENCE: float = 0.4
    FACE_MEDIUM_CONFIDENCE: float = 0.6
    FACE_MAX_EMBEDDINGS_PER_PLAYER: int = 20
    CAMERA_ACTIVE_INTERVAL_MS: int = 500
    CAMERA_IDLE_INTERVAL_MS: int = 2000

    # ── Embeddings (BGE-M3) ─────────────────────────────────
    EMBEDDING_MODEL: str = "BAAI/bge-m3"
    EMBEDDING_DIMENSION: int = 1024

    # ── Jailbreak Detection ─────────────────────────────────
    JAILBREAK_SIMILARITY_THRESHOLD: float = 0.85

    # ── Emotion Analysis ────────────────────────────────────
    SENTIMENT_MODEL: str = (
        "distilbert-base-uncased-finetuned-sst-2-english"
    )

    # ── Memory & Forgetting ─────────────────────────────────
    IMMEDIATE_MEMORY_TURNS: int = 10
    SHORT_TERM_SUMMARY_INTERVAL: int = 5
    SHORT_TERM_MAX_WORDS: int = 500
    FORGETTING_LAMBDA: float = 0.16

    # ── Elo System ──────────────────────────────────────────
    ELO_INITIAL_RATING: float = 1200.0
    ELO_K_FACTOR_NEW: float = 48.0
    ELO_K_FACTOR_STABLE: float = 16.0
    ELO_TARGET_SUCCESS_LOW: float = 0.65
    ELO_TARGET_SUCCESS_HIGH: float = 0.75

    # ── Hardware Lock ───────────────────────────────────────
    HARDWARE_LOCK_ENABLED: bool = False

    model_config = {
        "env_prefix": "PUZZLEMIND_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


settings = Settings()
