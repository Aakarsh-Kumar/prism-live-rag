from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip().strip("'\"")
        value = value.strip().strip("'\"")
        os.environ.setdefault(key, value)


def _env(*names: str) -> str | None:
    for name in names:
        value = os.getenv(name)
        if value:
            return value.strip().strip("'\"")
    return None


def _env_float(default: float, *names: str) -> float:
    value = _env(*names)
    if value is None:
        return default
    try:
        return float(value)
    except ValueError:
        return default


def _env_bool(default: bool, *names: str) -> bool:
    value = _env(*names)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(default: int, *names: str) -> int:
    value = _env(*names)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT
    data_dir: Path = ROOT / "data"
    lancedb_dir: Path = ROOT / ".cache" / "lancedb"
    table_name: str = "passages"
    embedding_dim: int = 128
    embedding_backend: str = "auto"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_cache_dir: Path = ROOT / ".cache" / "fastembed"
    embedding_local_files_only: bool = False
    embedding_device: str = "auto"
    embedding_batch_size: int = 128
    embedding_fixed_length: int = 512  # Fixed sequence length for stable CUDA memory usage (BGE-small max)
    rrf_k: int = 60
    sparse_weight: float = 0.35
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    top_k: int = 5
    groq_api_key: str | None = None
    deepseek_api_key: str | None = None
    groq_model: str = "openai/gpt-oss-120b"
    deepseek_model: str = "deepseek-chat"
    provider_timeout_s: float = 12.0

    @property
    def has_groq(self) -> bool:
        return bool(self.groq_api_key)

    @property
    def has_deepseek(self) -> bool:
        return bool(self.deepseek_api_key)


def load_settings() -> Settings:
    _load_env_file(ROOT / ".env")
    return Settings(
        groq_api_key=_env("GROQ_API_KEY", "GROQ-API-KEY"),
        deepseek_api_key=_env(
            "DEEPSEEK_API_KEY", "DEEPSEEK-API-KEY"
        ),
        groq_model=_env("GROQ_MODEL", "GROQ-MODEL") or "openai/gpt-oss-120b",
        deepseek_model=_env("DEEPSEEK_MODEL", "DEEPSEEK-MODEL") or "deepseek-chat",
        embedding_backend=_env("EMBEDDING_BACKEND", "EMBEDDING-BACKEND") or "auto",
        embedding_model=_env("EMBEDDING_MODEL", "EMBEDDING-MODEL")
        or "BAAI/bge-small-en-v1.5",
        embedding_cache_dir=Path(
            _env("EMBEDDING_CACHE_DIR", "EMBEDDING-CACHE-DIR")
            or (ROOT / ".cache" / "fastembed")
        ),
        embedding_local_files_only=_env_bool(
            False, "EMBEDDING_LOCAL_FILES_ONLY", "EMBEDDING-LOCAL-FILES-ONLY"
        ),
        embedding_device=_env("EMBEDDING_DEVICE", "EMBEDDING-DEVICE") or "auto",
        embedding_batch_size=_env_int(
            128, "EMBEDDING_BATCH_SIZE", "EMBEDDING-BATCH-SIZE"
        ),
        embedding_fixed_length=_env_int(
            512, "EMBEDDING_FIXED_LENGTH", "EMBEDDING-FIXED-LENGTH"
        ),
        provider_timeout_s=_env_float(12.0, "PROVIDER_TIMEOUT_S", "PROVIDER-TIMEOUT-S"),
        bm25_k1=_env_float(1.2, "BM25_K1", "BM25-K1"),
        bm25_b=_env_float(0.75, "BM25_B", "BM25-B"),
    )
