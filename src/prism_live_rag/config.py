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


@dataclass(frozen=True)
class Settings:
    root_dir: Path = ROOT
    data_dir: Path = ROOT / "data"
    lancedb_dir: Path = ROOT / ".cache" / "lancedb"
    table_name: str = "passages"
    embedding_dim: int = 128
    rrf_k: int = 60
    sparse_weight: float = 0.35
    top_k: int = 5
    groq_api_key: str | None = None
    deepseek_api_key: str | None = None

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
    )
