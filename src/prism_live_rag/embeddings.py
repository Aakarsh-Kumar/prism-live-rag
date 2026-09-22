from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Protocol


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
DEFAULT_NEURAL_MODEL = "BAAI/bge-small-en-v1.5"
HASH_DIM = 128


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def hash_embedding(text: str, dim: int = HASH_DIM) -> list[float]:
    vector = [0.0] * dim
    for token in tokenize(text):
        digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
        bucket = int.from_bytes(digest[:4], "little") % dim
        sign = 1.0 if digest[4] % 2 == 0 else -1.0
        vector[bucket] += sign
    norm = math.sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


class Encoder(Protocol):
    """Dense text encoder. Query and passage methods are separate because
    retrieval models such as bge apply an asymmetric instruction prefix."""

    name: str
    dim: int

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        ...

    def encode_query(self, text: str) -> list[float]:
        ...


class HashEncoder:
    """Deterministic, offline, dependency-free lexical hashing.

    This is a fallback, not a semantic encoder: it approximates token overlap
    and is used when the neural backend is unavailable."""

    def __init__(self, dim: int = HASH_DIM) -> None:
        self.dim = dim
        self.name = f"hash{dim}"

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        return [hash_embedding(text, self.dim) for text in texts]

    def encode_query(self, text: str) -> list[float]:
        return hash_embedding(text, self.dim)


class FastEmbedEncoder:
    """Neural encoder backed by fastembed (ONNX Runtime).

    `fastembed` is an optional dependency; it is imported lazily so the core
    package keeps working without it."""

    def __init__(
        self,
        model_name: str = DEFAULT_NEURAL_MODEL,
        *,
        cache_dir: Path | None = None,
        local_files_only: bool = False,
        batch_size: int = 256,
    ) -> None:
        from fastembed import TextEmbedding

        self.model_name = model_name
        self.batch_size = batch_size
        self._model = TextEmbedding(
            model_name=model_name,
            cache_dir=str(cache_dir) if cache_dir is not None else None,
            local_files_only=local_files_only,
        )
        self.dim = len(next(iter(self._model.passage_embed(["dimension probe"]))))
        self.name = f"{_slug(model_name)}-{self.dim}"

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        return [
            vector.tolist()
            for vector in self._model.passage_embed(texts, batch_size=self.batch_size)
        ]

    def encode_query(self, text: str) -> list[float]:
        return list(self._model.query_embed(text))[0].tolist()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def build_encoder(
    backend: str = "auto",
    *,
    model: str = DEFAULT_NEURAL_MODEL,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
    hash_dim: int = HASH_DIM,
) -> Encoder:
    """Build a dense encoder.

    - ``hash``      -> always the offline hashing fallback.
    - ``fastembed`` -> neural; raises if fastembed/model is unavailable.
    - ``auto``      -> neural when available, otherwise the hash fallback.
    """
    backend = (backend or "auto").lower()
    if backend == "hash":
        return HashEncoder(hash_dim)
    if backend == "fastembed":
        return FastEmbedEncoder(
            model, cache_dir=cache_dir, local_files_only=local_files_only
        )
    if backend != "auto":
        raise ValueError(f"Unknown embedding backend: {backend!r}")
    try:
        return FastEmbedEncoder(
            model, cache_dir=cache_dir, local_files_only=local_files_only
        )
    except Exception as exc:  # noqa: BLE001 - any failure means "fall back"
        import warnings

        warnings.warn(
            f"Neural embeddings unavailable ({exc}); falling back to {HASH_DIM}-dim "
            "hash embeddings. Install with `pip install -e '.[neural]'` for semantic "
            "retrieval.",
            RuntimeWarning,
            stacklevel=2,
        )
        return HashEncoder(hash_dim)
