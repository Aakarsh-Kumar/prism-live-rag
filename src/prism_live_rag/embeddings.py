from __future__ import annotations

import hashlib
import math
import re
from pathlib import Path
from typing import Protocol


TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")
DEFAULT_NEURAL_MODEL = "BAAI/bge-small-en-v1.5"
HASH_DIM = 128
DEVICE_CHOICES = ("auto", "cpu", "cuda")
DEFAULT_BATCH_SIZE = 64  # Conservative default for small GPU memory budgets
# Fixed sequence length for constant tensor shapes - prevents CUDA arena growth
# Set to BGE-small's maximum token length
FIXED_SEQUENCE_LENGTH = 512
_OOM_MARKERS = (
    "failed to allocate memory",
    "bfc_arena",
    "out of memory",
    "cuda_error_out_of_memory",
    "cuda out of memory",
)


def _is_cuda_oom(exc: Exception) -> bool:
    message = str(exc).lower()
    return any(marker in message for marker in _OOM_MARKERS)


def tokenize(text: str) -> list[str]:
    return [token.lower() for token in TOKEN_RE.findall(text)]


def cuda_available() -> bool:
    """True when the installed onnxruntime build exposes a CUDA provider.

    This is a build-level check: it does not prove a usable GPU is present, so
    ``device="auto"`` still falls back to CPU if the CUDA session fails to
    initialize at runtime."""
    try:
        import onnxruntime as ort
    except Exception:  # noqa: BLE001 - missing onnxruntime means no GPU either
        return False
    return "CUDAExecutionProvider" in ort.get_available_providers()


def available_device() -> str:
    return "cuda" if cuda_available() else "cpu"


def preload_gpu_libraries() -> None:
    """Make the CUDA/cuDNN libs shipped by the ``nvidia-*`` pip wheels visible.

    ``onnxruntime-gpu[cuda,cudnn]`` installs CUDA 12 and cuDNN 9 as separate pip
    packages, but its dynamic loader does not add those directories to the
    search path by default, so the CUDA provider fails with e.g.
    ``libcublasLt.so.12: cannot open shared object file``. onnxruntime exposes
    ``preload_dlls`` for exactly this. It is a no-op on CPU-only builds and on
    systems where the libraries are already resolvable."""
    try:
        import onnxruntime as ort
    except Exception:  # noqa: BLE001 - no onnxruntime means no GPU to preload
        return
    preload = getattr(ort, "preload_dlls", None)
    if preload is None:
        return
    try:
        preload()
    except Exception:  # noqa: BLE001 - missing libs surface later as CPU fallback
        pass


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
    device: str

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
        self.device = "cpu"

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
        batch_size: int = DEFAULT_BATCH_SIZE,
        device: str = "auto",
        fixed_length: int = FIXED_SEQUENCE_LENGTH,
    ) -> None:
        from fastembed import TextEmbedding

        device = (device or "auto").lower()
        if device not in DEVICE_CHOICES:
            raise ValueError(
                f"Unknown embedding device: {device!r}; expected one of {DEVICE_CHOICES}"
            )

        self.model_name = model_name
        self.batch_size = batch_size
        self.fixed_length = fixed_length
        self.requested_device = device
        self._use_fixed_shapes = device == "cuda" or (device == "auto" and cuda_available())
        
        model_kwargs: dict = {}
        if device != "cpu" and cuda_available():
            preload_gpu_libraries()
            # ORT's default kNextPowerOfTwo arena grows greedily and OOMs a 6 GB
            # card partway through a long indexing run. kSameAsRequested grows
            # exactly as needed and stays stable. CPU is kept as a fallback
            # provider so a failed CUDA session degrades instead of crashing.
            model_kwargs["providers"] = [
                ("CUDAExecutionProvider", {"arena_extend_strategy": "kSameAsRequested"}),
                "CPUExecutionProvider",
            ]
        else:
            # cuda=True surfaces a clear error when CUDA was explicitly requested
            # but is unavailable; auto/CPU fall through to CPUExecutionProvider.
            model_kwargs["cuda"] = device == "cuda"
        try:
            self._model = TextEmbedding(
                model_name=model_name,
                cache_dir=str(cache_dir) if cache_dir is not None else None,
                local_files_only=local_files_only,
                **model_kwargs,
            )
        except Exception as exc:  # noqa: BLE001
            if device == "cuda":
                raise RuntimeError(
                    f"CUDA was requested but the neural encoder could not start on GPU "
                    f"({exc}). Install a GPU build with `pip install onnxruntime-gpu`, "
                    "or use EMBEDDING_DEVICE=cpu."
                ) from exc
            raise
        self.device = _active_device(self._model, fallback=available_device())
        if device == "cuda" and self.device != "cuda":
            raise RuntimeError(
                "CUDA was requested but the ONNX session fell back to "
                "CPUExecutionProvider. Install a matching GPU build with "
                "`pip install onnxruntime-gpu`, or use EMBEDDING_DEVICE=cpu."
            )
        self.dim = len(next(iter(self._model.passage_embed(["dimension probe"]))))
        self.name = f"{_slug(model_name)}-{self.dim}"

        # For fixed-shape batching on GPU, we need access to the tokenizer and ONNX session
        if self._use_fixed_shapes:
            try:
                # Access fastembed internals safely - these paths may change in future versions
                # fastembed 0.8.0 structure: TextEmbedding.model (OnnxTextEmbedding) -> .tokenizer, .model (InferenceSession)
                fastembed_model = getattr(self._model, 'model', None)
                if fastembed_model is None:
                    raise AttributeError("Cannot access fastembed model internals")
                
                self._tokenizer = getattr(fastembed_model, 'tokenizer', None) 
                self._onnx_session = getattr(fastembed_model, 'model', None)
                
                if self._tokenizer is None or self._onnx_session is None:
                    raise AttributeError("Cannot access tokenizer or ONNX session from fastembed")
                
                # Validate that we can call the tokenizer (tokenizers.Tokenizer uses .encode_batch)
                # Test tokenization with dummy input
                if hasattr(self._tokenizer, 'encode_batch'):
                    test_result = self._tokenizer.encode_batch(["test"])
                    if not test_result or not hasattr(test_result[0], 'ids'):
                        raise ValueError("Tokenizer doesn't work as expected")
                elif hasattr(self._tokenizer, 'encode'):
                    test_result = self._tokenizer.encode("test")
                    if not hasattr(test_result, 'ids'):
                        raise ValueError("Tokenizer doesn't work as expected")
                else:
                    raise ValueError("Tokenizer doesn't have expected encode methods")
                    
            except Exception as exc:
                import warnings
                warnings.warn(
                    f"Could not access fastembed internals for fixed-shape batching ({type(exc).__name__}: {exc}); "
                    "falling back to variable-shape batches with OOM recovery. "
                    f"This may indicate a fastembed version incompatibility (current: 0.8.0).",
                    RuntimeWarning,
                    stacklevel=2,
                )
                self._use_fixed_shapes = False

    def encode_passages(self, texts: list[str]) -> list[list[float]]:
        if not self._use_fixed_shapes:
            return self._encode_passages_variable_shape(texts)
        return self._encode_passages_fixed_shape(texts)

    def _encode_passages_variable_shape(self, texts: list[str]) -> list[list[float]]:
        """Original variable-shape batching with length bucketing and OOM recovery."""
        if self.device != "cuda":
            return [
                vector.tolist()
                for vector in self._model.passage_embed(texts, batch_size=self.batch_size)
            ]
        # Bucket by length so every batch pads to a similar sequence length.
        # ONNX Runtime caches a GPU arena block per distinct tensor shape, so
        # feeding random-length batches makes the arena grow by one block per
        # shape until it exhausts a 6 GB card. Near-uniform batches keep the
        # number of shapes small and the arena stable. Order is restored below.
        order = sorted(range(len(texts)), key=lambda index: len(texts[index]))
        vectors: list[list[float] | None] = [None] * len(texts)
        for start in range(0, len(order), self.batch_size):
            positions = order[start : start + self.batch_size]
            chunk = [texts[index] for index in positions]
            for index, vector in zip(positions, self._embed_chunk(chunk, self.batch_size)):
                vectors[index] = vector
        return [vector for vector in vectors if vector is not None]

    def _encode_passages_fixed_shape(self, texts: list[str]) -> list[list[float]]:
        """Fixed-shape batching with constant padding to prevent CUDA arena growth."""
        import numpy as np
        
        vectors = []
        for start in range(0, len(texts), self.batch_size):
            batch_texts = texts[start : start + self.batch_size]
            batch_vectors = self._embed_fixed_batch(batch_texts)
            vectors.extend(batch_vectors)
        return vectors

    def _embed_fixed_batch(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch using fixed-length tokenization to maintain constant tensor shapes."""
        import numpy as np
        
        try:
            # Use tokenizers.Tokenizer.encode_batch for batch processing
            # Enable padding and truncation
            self._tokenizer.enable_padding(pad_id=0, pad_token="[PAD]", length=self.fixed_length)
            self._tokenizer.enable_truncation(max_length=self.fixed_length)
            
            # Tokenize the batch
            encoded_batch = self._tokenizer.encode_batch(texts)
            
            # Convert to numpy arrays with consistent shapes
            input_ids = np.array([encoding.ids for encoding in encoded_batch], dtype=np.int64)
            attention_mask = np.array([encoding.attention_mask for encoding in encoded_batch], dtype=np.int64)
            # BGE models typically need token_type_ids (all zeros for single sentences)
            token_type_ids = np.zeros_like(input_ids, dtype=np.int64)
            
            # Validate shapes are consistent
            batch_size = len(texts)
            expected_shape = (batch_size, self.fixed_length)
            if input_ids.shape != expected_shape:
                raise ValueError(f"Input shape {input_ids.shape} != expected {expected_shape}")
            
            # Prepare ONNX inputs - include token_type_ids for BERT-based models
            onnx_inputs = {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "token_type_ids": token_type_ids,
            }
            
            # Run inference with ONNX session
            outputs = self._onnx_session.run(None, onnx_inputs)
            
            # Robust output validation - check if this looks like sentence embeddings
            if len(outputs) == 0:
                raise ValueError("ONNX model returned no outputs")
            
            raw_embeddings = outputs[0]
            if len(raw_embeddings.shape) != 3:  # Expected: [batch_size, seq_len, hidden_size]
                # Try other outputs if first one doesn't look right
                for i, output in enumerate(outputs):
                    if len(output.shape) == 3 and output.shape[0] == batch_size:
                        raw_embeddings = output
                        break
                else:
                    raise ValueError(f"No suitable embedding output found. Shapes: {[o.shape for o in outputs]}")
            
            # Apply mean pooling over attention mask (fastembed's standard approach)
            attention_mask_expanded = np.expand_dims(attention_mask, -1)
            
            # Weighted mean pooling
            sum_embeddings = np.sum(raw_embeddings * attention_mask_expanded, axis=1)
            sum_mask = np.sum(attention_mask, axis=1, keepdims=True)
            mean_embeddings = sum_embeddings / np.maximum(sum_mask, 1e-9)
            
            # L2 normalization (standard for sentence embeddings)
            norms = np.linalg.norm(mean_embeddings, axis=1, keepdims=True)
            normalized = mean_embeddings / np.maximum(norms, 1e-9)
            
            # Validate output dimensions match encoder expectations
            if normalized.shape[1] != self.dim:
                raise ValueError(f"Output dimension {normalized.shape[1]} != expected {self.dim}")
            
            return normalized.tolist()
            
        except Exception as exc:
            # More specific error handling and fallback strategy
            import warnings
            if "input_ids" in str(exc).lower() or "tokenizer" in str(exc).lower():
                warnings.warn(
                    f"Fixed-shape tokenization failed ({exc}), falling back to variable-shape batching.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            elif "cuda" in str(exc).lower() or "memory" in str(exc).lower():
                warnings.warn(
                    f"Fixed-shape CUDA execution failed ({exc}), falling back with smaller batch.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            else:
                warnings.warn(
                    f"Fixed-shape batching failed unexpectedly ({exc}), falling back to variable-shape.",
                    RuntimeWarning,
                    stacklevel=3,
                )
            # Fall back to variable-shape embedding if fixed-shape fails
            return self._embed_chunk(texts, max(1, self.batch_size // 2))

    def _embed_chunk(self, chunk: list[str], batch_size: int) -> list[list[float]]:
        """Embed a chunk, halving the batch and retrying on a CUDA OOM.

        ONNX Runtime pads every batch to its longest sequence (up to the model's
        512-token limit), so a single long passage can spike VRAM past what a
        6 GB card holds. Halving the batch bounds that spike instead of failing
        the whole indexing run."""
        try:
            return [
                vector.tolist()
                for vector in self._model.passage_embed(chunk, batch_size=batch_size)
            ]
        except Exception as exc:  # noqa: BLE001 - only OOMs are retried
            if batch_size <= 1 or len(chunk) <= 1 or not _is_cuda_oom(exc):
                raise
            smaller = max(1, batch_size // 2)
            import warnings

            warnings.warn(
                f"CUDA out of memory at batch size {batch_size}; retrying with "
                f"{smaller}. Lower EMBEDDING_BATCH_SIZE to avoid this.",
                RuntimeWarning,
                stacklevel=2,
            )
            mid = len(chunk) // 2
            return self._embed_chunk(chunk[:mid], smaller) + self._embed_chunk(
                chunk[mid:], smaller
            )

    def encode_query(self, text: str) -> list[float]:
        return list(self._model.query_embed(text))[0].tolist()


def _slug(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "-", value).strip("-")


def _active_device(model: object, fallback: str = "cpu") -> str:
    """Read the execution provider the loaded ONNX session actually uses.

    fastembed exposes the session as ``TextEmbedding.model.model``; the chain is
    walked defensively so a fastembed refactor degrades to ``fallback`` instead
    of crashing."""
    session = model
    for attribute in ("model", "model"):
        session = getattr(session, attribute, None)
        if session is None:
            return fallback
    try:
        providers = session.get_providers()  # type: ignore[attr-defined]
    except Exception:  # noqa: BLE001
        return fallback
    return "cuda" if "CUDAExecutionProvider" in providers else "cpu"


def build_encoder(
    backend: str = "auto",
    *,
    model: str = DEFAULT_NEURAL_MODEL,
    cache_dir: Path | None = None,
    local_files_only: bool = False,
    hash_dim: int = HASH_DIM,
    device: str = "auto",
    batch_size: int = DEFAULT_BATCH_SIZE,
    fixed_length: int = FIXED_SEQUENCE_LENGTH,
) -> Encoder:
    """Build a dense encoder.

    - ``hash``      -> always the offline hashing fallback.
    - ``fastembed`` -> neural; raises if fastembed/model is unavailable.
    - ``auto``      -> neural when available, otherwise the hash fallback.

    ``device`` is ``auto`` (GPU when the onnxruntime build exposes CUDA, else
    CPU), ``cpu``, or ``cuda`` (raises if CUDA is not usable). The hash fallback
    is always CPU.
    
    ``fixed_length`` sets the constant sequence length for GPU batching to
    prevent CUDA BFC arena growth."""
    backend = (backend or "auto").lower()
    device = (device or "auto").lower()
    if device not in DEVICE_CHOICES:
        raise ValueError(
            f"Unknown embedding device: {device!r}; expected one of {DEVICE_CHOICES}"
        )
    if backend == "hash":
        return HashEncoder(hash_dim)
    if backend == "fastembed":
        return FastEmbedEncoder(
            model,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            device=device,
            batch_size=batch_size,
            fixed_length=fixed_length,
        )
    if backend != "auto":
        raise ValueError(f"Unknown embedding backend: {backend!r}")
    try:
        return FastEmbedEncoder(
            model,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
            device=device,
            batch_size=batch_size,
            fixed_length=fixed_length,
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
