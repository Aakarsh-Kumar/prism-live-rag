import importlib.util
import os

import pytest

from prism_live_rag.embeddings import (
    DEVICE_CHOICES,
    HASH_DIM,
    FastEmbedEncoder,
    HashEncoder,
    _is_cuda_oom,
    available_device,
    build_encoder,
    cuda_available,
    hash_embedding,
)


def test_hash_embedding_is_deterministic_and_normalized() -> None:
    first = hash_embedding("how do I deploy code engine")
    second = hash_embedding("how do I deploy code engine")
    assert first == second
    assert len(first) == HASH_DIM
    norm = sum(value * value for value in first) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-9)


def test_hash_encoder_exposes_query_and_passage_api() -> None:
    encoder = HashEncoder()
    assert encoder.dim == HASH_DIM
    assert encoder.name == f"hash{HASH_DIM}"
    assert encoder.device == "cpu"
    passages = encoder.encode_passages(["alpha beta", "gamma"])
    assert len(passages) == 2
    assert len(encoder.encode_query("alpha")) == HASH_DIM


def test_device_choices_and_detection_are_consistent() -> None:
    assert set(DEVICE_CHOICES) == {"auto", "cpu", "cuda"}
    assert available_device() == ("cuda" if cuda_available() else "cpu")


def test_build_encoder_rejects_unknown_device() -> None:
    with pytest.raises(ValueError):
        build_encoder("hash", device="tpu")


def test_explicit_cuda_raises_without_a_cuda_provider() -> None:
    if cuda_available():
        pytest.skip("CUDA is available; the fallback path is not exercised")
    with pytest.raises(Exception):
        build_encoder("fastembed", device="cuda")


def test_build_encoder_hash_backend() -> None:
    assert isinstance(build_encoder("hash"), HashEncoder)


def test_build_encoder_rejects_unknown_backend() -> None:
    with pytest.raises(ValueError):
        build_encoder("word2vec")


def test_auto_falls_back_to_hash_when_neural_unavailable() -> None:
    # A bogus model with local_files_only=True cannot resolve, so auto must fall
    # back regardless of whether fastembed is installed.
    encoder = build_encoder("auto", model="__prism_nonexistent__/nope", local_files_only=True)
    assert isinstance(encoder, HashEncoder)


def test_explicit_fastembed_failure_is_not_silent() -> None:
    if importlib.util.find_spec("fastembed") is None:
        with pytest.raises(Exception):
            build_encoder("fastembed")
    else:
        with pytest.raises(Exception):
            build_encoder(
                "fastembed",
                model="__prism_nonexistent__/nope",
                local_files_only=True,
            )


def test_is_cuda_oom_matches_onnxruntime_messages() -> None:
    assert _is_cuda_oom(RuntimeError("Failed to allocate memory for requested buffer"))
    assert _is_cuda_oom(RuntimeError("bfc_arena.cc:358 ..."))
    assert _is_cuda_oom(RuntimeError("CUDA out of memory"))
    assert not _is_cuda_oom(ValueError("unrelated failure"))


class _FakeVector:
    def __init__(self, value: float) -> None:
        self._value = value

    def tolist(self) -> list[float]:
        return [self._value]


class _FakeEmbedding:
    """Stands in for fastembed; OOMs at the configured batch sizes."""

    def __init__(self, fail_batch_sizes: set[int]) -> None:
        self.fail_batch_sizes = fail_batch_sizes
        self.calls: list[tuple[int, int]] = []

    def passage_embed(self, texts, batch_size):  # noqa: ANN001
        self.calls.append((len(texts), batch_size))
        if batch_size in self.fail_batch_sizes:
            raise RuntimeError("Failed to allocate memory for requested buffer of size N")
        return [_FakeVector(float(index)) for index, _ in enumerate(texts)]


def _cuda_encoder_stub(fail_batch_sizes: set[int], batch_size: int) -> FastEmbedEncoder:
    encoder = object.__new__(FastEmbedEncoder)
    encoder.device = "cuda"
    encoder.batch_size = batch_size
    encoder._use_fixed_shapes = False
    encoder._model = _FakeEmbedding(fail_batch_sizes)
    return encoder


def test_encode_passages_halves_batch_on_cuda_oom() -> None:
    encoder = _cuda_encoder_stub(fail_batch_sizes={4}, batch_size=4)
    vectors = encoder.encode_passages(["a", "b", "c", "d", "e", "f", "g", "h"])
    assert len(vectors) == 8
    model = encoder._model
    assert (4, 4) in model.calls
    assert (2, 2) in model.calls


def test_encode_passages_reraises_non_oom_errors() -> None:
    encoder = _cuda_encoder_stub(fail_batch_sizes=set(), batch_size=4)

    def boom(texts, batch_size):  # noqa: ANN001
        raise ValueError("not an OOM")

    encoder._model.passage_embed = boom
    with pytest.raises(ValueError):
        encoder.encode_passages(["a", "b"])


@pytest.mark.skipif(
    os.getenv("PRISM_TEST_NEURAL") != "1",
    reason="set PRISM_TEST_NEURAL=1 to run the neural encoder test (may download the model)",
)
def test_fastembed_encoder_dim_and_query() -> None:
    encoder = build_encoder("fastembed")
    assert isinstance(encoder, FastEmbedEncoder)
    assert encoder.dim == 384
    passages = encoder.encode_passages(["the toolchain is available"])
    assert len(passages) == 1
    assert len(passages[0]) == 384
    assert len(encoder.encode_query("toolchain availability")) == 384
