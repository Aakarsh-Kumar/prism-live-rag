import importlib.util
import os

import pytest

from prism_live_rag.embeddings import (
    HASH_DIM,
    FastEmbedEncoder,
    HashEncoder,
    build_encoder,
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
    passages = encoder.encode_passages(["alpha beta", "gamma"])
    assert len(passages) == 2
    assert len(encoder.encode_query("alpha")) == HASH_DIM


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
