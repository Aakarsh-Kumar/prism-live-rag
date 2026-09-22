from __future__ import annotations

from .embeddings import tokenize


REGION_REWRITES = (
    (("south", "america"), ("sao", "paulo", "br", "sao")),
)


def expand_query(query: str) -> str:
    """Apply deterministic corpus-vocabulary expansions before retrieval."""
    tokens = tokenize(query)
    expanded = tokens[:]
    for phrase, replacement in REGION_REWRITES:
        for index in range(0, len(tokens) - len(phrase) + 1):
            if tuple(tokens[index : index + len(phrase)]) == phrase:
                expanded[index : index + len(phrase)] = list(replacement)
                break
    return " ".join(expanded)
