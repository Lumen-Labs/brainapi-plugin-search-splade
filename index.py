from __future__ import annotations

from collections import defaultdict
from typing import Any

from encode import encode_text

_index: dict[str, dict[str, dict[str, float]]] = {}
_texts: dict[str, dict[str, str]] = {}
_inverted: dict[str, dict[str, list[tuple[str, float]]]] = {}


def reset(brain_id: str | None = None) -> None:
    if brain_id is None:
        _index.clear()
        _texts.clear()
        _inverted.clear()
        return
    _index.pop(brain_id, None)
    _texts.pop(brain_id, None)
    _inverted.pop(brain_id, None)


def stats(brain_id: str) -> dict[str, Any]:
    docs = _index.get(brain_id) or {}
    return {
        "brain_id": brain_id,
        "n_docs": len(docs),
        "n_terms": len(_inverted.get(brain_id) or {}),
    }


def _rebuild_inverted(brain_id: str) -> None:
    postings: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for chunk_id, weights in (_index.get(brain_id) or {}).items():
        for term, weight in weights.items():
            postings[term].append((chunk_id, float(weight)))
    _inverted[brain_id] = dict(postings)


def upsert_chunk(brain_id: str, chunk_id: str, text: str) -> None:
    weights = encode_text(text)
    _index.setdefault(brain_id, {})[chunk_id] = weights
    _texts.setdefault(brain_id, {})[chunk_id] = text
    _rebuild_inverted(brain_id)


def index_chunks(brain_id: str, chunks: list[dict[str, str]]) -> dict[str, Any]:
    reset(brain_id)
    for chunk in chunks:
        chunk_id = str(chunk.get("id") or "")
        text = str(chunk.get("text") or "")
        if not chunk_id:
            continue
        _index.setdefault(brain_id, {})[chunk_id] = encode_text(text)
        _texts.setdefault(brain_id, {})[chunk_id] = text
    _rebuild_inverted(brain_id)
    return stats(brain_id)


def retrieve(
    query: str,
    brain_id: str,
    k: int,
) -> tuple[list[str], dict[str, float], dict[str, str]]:
    docs = _index.get(brain_id) or {}
    if not docs:
        return [], {}, {}
    q_weights = encode_text(query)
    scores: dict[str, float] = defaultdict(float)
    inverted = _inverted.get(brain_id) or {}
    for term, q_weight in q_weights.items():
        for chunk_id, d_weight in inverted.get(term) or []:
            scores[chunk_id] += float(q_weight) * float(d_weight)
    ranked = sorted(scores.items(), key=lambda pair: pair[1], reverse=True)
    ids = [chunk_id for chunk_id, _ in ranked[: max(1, int(k))]]
    texts = _texts.get(brain_id) or {}
    return ids, {chunk_id: scores[chunk_id] for chunk_id in ids}, {
        chunk_id: texts.get(chunk_id, "") for chunk_id in ids
    }
