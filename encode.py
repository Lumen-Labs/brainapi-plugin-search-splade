from __future__ import annotations

import os
from typing import Callable, Optional, Protocol

DEFAULT_MODEL = "naver/splade-cocondenser-ensembledistil"

SparseEncoder = Callable[[str], dict[str, float]]

_encode: Optional[SparseEncoder] = None
_model_name = os.getenv("SEARCH_SPLADE_MODEL", DEFAULT_MODEL)
_load_error: Optional[str] = None


class SpladeEncoder(Protocol):
    def encode(self, text: str) -> dict[str, float]:
        ...


def set_encoder(fn: Optional[SparseEncoder]) -> None:
    global _encode, _load_error
    _encode = fn
    _load_error = None


def model_name() -> str:
    return _model_name


def status() -> dict:
    return {
        "plugin": "search-splade",
        "channel": "plugin:splade",
        "model": _model_name,
        "loaded": _encode is not None,
        "error": _load_error,
    }


def encode_text(text: str) -> dict[str, float]:
    encoder = _ensure_encoder()
    return encoder(text or "")


def _ensure_encoder() -> SparseEncoder:
    global _encode, _load_error
    if _encode is not None:
        return _encode
    try:
        import torch
        from transformers import AutoModelForMaskedLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(_model_name)
        model = AutoModelForMaskedLM.from_pretrained(_model_name)
        model.eval()
        skip = {
            tokenizer.cls_token,
            tokenizer.sep_token,
            tokenizer.pad_token,
            tokenizer.unk_token,
        }

        def _run(text: str) -> dict[str, float]:
            tokens = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=256,
            )
            with torch.no_grad():
                logits = model(**tokens).logits
                weights, _ = torch.max(torch.log1p(torch.relu(logits)), dim=1)
            vector = weights.squeeze(0)
            sparse: dict[str, float] = {}
            nz = torch.nonzero(vector > 0, as_tuple=False).squeeze(-1)
            if nz.ndim == 0:
                nz = nz.unsqueeze(0)
            ids = nz.tolist()
            values = vector[nz].tolist() if ids else []
            token_ids = [int(i) for i in (ids if isinstance(ids, list) else [ids])]
            for token_id, value in zip(token_ids, values):
                token = tokenizer.convert_ids_to_tokens(token_id)
                if not token or token in skip:
                    continue
                sparse[token] = float(value)
            return sparse

        _encode = _run
        _load_error = None
        return _encode
    except Exception as exc:
        _load_error = str(exc)
        raise RuntimeError(
            f"Failed to load SPLADE model {_model_name!r}: {exc}"
        ) from exc
