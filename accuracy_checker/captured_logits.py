"""Compatibility loader for one captured vLLM logits JSON.

The capture format is intentionally permissive: existing tools use slightly
different names (``positions``/``token_positions`` and ``logits``/
``captured_logits``).  The normalized object always has the recorded input
ids and positions; full vocabulary logits are optional.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import torch


_ID_KEYS = ("input_ids", "tokens")
_MASK_KEYS = ("attention_mask", "attn_mask", "key_padding_mask")
_POSITION_KEYS = ("token_positions", "positions", "position_ids", "position")
_LOGITS_KEYS = ("logits", "captured_logits", "vllm_logits", "vocab_logits")
_TOPK_KEYS = ("topk", "top_k_logits", "logprobs", "token_logprobs", "top_k")


@dataclass
class CapturedToken:
    token_id: int
    value: Optional[float] = None
    probability: Optional[float] = None
    token_str: str = ""


@dataclass
class CapturedLogits:
    input_ids: torch.Tensor
    token_positions: List[int]
    attention_mask: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None  # normalized [N, vocab], CPU fp32
    topk: Optional[List[List[CapturedToken]]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    source_path: str = ""

    @property
    def has_full_logits(self) -> bool:
        return self.logits is not None and self.logits.ndim == 2


def _first(mapping: Dict[str, Any], keys: Tuple[str, ...]):
    for key in keys:
        if key in mapping and mapping[key] is not None:
            return mapping[key]
    return None


def _unwrap(data: Any) -> Dict[str, Any]:
    if isinstance(data, dict):
        for key in ("data", "capture", "result", "payload"):
            child = data.get(key)
            if isinstance(child, dict) and (
                _first(child, _ID_KEYS) is not None
                or _first(child, _LOGITS_KEYS) is not None
                or _first(child, _TOPK_KEYS) is not None
            ):
                merged = dict(data)
                merged.update(child)
                return merged
        return data
    if isinstance(data, list):
        return {"records": data}
    raise ValueError("captured logits JSON must be an object or records array")


def _as_input_ids(value: Any) -> torch.Tensor:
    if isinstance(value, dict):
        value = _first(value, _ID_KEYS)
    if value is None:
        raise ValueError("captured logits JSON missing input_ids")
    ids = torch.as_tensor(value, dtype=torch.long)
    if ids.ndim == 1:
        ids = ids.unsqueeze(0)
    if ids.ndim != 2 or ids.shape[1] == 0:
        raise ValueError("captured input_ids must have shape [B,S]")
    return ids.cpu()


def _records_payload(records: Any):
    if not isinstance(records, list) or not records:
        return None
    positions: List[int] = []
    logits: List[Any] = []
    topk: List[Any] = []
    ids = None
    for index, record in enumerate(records):
        if not isinstance(record, dict):
            continue
        if ids is None:
            ids = _first(record, _ID_KEYS)
        position = _first(record, _POSITION_KEYS)
        positions.append(int(index if position is None else position))
        logits.append(_first(record, _LOGITS_KEYS))
        topk.append(_first(record, _TOPK_KEYS))
    if not positions or all(value is None for value in logits + topk):
        return None
    return ids, positions, logits, topk


def _normalize_positions(raw: Any, count: int) -> List[int]:
    if raw is None:
        return list(range(count))
    if isinstance(raw, int):
        return [int(raw)]
    positions = [int(x) for x in raw]
    if len(positions) != count:
        raise ValueError(
            f"captured positions count ({len(positions)}) != logits rows ({count})"
        )
    return positions


def _normalize_full_logits(raw: Any, positions: List[int], input_ids: torch.Tensor):
    if raw is None:
        return None
    if isinstance(raw, dict):
        # Some collectors serialize rows as {"position": [vocab logits]}.
        # Preserve the requested position order instead of relying on JSON
        # object insertion order.
        ordered = []
        for position in positions:
            value = raw.get(str(position), raw.get(position))
            if value is None:
                raise ValueError(f"captured logits missing position {position}")
            ordered.append(value)
        raw = ordered
    logits = torch.as_tensor(raw, dtype=torch.float32)
    if logits.ndim == 3:
        if logits.shape[0] != 1:
            raise ValueError("captured full logits currently support batch size 1")
        logits = logits[0]
    if logits.ndim != 2:
        raise ValueError("captured full logits must have shape [N,V] or [1,N,V]")
    if logits.shape[0] == input_ids.shape[1] and positions:
        if min(positions) < 0 or max(positions) >= logits.shape[0]:
            raise ValueError("captured position is outside the logits sequence")
        logits = logits[positions]
    elif logits.shape[0] != len(positions):
        raise ValueError(
            f"captured logits rows ({logits.shape[0]}) != positions ({len(positions)})"
        )
    return logits.cpu()


def _token_from_item(item: Any) -> Optional[CapturedToken]:
    if isinstance(item, dict):
        tid = _first(item, ("token_id", "id", "token"))
        if isinstance(tid, (list, dict)) or tid is None:
            return None
        value = _first(item, ("logit", "value", "score", "logprob"))
        probability = _first(item, ("probability", "prob", "p"))
        if probability is None and "logprob" in item and value is not None:
            # vLLM commonly stores Top-K log-probabilities.  Keep the raw
            # value for diagnostics and derive a display-only probability.
            import math
            probability = math.exp(float(value))
        return CapturedToken(
            token_id=int(tid),
            value=float(value) if value is not None else None,
            probability=float(probability) if probability is not None else None,
            token_str=str(item.get("token_str") or item.get("text") or ""),
        )
    if isinstance(item, (list, tuple)) and len(item) >= 2:
        return CapturedToken(token_id=int(item[0]), value=float(item[1]))
    return None


def _normalize_topk(raw: Any) -> Optional[List[List[CapturedToken]]]:
    if raw is None:
        return None
    # ``top_k`` is also a generation-config integer; it is not a row of
    # captured candidates unless an array/object is supplied.
    if isinstance(raw, (int, float, str)):
        return None
    rows = raw if isinstance(raw, list) else [raw]
    out: List[List[CapturedToken]] = []
    for row in rows:
        if isinstance(row, dict):
            # Common map form: {"123": {"logit": ...}, ...}
            items = []
            for key, value in row.items():
                if isinstance(value, dict):
                    value = dict(value)
                    value.setdefault("token_id", key)
                    items.append(value)
                else:
                    items.append((int(key), value))
        else:
            items = row if isinstance(row, list) else [row]
        tokens = [token for token in (_token_from_item(x) for x in items) if token]
        out.append(tokens)
    return out


def load_captured_logits(path: str) -> CapturedLogits:
    with open(path, encoding="utf-8") as handle:
        raw = json.load(handle)
    payload = _unwrap(raw)
    record_data = _records_payload(payload.get("records"))

    if record_data is not None:
        record_ids, positions, record_logits, record_topk = record_data
        input_ids = _as_input_ids(
            _first(payload, _ID_KEYS) if _first(payload, _ID_KEYS) is not None else record_ids
        )
        logits_raw = record_logits
        topk_raw = record_topk
    else:
        input_ids = _as_input_ids(_first(payload, _ID_KEYS))
        logits_raw = _first(payload, _LOGITS_KEYS)
        topk_raw = _first(payload, _TOPK_KEYS)
        positions_raw = _first(payload, _POSITION_KEYS)
        count = len(logits_raw) if isinstance(logits_raw, list) else 0
        if positions_raw is None and isinstance(topk_raw, list):
            count = len(topk_raw)
        positions = _normalize_positions(positions_raw, count or input_ids.shape[1])

    # Record-style logits are already one row per position.  For ordinary
    # [B,S,V] captures, _normalize_full_logits selects the recorded positions.
    if record_data is not None:
        if all(item is None for item in logits_raw):
            logits = None
        else:
            logits = torch.as_tensor(logits_raw, dtype=torch.float32)
            if logits.ndim == 3 and logits.shape[0] == 1:
                logits = logits[0]
            if logits.ndim != 2:
                raise ValueError("record logits must be one vector per position")
            logits = logits.cpu()
    else:
        logits = _normalize_full_logits(logits_raw, positions, input_ids)

    topk = _normalize_topk(topk_raw)
    if logits is None and topk is None:
        raise ValueError(
            "captured logits JSON must contain full logits or Top-K fields"
        )
    if topk is not None and len(topk) != len(positions):
        raise ValueError("captured Top-K rows count does not match positions")
    metadata = {
        key: payload[key]
        for key in ("model", "tokenizer", "tokenizer_path", "prompt", "generation_config", "temperature", "top_p", "top_k")
        if key in payload
    }
    attention_mask = _first(payload, _MASK_KEYS)
    if attention_mask is not None:
        attention_mask = torch.as_tensor(attention_mask, dtype=torch.long)
        if attention_mask.ndim == 1:
            attention_mask = attention_mask.unsqueeze(0)
        if attention_mask.shape != input_ids.shape:
            raise ValueError("captured attention_mask must have same shape as input_ids")
        attention_mask = attention_mask.cpu()
    return CapturedLogits(
        input_ids=input_ids,
        attention_mask=attention_mask,
        token_positions=positions,
        logits=logits,
        topk=topk,
        metadata=metadata,
        source_path=path,
    )


__all__ = ["CapturedToken", "CapturedLogits", "load_captured_logits"]
