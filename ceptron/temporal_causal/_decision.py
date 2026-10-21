"""The model's own edge decision (generated)."""
from __future__ import annotations
import numpy as np

RULE = 'graph_head_topk_expected_count_v1'
SMALL_MAX_COLUMNS = 10

def served_window(n_columns: int, window: int, small_window: int | None=None) -> int:
    if small_window and int(n_columns) <= SMALL_MAX_COLUMNS:
        return int(small_window)
    return int(window)

def decide_pairs(pair_logits, any_edge_logit) -> np.ndarray:
    s = np.asarray(pair_logits, dtype=np.float64).ravel()
    if any_edge_logit is None or not np.isfinite(float(any_edge_logit)):
        raise ValueError("the model's decision needs a finite graph-head logit")
    pred = np.zeros(s.size, bool)
    if float(any_edge_logit) < 0.0 or s.size == 0:
        return pred
    p = np.exp(-np.logaddexp(0.0, -s))
    k = min(s.size, max(1, int(round(float(p.sum())))))
    pred[np.argsort(-s, kind='mergesort')[:k]] = True
    return pred

def decide_matrix(logits, any_edge_logit) -> np.ndarray:
    L = np.asarray(logits)
    off = ~np.eye(L.shape[0], dtype=bool)
    out = np.zeros(L.shape, bool)
    out[off] = decide_pairs(L[off], any_edge_logit)
    return out

def expected_count(pair_logits) -> float:
    s = np.asarray(pair_logits, dtype=np.float64).ravel()
    return float(np.exp(-np.logaddexp(0.0, -s)).sum())
