"""Load a Ceptron Temporal Causal model (built on TCPFN) and ask it which variables of a time series cause which.

    from ceptron.temporal_causal import CausalModel

    model = CausalModel.from_pretrained("ProfitOpsAI/ceptron", model="XS-discovery")
    result = model.discover(df)            # one column per variable, one row per time step
    for e in result.edges:
        print(e["cause"], "->", e["effect"], "lag", e["lag"], round(e["probability"], 2))

The network decides the graph itself: its graph-level head says whether there is any edge, and if so the top
``round(sum of pair probabilities)`` pairs are marked. Nothing is fitted on the request and there is no statistical
fallback: an input the checkpoint was not trained for is refused with a ValueError.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from .._hub import WEIGHTS_FILE, model_folder, read_config
from ._decision import RULE, decide_matrix, expected_count, served_window
from ._model import JointPanelScorer

FAMILY = "temporal_causal"
#: the window grid this package implements (checked against each checkpoint's config)
WINDOW_POLICY = "all_half_overlap_v2_tail"
#: links listed in ``ranked_links`` when fewer are marked
DISPLAY_LINKS = 25


def _device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


@dataclass
class Discovery:
    """One discovery answer.

    ``edges``: the marked links, strongest first: cause, effect, lag (time steps), probability, logit.
    ``ranked_links``: the strongest links (all marked ones, at least 25) with their rank and whether they are marked.
    ``decision``: how the marked set was chosen (the graph head's logit, the expected edge count, the windows read).
    ``pair_logits``: (V, V) edge logit of every ordered pair, cause = row, effect = column; NaN on the diagonal.
    ``lags``: (V, V) most likely lag of every ordered pair.
    Probabilities are the network's readouts, not calibrated probabilities.
    """
    variables: list
    edges: list
    ranked_links: list
    decision: dict
    pair_logits: np.ndarray
    lags: np.ndarray

    def __repr__(self) -> str:
        lines = [f"Discovery: {len(self.edges)} edges over {len(self.variables)} variables "
                 f"(graph head logit {self.decision['graph_any_edge_logit']:+.2f})"]
        lines += [f"  {e['cause']} -> {e['effect']}  lag {e['lag']}  p {e['probability']:.2f}" for e in self.edges]
        return "\n".join(lines)

    def to_frame(self):
        """The marked edges as a pandas DataFrame (needs pandas)."""
        import pandas as pd
        return pd.DataFrame(self.edges, columns=["cause", "effect", "lag", "probability", "logit"])


class CausalModel:
    """A Ceptron Temporal Causal model, ready to answer ``discover``."""

    def __init__(self, network: JointPanelScorer, config: dict, device: torch.device, path: Path):
        self.network, self.config, self.device, self.path = network, config, device, path
        self.name = config["name"]
        serving = config["serving"]
        self.window = int(serving["window"])
        self.small_window = int(serving["small_window"]) if serving.get("small_window") else None
        self.col_type_codes = dict(serving["col_type_codes"])

    # ── loading ──

    @classmethod
    def from_pretrained(cls, name_or_path: str, *, model: str | None = None, revision: str | None = None,
                        token: str | None = None, cache_dir: str | None = None, device: str = "auto") -> "CausalModel":
        """A local model folder (holding config.json and model.safetensors), or a Hugging Face repo id with
        ``model=`` naming the model under ``models/temporal_causal/`` (downloaded once, then read from the cache)."""
        return cls.load(model_folder(name_or_path, FAMILY, model, revision=revision, token=token,
                                     cache_dir=cache_dir), device=device)

    @classmethod
    def load(cls, folder, device: str = "auto") -> "CausalModel":
        folder = Path(folder)
        config = read_config(folder, FAMILY)
        weights = folder / WEIGHTS_FILE
        serving = config["serving"]
        if serving.get("window_policy") != WINDOW_POLICY or serving.get("decision_rule") != RULE:
            raise ValueError(f"{config['name']} needs a newer ceptron (window policy "
                             f"{serving.get('window_policy')!r}, decision rule {serving.get('decision_rule')!r})")
        m = config["model"]
        if not m.get("graph_head"):
            raise ValueError(f"{config['name']} does not make its own edge decision; ceptron cannot serve it")
        net = JointPanelScorer.from_config({"model_type": config["model_type"], "model": m})
        if net.model_config["model"] != m:
            raise ValueError(f"{config['name']} needs a newer ceptron (architecture settings "
                             f"{sorted(set(m) ^ set(net.model_config['model']))} are not served)")
        from safetensors.torch import load_file
        dev = _device(device)
        net.load_state_dict(load_file(str(weights)), strict=True)
        return cls(net.to(dev).eval(), config, dev, folder)

    # ── discovery ──

    def window_starts(self, n_rows: int, window: int) -> list:
        """Where each read window starts: one tail-aligned window when the series is no longer than the training
        window (or its half-overlapping grid collapses to one), else windows overlapping by half."""
        p = self.network.patch_len
        w = min(window, n_rows) // p * p
        if w <= 0:
            raise ValueError(f"a series of {n_rows} steps is shorter than one patch ({p} steps)")
        if n_rows <= window:
            return [n_rows - w]
        n = max(1, (n_rows - w) // max(1, w // 2) + 1)
        if n == 1:
            return [n_rows - w]
        stride = (n_rows - w) / (n - 1)
        return sorted({int(round(min(n_rows - w, i * stride))) for i in range(n)})

    def _inputs(self, data, variables, col_types, time_gap):
        if hasattr(data, "columns"):                      # a pandas DataFrame
            if variables is None:
                variables = [str(c) for c in data.columns]
            if not all(np.issubdtype(t, np.number) for t in data.dtypes):
                raise ValueError("every column must be numeric: encode categories and drop the timestamp column")
            data = data.to_numpy()
        x = np.asarray(data, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError("expected one (time, variable) table: one row per time step, one column per variable")
        t, v = x.shape
        names = list(variables) if variables is not None else [f"var_{i}" for i in range(v)]
        if len(names) != v or len(set(names)) != v or not all(isinstance(n, str) for n in names):
            raise ValueError("variables must be unique strings, one per column")
        if v < 2 or t < self.network.patch_len:
            raise ValueError(f"discovery needs at least two columns and {self.network.patch_len} rows")
        if not np.isfinite(x).all():
            raise ValueError("missing or non-finite cells: this checkpoint has no trained missing-cell input, "
                             "so they are refused rather than filled")
        gaps = None
        if time_gap is not None:
            if not self.network.time_gap_input:
                raise ValueError("this checkpoint has no time-gap input")
            gaps = np.asarray(time_gap, dtype=np.float32)
            if gaps.shape != (t,) or not np.isfinite(gaps).all() or np.any(gaps < 0):
                raise ValueError("time_gap needs one finite, nonnegative gap per row, in steps of the regular clock")
        types = np.zeros(v, dtype=np.int64)
        if col_types is not None:
            if isinstance(col_types, dict):
                unknown = set(col_types) - set(names)
                if unknown:
                    raise ValueError(f"col_types names unknown columns: {sorted(unknown)}")
                col_types = [col_types.get(n, "continuous") for n in names]
            if len(col_types) != v:
                raise ValueError("col_types needs one type per column")
            codes = []
            for c in col_types:
                if isinstance(c, str):
                    if c not in self.col_type_codes:
                        raise ValueError(f"column type {c!r} is not served; one of {sorted(self.col_type_codes)}")
                    c = self.col_type_codes[c]
                codes.append(c)
            types = np.asarray(codes)
            if (not np.issubdtype(types.dtype, np.integer) or np.any(types < 0)
                    or np.any(types >= self.network.n_col_types)):
                raise ValueError("col_types must be type names or valid integer type codes")
            if np.any(types) and not self.network.categorical_input:
                raise ValueError("this checkpoint has no column-type input")
        return x, names, gaps, types

    @torch.inference_mode()
    def discover(self, data, variables=None, *, col_types=None, time_gap=None, target=None) -> Discovery:
        """Which variables cause which, at which lag.

        ``data``: a pandas DataFrame or a (time, variable) array, rows in time order on a regular clock.
        ``variables``: column names (taken from the DataFrame when omitted).
        ``col_types``: {column: type} or one type per column; types are the keys of ``config['serving']
        ['col_type_codes']`` (continuous, ordinal, binary, nominal, count, bounded); unnamed columns are continuous.
        ``time_gap``: for an irregular clock, the gap before each row in steps of the regular clock.
        ``target``: report only the links into this column (the decision is still made over all pairs).
        """
        x, names, gaps, types = self._inputs(data, variables, col_types, time_gap)
        t, v = x.shape
        if target is not None and target not in names:
            raise ValueError("target must name one of the columns")
        net = self.network
        w0 = served_window(v, self.window, self.small_window)
        starts = self.window_starts(t, w0)
        w = min(w0, t) // net.patch_len * net.patch_len
        sums = {}
        for start in starts:
            xw = torch.from_numpy(np.ascontiguousarray(x[start:start + w]))[None].to(self.device)
            xw = net.standardize_panel(xw)
            kw = {}
            if net.categorical_input:
                kw["col_type"] = torch.as_tensor(types, device=self.device)[None]
            if gaps is not None:
                kw["time_gap"] = torch.as_tensor(gaps[start:start + w], device=self.device)[None]
            z, h = net.encode_pool(xw, **kw)
            out = net.score_pairs_all(z, h=h)
            values = {"edge": out["edge_logits"][0].float(), "lag": out["lag_logits"][0].float().sigmoid(),
                      "any_edge": out["any_edge_logit"][0].float()}
            for key, value in values.items():
                if key not in sums:
                    sums[key] = value.clone()
                else:
                    sums[key].add_(value)
            del out, values, z, h
        logits = (sums.pop("edge") / len(starts)).cpu().numpy()
        lag_mean = sums.pop("lag") / len(starts)
        if not torch.isfinite(lag_mean).all().item():
            raise ValueError("non-finite lag scores")
        lags = lag_mean.argmax(-1).cpu().numpy()
        any_edge = float((sums.pop("any_edge") / len(starts)).item())
        return self._answer(logits, lags, names, any_edge, target, w0, len(starts))

    def _answer(self, logits, lags, names, any_edge, target, window, n_windows) -> Discovery:
        v = len(names)
        off = ~np.eye(v, dtype=bool)
        if not np.isfinite(logits[off]).all():
            raise ValueError("non-finite edge scores")
        probabilities = np.exp(-np.logaddexp(0., -logits.astype(np.float64)))
        np.fill_diagonal(probabilities, 0.)
        eligible = off.copy()
        if target is not None:
            eligible[:, np.arange(v) != names.index(target)] = False
        selected = eligible & decide_matrix(logits, any_edge)
        indices = np.argwhere(eligible)
        order = np.argsort(-logits[eligible], kind="stable")
        edges, ranked = [], []
        for rank, (i, j) in enumerate(indices[order[:max(DISPLAY_LINKS, int(selected.sum()))]], 1):
            link = dict(cause=names[i], effect=names[j], lag=int(lags[i, j]),
                        probability=float(probabilities[i, j]), logit=float(logits[i, j]))
            if selected[i, j]:
                edges.append(link)
            ranked.append(dict(link, rank=rank, marked=bool(selected[i, j])))
        pair_logits = logits.astype(np.float64)
        np.fill_diagonal(pair_logits, np.nan)
        decision = dict(rule=RULE, graph_any_edge_logit=any_edge, expected_edge_count=expected_count(logits[off]),
                        window_steps=int(window), window_policy=WINDOW_POLICY, n_windows=int(n_windows),
                        model=self.name)
        return Discovery(variables=names, edges=edges, ranked_links=ranked, decision=decision,
                         pair_logits=pair_logits, lags=lags)
