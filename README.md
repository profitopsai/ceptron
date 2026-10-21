# Ceptron

Open foundation models, one family per sub-package. The weights are on Hugging Face at
[ProfitOpsAI/ceptron](https://huggingface.co/ProfitOpsAI/ceptron), one folder per model under `models/<family>/`.

| family | answers |
|---|---|
| `ceptron.temporal_causal` | which variables of a time series cause which, at which lag (built on TCPFN) |

## Install

Install into a virtual environment (Python 3.10 or newer). Ubuntu, Debian and Homebrew Python refuse a plain
`pip install` outside one:

```
python3 -m venv ceptron-env
source ceptron-env/bin/activate        # Windows: ceptron-env\Scripts\activate
pip install "git+https://github.com/ProfitOpsAI/ceptron" pandas
```

On Ubuntu or Debian, if `python3 -m venv` says ensurepip is not available, run `sudo apt install python3-venv` first.
Or with [uv](https://docs.astral.sh/uv/), which needs no system packages:

```
uv venv ceptron-env
source ceptron-env/bin/activate
uv pip install "git+https://github.com/ProfitOpsAI/ceptron" pandas
```

## Temporal causal discovery

```python
import pandas as pd
from ceptron.temporal_causal import CausalModel

model = CausalModel.from_pretrained("ProfitOpsAI/ceptron", model="XS-discovery")

df = pd.read_csv("my_series.csv")          # one column per variable, one row per time step
result = model.discover(df.drop(columns=["timestamp"]))
print(result)                              # the marked edges
result.to_frame()                          # the same as a DataFrame
```

A Ceptron Temporal Causal model reads the whole series and answers in one forward pass: no fitting and no
statistical tests at answer time.

`discover` takes:

* `data`: a DataFrame or a (time, variable) array, rows in time order on a regular clock, numeric cells, nothing
  missing.
* `col_types`: optional, `{column: type}` with type one of `continuous`, `ordinal`, `binary`, `nominal`, `count`,
  `bounded`; columns not named are continuous.
* `time_gap`: optional, for an irregular clock: the gap before each row, in steps of the regular clock.
* `target`: optional, report only the links into this column.

It returns `edges` (the marked links: cause, effect, lag, probability, logit), `ranked_links` (the strongest links,
at least 25, with their rank), `decision` (how the marked set was chosen), `pair_logits` (every ordered pair) and
`lags`.

The model decides the graph itself: a graph-level head says whether there is any edge, and if so the top
`round(sum of pair probabilities)` pairs are marked. Probabilities are the network's readouts, not calibrated
probabilities. An input the model was not trained for (missing cells, text columns) is refused, never filled or
guessed.

`device="cpu"` or `"cuda"` in `from_pretrained` picks the device; by default CUDA is used when present. A local folder
holding `config.json` and `model.safetensors` loads with `CausalModel.from_pretrained("path/to/folder")`.

## License

The code in this repository is licensed under [Apache 2.0](https://www.apache.org/licenses/LICENSE-2.0) (see
`LICENSE` and `NOTICE`). The model weights on Hugging Face are licensed separately under
[CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/): anyone may use, study, modify and share them for
non-commercial purposes, including research, teaching and personal projects. Commercial use of the models needs a
separate license from ProfitOps Inc.
