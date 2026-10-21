"""Shared by every family: find a model folder (local or on Hugging Face) and check its weights.

A Hugging Face repo holds a root ``config.json``, the catalog of its models ({"library": "ceptron", "models":
{family: [name, ...]}}). Every load reads it: it names the models a request may ask for, and it is the file the Hub
counts downloads by (the Hub's default download query is a root config.json)."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

REPO = "ProfitOpsAI/ceptron"
CONFIG_FILE = "config.json"
WEIGHTS_FILE = "model.safetensors"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as stream:
        for block in iter(lambda: stream.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _check_catalog(root: Path, family: str, model: str, where: str):
    catalog = json.loads((root / CONFIG_FILE).read_text())
    names = (catalog.get("models") or {}).get(family, [])
    if model not in names:
        raise FileNotFoundError(f"{where} has no {family} model {model!r}; available: {sorted(names) or 'none'}")


def model_folder(name_or_path: str, family: str, model: str | None = None, *, revision: str | None = None,
                 token: str | None = None, cache_dir: str | None = None) -> Path:
    """A local model folder holding config.json, a local copy of a repo (``model=`` names the model), or a Hugging
    Face repo id with ``model=`` naming the model under ``models/<family>/`` (downloaded once, then read from the
    cache)."""
    sub = f"models/{family}/{model}" if model else None
    local = Path(name_or_path) / (sub or "")
    if (local / CONFIG_FILE).is_file():
        return local
    if not model:
        raise FileNotFoundError(f"{name_or_path} is not a model folder; for a Hugging Face repo name the model with "
                                "model=")
    from huggingface_hub import snapshot_download
    root = Path(snapshot_download(name_or_path, revision=revision, token=token, cache_dir=cache_dir,
                                  allow_patterns=[CONFIG_FILE, f"{sub}/*"]))
    if not (root / CONFIG_FILE).is_file():
        raise FileNotFoundError(f"{name_or_path} is not a ceptron repo (no root {CONFIG_FILE})")
    _check_catalog(root, family, model, name_or_path)
    return root / sub


def read_config(folder: Path, family: str) -> dict:
    """The folder's config, after checking it belongs to ``family`` and its weights match the recorded sha256."""
    config = json.loads((folder / CONFIG_FILE).read_text())
    if config.get("family") != family:
        raise ValueError(f"{folder} holds a {config.get('family')!r} model, not {family!r}")
    if sha256(folder / WEIGHTS_FILE) != config["weights_sha256"]:
        raise ValueError(f"{folder / WEIGHTS_FILE} does not match the sha256 recorded in {CONFIG_FILE}; "
                         "download it again")
    return config
