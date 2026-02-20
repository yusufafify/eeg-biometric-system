"""
Configuration Loader

Loads parameters from a YAML file and exposes them as nested attribute access.

Usage:
    from src.utils.config import load_config
    cfg = load_config()           # loads ./config.yaml
    cfg = load_config("other.yaml")
    print(cfg.model.lstm_hidden_size)
"""

from pathlib import Path
from typing import Any, Optional

import yaml


class _ConfigNode:
    """Thin wrapper that gives attribute access to a nested dict.

    Also supports dict-like ``get()``, ``keys()``, ``items()``, and
    ``**unpacking`` so it can be passed directly as kwargs.
    """

    def __init__(self, d: dict):
        for key, val in d.items():
            if isinstance(val, dict):
                setattr(self, key, _ConfigNode(val))
            else:
                setattr(self, key, val)

    def __repr__(self) -> str:
        items = ", ".join(f"{k}={v!r}" for k, v in self.__dict__.items())
        return f"Config({items})"

    def get(self, key: str, default: Any = None) -> Any:
        return getattr(self, key, default)

    def keys(self):
        return self.__dict__.keys()

    def values(self):
        return self.__dict__.values()

    def items(self):
        return self.__dict__.items()

    def __iter__(self):
        return iter(self.__dict__)

    def to_dict(self) -> dict:
        out = {}
        for k, v in self.__dict__.items():
            out[k] = v.to_dict() if isinstance(v, _ConfigNode) else v
        return out


def load_config(path: Optional[str] = None) -> _ConfigNode:
    """
    Load a YAML configuration file and return as a Config object.

    Args:
        path: Path to YAML file.  Defaults to ``config.yaml`` in the
              project root (i.e. the current working directory).

    Returns:
        A ``_ConfigNode`` with nested attribute access.

    Raises:
        FileNotFoundError: If the YAML file does not exist.
    """
    if path is None:
        path = "config.yaml"

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p.resolve()}")

    with open(p, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f)

    return _ConfigNode(raw)
