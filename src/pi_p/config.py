"""Configurări YAML din configs/, cu suprascrieri `cheie=valoare` din linia de comandă."""

from pathlib import Path

import yaml


def load(path: str, overrides: list[str]) -> dict:
    """Citește configurarea și aplică peste ea valorile date în linia de comandă."""
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"Config inexistent: {p}")
    cfg = yaml.safe_load(p.read_text()) or {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"Override invalid '{item}', format: key=value")
        key, value = item.split("=", 1)
        # yaml.safe_load transformă textul în tipul potrivit: "50" -> 50, "true" -> True.
        cfg[key] = yaml.safe_load(value)
    return cfg
