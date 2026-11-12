from pathlib import Path

import yaml


def load(path: str, overrides: list[str]) -> dict:
    p = Path(path)
    if not p.is_file():
        raise SystemExit(f"Config inexistent: {p}")
    cfg = yaml.safe_load(p.read_text()) or {}
    for item in overrides:
        if "=" not in item:
            raise SystemExit(f"Override invalid '{item}', format: key=value")
        key, value = item.split("=", 1)
        # yaml.safe_load transforma textul in tipul potrivit: "50" -> 50, "true" -> True.
        cfg[key] = yaml.safe_load(value)
    return cfg
