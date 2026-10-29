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
        k, v = item.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    return cfg
