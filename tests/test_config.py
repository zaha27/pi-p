from pi_p.config import load


def test_overrides_are_typed(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("epochs: 100\nimgsz: 640\n")
    assert load(str(p), ["imgsz=1024", "cache=true", "name=x"]) == {"epochs": 100, "imgsz": 1024, "cache": True, "name": "x"}
