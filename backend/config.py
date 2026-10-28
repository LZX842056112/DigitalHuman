from pathlib import Path
import os
import yaml


def project_root():
    return Path(os.environ.get("DIGITAL_HUMAN_ROOT", Path(__file__).resolve().parents[1])).resolve()


def load_config(root: Path):
    path = root / "configs" / "app.yaml"
    return yaml.safe_load(path.read_text(encoding="utf-8")) if path.exists() else {}
