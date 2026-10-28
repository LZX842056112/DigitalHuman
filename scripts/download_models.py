"""Explicit, resumable downloads. This module is never run during a session."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
# Keep historical helper imports available to existing callers.
from scripts.model_archives import (
    extract_archives, safe_extract as safe_extract, is_asset_archive as is_asset_archive,
)


def download(root: Path, selected=None):
    from huggingface_hub import HfApi, snapshot_download
    from backend.storage import sha256, atomic_json
    os.environ["HF_HOME"] = str(root / ".runtime" / "hf")
    catalog = json.loads((root / "configs" / "models.json").read_text(encoding="utf-8"))
    api = HfApi()
    lock_path = root / "models" / "lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    for key, item in catalog.items():
        if "hf_repo" not in item or selected and key not in selected:
            continue
        info = api.model_info(item["hf_repo"], revision=item.get("revision") or lock.get(key, {}).get("revision"), files_metadata=True)
        revision = info.sha
        destination = root / item["destination"]
        print(f"Download {key}: {item['hf_repo']} @ {revision}", flush=True)
        snapshot_download(item["hf_repo"], revision=revision, local_dir=str(destination),
                          allow_patterns=item.get("allow"), max_workers=3)
        files = {}
        for path in destination.rglob("*"):
            if path.is_file() and ".cache" not in path.parts:
                files[str(path.relative_to(destination)).replace("\\", "/")] = sha256(path)
        if not files:
            raise RuntimeError(f"模型下载为空: {key}")
        lock[key] = {"repo": item["hf_repo"], "revision": revision, "files": files, "license": item.get("license")}
        atomic_json(lock_path, lock)
        extract_archives(destination, files, legacy_suffix=True)
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--only", nargs="*")
    options = parser.parse_args()
    download(options.root.resolve(), options.only)
