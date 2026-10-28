"""Explicit, resumable downloads. This module is never run during a session."""
from __future__ import annotations

import argparse
import json
import os
import tarfile
import zipfile
from pathlib import Path


def is_asset_archive(path: Path):
    if not path.name.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
        return False
    # The publisher wraps this Torch checkpoint in an outer gzip tar with the
    # same basename. Inspect its container instead of inferring from the name.
    if path.name.lower() == 'lam_audio2exp_streaming.tar':
        return tarfile.is_tarfile(path)
    return True


def safe_extract(archive: Path, destination: Path):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    def checked(name):
        target = (destination / name).resolve()
        if target != destination and destination not in target.parents:
            raise ValueError(f"归档路径越界: {name}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                checked(entry.filename)
                if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("归档含符号链接")
            source.extractall(destination)
    else:
        with tarfile.open(archive) as source:
            for entry in source.getmembers():
                checked(entry.name)
                if entry.issym() or entry.islnk() or entry.isdev():
                    raise ValueError("归档含链接或设备文件")
            source.extractall(destination)


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
        for name in files:
            path = destination/name
            if not is_asset_archive(path):
                continue
            marker = path.with_suffix(".extracted")
            if not marker.exists():
                safe_extract(path, destination)
                marker.write_text(sha256(path), encoding="ascii")
    return lock


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--only", nargs="*")
    options = parser.parse_args()
    download(options.root.resolve(), options.only)
