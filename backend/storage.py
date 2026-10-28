from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path

from .errors import NotReady


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temp.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def safe_id(value: str) -> str:
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,96}", value):
        raise ValueError("无效的编号")
    return value


class Store:
    def __init__(self, root: Path):
        self.root = Path(root).resolve()
        self._lock = threading.RLock()

    def avatar_dir(self, avatar_id: str) -> Path:
        return self.root / "data" / "avatars" / safe_id(avatar_id)

    def manifest(self, avatar_id: str) -> dict:
        return read_json(self.avatar_dir(avatar_id) / "avatar.json")

    def write_manifest(self, avatar_id: str, manifest: dict) -> None:
        with self._lock:
            atomic_json(self.avatar_dir(avatar_id) / "avatar.json", manifest)

    def list_avatars(self) -> list[dict]:
        with self._lock:
            folder = self.root / "data" / "avatars"
            return [read_json(p) for p in sorted(folder.glob("*/avatar.json"))]

    def delete_avatar(self, avatar_id: str) -> None:
        with self._lock:
            folder=self.avatar_dir(avatar_id)
            base=self.root/'data/avatars'
            # Resolve the exact recursive-delete target before touching it.
            # A linked person directory must never delete another directory.
            if (folder.is_symlink() or folder.resolve()!=base.resolve()/avatar_id
                    or base.resolve()!=self.root/'data/avatars'):
                raise ValueError('人物目录路径越界或包含目录链接，不能删除')
            self.manifest(avatar_id)
            marker=self.root/'.runtime/default_avatar.txt'
            is_default=marker.is_file() and marker.read_text(encoding='utf-8-sig').strip()==avatar_id
            remaining=[a['id'] for a in self.list_avatars() if a['id']!=avatar_id] if is_default else []
            shutil.rmtree(folder)
            if is_default:
                if remaining:
                    temporary=marker.with_name(f'.default_avatar.{uuid.uuid4().hex}.tmp')
                    try:
                        temporary.write_text(safe_id(remaining[0]),encoding='ascii')
                        os.replace(temporary,marker)
                    finally:temporary.unlink(missing_ok=True)
                else:marker.unlink(missing_ok=True)

    def import_files(self, name: str, video: Path, body_photos: list[Path], face_photos: list[Path]) -> dict:
        files = [Path(video), *map(Path, body_photos), *map(Path, face_photos)]
        if not body_photos:
            raise ValueError("需要一张身体参考照片")
        if len({p.name for p in files}) != len(files):
            raise ValueError("素材文件名不能重复")
        for path in files:
            if not path.is_file():
                raise FileNotFoundError(str(path))
        avatar_id = uuid.uuid4().hex
        folder = self.avatar_dir(avatar_id) / "source"
        folder.mkdir(parents=True)
        hashes = {}
        for path in files:
            destination = folder / path.name
            shutil.copy2(path, destination)
            hashes[path.name] = sha256(destination)
            if hashes[path.name] != sha256(path):
                raise OSError(f"素材复制校验失败: {path.name}")
        manifest = {
            "schema_version": 1, "id": avatar_id, "name": name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "status": "imported", "video": Path(video).name,
            "body_photos": [p.name for p in body_photos],
            "face_photos": [p.name for p in face_photos], "sources": hashes,
            "estimated_regions": ["背面", "遮挡的手部", "服装遮挡的身体表面"],
            "voice": {"confirmed": False, "language": "zh", "text": ""},
            "validation": {"render": False, "expression": False, "voice": False},
        }
        self.write_manifest(avatar_id, manifest)
        return manifest

    def new_job(self, avatar_id: str, kind: str) -> dict:
        job = {"id": uuid.uuid4().hex, "avatar_id": avatar_id, "kind": kind,
               "status": "queued", "stage": "queued", "progress": 0, "error": None}
        atomic_json(self.root / "data" / "jobs" / f"{job['id']}.json", job)
        return job

    def job(self, job_id: str) -> dict:
        return read_json(self.root / "data" / "jobs" / f"{safe_id(job_id)}.json")

    def update_job(self, job_id: str, **updates) -> dict:
        with self._lock:
            job = self.job(job_id)
            job.update(updates)
            atomic_json(self.root / "data" / "jobs" / f"{safe_id(job_id)}.json", job)
            return job

    def require_ready(self, avatar_id: str) -> dict:
        try:
            manifest = self.manifest(avatar_id)
        except FileNotFoundError as exc:
            raise NotReady("人物尚未导入或构建") from exc
        folder = self.avatar_dir(avatar_id)
        if manifest.get("status") != "ready":
            raise NotReady("人物尚未通过重建、口型和声音检查")
        from backend.avatar.state import state_artifacts
        try:
            required = [*state_artifacts(folder), "expression_map.npz", "voice/reference.wav"]
        except (FileNotFoundError, ValueError, KeyError) as exc:
            raise NotReady('人物状态文件无效，请重新验证') from exc
        if any(not (folder / path).is_file() for path in required):
            raise NotReady("人物包缺少真实模型资产")
        if not manifest.get("voice", {}).get("confirmed") or not all(
            manifest.get("validation", {}).get(name) for name in ("render", "expression", "voice")
        ):
            raise NotReady("人物尚未通过重建、口型和声音检查")
        hashes = manifest.get('artifact_hashes', {})
        if not all(name in hashes for name in required):
            raise NotReady('人物资产缺少校验值，请重新验证')
        for name, expected in hashes.items():
            artifact = (folder / name).resolve()
            if folder.resolve() not in artifact.parents or not artifact.is_file() or sha256(artifact) != expected:
                raise NotReady(f"人物资产校验失败: {name}")
        return manifest
