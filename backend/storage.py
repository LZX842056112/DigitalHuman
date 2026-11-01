from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import threading
import uuid
from pathlib import Path



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


    def import_photo(self, name: str, photo: Path, voice_video: Path) -> dict:
        from backend.photo_profiles import import_photo
        return import_photo(self,name,photo,voice_video)

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
