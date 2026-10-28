from __future__ import annotations

import asyncio
import shutil
import uuid
import threading
import time
from typing import Literal
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import project_root, load_config
from backend.errors import NotReady
from backend.storage import Store


class VoiceUpdate(BaseModel):
    text: str
    confirmed: bool
    language: str = "zh"


class BuildRequest(BaseModel):
    stage: str = "all"


class SessionRequest(BaseModel):
    avatar_id: str
    render_size: Literal[512,768,1024] | None = None


def create_app(root: Path | None = None):
    root = Path(root or project_root()).resolve()
    store = Store(root)
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="avatar-build")
    sessions = {}
    building = set()
    operation_lock = threading.RLock()
    loading = set()
    deleting = set()
    config = load_config(root)

    @asynccontextmanager
    async def lifespan(app):
        async def reap():
            while True:
                await asyncio.sleep(10)
                for key, session in list(sessions.items()):
                    if session.closed or time.monotonic()-session.last_active > config.get('runtime',{}).get('idle_timeout_seconds',900):
                        await session.close()
                        sessions.pop(key, None)
        reaper = asyncio.create_task(reap())
        yield
        reaper.cancel()
        for session in list(sessions.values()):
            await session.close()
        pool.shutdown(wait=False, cancel_futures=True)

    app = FastAPI(title="本机三维数字人", lifespan=lifespan)
    app.state.store = store

    @app.exception_handler(NotReady)
    async def not_ready(request, exc):
        from fastapi.responses import JSONResponse
        return JSONResponse(status_code=409, content={"detail": str(exc)})

    @app.get("/api/health")
    def health():
        return {"status": "ok", "root": str(root), "offline_inference": True,
                "ready_avatars": sum(1 for a in store.list_avatars() if a["status"] == "ready")}

    @app.get("/api/system")
    def system():
        from scripts.doctor import inspect
        return inspect(root, verify_hashes=False)

    @app.get("/api/avatars")
    def avatars():
        return store.list_avatars()

    @app.post("/api/avatars/import", status_code=201)
    async def import_avatar(name: str = Form("新数字人"), video: UploadFile = File(...),
                            body_photos: list[UploadFile] = File(...), face_photos: list[UploadFile] = File(default=[])):
        temporary = root / "work/uploads" / uuid.uuid4().hex
        temporary.mkdir(parents=True)
        paths = []
        try:
            for upload in [video, *body_photos, *face_photos]:
                filename = upload.filename or ""
                if not filename or "/" in filename or "\\" in filename or filename.startswith("."):
                    raise HTTPException(400, "素材文件名无效")
                if (temporary / filename).exists():
                    raise HTTPException(400, "素材文件名不能重复")
                allowed = {".mp4", ".mov", ".mkv", ".webm"} if upload is video else {".png", ".jpg", ".jpeg"}
                if Path(filename).suffix.lower() not in allowed:
                    raise HTTPException(400, "素材格式不受支持")
                path, size = temporary / filename, 0
                with path.open("wb") as stream:
                    while block := await upload.read(1024 * 1024):
                        size += len(block)
                        if size > 1024 ** 3:
                            raise HTTPException(413, "单个素材不能超过1GB")
                        stream.write(block)
                paths.append(path)
            manifest = store.import_files(name[:80], paths[0], paths[1:1+len(body_photos)], paths[1+len(body_photos):])
            job = store.new_job(manifest["id"], "import")
            store.update_job(job["id"], status="complete", stage="import", progress=1)
            return {"avatar_id": manifest["id"], "job_id": job["id"]}
        finally:
            shutil.rmtree(temporary, ignore_errors=True)

    @app.get("/api/avatars/{avatar_id}")
    def avatar(avatar_id: str):
        try:
            return store.manifest(avatar_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "人物不存在")

    @app.put("/api/avatars/{avatar_id}/voice")
    def update_voice(avatar_id: str, update: VoiceUpdate):
        with operation_lock:
            if building or sessions or loading or deleting:
                raise HTTPException(409, '请先停止构建或交互会话，再修改声音参考')
            manifest = avatar(avatar_id)
            if update.confirmed and not update.text.strip():
                raise HTTPException(400, "请填写与参考录音一致的文字")
            manifest["voice"].update(update.model_dump())
            manifest["validation"]["voice"] = False
            if manifest["status"] == "ready":
                manifest["status"] = "voice_pending"
            store.write_manifest(avatar_id, manifest)
            return manifest["voice"]

    @app.delete('/api/avatars/{avatar_id}')
    def delete_avatar(avatar_id: str):
        from backend.avatar.lease import GpuLease
        lease=GpuLease(root)
        with operation_lock:
            if building or sessions or loading or deleting:
                raise HTTPException(409,'请先停止构建或交互会话，再删除人物')
            avatar(avatar_id)
            lease.acquire()
            deleting.add(avatar_id)
        try:
            store.delete_avatar(avatar_id)
            return {'deleted':True,'avatar_id':avatar_id}
        except FileNotFoundError:
            raise HTTPException(404,'人物不存在')
        except ValueError as exc:
            raise HTTPException(400,str(exc))
        except PermissionError:
            raise HTTPException(409,'人物文件正在被占用，请关闭相关文件后重试')
        finally:
            lease.release()
            with operation_lock:deleting.discard(avatar_id)

    @app.get("/api/avatars/{avatar_id}/files/{filename:path}")
    def avatar_file(avatar_id: str, filename: str):
        folder = store.avatar_dir(avatar_id).resolve()
        path = (folder / filename).resolve()
        if folder not in path.parents or path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".wav", ".mp4"} or not path.is_file():
            raise HTTPException(404, "素材不存在")
        return FileResponse(path)

    @app.post("/api/avatars/{avatar_id}/build", status_code=202)
    def build(avatar_id: str, request: BuildRequest):
        if request.stage not in {"all", "prepare", "reconstruct", "validate"}:
            raise HTTPException(400, "未知构建阶段")
        with operation_lock:
            if building or sessions or loading or deleting:
                raise HTTPException(409, "请先停止当前构建或交互会话")
            avatar(avatar_id)
            job = store.new_job(avatar_id, request.stage)
            building.add(avatar_id)
        def run():
            try:
                from backend.pipeline.build import build_avatar
                build_avatar(store, avatar_id, job["id"], request.stage)
            except Exception as exc:
                store.update_job(job["id"], status="failed", error=f"{type(exc).__name__}: {exc}")
            finally:
                with operation_lock:
                    building.discard(avatar_id)
        pool.submit(run)
        return {"job_id": job["id"]}

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str):
        try:
            return store.job(job_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "任务不存在")

    @app.post("/api/sessions", status_code=201)
    async def new_session(request: SessionRequest):
        with operation_lock:
            if sessions or building or loading or deleting:
                raise HTTPException(409, "本机模式只允许一个构建或交互会话")
            manifest = store.require_ready(request.avatar_id)
            loading.add(request.avatar_id)
        from backend.avatar.lease import GpuLease
        lease=GpuLease(root)
        retained=False
        renderer=None
        load_task=None
        render_executor=None
        try:
            lease.acquire()
            from backend.avatar.lhmpp import LHMRenderer
            from backend.transport.session import Session
            render_executor=ThreadPoolExecutor(1,thread_name_prefix='avatar-gpu')
            load_args=(root,store.avatar_dir(request.avatar_id))
            requested_size=getattr(request,'render_size',None)
            if requested_size is not None: load_args+= (requested_size,)
            load_task=asyncio.get_running_loop().run_in_executor(render_executor,LHMRenderer.load,*load_args)
            try:
                renderer = await asyncio.shield(load_task)
            except asyncio.CancelledError:
                await asyncio.gather(load_task,return_exceptions=True)
                raise
            presets=getattr(renderer,'camera_presets',None)
            renderer.views=await asyncio.get_running_loop().run_in_executor(render_executor,presets) if callable(presets) else {'full':renderer.default_camera}
            session = Session(root, manifest, renderer, config,render_executor)
            session.gpu_lease=lease
            sessions[session.id] = session
            retained=True
            return {"session_id": session.id, 'camera':renderer.default_camera,
                    'views':renderer.views,'render_size':getattr(renderer,'render_size',512),
                    'camera_limits':getattr(renderer,'camera_limits',{'min_distance':.3})}
        finally:
            if not retained:
                candidate=renderer
                if candidate is None and load_task is not None and load_task.done() and not load_task.cancelled() and load_task.exception() is None:
                    candidate=load_task.result()
                unload=getattr(candidate,'close',None)
                if callable(unload): await asyncio.get_running_loop().run_in_executor(render_executor,unload)
                candidate=unload=None
                if render_executor is not None:
                    render_executor.shutdown(wait=True,cancel_futures=True)
                renderer=None
                load_task=None
                import gc
                gc.collect()
                import torch
                if torch.cuda.is_initialized(): torch.cuda.empty_cache()
                lease.release()
            loading.discard(request.avatar_id)

    @app.post("/api/sessions/{session_id}/offer")
    async def offer(session_id: str, payload: dict):
        if session_id not in sessions:
            raise HTTPException(404, "会话不存在")
        if sessions[session_id].peer is not None:
            raise HTTPException(409, '会话已经建立连接，请先关闭会话')
        from backend.transport.webrtc import negotiate
        return await negotiate(sessions[session_id], payload, config.get("render", {}).get("fps", 15))

    @app.delete("/api/sessions/{session_id}")
    async def close(session_id: str):
        session = sessions.get(session_id)
        if session:
            await session.close()
            sessions.pop(session_id, None)
            import gc
            gc.collect()
        return {"closed": True}

    @app.get('/api/sessions/{session_id}/metrics')
    def metrics(session_id: str):
        session = sessions.get(session_id)
        if not session:
            raise HTTPException(404, '会话不存在')
        return {'frames':session.frames, 'render_fps':session.frames/max(session.render_seconds,.001),
                'seconds':time.monotonic()-session.started, 'queue_depth':len(session.timeline._queue),
                'video_queue_depth':getattr(session,'video_queue_depth',0),
                'video_frames_dropped':getattr(session,'video_frames_dropped',0),
                'av_sync':'not_measured'}

    dist = root / "frontend/dist"
    if dist.is_dir():
        app.mount("/", StaticFiles(directory=dist, html=True), name="studio")
    return app


app = create_app()
