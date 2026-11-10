"""Photo dialogue API. Model services own their GPU operation leases."""
from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager, suppress
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
import time
from backend import clock
import uuid

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from backend.config import load_config, project_root
from backend.errors import NotReady
from backend.gpu import GpuLease
from backend.photo_profiles import (add_variant, delete_variant, migrate_profile,
    persist_photo_profile, portrait_artifacts, portrait_gc, pruned_hashes, replace_source,
    replace_variant, required_photo_artifacts, require_photo_ready, resolve_variant,
    set_active_variant, source_path, status_for, sync_active_variant, validate_photo_sources,
    verify_artifact_hashes)
from backend.storage import Store, sha256, read_json
from backend import outfit, style


class VoiceUpdate(BaseModel):
    confirmed: bool


class SettingsUpdate(BaseModel):
    tone: str | None = None
    expression: str | None = None


class OutfitRequest(BaseModel):
    variant_id: str | None = None
    mode: str = 'recolor'
    color: str | None = None
    style: str | None = None
    prompt: str | None = None
    strength: float = 1.
    label: str | None = None


class SessionRequest(BaseModel):
    avatar_id: str


async def _prepare(store, avatar_id, job_id, config):
    from backend.media import prepare_voice, write_wav
    from backend.portrait.client import PortraitClient
    from backend.voice.client import VoiceClient
    import numpy as np

    person=persist_photo_profile(store,avatar_id)
    folder=store.avatar_dir(avatar_id)
    person.update(status='preparing',last_job_id=job_id)
    store.write_manifest(avatar_id,person)
    store.update_job(job_id,status='running',stage='audio',progress=.1)
    validate_photo_sources(folder,person)
    person=prepare_voice(store,avatar_id)

    def reusable(variant):
        if not variant.get('portrait'):return False
        try:
            verify_artifact_hashes(folder,person,
                portrait_artifacts(folder,person,root=store.root,variant=variant))
            ready=read_json(source_path(folder,variant['portrait']['folder'])/'ready.json')
            return ready.get('video_format')=='h264_rgb_lossless_v1'
        except (OSError,ValueError,KeyError,TypeError):
            return False

    client=PortraitClient(config.get('portrait',{}).get('base_url','http://127.0.0.1:9881'))
    pending=[variant for variant in person['variants'] if not reusable(variant)]
    total=max(1,len(pending))
    for index,variant in enumerate(pending):
        store.update_job(job_id,stage='photo',message=f'准备造型 {index+1}/{total}',
            progress=.35+.25*index/total)
        variant['portrait']=await client.prepare(avatar_id,variant['id'])
        variant['validation']={'photo':True}
        sync_active_variant(person)
        person['artifact_hashes']={**person.get('artifact_hashes',{}),
            **{name:sha256(source_path(folder,name)) for name in
               portrait_artifacts(folder,person,root=store.root,variant=variant)}}
        store.write_manifest(avatar_id,person)
    sync_active_variant(person)
    store.update_job(job_id,stage='voice',progress=.6)
    store.write_manifest(avatar_id,person)
    if not person['voice'].get('confirmed'):
        person['status']='voice_pending'
        person['validation']['voice']=False
        store.write_manifest(avatar_id,person)
        store.update_job(job_id,status='waiting',stage='reference_confirmation',progress=.6)
        return
    store.update_job(job_id,stage='voice',progress=.7)
    pcm,rate=await VoiceClient(config.get('voice',{}).get('base_url','http://127.0.0.1:9880')).synthesize(
        '你好，我们可以开始聊天了。',folder,person['voice'],emotion='neutral')
    pcm=np.asarray(pcm,dtype=np.float32)
    if (pcm.ndim!=1 or not pcm.size or not np.isfinite(pcm).all()
            or np.max(np.abs(pcm))<1e-4 or not isinstance(rate,int) or rate<=0):
        raise ValueError('声音服务未返回可播放的PCM音频')
    write_wav(folder/'voice/sample.wav',pcm,rate)
    person['validation']['voice']=True
    required=required_photo_artifacts(folder,person,root=store.root)
    person['artifact_hashes']={**person.get('artifact_hashes',{}),
        **{name:sha256(source_path(folder,name)) for name in required}}
    person['artifact_hashes']=pruned_hashes(person)
    person['status']=status_for(person)
    store.write_manifest(avatar_id,person)
    require_photo_ready(store,avatar_id)
    store.update_job(job_id,status='complete',stage='complete',progress=1)


def create_app(root=None):
    root=Path(root or project_root()).resolve()
    store=Store(root);config=load_config(root) or {}
    pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='photo-prepare')
    operation_lock=threading.RLock()
    sessions={};building=set();loading=set();deleting=set();outfitting=set()

    def busy():
        if building or sessions or loading or deleting or outfitting:
            raise HTTPException(409,'请先停止准备任务或交互会话')

    async def close_session(key,session):
        try:
            await session.close()
        finally:
            with operation_lock:sessions.pop(key,None)

    @asynccontextmanager
    async def lifespan(app):
        # Preparation runs in this process. Persisted active work cannot survive
        # a restart, but its source files and completed caches can be reused.
        for job_path in (root/'data/jobs').glob('*.json'):
            try:
                job=store.job(job_path.stem)
            except (OSError,ValueError):
                continue
            if job.get('status') in {'queued','running'}:
                store.update_job(job_path.stem,status='failed',
                    error='服务已停止，准备任务已中断。请重新准备人物。')
        for person in store.list_avatars():
            if (person.get('schema_version') in {2,3} and person.get('mode')=='photo'
                    and person.get('status')=='preparing'):
                person['status']='failed'
                store.write_manifest(person['id'],person)
        async def reap():
            while True:
                await asyncio.sleep(10)
                with operation_lock:current=list(sessions.items())
                for key,session in current:
                    timeout=config.get('runtime',{}).get('idle_timeout_seconds',900)
                    if session.closed or clock.monotonic()-session.last_active>timeout:
                        with suppress(Exception):await close_session(key,session)
        reaper=asyncio.create_task(reap())
        try:
            yield
        finally:
            reaper.cancel()
            with suppress(asyncio.CancelledError):await reaper
            for key,session in list(sessions.items()):
                with suppress(Exception):await close_session(key,session)
            pool.shutdown(wait=False,cancel_futures=True)

    app=FastAPI(title='本机照片数字人',lifespan=lifespan)
    app.state.store=store

    @app.exception_handler(NotReady)
    async def not_ready(request,error):
        return JSONResponse(status_code=409,content={'detail':str(error)})

    def avatar(avatar_id):
        try:return migrate_profile(store.manifest(avatar_id))
        except (OSError,ValueError,KeyError,IndexError):raise HTTPException(404,'人物不存在或资料无效')

    async def upload_file(upload,temporary,role,limit=1024**3):
        filename=upload.filename or ''
        if not filename or '/' in filename or '\\' in filename or ':' in filename or filename.startswith('.'):
            raise HTTPException(400,'素材文件名无效')
        allowed={'.png','.jpg','.jpeg'} if role=='photo' else {'.mp4','.mov','.mkv','.webm'}
        if Path(filename).suffix.lower() not in allowed:raise HTTPException(400,'素材格式不受支持')
        path=temporary/role/filename;path.parent.mkdir(parents=True,exist_ok=True)
        size=0
        with path.open('wb') as stream:
            while block:=await upload.read(1024*1024):
                size+=len(block)
                if size>limit:raise HTTPException(413,f'单个素材不能超过{limit//1024**2}MB')
                stream.write(block)
        if size==0:raise HTTPException(400,'素材文件为空')
        return path

    @app.get('/api/health')
    def health():
        return {'status':'ok','root':str(root),'offline_inference':True,
            'ready_avatars':sum(migrate_profile(a)['status']=='ready' for a in store.list_avatars())}

    @app.get('/api/system')
    def system():
        from scripts.doctor import inspect
        return inspect(root,verify_hashes=False)

    @app.get('/api/avatars')
    def avatars():return [migrate_profile(a) for a in store.list_avatars()]

    @app.get('/api/avatars/{avatar_id}')
    def get_avatar(avatar_id:str):return avatar(avatar_id)

    @app.post('/api/avatars/import',status_code=201)
    async def import_avatar(name:str=Form('新数字人'),photo:UploadFile=File(...),voice_video:UploadFile=File(...)):
        temporary=root/'work/uploads'/uuid.uuid4().hex
        try:
            image=await upload_file(photo,temporary,'photo')
            video=await upload_file(voice_video,temporary,'voice_video')
            with operation_lock:
                busy()
                person=store.import_photo(name,image,video)
                job=store.new_job(person['id'],'import')
                person['last_job_id']=job['id'];store.write_manifest(person['id'],person)
                store.update_job(job['id'],status='complete',stage='import',progress=1)
            return {'avatar_id':person['id'],'job_id':job['id']}
        finally:shutil.rmtree(temporary,ignore_errors=True)

    async def replace(avatar_id,role,upload):
        with operation_lock:busy();avatar(avatar_id)
        temporary=root/'work/uploads'/uuid.uuid4().hex
        try:
            path=await upload_file(upload,temporary,role)
            with operation_lock:
                busy()
                return replace_source(store,avatar_id,role,path)
        finally:shutil.rmtree(temporary,ignore_errors=True)

    @app.put('/api/avatars/{avatar_id}/photo')
    async def update_photo(avatar_id:str,photo:UploadFile=File(...)):
        return await replace(avatar_id,'photo',photo)

    @app.put('/api/avatars/{avatar_id}/voice-video')
    async def update_video(avatar_id:str,voice_video:UploadFile=File(...)):
        return await replace(avatar_id,'voice_video',voice_video)

    @app.put('/api/avatars/{avatar_id}/voice')
    def update_voice(avatar_id:str,update:VoiceUpdate):
        with operation_lock:
            busy();avatar(avatar_id)
            person=persist_photo_profile(store,avatar_id)
            person['voice']['confirmed']=update.confirmed
            person['validation']['voice']=False
            if person['status']=='ready':person['status']='voice_pending'
            store.write_manifest(avatar_id,person)
            return person['voice']

    def variant_limit():
        return int((config.get('runtime') or {}).get('max_photo_variants') or 8)

    async def photo_upload(upload):
        temporary=root/'work/uploads'/uuid.uuid4().hex
        try:return await upload_file(upload,temporary,'photo',limit=20*1024**2),temporary
        except BaseException:
            shutil.rmtree(temporary,ignore_errors=True);raise

    @app.post('/api/avatars/{avatar_id}/variants',status_code=201)
    async def create_variant(avatar_id:str,photo:UploadFile=File(...),label:str=Form('')):
        with operation_lock:busy();avatar(avatar_id)
        path,temporary=await photo_upload(photo)
        try:
            with operation_lock:
                busy();avatar(avatar_id)
                person=add_variant(store,avatar_id,path,label or None,limit=variant_limit())
            return person
        except ValueError as error:raise HTTPException(409,str(error)) from error
        finally:shutil.rmtree(temporary,ignore_errors=True)

    def outfit_model_ready():
        return (root/'models/outfit/inpaint/model_index.json').is_file()

    def outfit_adapter_ready():
        return (outfit_model_ready()
                and (root/'models/outfit/ip-adapter/models'/outfit.ADAPTER_WEIGHT).is_file()
                and (root/'models/outfit/ip-adapter/models/image_encoder/model.safetensors').is_file())

    def run_outfit(photo,out,spec,timeout):
        """One photo edit runs in its own process: torch is released on exit."""
        env={**os.environ,'PYTHONPATH':str(root),'DIGITAL_HUMAN_ROOT':str(root),
             'HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1'}
        command=[sys.executable,'-m','backend.outfit','--photo',str(photo),'--out',str(out),
                 '--spec',json.dumps(spec,ensure_ascii=False)]
        result=subprocess.run(command,cwd=str(root),env=env,capture_output=True,text=True,
                              timeout=timeout)
        lines=[line for line in (result.stdout or '').strip().splitlines() if line.strip()]
        if not lines:
            raise HTTPException(500,f'换装进程没有输出：{(result.stderr or "")[-200:]}')
        try:report=json.loads(lines[-1])
        except json.JSONDecodeError:
            raise HTTPException(500,'换装进程返回了无法解析的结果') from None
        if report.get('error'):raise HTTPException(400,report['error'])
        return report

    @app.get('/api/outfit/options')
    def outfit_options():
        return {'colors':[{'name':name,'value':code,'label':outfit.COLOR_LABELS[name]}
                          for name,code in outfit.COLORS.items()],
                'styles':[{'name':name,'label':outfit.STYLE_LABELS[name]}
                          for name in outfit.STYLES],
                'generate':outfit_model_ready(),
                'wear':outfit_adapter_ready()}

    def outfit_subject(avatar_id, variant_id):
        """Validate the avatar, pick the variant and return its source photo."""
        with operation_lock:
            busy();avatar(avatar_id)
            person=persist_photo_profile(store,avatar_id)
            try:variant=resolve_variant(person,variant_id or None)
            except (KeyError,ValueError) as error:
                raise HTTPException(400,'找不到这套造型') from error
            folder=store.avatar_dir(avatar_id)
            photo=source_path(folder,variant['photo'])
            if not photo.is_file() or sha256(photo)!=person['sources'].get(variant['photo']):
                raise HTTPException(400,'照片来源校验失败')
            outfitting.add(avatar_id)
        return person,photo

    def outfit_finish(avatar_id, produced, label):
        with operation_lock:
            outfitting.discard(avatar_id)
            busy()
            return add_variant(store,avatar_id,produced,label or '换装',limit=variant_limit())

    @app.post('/api/avatars/{avatar_id}/outfit/wear',status_code=201)
    async def wear_outfit(avatar_id:str,garment:UploadFile=File(...),label:str=Form(''),
                          prompt:str=Form(''),variant_id:str=Form(''),scale:float=Form(.9)):
        if not outfit_adapter_ready():
            raise HTTPException(503,'衣服参考模型尚未安装，请先换色或选款式')
        description=prompt.strip()[:80]
        person,photo=outfit_subject(avatar_id,variant_id)
        temporary=root/'work/uploads'/uuid.uuid4().hex
        try:
            path=await upload_file(garment,temporary,'photo',limit=20*1024**2)
            head=description.replace('，',',').split(',')[0][:8]
            spec={'mode':'wear','garment':str(path),
                  'describe':description or None,
                  'label':(label or (('换装·'+head) if head else '')).strip() or None,
                  'scale':max(.3,min(1.2,float(scale)))}
            report=run_outfit(photo,temporary/'out.png',spec,timeout=900)
            return outfit_finish(avatar_id,temporary/'out.png',report.get('label'))
        except ValueError as error:
            raise HTTPException(409,str(error)) from error
        except subprocess.TimeoutExpired as error:
            raise HTTPException(504,'换装超时，请换一张更小的衣服图片') from error
        finally:
            with operation_lock:outfitting.discard(avatar_id)
            shutil.rmtree(temporary,ignore_errors=True)

    @app.post('/api/avatars/{avatar_id}/outfit',status_code=201)
    def edit_outfit(avatar_id:str,body:OutfitRequest):
        if body.mode not in ('recolor','generate'):
            raise HTTPException(400,'未知的换装方式')
        if body.mode=='generate' and not outfit_model_ready():
            raise HTTPException(503,'换装模型尚未安装，当前只能换色')
        with operation_lock:
            busy();avatar(avatar_id)
            person=persist_photo_profile(store,avatar_id)
            try:variant=resolve_variant(person,body.variant_id)
            except (KeyError,ValueError) as error:
                raise HTTPException(400,'找不到这套造型') from error
            folder=store.avatar_dir(avatar_id)
            photo=source_path(folder,variant['photo'])
            if not photo.is_file() or sha256(photo)!=person['sources'].get(variant['photo']):
                raise HTTPException(400,'照片来源校验失败')
            outfitting.add(avatar_id)
        temporary=root/'work/outfit'/uuid.uuid4().hex
        try:
            temporary.mkdir(parents=True,exist_ok=True)
            spec={'mode':body.mode,'color':body.color,'style':body.style,'prompt':body.prompt,
                  'strength':body.strength,'label':body.label}
            report=run_outfit(photo,temporary/'out.png',spec,
                              timeout=900 if body.mode=='generate' else 180)
            with operation_lock:
                outfitting.discard(avatar_id)
                busy()
                person=add_variant(store,avatar_id,temporary/'out.png',
                    report.get('label') or '换装',limit=variant_limit())
            return person
        except ValueError as error:
            raise HTTPException(409,str(error)) from error
        except subprocess.TimeoutExpired as error:
            raise HTTPException(504,'换装超时，请改用换色或减少采样步数') from error
        finally:
            with operation_lock:outfitting.discard(avatar_id)
            shutil.rmtree(temporary,ignore_errors=True)

    @app.put('/api/avatars/{avatar_id}/variants/{variant_id}')
    async def update_variant(avatar_id:str,variant_id:str,photo:UploadFile=File(...)):
        with operation_lock:busy();avatar(avatar_id)
        path,temporary=await photo_upload(photo)
        try:
            with operation_lock:
                busy();avatar(avatar_id)
                person=replace_variant(store,avatar_id,variant_id,path)
            return person
        except ValueError as error:raise HTTPException(400,str(error)) from error
        finally:shutil.rmtree(temporary,ignore_errors=True)

    @app.delete('/api/avatars/{avatar_id}/variants/{variant_id}')
    def remove_variant(avatar_id:str,variant_id:str):
        with operation_lock:
            busy();avatar(avatar_id)
            try:person=delete_variant(store,avatar_id,variant_id)
            except ValueError as error:raise HTTPException(409,str(error)) from error
        return person

    @app.put('/api/avatars/{avatar_id}/variants/{variant_id}/active')
    def activate_variant(avatar_id:str,variant_id:str):
        with operation_lock:
            busy();avatar(avatar_id)
            try:person=set_active_variant(store,avatar_id,variant_id)
            except ValueError as error:raise HTTPException(409,str(error)) from error
        return person

    @app.put('/api/avatars/{avatar_id}/settings')
    def update_settings(avatar_id:str,update:SettingsUpdate):
        with operation_lock:
            busy();avatar(avatar_id)
            person=persist_photo_profile(store,avatar_id)
            try:person['settings']=style.merge_settings(person,update.model_dump(exclude_none=True))
            except ValueError as error:raise HTTPException(400,str(error)) from error
            store.write_manifest(avatar_id,person)
            return {'settings':person['settings']}

    @app.post('/api/avatars/{avatar_id}/portrait/gc')
    def collect_portraits(avatar_id:str):
        with operation_lock:busy();avatar(avatar_id)
        return portrait_gc(store,avatar_id)

    @app.delete('/api/avatars/{avatar_id}')
    def delete_avatar(avatar_id:str):
        lease=GpuLease(root)
        with operation_lock:
            busy();avatar(avatar_id)
            lease.acquire();deleting.add(avatar_id)
        try:
            store.delete_avatar(avatar_id)
            return {'deleted':True,'avatar_id':avatar_id}
        except FileNotFoundError:raise HTTPException(404,'人物不存在')
        except ValueError as error:raise HTTPException(400,str(error))
        except PermissionError:raise HTTPException(409,'人物文件正在被占用')
        finally:
            lease.release()
            with operation_lock:deleting.discard(avatar_id)

    @app.get('/api/avatars/{avatar_id}/files/{filename:path}')
    def avatar_file(avatar_id:str,filename:str):
        try:
            avatar(avatar_id)
            path=source_path(store.avatar_dir(avatar_id),filename)
            if path.suffix.lower() not in {'.png','.jpg','.jpeg','.wav'} or not path.is_file():
                raise ValueError('素材不存在')
        except (ValueError,OSError):raise HTTPException(404,'素材不存在')
        return FileResponse(path)

    @app.post('/api/avatars/{avatar_id}/build',status_code=202)
    def build(avatar_id:str):
        with operation_lock:
            busy();avatar(avatar_id)
            person=persist_photo_profile(store,avatar_id)
            job=store.new_job(avatar_id,'prepare')
            person['last_job_id']=job['id'];store.write_manifest(avatar_id,person)
            building.add(avatar_id)
        def run():
            try:asyncio.run(_prepare(store,avatar_id,job['id'],config))
            except Exception as error:
                person=store.manifest(avatar_id);person['status']='failed'
                store.write_manifest(avatar_id,person)
                store.update_job(job['id'],status='failed',error=f'{type(error).__name__}: {error}')
            finally:
                with operation_lock:building.discard(avatar_id)
        pool.submit(run)
        return {'job_id':job['id']}

    @app.get('/api/jobs/{job_id}')
    def job(job_id:str):
        try:return store.job(job_id)
        except (ValueError,OSError):raise HTTPException(404,'任务不存在')

    @app.post('/api/sessions',status_code=201)
    async def new_session(request:SessionRequest):
        with operation_lock:
            busy()
            person=require_photo_ready(store,request.avatar_id)
            loading.add(request.avatar_id)
        session=None;retained=False;task=None
        try:
            from backend.transport.photo_session import PhotoSession
            task=asyncio.create_task(asyncio.to_thread(PhotoSession,root,person,config))
            try:session=await asyncio.shield(task)
            except asyncio.CancelledError:
                result=await asyncio.gather(task,return_exceptions=True)
                if not isinstance(result[0],BaseException):session=result[0]
                raise
            with operation_lock:sessions[session.id]=session
            retained=True
            return {'session_id':session.id,'width':person['portrait']['width'],
                'height':person['portrait']['height'],'fps':25}
        finally:
            if session is not None and not retained:await session.close()
            with operation_lock:loading.discard(request.avatar_id)

    @app.post('/api/sessions/{session_id}/offer')
    async def offer(session_id:str,payload:dict):
        session=sessions.get(session_id)
        if session is None or session.closed:raise HTTPException(404,'会话不存在')
        if getattr(session,'peer',None) is not None:raise HTTPException(409,'会话已经连接')
        from backend.transport.photo_webrtc import negotiate
        return await negotiate(session,payload,fps=25)

    @app.delete('/api/sessions/{session_id}')
    async def close(session_id:str):
        session=sessions.get(session_id)
        if session is not None:await close_session(session_id,session)
        return {'closed':True}

    @app.get('/api/sessions/{session_id}/metrics')
    def metrics(session_id:str):
        session=sessions.get(session_id)
        if session is None:raise HTTPException(404,'会话不存在')
        return {**getattr(session,'metrics',{}),
            'queue_depth':session.timeline.queue_depth,'av_sync':'not_measured'}

    dist=root/'frontend/dist'
    if dist.is_dir():app.mount('/',StaticFiles(directory=dist,html=True),name='studio')
    return app


app=create_app()
