"""Private local photo worker. Only image sources can reach Ditto."""
import hashlib
import json
import threading
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from backend.config import project_root
from backend.storage import Store,read_json,sha256
from backend.photo_profiles import migrate_profile,resolve_variant,source_path
from backend.model_jobs import expired,sentence_paths
from backend.errors import NotReady
from backend.portrait.assets import portrait_version
from backend.portrait.engine import PhotoEngine
from backend.gpu import stream_owner,StreamCancellation
from backend.stream_jobs import stream_folder
from backend.stream_worker import StreamWorker,worker_lifespan

ROOT=project_root();store=Store(ROOT);engine=PhotoEngine(ROOT);lock=threading.Lock()


@asynccontextmanager
async def lifespan(application):
    from backend import warmup
    warmup.start(lambda: engine.warmup(), lock, 'portrait-warmup')
    async with worker_lifespan(application):yield


app=FastAPI(title='Local photo animation',lifespan=lifespan)


class PrepareRequest(BaseModel):
    avatar_id:str
    variant_id:str|None=None


class RenderRequest(PrepareRequest):
    session_id:str
    generation:int=Field(ge=0)
    index:int=Field(ge=0)
    emotion:str='neutral'
    expression:str='natural'


class StreamRenderRequest(RenderRequest):
    token:str=Field(min_length=64,max_length=64)


@app.exception_handler(NotReady)
async def not_ready(request,error):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=409,content={'detail':str(error),'code':getattr(error,'code',None)})


@app.get('/health')
def health():return {'status':'ok','loaded':engine.loaded,'mode':'photo'}


def profile(avatar_id,variant_id=None):
    manifest=migrate_profile(store.manifest(avatar_id));folder=store.avatar_dir(avatar_id)
    variant=resolve_variant(manifest,variant_id)
    if variant is None:raise ValueError('人物没有可用造型')
    photo=source_path(folder,variant['photo'])
    if sha256(photo)!=manifest['sources'][variant['photo']]:raise ValueError('照片来源校验失败')
    return manifest,folder,photo,variant


def cache_for(folder,variant):
    if not variant.get('portrait'):raise NotReady('该造型尚未准备')
    return source_path(folder,variant['portrait']['folder'])


@app.post('/render/stream')
def render_stream(body:StreamRenderRequest):
    stream_owner(ROOT,body.session_id,body.token)
    manifest,avatar,photo,variant=profile(body.avatar_id)
    cache=cache_for(avatar,variant)
    if read_json(cache/'source.json')['metadata']['photo_sha256']!=sha256(photo):
        raise HTTPException(400,'照片缓存与当前照片不匹配')
    folder=stream_folder(ROOT,body.session_id,body.generation,body.index)
    def produce(disconnected):
        cancelled=StreamCancellation(ROOT,body.session_id,body.generation,body.token,disconnected)
        with lock:
            yield dict(type='start',fps=25)
            yield from engine.stream_render(cache,folder,body.emotion,cancelled,
                session_id=body.session_id,token=body.token,expression=body.expression)
            if cancelled():raise InterruptedError('当前回答已停止')
            yield dict(type='complete',report=engine.last_stream_report)
    return StreamWorker(produce,dict(generation=body.generation,sentence_index=body.index),
                        job_path=folder/'portrait-job.json').response()


@app.post('/prepare')
def prepare(body:PrepareRequest):
    with lock:
        try:
            manifest,folder,photo,variant=profile(body.avatar_id,body.variant_id)
            version=portrait_version(ROOT)
            identity={'photo':sha256(photo),'model':version['model_revision'],
                      'code':version['code_commit'],'size':version['max_size']}
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
            relative='portrait/'+key
            report=engine.prepare(photo,folder/relative)
            return {'folder':relative,'width':report['width'],'height':report['height'],'fps':25}
        except (ValueError,FileNotFoundError) as error:raise HTTPException(400,str(error)) from error


@app.post('/render')
def render(body:RenderRequest):
    with lock:
        audio,video=sentence_paths(ROOT,body.session_id,body.generation,body.index)
        try:
            cancelled=StreamCancellation(ROOT,body.session_id,body.generation,None,lambda:False)
            if cancelled():raise InterruptedError('当前回答已停止')
            manifest,folder,photo,variant=profile(body.avatar_id)
            cache=cache_for(folder,variant)
            source=read_json(cache/'source.json')
            if source['metadata']['photo_sha256']!=sha256(photo):raise ValueError('照片缓存与当前照片不匹配')
            report=engine.render(cache,audio,video,body.emotion,cancelled,expression=body.expression)
            if cancelled():raise InterruptedError('当前回答已停止')
            return {'report':report}
        except InterruptedError as error:
            video.unlink(missing_ok=True);audio.unlink(missing_ok=True)
            raise HTTPException(409,str(error)) from error
        except (ValueError,FileNotFoundError,KeyError) as error:
            video.unlink(missing_ok=True)
            raise HTTPException(400,str(error)) from error
