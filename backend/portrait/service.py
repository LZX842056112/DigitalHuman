"""Private local photo worker. Only image sources can reach Ditto."""
import hashlib
import json
import threading
from pathlib import Path
from fastapi import FastAPI,HTTPException
from pydantic import BaseModel,Field
from backend.config import project_root
from backend.storage import Store,read_json,sha256
from backend.photo_profiles import migrate_profile,source_path
from backend.model_jobs import expired,sentence_paths
from backend.errors import NotReady
from backend.portrait.engine import PhotoEngine

ROOT=project_root();store=Store(ROOT);engine=PhotoEngine(ROOT);lock=threading.Lock()
app=FastAPI(title='Local photo animation')


class PrepareRequest(BaseModel):
    avatar_id:str


class RenderRequest(PrepareRequest):
    session_id:str
    generation:int=Field(ge=0)
    index:int=Field(ge=0)
    emotion:str='neutral'


@app.exception_handler(NotReady)
async def not_ready(request,error):
    from fastapi.responses import JSONResponse
    return JSONResponse(status_code=409,content={'detail':str(error)})


@app.get('/health')
def health():return {'status':'ok','loaded':engine.loaded,'mode':'photo'}


def profile(avatar_id):
    manifest=migrate_profile(store.manifest(avatar_id));folder=store.avatar_dir(avatar_id)
    photo=source_path(folder,manifest['photo'])
    if sha256(photo)!=manifest['sources'][manifest['photo']]:raise ValueError('照片来源校验失败')
    return manifest,folder,photo


@app.post('/prepare')
def prepare(body:PrepareRequest):
    with lock:
        try:
            manifest,folder,photo=profile(body.avatar_id)
            catalog=read_json(ROOT/'configs/photo-models.json')
            identity={'photo':sha256(photo),'model':catalog['models']['ditto']['revision'],
                      'code':catalog['sources']['ditto']['commit'],'size':1024}
            key=hashlib.sha256(json.dumps(identity,sort_keys=True).encode()).hexdigest()[:24]
            relative='portrait/'+key
            report=engine.prepare(photo,folder/relative)
            return {'folder':relative,'width':report['width'],'height':report['height'],'fps':25}
        except (ValueError,FileNotFoundError) as error:raise HTTPException(400,str(error)) from error


@app.post('/render')
def render(body:RenderRequest):
    cancelled=lambda:expired(ROOT,body.session_id,body.generation)
    with lock:
        audio,video=sentence_paths(ROOT,body.session_id,body.generation,body.index)
        try:
            if cancelled():raise InterruptedError('当前回答已停止')
            manifest,folder,photo=profile(body.avatar_id)
            cache=source_path(folder,manifest['portrait']['folder'])
            source=read_json(cache/'source.json')
            if source['metadata']['photo_sha256']!=sha256(photo):raise ValueError('照片缓存与当前照片不匹配')
            report=engine.render(cache,audio,video,body.emotion,cancelled)
            if cancelled():raise InterruptedError('当前回答已停止')
            return {'report':report}
        except InterruptedError as error:
            video.unlink(missing_ok=True);audio.unlink(missing_ok=True)
            raise HTTPException(409,str(error)) from error
        except (ValueError,FileNotFoundError,KeyError) as error:
            video.unlink(missing_ok=True)
            raise HTTPException(400,str(error)) from error
