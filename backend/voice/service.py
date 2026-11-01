"""Private local emotional voice worker; references are resolved by profile ID."""
import io
import threading
from fastapi import FastAPI,HTTPException
from fastapi.responses import Response,JSONResponse
from pydantic import BaseModel,Field
from backend.media import write_wav_stream
from backend.config import project_root
from backend.errors import NotReady
from backend.model_jobs import expired
from backend.storage import Store,sha256
from backend.photo_profiles import migrate_profile,source_path
from backend.voice.engine import VoiceEngine

ROOT=project_root();store=Store(ROOT);engine=VoiceEngine(ROOT);lock=threading.Lock()
app=FastAPI(title='Local emotional voice')


class VoiceRequest(BaseModel):
    avatar_id:str
    text:str=Field(min_length=1,max_length=300)
    emotion:str='neutral'
    session_id:str|None=None
    generation:int|None=Field(default=None,ge=0)


@app.exception_handler(NotReady)
async def not_ready(request,error):
    return JSONResponse(status_code=409,content={'detail':str(error)})


@app.get('/health')
def health():return {'status':'ok','loaded':engine.voice is not None,'engine':'CosyVoice3'}


@app.post('/tts')
def tts(body:VoiceRequest):
    cancelled=lambda:expired(ROOT,body.session_id,body.generation)
    with lock:
        try:
            if cancelled():raise InterruptedError('当前回答已停止')
            manifest=migrate_profile(store.manifest(body.avatar_id));folder=store.avatar_dir(body.avatar_id)
            if not manifest['voice'].get('confirmed'):raise NotReady('请先试听并确认参考声音')
            reference=source_path(folder,manifest['voice']['reference'])
            if sha256(reference)!=manifest['voice']['reference_sha256']:raise ValueError('参考声音校验失败')
            pcm,rate,report=engine.synthesize(body.text,reference,body.emotion,folder/'voice',cancelled)
            if cancelled():raise InterruptedError('当前回答已停止')
            output=io.BytesIO()
            write_wav_stream(output,pcm,rate)
            return Response(output.getvalue(),media_type='audio/wav',headers={
                'X-Generation-Seconds':str(report['generation_seconds']),
                'X-Peak-VRAM-MiB':str(report['peak_vram_mib'])})
        except InterruptedError as error:raise HTTPException(409,str(error)) from error
        except (ValueError,FileNotFoundError,KeyError) as error:raise HTTPException(400,str(error)) from error
