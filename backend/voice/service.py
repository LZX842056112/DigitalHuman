"""Private local emotional voice worker; references are resolved by profile ID."""
import io
import threading
import itertools
from contextlib import asynccontextmanager
from fastapi import FastAPI,HTTPException
from fastapi.responses import Response,JSONResponse
from pydantic import BaseModel,Field
from backend.media import write_wav_stream
from backend.config import load_config, project_root
from backend.emotion import normalize_emotion
from backend.errors import NotReady
from backend.style import normalize_tone,settings_of
from backend.model_jobs import expired
from backend.storage import Store,sha256
from backend.photo_profiles import migrate_profile,source_path
from backend.voice.engine import VoiceEngine
from backend.gpu import stream_owner,StreamCancellation
from backend.stream_jobs import AudioSpool,stream_folder,check_space
from backend.stream_worker import StreamWorker,worker_lifespan

ROOT=project_root();store=Store(ROOT);engine=VoiceEngine(ROOT);lock=threading.Lock()
# hybrid keeps the reference melody on neutral turns and only falls back to an
# instruction when the answer asks for a different emotion.
CLONE_MODE=str(((load_config(ROOT) or {}).get('runtime') or {}).get('voice_clone_mode','hybrid') or 'hybrid')
# On by default: choosing a non-natural tone is an explicit request for spoken
# delivery, so neutral turns trade a little zero-shot similarity for it.
TONE_VOICE=bool(((load_config(ROOT) or {}).get('runtime') or {}).get('tone_voice',True))


def clone_settings(emotion,text,tone='natural',mode=None,tone_prompt=''):
    """Pick the cloning path: zero_shot needs a real transcript.

    zero-shot ignores the instruction, so a non-neutral tone has to move
    neutral turns to instruct2 for the tone to be audible at all. A free text
    tone counts as a tone here: without this check "natural" plus a custom
    sentence would silently stay zero-shot and change nothing.
    """
    prompt=(text or '').strip()
    mode=mode or CLONE_MODE
    if not prompt or mode=='instruct':return 'instruct',''
    if mode=='zero_shot':return 'zero_shot',prompt
    if normalize_emotion(emotion)!='neutral':return 'instruct',''
    if TONE_VOICE and (normalize_tone(tone)!='natural' or (tone_prompt or '').strip()):
        return 'instruct',''
    return 'zero_shot',prompt


@asynccontextmanager
async def lifespan(application):
    from backend import warmup
    warmup.start(lambda: engine.warmup(), lock, 'voice-warmup')
    async with worker_lifespan(application):yield


app=FastAPI(title='Local emotional voice',lifespan=lifespan)


class VoiceRequest(BaseModel):
    avatar_id:str
    text:str=Field(min_length=1,max_length=300)
    emotion:str='neutral'
    session_id:str|None=None
    generation:int|None=Field(default=None,ge=0)


class VoiceStreamRequest(VoiceRequest):
    session_id:str
    generation:int=Field(ge=0)
    index:int=Field(ge=0)
    token:str=Field(min_length=64,max_length=64)


@app.exception_handler(NotReady)
async def not_ready(request,error):
    return JSONResponse(status_code=409,content={'detail':str(error),'code':getattr(error,'code',None)})


@app.get('/health')
def health():return {'status':'ok','loaded':engine.voice is not None,'engine':'CosyVoice3',
        'clone_mode':CLONE_MODE,'tone_voice':TONE_VOICE}


@app.post('/tts/stream')
def tts_stream(body:VoiceStreamRequest):
    stream_owner(ROOT,body.session_id,body.token)
    manifest=migrate_profile(store.manifest(body.avatar_id));avatar=store.avatar_dir(body.avatar_id)
    if not manifest['voice'].get('confirmed'):raise NotReady('请先试听并确认参考声音')
    reference=source_path(avatar,manifest['voice']['reference'])
    if sha256(reference)!=manifest['voice']['reference_sha256']:raise HTTPException(400,'参考声音校验失败')
    folder=stream_folder(ROOT,body.session_id,body.generation,body.index)
    settings=settings_of(manifest)
    mode,prompt=clone_settings(body.emotion,manifest['voice'].get('text'),settings['tone'],
        tone_prompt=settings['tone_prompt'])
    def produce(disconnected):
        cancelled=StreamCancellation(ROOT,body.session_id,body.generation,body.token,disconnected)
        spool=None;native=None
        with lock:
            try:
                if cancelled():raise InterruptedError('当前回答已停止')
                check_space(ROOT,folder)
                native=iter(engine.stream_synthesize(body.text,reference,body.emotion,avatar/'voice',cancelled,
                    session_id=body.session_id,token=body.token,prompt_text=prompt,mode=mode,
                    tone=settings['tone'],tone_prompt=settings['tone_prompt']))
                first=next(native)
                spool=AudioSpool(folder,engine.voice.sample_rate)
                yield dict(type='start',rate=engine.voice.sample_rate)
                for pcm in itertools.chain([first],native):
                    if cancelled():raise InterruptedError('当前回答已停止')
                    if len(pcm):
                        check_space(ROOT,folder)
                        yield spool.append(pcm)
                report=engine.last_stream_report
                spool.finish(report=report)
                state=spool.state()
                yield dict(type='complete',samples=state['samples'],chunks=state['chunks'],report=report)
            except BaseException as error:
                if spool is not None:spool.finish(error=str(error)[:500])
                raise
            finally:
                if native is not None:native.close()
    return StreamWorker(produce,dict(generation=body.generation,sentence_index=body.index),
                        job_path=folder/'voice-job.json').response()


@app.post('/tts')
def tts(body:VoiceRequest):
    with lock:
        try:
            cancelled=StreamCancellation(ROOT,body.session_id,body.generation,None,lambda:False)
            if cancelled():raise InterruptedError('当前回答已停止')
            manifest=migrate_profile(store.manifest(body.avatar_id));folder=store.avatar_dir(body.avatar_id)
            if not manifest['voice'].get('confirmed'):raise NotReady('请先试听并确认参考声音')
            reference=source_path(folder,manifest['voice']['reference'])
            if sha256(reference)!=manifest['voice']['reference_sha256']:raise ValueError('参考声音校验失败')
            settings=settings_of(manifest)
            mode,prompt=clone_settings(body.emotion,manifest['voice'].get('text'),settings['tone'],
                tone_prompt=settings['tone_prompt'])
            pcm,rate,report=engine.synthesize(body.text,reference,body.emotion,folder/'voice',cancelled,
                prompt_text=prompt,mode=mode,tone=settings['tone'],tone_prompt=settings['tone_prompt'])
            if cancelled():raise InterruptedError('当前回答已停止')
            output=io.BytesIO()
            write_wav_stream(output,pcm,rate)
            return Response(output.getvalue(),media_type='audio/wav',headers={
                'X-Generation-Seconds':str(report['generation_seconds']),
                'X-Peak-VRAM-MiB':str(report['peak_vram_mib'])})
        except InterruptedError as error:raise HTTPException(409,str(error)) from error
        except (ValueError,FileNotFoundError,KeyError) as error:raise HTTPException(400,str(error)) from error
