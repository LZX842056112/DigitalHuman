"""Audio-track ingestion. Video frames never enter the photo animation path."""
import wave
import os
import uuid
from pathlib import Path
import av
import numpy as np
from backend.errors import PipelineError
from backend.storage import sha256
from backend.photo_profiles import migrate_profile, source_path


REFERENCE_AUDIO_MAX_SECONDS = 600
# CosyVoice reads the reference at 24 kHz. Decoding the source at 16 kHz and
# letting the model upsample it back throws away everything above 8 kHz, which
# is exactly the band that carries a speaker's brightness and sibilance.
REFERENCE_RATE = 24000
REFERENCE_MIN_SECONDS = 3
# Bump when extraction changes: avatars prepared by an older pipeline rebuild
# their reference once instead of silently keeping a 16 kHz six-second clip.
REFERENCE_VERSION = 2


def read_audio(path, rate=16000, *, max_samples=None):
    chunks=[];resampler=av.AudioResampler(format='fltp',layout='mono',rate=rate)
    samples=0
    def append(frames):
        nonlocal samples
        for out in frames:
            if max_samples is not None and samples+out.samples>max_samples:
                raise PipelineError('audio','音轨解码采样超过上限，请提供不超过十分钟的声音视频')
            samples+=out.samples
            chunks.append(out.to_ndarray().reshape(-1))
    with av.open(str(path)) as container:
        if not container.streams.audio:raise PipelineError('audio','视频中没有音轨')
        for frame in container.decode(audio=0):
            append(resampler.resample(frame))
        append(resampler.resample(None))
    if not chunks:raise PipelineError('audio','音轨无法解码')
    return np.concatenate(chunks).astype(np.float32),rate


def write_wav_stream(target, pcm, rate):
    """Write mono PCM16 WAV to a filename or binary stream."""
    with wave.open(target,'wb') as output:
        output.setnchannels(1);output.setsampwidth(2);output.setframerate(rate)
        output.writeframes((np.clip(pcm,-1,1)*32767).astype('<i2').tobytes())


def write_wav(path, pcm, rate):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    write_wav_stream(str(path),pcm,rate)


def _frame_rms(pcm, rate, hop=.02):
    width=max(1,int(rate*hop))
    usable=len(pcm)//width*width
    if not usable:return np.zeros(0,np.float32)
    return np.sqrt(np.mean(pcm[:usable].reshape(-1,width)**2,axis=1)+1e-12)


def _most_typical(starts, vectors):
    """Return the window whose speaker vector sits closest to the shortlist mean.

    A music bed, a second speaker or an odd mic position pulls a window away
    from the centroid; the closest one is the most representative take.
    """
    centroid=np.mean(vectors,axis=0)
    centroid/=max(float(np.linalg.norm(centroid)),1e-8)
    order=max(range(len(starts)),key=lambda index:(float(vectors[index]@centroid),-index))
    return starts[order]


def choose_reference(pcm, rate, duration=6, *, embedder=None):
    """Pick one clean, representative window of the speaker.

    Two things go wrong with a plain loudness scan: quiet videos fail the fixed
    threshold outright, and a music bed or a second speaker can score as well as
    the person we want. So the gate is relative to the track's own level, and
    the final choice among clean windows prefers the one closest to the average
    speaker embedding - the most typical segment of this voice.
    """
    pcm=np.asarray(pcm,np.float32).reshape(-1)
    if not len(pcm) or rate<=0:raise PipelineError('audio','音轨为空')
    duration=float(duration)
    total=len(pcm)/rate
    if total<duration:duration=total
    if duration<REFERENCE_MIN_SECONDS:
        raise PipelineError('audio','需要至少三秒清晰语音')
    frames=_frame_rms(pcm,rate)
    if not len(frames):raise PipelineError('audio','音轨无法解码')
    levels=frames[frames<.98]
    level=float(np.percentile(levels,90)) if len(levels) else 0.
    if level<1e-4:raise PipelineError('audio','未找到清晰参考语音，请更换声音视频')
    gate=max(.003,level*.12)
    active=frames>gate
    candidates=[]
    for start in np.arange(0,total-duration+.001,.25):
        first=int(round(start/.02));last=min(len(frames),int(round((start+duration)/.02)))
        if last<=first:continue
        speech=float(np.mean(active[first:last]))
        segment=pcm[int(start*rate):int((start+duration)*rate)]
        clipped=float(np.mean(np.abs(segment)>.985))
        if speech<.55 or clipped>=.01:continue
        candidates.append((speech-4*clipped,float(start)))
    if not candidates:
        raise PipelineError('audio','未找到清晰参考语音，请更换声音视频')
    candidates.sort(key=lambda item:-item[0])
    best=candidates[0][1]
    # Scores tie for long stretches of clean speech, so take a spread across the
    # whole track instead of the first eight windows: the point of the embedding
    # step is to compare different parts of the video, not one small corner.
    pool=[item for item in candidates if item[0]>=candidates[0][0]-.05]
    shortlist=pool[::max(1,len(pool)//8)][:8]
    if embedder is not None and len(shortlist)>1:
        vectors=[]
        for _,start in shortlist:
            left=int(start*rate)
            try:vectors.append(embedder.embedding(pcm[left:left+int(duration*rate)],rate))
            except Exception:vectors=[];break
        if len(vectors)==len(shortlist):
            best=_most_typical([start for _,start in shortlist],vectors)
    return best,duration


def clean_reference(pcm, rate, target_rms=.08, peak_limit=.95):
    """Remove DC, soften both edges and put the prompt at a normal level.

    The flow model conditions on the raw log-mel of this clip, so a quiet
    reference pulls the generated voice thin and a hot one makes it hard.
    """
    pcm=np.asarray(pcm,np.float32).reshape(-1).copy()
    if not len(pcm) or not np.isfinite(pcm).all():raise PipelineError('audio','参考语音无效')
    pcm-=float(np.mean(pcm))
    peak=float(np.max(np.abs(pcm)))
    if peak<1e-5:raise PipelineError('audio','参考语音过轻')
    fade=min(len(pcm)//2,int(rate*.01))
    if fade>1:
        ramp=np.linspace(0.,1.,fade,dtype=np.float32)
        pcm[:fade]*=ramp;pcm[-fade:]*=ramp[::-1]
    voiced=pcm[np.abs(pcm)>peak*.05].astype(np.float64)
    rms=float(np.sqrt(np.mean(voiced**2)))
    gain=min(target_rms/max(rms,1e-9),peak_limit/peak)
    return (pcm*gain).astype(np.float32)


def _reference_transcript(root, path):
    from backend import asr
    try:
        text,language=asr.transcribe(root,path)
    except Exception as error:
        # A missing or broken recognizer must not block preparing the avatar;
        # without a transcript the cloner simply stays in instruct mode.
        return None,None,str(error)[:200]
    return (text if asr.usable(text) else None),language,None


def prepare_voice(store, avatar_id):
    from backend.config import load_config
    manifest=migrate_profile(store.manifest(avatar_id));folder=store.avatar_dir(avatar_id)
    video=source_path(folder,manifest['voice_video']);digest=sha256(video)
    if digest!=manifest['sources'][manifest['voice_video']]:raise ValueError('声音来源校验失败')
    target=folder/'voice/reference.wav';voice=manifest['voice']
    if (voice.get('source_sha256')==digest and target.is_file()
            and voice.get('reference_version')==REFERENCE_VERSION
            and voice.get('reference_sha256')==sha256(target)):return manifest
    config=load_config(store.root) or {}
    seconds=float((config.get('runtime') or {}).get('voice_reference_seconds',6) or 6)
    # The speech tokenizer refuses anything longer than 30 s.
    seconds=min(max(seconds,REFERENCE_MIN_SECONDS),30.)
    from backend.voice import similarity
    previous_hash=sha256(target) if target.is_file() else None
    pcm,rate=read_audio(video,REFERENCE_RATE,max_samples=REFERENCE_RATE*REFERENCE_AUDIO_MAX_SECONDS)
    start,duration=choose_reference(pcm,rate,seconds,embedder=similarity.load(store.root))
    segment=clean_reference(pcm[int(start*rate):int((start+duration)*rate)],rate)
    temporary=target.with_name('.reference-'+uuid.uuid4().hex+'.wav')
    try:
        write_wav(temporary,segment,rate)
        # Recognise before the swap so a failed transcript never leaves a new
        # reference behind with no way to describe it.
        text,language,error=_reference_transcript(store.root,temporary)
        os.replace(temporary,target)
    finally:temporary.unlink(missing_ok=True)
    # A legacy confirmation is retained only if the extracted waveform matches.
    confirmed=(bool(voice.get('confirmed')) and previous_hash==sha256(target)
               and voice.get('start_seconds')==start and voice.get('duration_seconds')==duration)
    manifest['voice']={key:value for key,value in voice.items()
                       if key not in {'transcription_error','text_language','text_model'}}
    manifest['voice'].update(confirmed=confirmed,reference='voice/reference.wav',source_sha256=digest,
        reference_sha256=sha256(target),start_seconds=start,duration_seconds=duration,sample_rate=rate,
        reference_version=REFERENCE_VERSION,
        text=text,text_language=language,text_model='faster-whisper-small' if text else None)
    if error:manifest['voice']['transcription_error']=error
    manifest['validation']['voice']=False
    store.write_manifest(avatar_id,manifest)
    return manifest
