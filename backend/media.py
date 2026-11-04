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


def choose_reference(pcm, rate, duration=6):
    if len(pcm)<rate*duration:raise PipelineError('audio','需要至少六秒清晰语音')
    best=None
    for start in np.arange(0,len(pcm)/rate-duration+.01,.5):
        segment=pcm[int(start*rate):int((start+duration)*rate)]
        frames=segment[:len(segment)//320*320].reshape(-1,320)
        rms=np.sqrt(np.mean(frames**2,axis=1)+1e-12)
        active=float(np.mean(rms>.008));clipped=float(np.mean(np.abs(segment)>.98))
        score=active-clipped*5-abs(float(np.median(rms))-.1)*.1
        if active>.55 and clipped<.01 and (best is None or score>best[0]):best=(score,float(start))
    if best is None:raise PipelineError('audio','未找到清晰参考语音，请更换声音视频')
    return best[1],duration


def prepare_voice(store, avatar_id):
    manifest=migrate_profile(store.manifest(avatar_id));folder=store.avatar_dir(avatar_id)
    video=source_path(folder,manifest['voice_video']);digest=sha256(video)
    if digest!=manifest['sources'][manifest['voice_video']]:raise ValueError('声音来源校验失败')
    target=folder/'voice/reference.wav';voice=manifest['voice']
    if (voice.get('source_sha256')==digest and target.is_file()
            and voice.get('reference_sha256')==sha256(target)):return manifest
    previous_hash=sha256(target) if target.is_file() else None
    pcm,rate=read_audio(video,max_samples=16000*REFERENCE_AUDIO_MAX_SECONDS)
    start,duration=choose_reference(pcm,rate)
    temporary=target.with_name('.reference-'+uuid.uuid4().hex+'.wav')
    try:
        write_wav(temporary,pcm[int(start*rate):int((start+duration)*rate)],rate)
        os.replace(temporary,target)
    finally:temporary.unlink(missing_ok=True)
    # A legacy confirmation is retained only if the extracted waveform matches.
    confirmed=(bool(voice.get('confirmed')) and previous_hash==sha256(target)
               and voice.get('start_seconds')==start and voice.get('duration_seconds')==duration)
    manifest['voice']={key:value for key,value in voice.items() if key not in {'text','transcription_error'}}
    manifest['voice'].update(confirmed=confirmed,reference='voice/reference.wav',source_sha256=digest,
        reference_sha256=sha256(target),start_seconds=start,duration_seconds=duration,sample_rate=rate)
    manifest['validation']['voice']=False
    store.write_manifest(avatar_id,manifest)
    return manifest
