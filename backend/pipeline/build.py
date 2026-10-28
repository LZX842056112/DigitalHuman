"""Build real assets and record evidence. No synthetic success or fallback avatar."""
from __future__ import annotations

import asyncio
import gc
import time
import traceback
from pathlib import Path
import numpy as np
from PIL import Image

from backend.errors import PipelineError
from backend.storage import atomic_json, read_json, sha256
from backend.avatar.lease import GpuLease

PIPELINE_VERSION = 7
RECONSTRUCTION_VERSION = 1


def build_signatures(manifest,lock):
    common={'sources':manifest['sources'],'model_lock':sha256(lock) if lock.is_file() else None}
    return ({**common,'pipeline_version':PIPELINE_VERSION},
            {**common,'reconstruction_version':RECONSTRUCTION_VERSION,'precision':'bfloat16'})


def body_checkpoint_matches(folder,signature):
    path=folder/'body_checkpoint.json'
    if not path.is_file() or not (folder/'state.json').is_file(): return False
    checkpoint=read_json(path)
    return checkpoint.get('signature')==signature and checkpoint.get('state_descriptor_sha256')==sha256(folder/'state.json')


def build_avatar(store, avatar_id, job_id, stage='all'):
    root, folder = store.root, store.avatar_dir(avatar_id)
    renderer = None
    lease=GpuLease(root)
    try:
        lease.acquire()
    except Exception as exc:
        store.update_job(job_id,status='failed',stage='gpu_busy',error=str(exc))
        raise
    def progress(name, fraction, **extra):
        store.update_job(job_id, status='running', stage=name, progress=fraction, **extra)
    try:
        progress('source_integrity', .01)
        manifest = store.manifest(avatar_id)
        manifest['last_job_id']=job_id
        store.write_manifest(avatar_id,manifest)
        for name, digest in manifest['sources'].items():
            if sha256(folder/'source'/name) != digest:
                raise PipelineError('source_integrity', f'原始素材校验失败: {name}')
        if stage == 'validate' and not (folder/'state.json').exists():
            raise PipelineError('reconstruction', '人物尚未完成三维重建')
        if stage in {'all', 'prepare', 'reconstruct'} and not (folder/'processed/faces/frames.json').exists():
            progress('media', .05)
            from .media import prepare_media
            manifest = prepare_media(store, avatar_id)
        if stage in {'all','prepare','reconstruct'}:
            from .media import append_face_photos
            manifest['face_frames']=append_face_photos(folder,manifest['face_photos'])
            store.write_manifest(avatar_id,manifest)
        if stage == 'prepare':
            store.update_job(job_id, status='complete', stage='prepared', progress=1)
            return
        from backend.avatar.lhmpp import reconstruct, LHMRenderer
        lock = root/'models/lock.json'
        signature,body_signature=build_signatures(manifest,lock)
        cached = folder/'build_signature.json'
        reusable = cached.is_file() and read_json(cached) == signature and (folder/'state.json').is_file()
        if not reusable and stage == 'validate':
            raise PipelineError('reconstruction', '人物模型与素材或权重版本不一致，请重新构建')
        if not reusable:
            manifest['status'] = 'building'
            manifest['validation'] = dict(render=False, expression=False, voice=False)
            manifest.pop('artifact_hashes', None)
            store.write_manifest(avatar_id, manifest)
            progress('body_reconstruction', .15)
            if body_checkpoint_matches(folder,body_signature):
                renderer=LHMRenderer.load(root,folder)
            else:
                renderer = reconstruct(root, folder, manifest)
                atomic_json(folder/'body_checkpoint.json',{'signature':body_signature,
                    'state_descriptor_sha256':sha256(folder/'state.json'),'source_job':job_id})
            progress('head_calibration', .40)
            from backend.avatar.calibrate import calibrate_head
            calibrate_head(root, folder, renderer)
            progress('expression_mapping', .60)
            from scripts.build_expression_map import build_expression_map
            build_expression_map(root, folder, renderer)
            # Prove unload/reload, using the persisted safe state.
            renderer.save_preview(folder)
            del renderer
            renderer = None
            gc.collect()
            import torch
            torch.cuda.empty_cache()
            atomic_json(cached, signature)
        progress('reload_and_render', .70)
        renderer = LHMRenderer.load(root, folder)
        report = validate_render(folder, renderer)
        manifest = store.manifest(avatar_id)
        manifest['validation'].update(render=True, expression=True)
        manifest['render_report'] = report
        manifest['model_versions'] = read_json(lock) if lock.is_file() else {}
        manifest['status'] = 'voice_pending'
        store.write_manifest(avatar_id, manifest)
        del renderer
        renderer = None
        gc.collect()
        import torch
        torch.cuda.empty_cache()
        if stage == 'reconstruct':
            store.update_job(job_id,status='complete',stage='reconstructed',progress=1,
                             message='三维状态和表情映射已验证，声音验证尚未执行。')
            return
        progress('reference_transcription', .85)
        if not manifest['voice'].get('text'):
            from backend.dialogue.client import Dialogue
            from backend.config import load_config
            from .media import read_audio
            pcm, _ = read_audio(folder/'voice/reference.wav')
            try:
                manifest['voice']['text'] = Dialogue(root, load_config(root)).transcribe(pcm)
            except Exception as exc:
                manifest['voice']['transcription_error'] = str(exc)
            manifest['voice']['confirmed'] = False
            store.write_manifest(avatar_id, manifest)
        if not manifest['voice'].get('confirmed'):
            store.update_job(job_id, status='waiting', stage='reference_confirmation', progress=.88,
                             message='请试听参考录音，校对文字，并确认。然后执行验证。')
            return
        progress('voice_and_audio_expression', .90)
        asyncio.run(validate_voice(root, folder, manifest))
        manifest['validation']['voice'] = True
        from backend.avatar.state import state_artifacts
        names = [*state_artifacts(folder), 'expression_map.npz', 'voice/reference.wav',
                 'face_calibration.json', 'expression_map_report.json', 'render_report.json', 'voice_report.json']
        manifest['artifact_hashes'] = {name: sha256(folder/name) for name in names}
        manifest['status'] = 'ready'
        manifest['acceptance'] = {'source_appearance': 'pending_user_review', 'voice_identity': 'pending_user_review',
                                  'ten_rounds': 'not_run', 'ten_minutes': 'not_run', 'av_sync': 'not_measured'}
        store.write_manifest(avatar_id, manifest)
        store.require_ready(avatar_id)
        store.update_job(job_id, status='complete', stage='ready', progress=1)
    except Exception as exc:
        (folder/'build_error.log').write_text(traceback.format_exc(), encoding='utf-8')
        manifest = store.manifest(avatar_id)
        manifest['status'] = 'failed'
        store.write_manifest(avatar_id, manifest)
        store.update_job(job_id, status='failed', stage=getattr(exc, 'stage', store.job(job_id)['stage']),
                         error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        if renderer is not None:
            del renderer
        gc.collect()
        lease.release()


def validate_render(folder: Path, renderer):
    import torch
    from backend.avatar.expressions import ExpressionMap
    mapping = ExpressionMap.load(folder/'expression_map.npz')
    neutral = np.zeros(100, np.float32)
    jaw = np.zeros(3, np.float32)
    camera = renderer.default_camera
    image = renderer.render('idle', 0, camera, neutral, jaw)
    if image.shape != (512,512,3) or image.std() < 2:
        raise PipelineError('render', '重新加载后的人物渲染为空或尺寸无效')
    previous = np.asarray(Image.open(folder/'front.png').convert('RGB'))
    error = float(np.abs(image.astype(float)-previous).mean())
    if error > 2:
        raise PipelineError('reload', f'保存与重新加载的图像不一致: MAE={error:.3f}')
    renderer.save_preview(folder)
    # Record every published ARKit basis as a contact sheet. These are test drives,
    # not evidence that the resulting visemes look correct to a human observer.
    from backend.avatar.head_region import head_mask
    coords=renderer.state['query_points']['neutral_coords'].reshape(-1,3)
    head=coords[head_mask(renderer.state)]
    low,high=head.min(0).values,head.max(0).values
    face_camera=dict(camera,target=((low+high)/2).cpu().tolist(),
                     distance=max(.5,float((high-low)[:2].max().cpu())*.65/np.tan(np.deg2rad(17.5))))
    face_neutral=renderer.render('idle',0,face_camera,neutral,jaw)
    Image.fromarray(face_neutral).save(folder/'face_neutral.png')
    channels=read_json(folder/'expression_map_report.json')['channels']
    drive_dir = folder/'expression_checks'
    drive_dir.mkdir(exist_ok=True)
    changes = []
    for index in range(52):
        weight = np.zeros(52, np.float32)
        weight[index] = .7
        expression, pose = mapping.apply(weight)
        driven = renderer.render('idle', 0, face_camera, expression, pose)
        Image.fromarray(driven).save(drive_dir/f'{index:02d}_{channels[index]}.png')
        changes.append(float(np.abs(driven.astype(float)-face_neutral).mean()))
    if not np.isfinite(changes).all() or max(changes) < .01:
        raise PipelineError('expression', '表情驱动未产生可测量的图像变化')
    required=['jawOpen','mouthFunnel','eyeBlinkLeft','eyeBlinkRight']
    inactive=[name for name in required if changes[channels.index(name)]<.001]
    if inactive: raise PipelineError('expression','以下口型或表情未产生有效变化: '+', '.join(inactive))
    close=np.zeros(52,np.float32);close[channels.index('jawOpen')]=.4;close[channels.index('mouthClose')]=.8
    close_expr,close_jaw=mapping.apply(close)
    Image.fromarray(renderer.render('idle',0,face_camera,close_expr,close_jaw)).save(drive_dir/'mouthClose_with_jawOpen.png')
    torch.cuda.reset_peak_memory_stats()
    times = []
    for i in range(30):
        torch.cuda.synchronize()
        start = time.perf_counter()
        renderer.render(('idle','walk','wave')[i%3], i/15, camera, neutral, jaw)
        torch.cuda.synchronize()
        times.append(time.perf_counter()-start)
    report = {'resolution':[512,512], 'frames':len(times), 'fps':1/float(np.mean(times)),
              'frame_seconds_p95':float(np.percentile(times,95)), 'reload_mae':error,
              'peak_vram_mib':torch.cuda.max_memory_allocated()/1024**2,
              'expression_pixel_changes':dict(zip(channels,changes)), 'visual_approval':'pending'}
    if getattr(renderer,'runtime_cache',None) is not None:
        report['runtime_optimizations']=dict(renderer.runtime_cache.stats)
    atomic_json(folder/'render_report.json', report)
    return report


async def validate_voice(root, folder, manifest):
    from backend.voice.client import VoiceClient
    from backend.config import load_config
    from backend.pipeline.media import write_wav
    from backend.avatar.expressions import ExpressionMap
    text = '今天我正在本机进行三维数字人测试。现在请观察我的口型和表情。'
    config = load_config(root)
    start = time.perf_counter()
    pcm, rate = await VoiceClient(config['voice']['base_url']).synthesize(text, folder, manifest['voice'])
    write_wav(folder/'voice/validation.wav', pcm, rate)
    from backend.avatar.a2e_client import request_weights
    weights = await request_weights((folder/'voice/validation.wav').read_bytes(), timeout=300)
    ExpressionMap.load(folder/'expression_map.npz').apply(weights)
    if len(weights) < 2 or np.ptp(weights) < .001:
        raise PipelineError('a2e', '音频未产生有效表情序列')
    np.save(folder/'voice/validation_arkit.npy', weights)
    atomic_json(folder/'voice_report.json', {'text':text, 'seconds':len(pcm)/rate,
                'generation_seconds':time.perf_counter()-start, 'sample_rate':rate,
                'expression_frames':len(weights), 'identity_approval':'pending'})
