"""Qualify same-session GPU sharing with real native voice and portrait workers.

This is a memory probe, not playback, quality, or latency acceptance.
Run only after existing conversations and builds have ended.
"""
import argparse
import asyncio
import copy
import hashlib
import json
import math
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from backend import clock
from backend.gpu import SessionGpuLease, stream_code_key
from backend.storage import atomic_json


CASES = [
    ('neutral', '今天我们一起检查声音和画面的连续生成过程，请保持自然的语速，先说明天气变化时应该怎样安排日常活动，再介绍出门前需要准备的物品，最后提醒大家按时休息并保持轻松的心情，让每一句话都清楚完整。'),
    ('happy', '很高兴再次见到你，希望今天的交流能带来新的想法。'),
    ('sad', '遇到困难时可以先休息一会儿，再慢慢整理自己的想法。'),
    ('angry', '请认真检查每一个步骤，发现问题后要及时说明原因。'),
]
# The cold portrait load can finish after speech. Repeat the long sentence
# with both models warm so the probe must exercise concurrent computation.
CASES.insert(1, CASES[0])


def admission_record(root, gpu_uuid, idle_used_mib, samples, cases):
    if not samples or not cases:
        raise RuntimeError('缺少显存样本或双模型结果')
    if any(row['free_mib'] < 1024 for row in samples):
        raise RuntimeError('采样期间显存余量不足1 GiB')
    peaks = [];overlaps=[]
    for case in cases:
        timing = case.get('last_turn_timing') or {}
        if (case.get('delivery_mode') != 'streaming_parallel'
                or timing.get('status') != 'completed' or timing.get('stream_retries', 0)):
            raise RuntimeError('双模型并行采样未完成，或发生了串行回退')
        pair = [(timing.get(name) or {}).get('peak_vram_mib')
                for name in ('voice_worker', 'portrait_worker')]
        if any(not isinstance(value, (int, float)) or value <= 0 for value in pair):
            raise RuntimeError('缺少声音或画面模型的显存峰值')
        peaks.append(pair)
        voice=(timing.get('voice_worker') or {}).get('timing_spans',{}).get('inference_ms')
        portrait=(timing.get('portrait_worker') or {}).get('timing_spans',{})
        if voice:
            for name in ('motion_ms','frame_gpu_ms'):
                span=portrait.get(name)
                if span:
                    values=[*voice,*span]
                    if (len(values)!=4 or any(type(value) not in (int,float) or not math.isfinite(value) for value in values)
                            or voice[1]<voice[0] or span[1]<span[0]):
                        raise RuntimeError('原生任务活动区间无效')
                    overlaps.append(max(0.,min(voice[1],span[1])-max(voice[0],span[0])))
    overlap=max(overlaps,default=0.)
    if overlap<=0:raise RuntimeError('未测到声音与画面模型的实际并行活动区间')
    # CUDA allocator peaks omit driver contexts; retain an additional 512 MiB.
    bound = max(pair[0] for pair in peaks) + max(pair[1] for pair in peaks) + 512
    measured = max(row['used_mib'] for row in samples) - idle_used_mib
    return dict(status='passed', qualification_version=2, overlap_seconds=overlap,
                code_key=stream_code_key(root), gpu_uuid=gpu_uuid,
                peak_delta_mib=max(measured, bound), min_free_mib=min(row['free_mib'] for row in samples),
                cases=len(cases), samples=len(samples), qualified_at=time.strftime('%Y-%m-%dT%H:%M:%S'))


def gpu_sample():
    result = subprocess.run(['nvidia-smi', '--query-gpu=uuid,memory.used,memory.free',
        '--format=csv,noheader,nounits'], capture_output=True, text=True, timeout=5, check=True)
    identity, used, free = result.stdout.strip().splitlines()[0].split(',')
    return dict(gpu_uuid=identity.strip(), used_mib=int(used), free_mib=int(free), seconds=clock.monotonic())


def require_idle(root):
    import httpx
    # Historical markers can outlive their process. Ask the live API about them.
    with httpx.Client(base_url='http://127.0.0.1:7860', timeout=5, trust_env=False) as client:
        for marker in (root/'data/sessions').glob('*/generation.json'):
            if json.loads(marker.read_text()).get('closed'):
                continue
            try:
                response = client.get(f'/api/sessions/{marker.parent.name}/metrics')
            except httpx.ConnectError:
                break
            if response.status_code == 200:
                raise RuntimeError('已有对话连接。请先结束该对话')
            response.raise_for_status() if response.status_code != 404 else None
    for path in (root/'data/jobs').glob('*.json'):
        if json.loads(path.read_text()).get('status') in {'queued', 'running'}:
            raise RuntimeError('已有构建任务，不能开始显存采样')


async def profile(root, avatar_id, folder):
    from backend.config import load_config
    from backend.emotion import Reply
    from backend.transport.photo_session import PhotoSession

    require_idle(root)
    source = root/'data/avatars'/avatar_id/'avatar.json'
    before = hashlib.sha256(source.read_bytes()).hexdigest()
    manifest = json.loads(source.read_text())
    if manifest['status'] != 'ready' or not manifest['voice']['confirmed']:
        raise RuntimeError('显存采样需要已确认声音的就绪人物')
    folder.mkdir(parents=True, exist_ok=False)
    report = dict(status='running', avatar_id=avatar_id, avatar_manifest_sha256=before,
                  code_key=stream_code_key(root), samples=[], cases=[], purpose='memory_admission_only')
    first = await asyncio.to_thread(gpu_sample)
    report['samples'].append(first)
    stopped = asyncio.Event()

    async def sample():
        while not stopped.is_set():
            report['samples'].append(await asyncio.to_thread(gpu_sample))
            await asyncio.sleep(.1)

    sampler = asyncio.create_task(sample())
    instance = None
    try:
        config = copy.deepcopy(load_config(root))
        config.setdefault('runtime', {})['photo_streaming'] = True
        for emotion, text in CASES:
            events = []
            instance = await asyncio.to_thread(PhotoSession, root, manifest, config)
            # Explicitly granted only inside this controlled probe. Ordinary
            # sessions must use stream_admission and never bypass qualification.
            instance._stream_lease = await asyncio.to_thread(
                lambda: SessionGpuLease(root, instance.id, concurrent=True).acquire())
            instance._delivery_mode = 'streaming_parallel'
            instance.channel = SimpleNamespace(readyState='open', send=lambda raw: events.append(json.loads(raw)))

            async def answer(*args):
                return Reply(text, emotion)

            instance.dialogue = SimpleNamespace(answer=answer)
            instance.message(json.dumps(dict(type='chat_text', text='固定显存采样问题'), ensure_ascii=False))

            async def playback():
                started = clock.monotonic()
                count = 0
                while not instance.task.done():
                    instance.timeline.read_audio(960)
                    if count % 2 == 0:
                        await asyncio.to_thread(instance.timeline.video_frame)
                    count += 1
                    await asyncio.sleep(max(0, started + count*.02 - clock.monotonic()))

            driver = asyncio.create_task(playback())
            try:
                await asyncio.wait_for(asyncio.shield(instance.task), 600)
                await driver
                metrics = instance.metrics
                row = dict(emotion=emotion, text=text, events=events, **metrics)
                report['cases'].append(row)
                if any(event.get('type') == 'error' for event in events):
                    raise RuntimeError('真实模型采样失败：' + str(events[-1]))
                print(f"{emotion}: {metrics['delivery_mode']} {metrics['last_turn_timing']}", flush=True)
            finally:
                driver.cancel()
                await asyncio.gather(driver, return_exceptions=True)
                await instance.close()
                instance = None
                atomic_json(folder/'report.json', report)
        record = admission_record(root, first['gpu_uuid'], first['used_mib'], report['samples'], report['cases'])
        report.update(status='passed', admission=record)
        if hashlib.sha256(source.read_bytes()).hexdigest() != before:
            raise RuntimeError('采样期间人物记录发生变化')
        if stream_code_key(root) != report['code_key']:
            raise RuntimeError('采样期间生成代码发生变化')
        atomic_json(root/'.runtime/stream-admission.json', record)
        return report
    except BaseException as error:
        report.update(status='failed', error=f'{type(error).__name__}: {error}')
        raise
    finally:
        try:
            if instance is not None:
                await instance.close()
        finally:
            stopped.set()
            await sampler
            atomic_json(folder/'report.json', report)


def main():
    from backend.config import project_root
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=project_root())
    parser.add_argument('--avatar', required=True)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    folder = args.output or root/'outputs/streaming-profile'/time.strftime('%Y%m%d-%H%M%S')
    asyncio.run(profile(root, args.avatar, folder))


if __name__ == '__main__':
    main()
