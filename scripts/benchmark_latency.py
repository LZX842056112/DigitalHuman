"""Measure first non-silent WebRTC reception; never certify browser audibility."""
import argparse
import asyncio
import copy
import fractions
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import time

import av
import httpx
import numpy as np
from aiortc import MediaStreamTrack,RTCConfiguration,RTCPeerConnection,RTCSessionDescription

from backend import clock
from backend.media import read_audio,write_wav
from backend.storage import atomic_json
from scripts.acceptance import QUESTIONS
from scripts import services


class SpeechMeter:
    def __init__(self):
        self.started=self.first=self.last_speech=None
        self.last_audio=None
        self.continuous_since=None

    def arm(self,started):
        self.started=started;self.first=None

    def observe(self,data):
        now=clock.monotonic()
        if self.last_audio is None or now-self.last_audio>.1:self.continuous_since=now
        self.last_audio=now
        scale=np.iinfo(data.dtype).max if np.issubdtype(data.dtype,np.integer) else 1
        if np.max(np.abs(data.astype(float)/scale))>.001:
            self.last_speech=now
            if self.started is not None and now>=self.started and self.first is None:self.first=now

    @property
    def first_ms(self):
        return None if self.first is None else (self.first-self.started)*1000

    @property
    def quiet_for(self):
        now=clock.monotonic()
        if self.last_audio is None or now-self.last_audio>.1:return 0.
        since=self.continuous_since
        if self.last_speech is not None:since=max(since,self.last_speech)
        return max(0.,now-since)


async def wait_for_quiet(meter,timeout=10):
    started=clock.monotonic()
    while True:
        elapsed=clock.monotonic()-started
        if min(elapsed,meter.quiet_for)>=.5:return
        if elapsed>=timeout:raise TimeoutError('Fresh continuous audio did not establish a quiet interval')
        await asyncio.sleep(.02)


def load_audio_fixtures(folder):
    folder=Path(folder).resolve();rows=json.loads((folder/'manifest.json').read_text())
    if not isinstance(rows,list) or len(rows)!=len(QUESTIONS):raise ValueError('Expected ten fixed audio questions')
    verified=[]
    for question,row in zip(QUESTIONS,rows):
        relative=Path(row['file']);path=(folder/relative).resolve()
        if row['question']!=question:raise ValueError('Audio question order changed')
        if relative.is_absolute() or folder not in path.parents or path.suffix.lower()!='.wav':
            raise ValueError('Audio fixture path is outside its input directory')
        if hashlib.sha256(path.read_bytes()).hexdigest()!=row['sha256']:raise ValueError('Audio fixture SHA-256 mismatch')
        verified.append({**row,'file':str(path)})
    return verified


def summarize_warm(rows):
    warm=[row for row in rows if row['phase']=='warm']
    failed=sum(row['status']!='completed' or row.get('first_non_silent_receive_ms') is None for row in warm)
    values=sorted(row['first_non_silent_receive_ms'] for row in warm) if warm and not failed else []
    return {'samples':len(warm),'failed':failed,
            'median_ms':statistics.median(values) if values else None,
            'p95_ms':float(values[math.ceil(.95*len(values))-1]) if values else None}


class RecordedMicrophone(MediaStreamTrack):
    kind='audio'

    def __init__(self):
        super().__init__();self.pcm=np.zeros(0,np.int16);self.position=self.pts=0
        self.started=None;self.finished=asyncio.Event()

    async def recv(self):
        if self.started is None:self.started=clock.monotonic()
        await asyncio.sleep(max(0,self.started+self.pts/48000-clock.monotonic()))
        values=self.pcm[self.position:self.position+960];self.position+=len(values)
        pcm=np.zeros(960,np.int16);pcm[:len(values)]=values
        if self.position>=len(self.pcm):self.finished.set()
        frame=av.AudioFrame.from_ndarray(pcm[None],format='s16',layout='mono')
        frame.sample_rate=48000;frame.pts=self.pts;frame.time_base=fractions.Fraction(1,48000)
        self.pts+=960
        return frame

    def play(self,pcm):
        self.pcm=(np.clip(pcm,-1,1)*32767).astype(np.int16);self.position=0;self.finished.clear()


def resource_snapshot(root):
    records=json.loads((root/'.runtime/services.json').read_text());rss={}
    for name,record in records.items():
        try:
            lines=Path(f'/proc/{record["pid"]}/status').read_text().splitlines()
            rss[name]=int(next(line for line in lines if line.startswith('VmRSS:')).split()[1])
        except (OSError,StopIteration):rss[name]=None
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,timeout=5)
    if result.returncode:raise RuntimeError(result.stderr)
    return {'seconds':clock.monotonic(),'rss_kib':rss,
            'system_vram_mib':int(result.stdout.strip().splitlines()[0])}


def source_fingerprint(root):
    paths=[path for directory in ('backend','scripts','configs','frontend/src')
           for path in (root/directory).rglob('*') if path.suffix in {'.py','.yaml','.json','.ts','.tsx'}]
    return {path.relative_to(root).as_posix():hashlib.sha256(path.read_bytes()).hexdigest()
            for path in sorted(paths)}


async def measure(root,avatar_id,cases,folder):
    """Collect one connection's samples. Full cold/warm policy belongs to main."""
    clock.install_rtc_clock();folder.mkdir(parents=True)
    report={'status':'running','avatar_id':avatar_id,'clock_source':clock.source_name(),
            'measurement':'control send to first non-silent received audio frame',
            'browser_playback':'not_measured','physical_microphone':False,'samples':[],
            'system_vram_sample_peak_mib':0,'resources':[]}
    peer=RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    microphone=RecordedMicrophone();meter=SpeechMeter();events=asyncio.Queue()
    tasks=[];errors=[];sid=None;stopping=False
    async def receive(track):
        try:
            while True:
                frame=await track.recv()
                if track.kind=='audio':meter.observe(frame.to_ndarray())
        except asyncio.CancelledError:raise
        except Exception as error:
            if not stopping:errors.append(f'{track.kind} receiver: {error}')
    @peer.on('track')
    def track_ready(track):tasks.append(asyncio.create_task(receive(track)))
    channel=peer.createDataChannel('control')
    @channel.on('message')
    def message(raw):events.put_nowait(json.loads(raw))
    async def pause(seconds):
        deadline=clock.monotonic()+seconds
        while clock.monotonic()<deadline:
            if errors:raise RuntimeError(errors[-1])
            await asyncio.sleep(min(.05,deadline-clock.monotonic()))
    async def monitor():
        while True:
            sample=await asyncio.to_thread(resource_snapshot,root)
            report['system_vram_sample_peak_mib']=max(report['system_vram_sample_peak_mib'],sample['system_vram_mib'])
            report['resources'].append(sample)
            await asyncio.sleep(5)
    async with httpx.AsyncClient(base_url='http://127.0.0.1:7860',trust_env=False,timeout=180) as client:
        try:
            created=await client.post('/api/sessions',json={'avatar_id':avatar_id});created.raise_for_status()
            sid=created.json()['session_id']
            peer.addTransceiver('video',direction='recvonly');peer.addTrack(microphone)
            await peer.setLocalDescription(await peer.createOffer())
            answer=await client.post(f'/api/sessions/{sid}/offer',json={'sdp':peer.localDescription.sdp,'type':'offer'})
            answer.raise_for_status();await peer.setRemoteDescription(RTCSessionDescription(**answer.json()))
            deadline=clock.monotonic()+30
            while channel.readyState!='open' or meter.last_audio is None:
                if clock.monotonic()>=deadline:raise TimeoutError('Media connection did not become ready')
                await pause(.05)
            resource_task=asyncio.create_task(monitor());tasks.append(resource_task)
            for index,case in enumerate(cases):
                row={**case,'status':'running','first_non_silent_receive_ms':None}
                report['samples'].append(row)
                try:
                    await wait_for_quiet(meter)
                    if errors:raise RuntimeError(errors[-1])
                    while not events.empty():events.get_nowait()
                    if case['input_kind']=='audio':
                        if hashlib.sha256((root/case['audio_file']).read_bytes()).hexdigest()!=case['audio_sha256']:
                            raise ValueError('Audio fixture changed before recording')
                        pcm,_=await asyncio.to_thread(read_audio,root/case['audio_file'],48000)
                        recording=clock.monotonic();channel.send(json.dumps({'type':'start_recording'}))
                        await pause(.15);microphone.play(pcm)
                        await asyncio.wait_for(microphone.finished.wait(),35);await pause(.2)
                        row['recording_and_uplink_ms']=(clock.monotonic()-recording)*1000
                    sent=clock.monotonic();meter.arm(sent)
                    channel.send(json.dumps({'type':'stop_recording'} if case['input_kind']=='audio' else
                                            {'type':'chat_text','text':case['question']},ensure_ascii=False))
                    deadline=sent+300;reply=None
                    while True:
                        if errors:raise RuntimeError(errors[-1])
                        if resource_task.done():resource_task.result()
                        if clock.monotonic()>=deadline:raise TimeoutError('No completed answer within300s')
                        try:event=await asyncio.wait_for(events.get(),.25)
                        except asyncio.TimeoutError:continue
                        if event.get('type')=='error':raise RuntimeError(event.get('text'))
                        if event.get('type')=='transcript':
                            if event.get('role')=='assistant':reply=event['text']
                            else:row['recognized_text']=event['text']
                        if event.get('type')=='turn_complete':
                            row['generation']=event['generation'];break
                    if reply is None or meter.first_ms is None:raise RuntimeError('Answer or received speech missing')
                    response=await client.get(f'/api/sessions/{sid}/metrics');response.raise_for_status()
                    timing=response.json()['last_turn_timing']
                    if timing['generation']!=row['generation'] or timing['status']!='completed':
                        raise RuntimeError('Server timing belongs to another or unfinished turn')
                    row.update(status='completed',first_non_silent_receive_ms=meter.first_ms,
                        total_ms=(clock.monotonic()-sent)*1000,server_timing=timing,
                        answer=reply,answer_chars=len(reply),
                        first_sentence_chars=len(re.split(r'[。！？!?\n]',reply)[0]))
                    print(f"{case['input_kind']} {case['phase']} {index+1}/{len(cases)}: first={meter.first_ms/1000:.3f}s",flush=True)
                except BaseException as error:
                    row.update(status='failed',error=f'{type(error).__name__}: {error}',first_non_silent_receive_ms=meter.first_ms)
                    raise
                finally:
                    await asyncio.to_thread(atomic_json,folder/'report.json',copy.deepcopy(report))
            report['status']='completed'
        except BaseException as error:
            report.update(status='failed',error=f'{type(error).__name__}: {error}')
            raise
        finally:
            stopping=True;microphone.stop()
            for task in tasks:task.cancel()
            results=await asyncio.gather(*tasks,return_exceptions=True)
            for result in results:
                if isinstance(result,Exception):errors.append(str(result))
            try:await peer.close()
            except Exception as error:errors.append(f'Peer cleanup: {error}')
            if sid:
                try:
                    response=await client.delete(f'/api/sessions/{sid}');response.raise_for_status()
                except Exception as error:errors.append(f'Session cleanup: {error}')
            if errors:report.update(status='failed',cleanup_errors=errors)
            report['warm_summary']=summarize_warm(report['samples'])
            await asyncio.to_thread(atomic_json,folder/'report.json',report)
    if report['status']!='completed':raise RuntimeError('Latency measurement or cleanup failed')
    return report


async def benchmark(root,avatar_id,audio_fixtures=None):
    folder=root/'outputs/latency'/time.strftime('%Y%m%d-%H%M%S');folder.mkdir(parents=True)
    manifest=json.loads((root/'data/avatars'/avatar_id/'avatar.json').read_text())
    if manifest['status']!='ready' or not manifest['voice']['confirmed']:raise ValueError('A ready confirmed avatar is required')
    before=hashlib.sha256((root/'data/avatars'/avatar_id/'avatar.json').read_bytes()).hexdigest()
    summary={'status':'running','avatar_id':avatar_id,
             'git_head':subprocess.run(['git','rev-parse','HEAD'],cwd=root,check=True,capture_output=True,text=True).stdout.strip(),
             'source_sha256':source_fingerprint(root),'avatar_manifest_sha256':before,
             'source_media_sha256':manifest['sources'],
             'input_audio_reused':audio_fixtures is not None,
             'policy':{'cold_samples_per_input':3,'warm_samples_per_input':20,'p95':'nearest rank'},
             'measurement':'control send to first non-silent received audio frame','runs':[],'input_audio':[]}
    atomic_json(folder/'summary.json',summary)
    try:
        from backend.voice.client import VoiceClient
        fixture_started=clock.monotonic()
        fixtures=Path(audio_fixtures).resolve() if audio_fixtures is not None else folder/'input-audio'
        if audio_fixtures is None:
            services.start(root);fixtures.mkdir()
            rows=[]
            for index,question in enumerate(QUESTIONS):
                pcm,rate=await VoiceClient().synthesize(question,root/'data/avatars'/avatar_id,manifest['voice'])
                path=fixtures/f'{index:02}.wav'
                await asyncio.to_thread(write_wav,path,pcm,rate)
                rows.append({'question':question,'file':path.name,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()})
                print(f'Prepared audio question{index+1}/10',flush=True)
            atomic_json(fixtures/'manifest.json',rows)
        summary['input_audio']=load_audio_fixtures(fixtures)
        summary['fixture_preparation_seconds']=clock.monotonic()-fixture_started
        for kind in ('text','audio'):
            for cold in range(3):
                restart=clock.monotonic();services.stop(root);services.start(root)
                startup_ms=(clock.monotonic()-restart)*1000
                cases=[{'phase':'cold','input_kind':kind,'question':QUESTIONS[0],
                        'audio_file':summary['input_audio'][0]['file'],
                        'audio_sha256':summary['input_audio'][0]['sha256']}]
                if cold==2:
                    cases.extend({'phase':'warm','input_kind':kind,'question':QUESTIONS[index%10],
                                  'audio_file':summary['input_audio'][index%10]['file'],
                                  'audio_sha256':summary['input_audio'][index%10]['sha256']}
                                 for index in range(20))
                result=await measure(root,avatar_id,cases,folder/f'{kind}-cold{cold+1}')
                summary['runs'].append({'input_kind':kind,'cold_index':cold+1,'startup_ms':startup_ms,
                                       'samples':result['samples'],'warm_summary':result['warm_summary'],
                                       'system_vram_sample_peak_mib':result['system_vram_sample_peak_mib']})
                atomic_json(folder/'summary.json',summary)
        assert hashlib.sha256((root/'data/avatars'/avatar_id/'avatar.json').read_bytes()).hexdigest()==before
        assert source_fingerprint(root)==summary['source_sha256'],'Source changed during measurement'
        assert load_audio_fixtures(fixtures)==summary['input_audio'],'Audio fixtures changed during measurement'
        summary.update(status='completed',avatar_manifest_unchanged=True)
    except BaseException as error:
        summary.update(status='failed',error=f'{type(error).__name__}: {error}')
        raise
    finally:atomic_json(folder/'summary.json',summary)
    print(f'Latency evidence: {folder}',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--avatar-id',required=True)
    parser.add_argument('--audio-fixtures',type=Path,help='Reuse a verified input-audio directory from a prior run')
    arguments=parser.parse_args()
    asyncio.run(benchmark(Path(__file__).resolve().parents[1],arguments.avatar_id,arguments.audio_fixtures))
