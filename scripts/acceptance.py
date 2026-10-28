"""Real local WebRTC soak test. Does not approve appearance or voice identity."""
import argparse
import asyncio
import json
import time
import subprocess
from pathlib import Path
import numpy as np
import httpx
from aiortc import RTCPeerConnection,RTCConfiguration,RTCSessionDescription
from backend.storage import atomic_json

QUESTIONS=['请用一句话问候观众。','请简单介绍本次数字人演示。','请用一句话介绍春天。','请解释什么是三维模型。',
           '请告诉我保持学习习惯的方法。','请用一句话描述晴朗的天空。','请解释为什么需要休息。',
           '请给我一句简短的鼓励。','请说明本次测试使用本机推理。','请向观众说再见。']


def video_rate_target_met(average,duration,windows,target=15):
    # A timestamp window can include/exclude one boundary frame. Apply that
    # counting uncertainty to the whole run as well, rather than lowering FPS.
    return average*duration+1+1e-8>=target*duration and all(
        window['fps']*window['seconds']+1+1e-8>=target*window['seconds'] for window in windows)


async def run(root,avatar_id,minutes,rounds):
    folder=root/'outputs/acceptance'/time.strftime('%Y%m%d-%H%M%S')
    folder.mkdir(parents=True)
    peer=RTCPeerConnection(configuration=RTCConfiguration(iceServers=[]))
    tasks=[];session_id=None;started=time.monotonic();media_started=None
    report={'status':'running','avatar_id':avatar_id,'rounds':[],'frames':0,'audio_frames':0,
            'non_silent_audio_frames':0,'resolution':[],'queue_depth_max':0,'system_vram_peak_mib':0,
            'source_appearance':'pending_user_review','voice_identity':'pending_user_review',
            'browser_playback_av_offset_ms':'not_measured','hardware_microphone':'not_tested'}
    events=asyncio.Queue();errors=[];signal=asyncio.Event()
    last_video=None;last_audio=None;last_monitor=None;video_times=[]
    failure=None
    def check_health():
        for task in tasks:
            if task.done():
                exc=None if task.cancelled() else task.exception()
                raise RuntimeError(f'{task.get_name()} stopped: {exc or "unexpected end"}')
        if errors: raise RuntimeError(errors[-1])
        if media_started is not None:
            now=time.monotonic()
            for name,stamp,limit in [('video',last_video,2),('audio',last_audio,2),('monitor',last_monitor,20)]:
                if stamp is None or now-stamp>limit:
                    raise RuntimeError(f'{name} evidence stopped for more than {limit} seconds')
    async def pause(seconds):
        deadline=time.monotonic()+seconds
        while True:
            check_health()
            remaining=deadline-time.monotonic()
            if remaining<=0: return
            await asyncio.sleep(min(.1,remaining))
    async def next_event(deadline):
        while True:
            check_health()
            remaining=deadline-time.monotonic()
            if remaining<=0: raise TimeoutError('Dialogue did not complete within 300 seconds')
            try: return await asyncio.wait_for(events.get(),timeout=min(.25,remaining))
            except asyncio.TimeoutError: pass
    async with httpx.AsyncClient(base_url='http://127.0.0.1:7860',timeout=180,trust_env=False) as client:
        try:
            response=await client.post('/api/sessions',json={'avatar_id':avatar_id});response.raise_for_status()
            created=response.json();session_id=created['session_id']
            camera=created['camera']
            async def receive(track):
                nonlocal last_video,last_audio
                try:
                    while True:
                        frame=await track.recv()
                        if track.kind=='video':
                            now=time.monotonic();last_video=now;video_times.append(now)
                            if (frame.width,frame.height)!=(512,512):
                                raise RuntimeError('Video resolution changed from 512x512')
                            report['frames']+=1;report['resolution']=[frame.width,frame.height]
                            if report['frames'] in {1,100,500,1000}:
                                frame.to_image().save(folder/f'frame_{report["frames"]:05d}.png')
                        else:
                            last_audio=time.monotonic()
                            report['audio_frames']+=1
                            data=frame.to_ndarray()
                            scale=np.iinfo(data.dtype).max if np.issubdtype(data.dtype,np.integer) else 1
                            if np.max(np.abs(data.astype(float)/scale))>.001:
                                report['non_silent_audio_frames']+=1
                except asyncio.CancelledError: raise
                except Exception as exc:
                    if not signal.is_set(): errors.append(f'{track.kind}: {type(exc).__name__}: {exc}')
            @peer.on('track')
            def on_track(track): tasks.append(asyncio.create_task(receive(track),name=f'{track.kind} receiver'))
            channel=peer.createDataChannel('control')
            @channel.on('message')
            def message(raw):
                data=json.loads(raw)
                if data.get('type')=='error': errors.append(data.get('text','unknown service error'))
                events.put_nowait(data)
            peer.addTransceiver('video',direction='recvonly');peer.addTransceiver('audio',direction='recvonly')
            await peer.setLocalDescription(await peer.createOffer())
            response=await client.post(f'/api/sessions/{session_id}/offer',json={'sdp':peer.localDescription.sdp,'type':'offer'})
            response.raise_for_status();answer=response.json()
            await peer.setRemoteDescription(RTCSessionDescription(**answer))
            deadline=time.monotonic()+30
            while channel.readyState!='open':
                if time.monotonic()>deadline: raise TimeoutError('WebRTC data channel did not open')
                await asyncio.sleep(.1)
            def send(data): channel.send(json.dumps(data,ensure_ascii=False))
            async def monitor():
                nonlocal last_monitor
                while not signal.is_set():
                    response=await client.get(f'/api/sessions/{session_id}/metrics',timeout=10);response.raise_for_status()
                    metrics=response.json();report['queue_depth_max']=max(report['queue_depth_max'],metrics['queue_depth'])
                    if metrics['queue_depth']>2: errors.append('Playback queue exceeded two pending chunks')
                    report['video_queue_depth_max']=max(report.get('video_queue_depth_max',0),metrics.get('video_queue_depth',0))
                    report['video_frames_dropped']=metrics.get('video_frames_dropped',0)
                    if metrics.get('video_queue_depth',0)>1: errors.append('Video queue exceeded one pending frame')
                    gpu=await asyncio.to_thread(subprocess.run,['nvidia-smi','--query-gpu=memory.used','--format=csv,noheader,nounits'],capture_output=True,text=True,timeout=5)
                    if gpu.returncode!=0: raise RuntimeError(f'nvidia-smi failed: {gpu.stderr}')
                    report['system_vram_peak_mib']=max(report['system_vram_peak_mib'],int(gpu.stdout.strip().splitlines()[0]))
                    last_monitor=time.monotonic()
                    report['monitor_samples']=report.get('monitor_samples',0)+1
                    report['elapsed_seconds']=time.monotonic()-started
                    atomic_json(folder/'report.json',report)
                    await asyncio.sleep(5)
            tasks.append(asyncio.create_task(monitor(),name='monitor'))
            # Preparation does not count toward the continuous media run.
            deadline=time.monotonic()+30
            while last_video is None or last_audio is None or last_monitor is None:
                check_health()
                if time.monotonic()>deadline: raise TimeoutError('Audio, video and monitoring did not start')
                await asyncio.sleep(.1)
            media_started=time.monotonic()
            report['preparation_seconds']=media_started-started
            for index in range(rounds):
                check_health()
                send({'type':'action','action':['idle','walk','wave'][index%3]})
                send({'type':'camera',**camera,'yaw':[-30,0,45][index%3]})
                baseline=report['non_silent_audio_frames'];turn_start=time.monotonic();answer_text=None
                send({'type':'chat_text','text':QUESTIONS[index%len(QUESTIONS)]})
                deadline=time.monotonic()+300
                while True:
                    event=await next_event(deadline)
                    if event.get('type')=='error': raise RuntimeError(event.get('text'))
                    if event.get('type')=='transcript' and event.get('role')=='assistant': answer_text=event.get('text')
                    if event.get('type')=='turn_complete': break
                if not answer_text or report['non_silent_audio_frames']<=baseline:
                    raise RuntimeError('Round completed without an answer and received speech')
                report['rounds'].append({'index':index+1,'seconds':time.monotonic()-turn_start,
                                         'received_speech_frames':report['non_silent_audio_frames']-baseline})
                print(f'Completed real dialogue round {index+1}/{rounds}',flush=True)
            # Verify interruption, then require another successful round on the same peer.
            send({'type':'chat_text','text':'请用三句话介绍这次本机数字人演示。'})
            baseline=report['non_silent_audio_frames'];deadline=time.monotonic()+300
            while report['non_silent_audio_frames']==baseline:
                if time.monotonic()>deadline: raise RuntimeError('No speech arrived for interruption test')
                await pause(.1)
            send({'type':'interrupt'})
            await pause(.5);quiet=report['non_silent_audio_frames'];await pause(.5)
            report['interrupt_silent_after_half_second']=report['non_silent_audio_frames']==quiet
            if not report['interrupt_silent_after_half_second']: raise RuntimeError('Speech continued after interruption')
            while not events.empty(): events.get_nowait()
            send({'type':'chat_text','text':'请回复测试继续正常。'})
            deadline=time.monotonic()+300
            while True:
                event=await next_event(deadline)
                if event.get('type')=='error': raise RuntimeError(event.get('text'))
                if event.get('type')=='turn_complete': break
            report['recovered_after_interrupt']=report['non_silent_audio_frames']>quiet
            if not report['recovered_after_interrupt']: raise RuntimeError('No speech after interruption recovery')
            await pause(max(0,minutes*60-(time.monotonic()-media_started)))
            check_health()
            if report['frames']<2 or report['resolution']!=[512,512]: raise RuntimeError('No valid 512x512 video stream')
            measured_until=time.monotonic()
            duration=measured_until-media_started
            measured_frames=[stamp for stamp in video_times if media_started<=stamp<=measured_until]
            report['media_elapsed_seconds']=duration
            report['measured_video_frames']=len(measured_frames)
            report['received_video_fps']=len(measured_frames)/duration
            # A fast opening segment must not hide a later slow or frozen segment.
            windows=[]
            for offset in range(0,int(duration),10):
                length=min(10,duration-offset)
                if length<1: continue
                count=sum(media_started+offset<=stamp<media_started+offset+length for stamp in measured_frames)
                windows.append({'start_second':offset,'seconds':length,'fps':count/length})
            report['video_windows']=windows
            report['frame_count_uncertainty']=1
            report['fps_average_count_tolerance']=1/duration
            report['fps_target_met']=video_rate_target_met(report['received_video_fps'],duration,windows)
            report['status']='passed' if report['fps_target_met'] else 'failed_performance'
        except Exception as exc:
            failure=exc
            report.update(status='failed',error=f'{type(exc).__name__}: {exc}')
        finally:
            signal.set()
            for task in tasks: task.cancel()
            results=await asyncio.gather(*tasks,return_exceptions=True)
            for result in results:
                if isinstance(result,Exception): errors.append(f'Background task: {type(result).__name__}: {result}')
            try: await peer.close()
            except Exception as exc: errors.append(f'Peer cleanup: {exc}')
            if session_id:
                try:
                    response=await client.delete(f'/api/sessions/{session_id}');response.raise_for_status()
                except Exception as exc: errors.append(f'Session cleanup: {exc}')
            if errors:
                report['status']='failed'
                report.setdefault('error',errors[-1])
            if media_started is not None: report.setdefault('media_elapsed_seconds',time.monotonic()-media_started)
            report['elapsed_seconds']=time.monotonic()-started;report['errors']=errors
            atomic_json(folder/'report.json',report)
            print(f'Actual acceptance evidence: {folder}',flush=True)
    if failure is not None: raise RuntimeError(str(failure)) from failure
    if report['status']!='passed': raise RuntimeError('Performance target was not reached; evidence retained')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--avatar-id');parser.add_argument('--minutes',type=float,default=10);parser.add_argument('--rounds',type=int,default=10)
    args=parser.parse_args();root=Path(__file__).resolve().parents[1]
    if args.minutes<10 or args.rounds<10: parser.error('Actual acceptance requires at least ten minutes and ten rounds')
    avatar=args.avatar_id or (root/'.runtime/default_avatar.txt').read_text().strip()
    asyncio.run(run(root,avatar,args.minutes,args.rounds))
