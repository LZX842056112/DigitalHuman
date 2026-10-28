"""Real 768px head/voice component diagnostic; excludes browser playback claims."""
import asyncio,json,time
from pathlib import Path
import httpx
import numpy as np
from aiortc import RTCPeerConnection,RTCSessionDescription,RTCConfiguration
from backend.storage import atomic_json
from backend.pipeline.media import write_wav

root=Path(__file__).resolve().parents[1]

async def run():
    output=root/'outputs'/('head-session-'+time.strftime('%Y%m%d-%H%M%S'));output.mkdir()
    peer=RTCPeerConnection(RTCConfiguration(iceServers=[]))
    tasks=[];sid=None;done=asyncio.Event();errors=[];video=[];pcm=[];closed=False
    latest=None;audio_count=0;speech_count=0;has_answer=False
    report={'kind':'short_head_768_component_diagnostic','browser_sound':'user_confirmed_native_player',
            'browser_access':'not_used','browser_av_sync':'not_measured','errors':errors,'complete':False}
    start=time.monotonic()
    async with httpx.AsyncClient(base_url='http://127.0.0.1:7860',timeout=240,trust_env=False) as client:
        try:
            # Check the CPU services before spending time loading a GPU avatar.
            for url in ('http://127.0.0.1:8081/health','http://127.0.0.1:9880/docs','http://127.0.0.1:9881/health'):
                ready=await client.get(url,timeout=5);ready.raise_for_status()
            response=await client.post('/api/sessions',json={'avatar_id':(root/'.runtime/default_avatar.txt').read_text().strip(),'render_size':768})
            response.raise_for_status();created=response.json();sid=created['session_id']
            assert created['render_size']==768 and 'portrait' in created['views']
            report['session_id']=sid
            async def receive(track):
                nonlocal latest,audio_count,speech_count
                try:
                    while not closed:
                        frame=await track.recv()
                        if track.kind=='video':
                            assert (frame.width,frame.height)==(768,768)
                            latest=frame;video.append(time.monotonic())
                        else:
                            audio_count+=1;data=frame.to_ndarray().reshape(-1)
                            scale=np.iinfo(data.dtype).max if np.issubdtype(data.dtype,np.integer) else 1
                            values=data.astype(np.float32)/scale;pcm.append(values)
                            if np.max(np.abs(values))>.001:speech_count+=1
                except asyncio.CancelledError:raise
                except Exception as error:
                    if not closed:errors.append(f'{track.kind}: {error}')
            @peer.on('track')
            def track_received(track):tasks.append(asyncio.create_task(receive(track)))
            channel=peer.createDataChannel('control')
            @channel.on('message')
            def message(raw):
                nonlocal has_answer
                event=json.loads(raw)
                if event.get('type')=='error':
                    errors.append(event.get('text'));done.set()
                if event.get('type')=='transcript' and event.get('role')=='assistant':has_answer=bool(event.get('text'))
                if event.get('type')=='turn_complete':done.set()
            peer.addTransceiver('video',direction='recvonly');peer.addTransceiver('audio',direction='recvonly')
            await peer.setLocalDescription(await peer.createOffer())
            response=await client.post(f'/api/sessions/{sid}/offer',json={'sdp':peer.localDescription.sdp,'type':'offer'})
            response.raise_for_status();await peer.setRemoteDescription(RTCSessionDescription(**response.json()))
            deadline=time.monotonic()+30
            while channel.readyState!='open' or len(video)<5:
                if time.monotonic()>deadline:raise TimeoutError('Media startup timeout')
                await asyncio.sleep(.1)
            latest.to_image().save(output/'received-full.png')
            channel.send(json.dumps({'type':'view','view':'portrait'}));count=len(video)
            while len(video)<count+5:await asyncio.sleep(.1)
            latest.to_image().save(output/'received-portrait.png')
            channel.send(json.dumps({'type':'camera',**created['views']['portrait'],'yaw':25}));count=len(video)
            while len(video)<count+5:await asyncio.sleep(.1)
            latest.to_image().save(output/'received-side.png')
            channel.send(json.dumps({'type':'chat_text','text':'请只说一句：你好，欢迎使用数字人。'},ensure_ascii=False))
            await asyncio.wait_for(done.wait(),timeout=180)
            assert not errors,f'Interaction failed: {errors}'
            assert has_answer and speech_count>0,'No answer or received non-silent speech'
            values=np.concatenate(pcm);active=np.where(np.abs(values)>.001)[0]
            values=values[max(0,active[0]-4800):min(len(values),active[-1]+4800)]
            assert np.max(np.abs(values))>.1 and np.sqrt(np.mean(values**2))>.02,'Received speech is too quiet for playback'
            write_wav(output/'received-reply.wav',values,48000)
            response=await client.get(f'/api/sessions/{sid}/metrics');response.raise_for_status()
            report.update(complete=True,received_resolution=[768,768],frames=len(video),audio_frames=audio_count,
                non_silent_audio_frames=speech_count,answer_received=has_answer,audio_peak=float(np.max(np.abs(values))),
                audio_rms=float(np.sqrt(np.mean(values**2))),metrics=response.json(),
                received_fps=(len(video)-1)/(video[-1]-video[0]),media_seconds=video[-1]-video[0])
        except Exception as error:
            report['error']=f'{type(error).__name__}: {error}';raise
        finally:
            closed=True
            for task in tasks:task.cancel()
            await asyncio.gather(*tasks,return_exceptions=True);await peer.close()
            if sid:
                response=await client.delete(f'/api/sessions/{sid}');response.raise_for_status()
            report['seconds']=time.monotonic()-start
            atomic_json(output/'report.json',report);print(output,flush=True)

asyncio.run(run())
