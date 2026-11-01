"""Keep real HTTPX initialization, replace only network I/O."""
import asyncio
import io
import json
import ssl
import time
import wave
from pathlib import Path

import httpx
import pytest


@pytest.mark.asyncio
@pytest.mark.parametrize('kind',['dialogue','voice','portrait'])
async def test_service_request_does_not_parse_certificates_on_media_loop(kind,monkeypatch):
    from backend.dialogue.client import Dialogue
    from backend.voice.client import VoiceClient
    from backend.portrait.client import PortraitClient
    original=ssl.create_default_context
    def slow_certificates(*args,**kwargs):
        time.sleep(.15)
        return original(*args,**kwargs)
    monkeypatch.setattr(ssl,'create_default_context',slow_certificates)
    buffer=io.BytesIO()
    with wave.open(buffer,'wb') as output:
        output.setnchannels(1);output.setsampwidth(2);output.setframerate(24000)
        output.writeframes(b'\x00\x20'*2400)
    async def network(transport,request):
        if request.url.path.endswith('/chat/completions'):
            value={'choices':[{'message':{'content':json.dumps({'text':'你好','emotion':'neutral'})}}]}
            return httpx.Response(200,json=value,request=request)
        if request.url.path.endswith('/tts'):return httpx.Response(200,content=buffer.getvalue(),request=request)
        return httpx.Response(200,json={'report':{}},request=request)
    monkeypatch.setattr(httpx.AsyncHTTPTransport,'handle_async_request',network)
    # The API already constructs PhotoSession and these clients in a thread.
    if kind=='dialogue':client=await asyncio.to_thread(Dialogue,Path('/tmp'),{})
    elif kind=='voice':client=await asyncio.to_thread(VoiceClient)
    else:client=await asyncio.to_thread(PortraitClient)
    stamps=[]
    async def heartbeat():
        while True:
            stamps.append(time.monotonic());await asyncio.sleep(.005)
    pulse=asyncio.create_task(heartbeat())
    try:
        await asyncio.sleep(.02)
        if kind=='dialogue':await client.answer('你好',[])
        elif kind=='voice':await client.synthesize('你好',Path('/tmp/person'),{'confirmed':True})
        else:await client.animate('person','session',1,0,'neutral')
        await asyncio.sleep(.02)
        assert max(b-a for a,b in zip(stamps,stamps[1:]))<.08
    finally:
        pulse.cancel();await asyncio.gather(pulse,return_exceptions=True)
