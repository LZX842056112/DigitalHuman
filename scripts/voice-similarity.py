"""Report how closely a generated voice matches the avatar reference.

Uses the cloner's own campplus encoder, so the number reflects what CosyVoice
conditions on. Same speaker usually scores 0.75-0.95; below 0.6 means the
reference or the clone settings drifted.

    python scripts/voice-similarity.py --avatar <id>
    python scripts/voice-similarity.py --avatar <id> --text "你好，我们可以开始聊天了。"
    python scripts/voice-similarity.py --avatar <id> --wav a.wav --wav b.wav
"""
import argparse
import io
import json
import os
import urllib.request
import wave
from pathlib import Path

import numpy as np

ROOT=Path(os.environ.get('DIGITAL_HUMAN_ROOT') or Path(__file__).resolve().parents[1]).resolve()


def read_wav(source):
    if isinstance(source,(bytes,bytearray)):stream=io.BytesIO(source)
    elif isinstance(source,(str,os.PathLike)):stream=str(source)
    else:stream=source
    with wave.open(stream) as handle:
        rate=handle.getframerate();channels=handle.getnchannels()
        if handle.getsampwidth()!=2:raise SystemExit('只支持 16 位 PCM wav')
        pcm=np.frombuffer(handle.readframes(handle.getnframes()),dtype='<i2').astype(np.float32)/32768
    if channels>1:pcm=pcm.reshape(-1,channels).mean(axis=1)
    return pcm,rate


def level(pcm):
    if not len(pcm):return float('-inf')
    return 20*np.log10(max(float(np.sqrt(np.mean(pcm.astype(np.float64)**2))),1e-9))


def synthesize(base_url,avatar_id,text,emotion):
    body=json.dumps({'avatar_id':avatar_id,'text':text,'emotion':emotion}).encode()
    request=urllib.request.Request(base_url.rstrip('/')+'/tts',data=body,
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(request,timeout=600) as response:
        return response.read()


def main():
    parser=argparse.ArgumentParser(description='Compare generated speech with the avatar reference')
    parser.add_argument('--avatar',required=True,help='avatar id under data/avatars')
    parser.add_argument('--text',default='',help='synthesize this sentence through the voice service first')
    parser.add_argument('--emotion',default='neutral')
    parser.add_argument('--wav',action='append',default=[],help='extra wav to score (repeatable)')
    parser.add_argument('--base-url',default='http://127.0.0.1:9880')
    parser.add_argument('--diagnose',action='store_true',help='print reference facts before scoring')
    options=parser.parse_args()

    folder=ROOT/'data/avatars'/options.avatar
    if not (folder/'avatar.json').is_file():raise SystemExit('找不到人物：'+str(folder))
    manifest=json.loads((folder/'avatar.json').read_text(encoding='utf-8'))
    voice=manifest['voice']
    reference=folder/voice['reference']
    if not reference.is_file():raise SystemExit('缺少参考音频：'+str(reference))

    from backend.voice import similarity
    embedder=similarity.load(ROOT)
    if embedder is None:
        raise SystemExit('找不到 campplus.onnx，或缺 onnxruntime / torchaudio')

    pcm,rate=read_wav(reference)
    target=embedder.embedding(pcm,rate)
    print(f'参考 {reference.name}: {len(pcm)/rate:.2f} s, {rate} Hz, {level(pcm):.1f} dBFS')
    if options.diagnose:
        print(f'  峰值 {float(np.max(np.abs(pcm))):.3f}, 截断比例 {float(np.mean(np.abs(pcm)>.985)):.4f}')
        print(f'  起始 {voice.get("start_seconds")} s, 长度 {voice.get("duration_seconds")} s, '
              f'取样版本 {voice.get("reference_version")}')
        print(f'  转写 {voice.get("text")!r} ({voice.get("text_language")})')
        try:
            with urllib.request.urlopen(options.base_url.rstrip('/')+'/health',timeout=3) as response:
                print('  服务克隆方式', json.loads(response.read()).get('clone_mode'))
        except OSError:
            print('  声音服务未启动，只比对本地 wav')

    clips=[]
    if options.text:
        try:raw=synthesize(options.base_url,options.avatar,options.text,options.emotion)
        except OSError as error:raise SystemExit(f'声音服务不可用：{error}')
        clips.append(('服务合成',raw))
    for item in options.wav:clips.append((Path(item).name,str(item)))
    if not clips:clips.append(('sample.wav',str(folder/'voice/sample.wav')))

    for name,source in clips:
        try:other,other_rate=read_wav(source)
        except (OSError,wave.Error) as error:
            print(f'{name:>14}  读取失败：{error}');continue
        score=similarity.cosine(target,embedder.embedding(other,other_rate))
        print(f'{name:>14}  cosine {score:.3f}  ({len(other)/other_rate:.2f} s)')


if __name__=='__main__':
    main()
