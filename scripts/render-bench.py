"""Picture-render A/B bench for the local photo worker.

Runs with the plain Windows Python (standard library only): it rewrites the
three speed knobs in configs/app.yaml, asks the running portrait worker to
render one fixed WAV, and records the per-stage timings it returns. The engine
re-reads those knobs on every render, so no service restart is needed between
presets. The original config text is always restored.
"""
import argparse
import json
import re
import shutil
import time
import urllib.error
import urllib.request
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / 'configs' / 'app.yaml'
STUDIO = 'http://127.0.0.1:7860'
PORTRAIT = 'http://127.0.0.1:9881'
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def post(url, payload, timeout=600):
    data = json.dumps(payload).encode()
    request = urllib.request.Request(url, data=data, headers={'Content-Type': 'application/json'})
    with OPENER.open(request, timeout=timeout) as response:
        return json.loads(response.read().decode())


def get(url, timeout=30):
    with OPENER.open(urllib.request.Request(url), timeout=timeout) as response:
        return json.loads(response.read().decode())


def set_knobs(text, steps, batch, stride):
    for key, value in (('portrait_sampling_steps', steps), ('portrait_frame_batch', batch),
                       ('portrait_frame_stride', stride)):
        pattern = re.compile(r'^(\s*' + key + r':\s*).*$', re.M)
        if pattern.search(text):
            text = pattern.sub(lambda match: match.group(1) + str(value), text, count=1)
        else:
            text = text.replace('runtime:\n', 'runtime:\n  %s: %s\n' % (key, value), 1)
    return text


def parse(text):
    return [int(part) for part in text.split(',')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--presets', default='50,1,1;12,4,1;12,4,2')
    parser.add_argument('--wav', default=None)
    parser.add_argument('--avatar', default=None)
    parser.add_argument('--out', default=None)
    args = parser.parse_args()

    avatar = args.avatar
    if not avatar:
        avatars = get(STUDIO + '/api/avatars')
        ready = [item for item in avatars if item.get('status') == 'ready']
        if not ready:
            raise SystemExit('no ready avatar')
        avatar = ready[0]['id']
    if not args.wav:
        candidate = ROOT / 'data/avatars' / avatar / 'voice' / 'reference.wav'
        if not candidate.is_file():
            raise SystemExit('pass --wav; the avatar has no reference.wav')
        args.wav = str(candidate)

    stamp = datetime.now().strftime('%Y%m%d-%H%M%S')
    out = Path(args.out) if args.out else ROOT / 'outputs/render-bench' / stamp
    out.mkdir(parents=True, exist_ok=True)

    session = post(STUDIO + '/api/sessions', {'avatar_id': avatar})['session_id']
    folder = ROOT / 'data/sessions' / session
    folder.mkdir(parents=True, exist_ok=True)
    (folder / 'generation.json').write_text(json.dumps({'generation': 0, 'closed': False}), encoding='utf-8')

    original = CONFIG.read_text(encoding='utf-8')
    rows = []
    try:
        for offset, preset in enumerate(args.presets.split(';')):
            steps, batch, stride = parse(preset)
            CONFIG.write_text(set_knobs(original, steps, batch, stride), encoding='utf-8')
            index = offset
            wav = folder / ('0_%d.wav' % index)
            shutil.copyfile(args.wav, wav)
            started = time.monotonic()
            try:
                payload = post(PORTRAIT + '/render', {'avatar_id': avatar, 'session_id': session,
                                                      'generation': 0, 'index': index, 'emotion': 'neutral'})
                report = payload.get('report') or {}
            except urllib.error.HTTPError as error:
                rows.append({'preset': preset, 'error': error.read().decode()[:400]})
                continue
            elapsed = time.monotonic() - started
            frames = report.get('frames') or 0
            timings = report.get('timings') or {}
            video = folder / ('0_%d.mkv' % index)
            row = {'preset': preset, 'steps': steps, 'batch': batch, 'stride': stride,
                   'seconds': report.get('seconds'), 'frames': frames,
                   'wall_ms': round(elapsed * 1000),
                   'render_fps': round(frames / elapsed, 2) if elapsed else None,
                   'realtime_ratio': round(elapsed / report['seconds'], 3) if report.get('seconds') else None,
                   'timings': {key: round(value) for key, value in timings.items()},
                   'video': str(video), 'video_bytes': video.stat().st_size if video.exists() else None}
            rows.append(row)
            print(json.dumps(row, ensure_ascii=False))
    finally:
        CONFIG.write_text(original, encoding='utf-8')
    (out / 'report.json').write_text(json.dumps({'session': session, 'wav': args.wav, 'rows': rows},
                                                ensure_ascii=False, indent=1), encoding='utf-8')
    print('written', out / 'report.json')


if __name__ == '__main__':
    main()
