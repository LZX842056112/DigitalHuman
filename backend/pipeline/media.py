from __future__ import annotations

import wave
from pathlib import Path
import av
import cv2
import numpy as np

from backend.errors import PipelineError
from backend.storage import atomic_json


def read_audio(path: Path, rate=16000):
    chunks = []
    resampler = av.AudioResampler(format="fltp", layout="mono", rate=rate)
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise PipelineError("audio", "视频中没有音轨，请提供本人录音")
        for frame in container.decode(audio=0):
            for out in resampler.resample(frame):
                chunks.append(out.to_ndarray().reshape(-1))
        for out in resampler.resample(None):
            chunks.append(out.to_ndarray().reshape(-1))
    if not chunks:
        raise PipelineError("audio", "音轨无法解码")
    return np.concatenate(chunks).astype(np.float32), rate


def write_wav(path: Path, pcm, rate):
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(rate)
        output.writeframes((np.clip(pcm, -1, 1) * 32767).astype("<i2").tobytes())


def choose_reference(pcm, rate, duration=6):
    if len(pcm) < rate * duration:
        raise PipelineError("audio", "连续语音不足，请提供至少六秒清晰语音")
    best = None
    for start in np.arange(0, len(pcm) / rate - duration + .01, .5):
        segment = np.asarray(pcm[int(start * rate):int((start + duration) * rate)])
        frames = segment[:len(segment) // 320 * 320].reshape(-1, 320)
        rms = np.sqrt(np.mean(frames ** 2, axis=1) + 1e-12)
        active = float(np.mean(rms > .008))
        clipped = float(np.mean(np.abs(segment) > .98))
        score = active - clipped * 5 - abs(float(np.median(rms)) - .10) * .1
        if active > .55 and clipped < .01 and (best is None or score > best[0]):
            best = score, float(start)
    if best is None:
        raise PipelineError("audio", "未找到可用语音，请选择参考片段或补录")
    return best[1], duration


def inspect_video(path: Path):
    with av.open(str(path)) as source:
        if not source.streams.video:
            raise PipelineError("video", "文件没有视频画面")
        stream = source.streams.video[0]
        return {"width": stream.width, "height": stream.height,
                "fps": float(stream.average_rate or 0),
                "seconds": float(stream.duration * stream.time_base) if stream.duration else None,
                "has_audio": bool(source.streams.audio)}


def extract_faces(video: Path, destination: Path, count=16):
    destination.mkdir(parents=True, exist_ok=True)
    detector = cv2.CascadeClassifier(str(Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"))
    candidates = []
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    index = 0
    while True:
        ok, bgr = cap.read()
        if not ok:
            break
        if index % max(1, round(fps * 2)) == 0:
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            boxes = detector.detectMultiScale(gray, 1.1, 5, minSize=(64, 64))
            if len(boxes) == 1:
                x, y, w, h = map(int, boxes[0])
                margin = int(.35 * max(w, h))
                x0, y0 = max(0, x - margin), max(0, y - margin)
                x1, y1 = min(bgr.shape[1], x + w + margin), min(bgr.shape[0], y + h + margin)
                sharpness = float(cv2.Laplacian(gray[y:y+h, x:x+w], cv2.CV_64F).var())
                name = f"{index:06d}.jpg"
                cv2.imwrite(str(destination / name), bgr[y0:y1, x0:x1])
                # Full, unchanged frame is retained for the 3D face tracker.
                full = destination.parent / "tracking" / "imgs_png"
                full.mkdir(parents=True, exist_ok=True)
                cv2.imwrite(str(full / f"{len(candidates)+1:05d}.png"), bgr)
                candidates.append({"file": name, "seconds": index / fps,
                                   "bbox": [x0, y0, x1, y1], "score": sharpness,
                                   "tracking_frame": len(candidates) + 1})
        index += 1
    cap.release()
    if not candidates:
        raise PipelineError("face", "未检测到可用的单人面部，请检查素材")
    selected = sorted(candidates, key=lambda row: row["score"], reverse=True)[:count]
    atomic_json(destination / "frames.json", {"selected": selected, "all": candidates})
    return selected


def prepare_media(store, avatar_id):
    manifest = store.manifest(avatar_id)
    folder = store.avatar_dir(avatar_id)
    video = folder / "source" / manifest["video"]
    processed = folder / "processed"
    processed.mkdir(exist_ok=True)
    metadata = inspect_video(video)
    atomic_json(processed / "video.json", metadata)
    pcm, rate = read_audio(video)
    start, duration = choose_reference(pcm, rate)
    write_wav(folder / "voice" / "reference.wav", pcm[int(start * rate):int((start + duration) * rate)], rate)
    write_wav(processed / "source.wav", pcm, rate)
    selected = extract_faces(video, processed / "faces")
    manifest["video_metadata"] = metadata
    manifest["face_frames"] = len(selected)
    manifest["voice"].update({"reference": "voice/reference.wav", "start_seconds": start,
                              "duration_seconds": duration, "sample_rate": rate})
    manifest["status"] = "prepared"
    store.write_manifest(avatar_id, manifest)
    return manifest


def append_face_photos(folder, names):
    """Use only face crops from auxiliary photos. Never mix their clothing."""
    path=folder/'processed/faces/frames.json'
    from backend.storage import read_json
    frames=read_json(path)
    known={row.get('source_photo') for row in frames['selected']}
    detector=cv2.CascadeClassifier(str(Path(cv2.data.haarcascades)/'haarcascade_frontalface_default.xml'))
    skipped=[]
    for number,name in enumerate(names):
        if name in known: continue
        source=folder/'source'/name
        image=cv2.imdecode(np.frombuffer(source.read_bytes(),np.uint8),cv2.IMREAD_COLOR)
        if image is None: raise PipelineError('photo',f'无法解析面部照片: {name}')
        gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
        boxes=detector.detectMultiScale(gray,1.1,5,minSize=(32,32))
        boxes=sorted(boxes,key=lambda box:int(box[2])*int(box[3]),reverse=True)
        # A small false detection on clothing must not discard a clear portrait.
        # Comparable faces remain ambiguous and are never selected automatically.
        if not boxes or (len(boxes)>1 and boxes[0][2]*boxes[0][3]<4*boxes[1][2]*boxes[1][3]):
            skipped.append({'source':name,'reason':'未检测到唯一清晰面部'})
            continue
        x,y,w,h=map(int,boxes[0]); margin=int(.35*max(w,h))
        x0,y0=max(0,x-margin),max(0,y-margin)
        x1,y1=min(image.shape[1],x+w+margin),min(image.shape[0],y+h+margin)
        filename=f'photo_{number:03d}.jpg'
        ok,encoded=cv2.imencode('.jpg',image[y0:y1,x0:x1])
        if not ok: raise PipelineError('photo','面部照片裁剪失败')
        (path.parent/filename).write_bytes(encoded.tobytes())
        item={'file':filename,'source_photo':name,'bbox':[x0,y0,x1,y1],
              'detected_bbox':[x,y,w,h],'detection_count':len(boxes),
              'tracking_frame':1000000+number,'role':'face_only',
              'score':float(cv2.Laplacian(gray[y:y+h,x:x+w],cv2.CV_64F).var())}
        frames['selected'].append(item)
        frames['all'].append(item)
    frames['selected'].sort(key=lambda item:item['tracking_frame'])
    atomic_json(path,frames)
    atomic_json(path.parent/'photo_checks.json',{'skipped':skipped,'selected':[r.get('source_photo') for r in frames['selected'] if r.get('source_photo')]})
    return len(frames['selected'])
