import hashlib
import json
from pathlib import Path


def check(root):
    folder=root/'frontend'
    marker=folder/'dist/build.json'
    if not marker.is_file(): return False
    digest=hashlib.sha256()
    paths=[folder/p for p in ['package.json','package-lock.json','tsconfig.json','vite.config.ts','index.html']]
    paths+= [p for p in (folder/'src').rglob('*') if p.is_file()]
    for path in sorted(paths,key=lambda p:p.relative_to(folder).as_posix()):
        digest.update(path.relative_to(folder).as_posix().encode());digest.update(b'\0')
        digest.update(path.read_bytes());digest.update(b'\0')
    data=json.loads(marker.read_text())
    if data['source_sha256']!=digest.hexdigest(): return False
    for name,expected in data['files'].items():
        path=(folder/'dist'/name).resolve()
        if (folder/'dist').resolve() not in path.parents or not path.is_file(): return False
        if hashlib.sha256(path.read_bytes()).hexdigest()!=expected: return False
    return True


if __name__=='__main__':
    raise SystemExit(0 if check(Path(__file__).resolve().parents[1]) else 1)
