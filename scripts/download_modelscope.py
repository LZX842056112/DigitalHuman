"""Public ModelScope downloads pinned to file revisions and publisher SHA-256."""
import argparse
import fnmatch
import json
import os
import shutil
import urllib.request
import urllib.parse
import time
import http.client
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from backend.storage import sha256, atomic_json
from scripts.download_models import safe_extract, is_asset_archive


def fetch_file(url, destination: Path, digest, size):
    if size>=8*1024**2 and urllib.parse.urlparse(url).scheme in {'http','https'}:
        from scripts.download_parallel import fetch_parallel
        return fetch_parallel(url,destination,digest,size,workers=12)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.stat().st_size == size and sha256(destination) == digest:
        return
    partial = destination.with_name(destination.name + ".partial")
    if partial.exists() and partial.stat().st_size >= size:
        if partial.stat().st_size == size and sha256(partial) == digest:
            os.replace(partial, destination)
            return
        partial.unlink()
    while not partial.exists() or partial.stat().st_size < size:
        for attempt in range(5):
            before = partial.stat().st_size if partial.exists() else 0
            try:
                _transfer(url, partial, size)
                break
            except (OSError, TimeoutError, http.client.HTTPException) as exc:
                if attempt == 4:
                    raise
                print(f"Resume {destination.name}: {exc}", flush=True)
                time.sleep(min(2 ** attempt, 15))
        after = partial.stat().st_size
        if after <= before:
            raise OSError(f"Download made no progress: {destination}")
        if after // (64*1024**2) > before // (64*1024**2):
            print(f"{destination.name}: {after}/{size}", flush=True)
    if sha256(partial) != digest:
        partial.unlink()
        raise ValueError(f"SHA-256 mismatch: {destination.name}")
    os.replace(partial, destination)
    print(f"Verified {destination.name}: {size} bytes", flush=True)


def _transfer(url, partial, size):
    offset = partial.stat().st_size if partial.exists() else 0
    headers = {"User-Agent": "DigitalHuman/0.1"}
    end = min(offset + 1024**2, size) - 1
    headers["Range"] = f"bytes={offset}-{end}"
    with urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=90) as response:
        append = offset > 0 and response.status == 206
        if offset and response.status not in {206, None}:
            raise OSError("Server ignored the resume range")
        if append and not response.headers.get("Content-Range", "").startswith(f"bytes {offset}-"):
            raise ValueError("Server returned an inconsistent resume range")
        with partial.open("ab" if append else "wb") as output:
            shutil.copyfileobj(response, output, length=64*1024)


def list_files(repo):
    url = "https://modelscope.cn/api/v1/models/" + repo + "/repo/files?Revision=master&Recursive=true"
    with urllib.request.urlopen(url, timeout=45) as response:
        result = json.load(response)
    if not result.get("Success"):
        raise ValueError(f"ModelScope repository unavailable: {repo}: {result.get('Message')}")
    return [row for row in result["Data"]["Files"] if row["Type"] == "blob"]


def download(root: Path, selected=None, list_only=False):
    catalog = json.loads((root / "configs/models.json").read_text(encoding="utf-8"))
    lock_path = root / "models/lock.json"
    lock = json.loads(lock_path.read_text(encoding="utf-8")) if lock_path.exists() else {}
    for key, item in catalog.items():
        if not item.get("modelscope_repo") or selected and key not in selected:
            continue
        repo = item["modelscope_repo"]
        patterns = item.get("allow", ["*"])
        prefix = item.get('modelscope_prefix','')
        rows = [{**r, 'RemotePath':r['Path'], 'Path':r['Path'][len(prefix):]} for r in list_files(repo)
                if r['Path'].startswith(prefix) and any(fnmatch.fnmatchcase(r['Path'][len(prefix):],p) for p in patterns)
                and not any(fnmatch.fnmatchcase(r['Path'][len(prefix):],p) for p in item.get('ignore',[]))]
        pinned=lock.get(key)
        if not list_only and pinned and pinned.get('provider')=='modelscope':
            current_hashes={row['Path']:row['Sha256'] for row in rows}
            if pinned.get('repo')!=repo or current_hashes!=pinned.get('files'):
                raise RuntimeError(f'{key}: 上游模型清单与已固定的权重不一致。原权重和锁文件已保留。')
            revisions=pinned.get('revision',{})
            if not isinstance(revisions,dict) or any(row['Path'] not in revisions for row in rows):
                raise RuntimeError(f'{key}: 已固定模型的文件版本不完整')
            rows=[{**row,'Revision':revisions[row['Path']]} for row in rows]
        total = sum(row["Size"] for row in rows)
        metadata={"model":key,"repo":repo,"bytes":total,"files":[
            {"path":r["Path"],"size":r["Size"],"sha256":r["Sha256"],"revision":r["Revision"]} for r in rows]}
        atomic_json(root/'.runtime/model-files'/f'{key}.json',metadata)
        print(json.dumps(metadata if list_only else {'model':key,'repo':repo,'bytes':total,'file_count':len(rows)}),flush=True)
        if list_only:
            continue
        if not rows:
            raise ValueError(f"No files selected: {repo}")
        destination = root / item["destination"]
        destination.mkdir(parents=True, exist_ok=True)
        missing = sum(r["Size"] for r in rows if not (destination/r["Path"]).exists())
        if shutil.disk_usage(root).free < missing + 10 * 1024**3:
            raise OSError(f"Insufficient disk space for {key}: need payload plus 10 GiB reserve")
        def fetch(row):
            target = (destination / row["Path"]).resolve()
            if destination.resolve() not in target.parents:
                raise ValueError("Remote file path leaves model directory")
            query = urllib.parse.urlencode({"Revision": row["Revision"], "FilePath": row.get('RemotePath',row["Path"])})
            fetch_file("https://modelscope.cn/api/v1/models/"+repo+"/repo?"+query, target,row["Sha256"],row["Size"])
        with ThreadPoolExecutor(max_workers=3) as pool:
            list(pool.map(fetch, rows))
        files = {r["Path"]:r["Sha256"] for r in rows}
        lock[key] = {"repo":repo,"provider":"modelscope","revision":{r["Path"]:r["Revision"] for r in rows},
                     "files":files,"license":item.get("license")}
        atomic_json(lock_path,lock)
        for name,digest in files.items():
            path = destination/name
            if is_asset_archive(path):
                marker=path.with_name(path.name+".extracted")
                if not marker.exists():
                    safe_extract(path,destination)
                    marker.write_text(digest,encoding="ascii")
    return lock


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--only",nargs="*")
    parser.add_argument("--list",action="store_true")
    args=parser.parse_args()
    download(args.root.resolve(),args.only,args.list)
