"""Public ModelScope downloads pinned to file revisions and publisher SHA-256."""
import argparse
import fnmatch
import json
import shutil
import urllib.request
import urllib.parse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from backend.storage import atomic_json
from scripts.fetch_artifact import fetch_file as fetch_file
# Keep historical helper imports available to existing callers.
from scripts.model_archives import (
    extract_archives, safe_extract as safe_extract, is_asset_archive as is_asset_archive,
)


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
        extract_archives(destination, files)
    return lock


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--root",type=Path,default=Path(__file__).resolve().parents[1])
    parser.add_argument("--only",nargs="*")
    parser.add_argument("--list",action="store_true")
    args=parser.parse_args()
    download(args.root.resolve(),args.only,args.list)
