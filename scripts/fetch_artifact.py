"""Download a publisher-verified runtime artifact with resumable ranges."""
import argparse
import http.client
import os
import shutil
import time
import urllib.parse
import urllib.request
from pathlib import Path
from backend.storage import sha256


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


if __name__ == '__main__':
    p=argparse.ArgumentParser()
    p.add_argument('url')
    p.add_argument('destination',type=Path)
    p.add_argument('sha256')
    p.add_argument('size',type=int)
    a=p.parse_args()
    fetch_file(a.url,a.destination,a.sha256,a.size)
