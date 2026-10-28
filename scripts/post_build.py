import asyncio
from pathlib import Path
from backend.storage import Store
from backend.errors import NotReady


if __name__=='__main__':
    root=Path(__file__).resolve().parents[1]
    avatar=(root/'.runtime/default_avatar.txt').read_text().strip()
    try:
        Store(root).require_ready(avatar)
    except NotReady as exc:
        print(f'Acceptance has not started: {exc}',flush=True)
    else:
        from scripts.acceptance import run
        asyncio.run(run(root,avatar,minutes=10,rounds=10))
