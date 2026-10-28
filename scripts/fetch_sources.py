import argparse
import json
import subprocess
from pathlib import Path

NAMES = {"lhmpp": "lhmpp", "a2e": "a2e", "lam": "lam", "tracker": "tracker", "gaga_track": "gaga_track", "gpt_sovits": "gpt_sovits"}


def fetch(root):
    catalog = json.loads((root / "configs/models.json").read_text(encoding="utf-8"))
    for key, folder in NAMES.items():
        item = catalog[key]
        destination = root / "third_party" / folder
        if (destination / ".git").is_dir():
            found = subprocess.run(["git", "-C", str(destination), "rev-parse", "--verify", "HEAD"], text=True, capture_output=True)
            if found.returncode == 0:
                actual = found.stdout.strip()
                if actual != item["commit"]:
                    raise RuntimeError(f"Upstream checkout was changed: {destination}. Expected {item['commit']}, found {actual}.")
                continue
        else:
            destination.mkdir(parents=True, exist_ok=True)
            subprocess.run(["git", "init", str(destination)], check=True)
            subprocess.run(["git", "-C", str(destination), "remote", "add", "origin", f"https://github.com/{item['repo']}.git"], check=True)
        subprocess.run(["git", "-C", str(destination), "fetch", "--depth", "1", "origin", item["commit"]], check=True)
        subprocess.run(["git", "-C", str(destination), "checkout", "--detach", "FETCH_HEAD"], check=True)
    # Never fetch the private Alibaba submodule declared by the upstream repository.


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    fetch(p.parse_args().root)
