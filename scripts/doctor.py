import argparse
import importlib.util
import json
import platform
import shutil
import subprocess
from pathlib import Path

from backend.storage import sha256


def inspect(root: Path, verify_hashes=True):
    checks = []
    def add(name, ok, detail):
        checks.append({"name": name, "ok": bool(ok), "detail": str(detail)})
    add("disk", shutil.disk_usage(root).free > 10 * 1024**3,
        f"{shutil.disk_usage(root).free / 1024**3:.1f} GiB available")
    add("platform", platform.system() == "Linux", platform.platform())
    for module in ("torch", "pytorch3d", "gsplat", "pointops_cuda", "aiortc", "faster_whisper"):
        try:
            available = importlib.util.find_spec(module) is not None
        except (ImportError, ValueError):
            available = False
        add(module, available, "installed" if available else "missing")
    add("ffmpeg", shutil.which("ffmpeg") is not None, shutil.which("ffmpeg") or "missing")
    try:
        gpu = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=5)
        add("gpu", gpu.returncode == 0, gpu.stdout.strip() or gpu.stderr.strip())
    except (OSError, subprocess.SubprocessError) as exc:
        add("gpu", False, str(exc))
    for name in ("lhmpp", "a2e", "gpt_sovits", "gaga_track"):
        add("source_" + name, (root / "third_party" / name / '.source-lock.json').is_file(), name)
    for name, file in (("lhmpp_weights", "models/lhmpp/model.safetensors"),
                       ("assets_index", "models/assets-index.json"),
                       ("asr_weights", "models/asr/small/model.bin")):
        add(name, (root / file).is_file(), file)
    lock_file = root / "models/lock.json"
    add("model_lock", lock_file.is_file(), str(lock_file))
    from scripts.check_frontend import check
    add('frontend',check(root),'production files match source digest')
    runtime_lock=root/'models/runtime-assets-lock.json'
    add('runtime_assets',runtime_lock.is_file(),'FAN/SFD and LPIPS assets')
    if verify_hashes and runtime_lock.exists():
        runtime=json.loads(runtime_lock.read_text())
        invalid=[name for name,digest in runtime['files'].items() if not (root/name).is_file() or sha256(root/name)!=digest]
        add('runtime_asset_hashes',not invalid,', '.join(invalid) or 'verified')
    if verify_hashes and lock_file.exists():
        catalog = json.loads((root / "configs/models.json").read_text(encoding="utf-8"))
        lock = json.loads(lock_file.read_text(encoding="utf-8"))
        for name, item in lock.items():
            base = root / catalog[name]["destination"]
            invalid = [file for file, digest in item["files"].items()
                       if not (base / file).is_file() or sha256(base / file) != digest]
            add("hash_" + name, not invalid, ", ".join(invalid) if invalid else item.get("revision", "verified"))
    return {"ready": all(check["ok"] for check in checks), "checks": checks}


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    report = inspect(args.root.resolve())
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ready"] else 2)
