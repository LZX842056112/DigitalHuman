import subprocess
import sys
from pathlib import Path


def test_gpu_lease_excludes_a_second_process(tmp_path):
    from backend.avatar.lease import GpuLease
    lease=GpuLease(tmp_path).acquire()
    source="from pathlib import Path\nimport sys\nfrom backend.avatar.lease import GpuLease\nfrom backend.errors import NotReady\ntry:\n GpuLease(Path(sys.argv[1])).acquire()\nexcept NotReady:\n raise SystemExit(23)\n"
    def probe():
        return subprocess.run([sys.executable,'-c',source,str(tmp_path)],cwd=Path(__file__).resolve().parents[1],timeout=15).returncode
    try: assert probe()==23
    finally: lease.release()
    assert probe()==0
