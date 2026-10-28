import subprocess
import sys
import pytest
from pathlib import Path

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="Windows PowerShell path integration")


def test_windows_drive_and_spaces_are_preserved_for_wsl():
    script = Path(__file__).resolve().parents[1] / "scripts/common.ps1"
    result = subprocess.run(["pwsh", "-NoProfile", "-Command",
                             f". '{script}'; ConvertTo-ProjectLinuxPath 'D:\\Projects\\3d\\DigitalHuman'"],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/mnt/d/Projects/3d/DigitalHuman"


def test_windows_path_with_spaces_converts_without_a_linux_shell():
    script = Path(__file__).resolve().parents[1] / "scripts/common.ps1"
    result = subprocess.run(["pwsh", "-NoProfile", "-Command",
                             f". '{script}'; ConvertTo-ProjectLinuxPath 'C:\\My Projects\\digital human'"],
                            text=True, capture_output=True)
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "/mnt/c/My Projects/digital human"
