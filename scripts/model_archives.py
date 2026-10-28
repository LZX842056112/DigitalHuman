"""Shared model archive handling, including legacy extraction markers."""
import tarfile
import zipfile
from pathlib import Path


def is_asset_archive(path: Path):
    if not path.name.endswith((".tar", ".tar.gz", ".tgz", ".zip")):
        return False
    # The publisher uses this name for both a checkpoint and an outer archive.
    if path.name.lower() == 'lam_audio2exp_streaming.tar':
        return tarfile.is_tarfile(path)
    return True


def safe_extract(archive: Path, destination: Path):
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    def checked(name):
        target = (destination / name).resolve()
        if target != destination and destination not in target.parents:
            raise ValueError(f"归档路径越界: {name}")
    if zipfile.is_zipfile(archive):
        with zipfile.ZipFile(archive) as source:
            for entry in source.infolist():
                checked(entry.filename)
                if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                    raise ValueError("归档含符号链接")
            source.extractall(destination)
    else:
        with tarfile.open(archive) as source:
            for entry in source.getmembers():
                checked(entry.name)
                if entry.issym() or entry.islnk() or entry.isdev():
                    raise ValueError("归档含链接或设备文件")
            source.extractall(destination)


def extract_archives(destination: Path, files: dict, *, legacy_suffix=False):
    for name, digest in files.items():
        path = destination/name
        if not is_asset_archive(path):
            continue
        # Preserve both providers' historical markers so reruns reuse assets.
        marker = path.with_suffix('.extracted') if legacy_suffix else path.with_name(path.name+'.extracted')
        if not marker.exists():
            safe_extract(path, destination)
            marker.write_text(digest, encoding='ascii')
