import hashlib
import pytest


def test_download_checks_publisher_hash_before_accepting(tmp_path):
    from scripts.download_modelscope import fetch_file
    source = tmp_path / "source.bin"
    source.write_bytes(b"publisher model")
    destination = tmp_path / "cache/model.bin"
    digest = hashlib.sha256(b"publisher model").hexdigest()
    fetch_file(source.as_uri(), destination, digest, len(b"publisher model"))
    assert destination.read_bytes() == b"publisher model"
    source.write_bytes(b"tampered model!")
    # Existing verified files are reused, even if the remote endpoint changes.
    fetch_file(source.as_uri(), destination, digest, len(b"publisher model"))
    assert destination.read_bytes() == b"publisher model"


def test_download_rejects_corrupt_payload(tmp_path):
    from scripts.download_modelscope import fetch_file
    source = tmp_path / "bad.bin"
    source.write_bytes(b"bad")
    with pytest.raises(ValueError, match="SHA"):
        fetch_file(source.as_uri(), tmp_path / "download.bin", "0"*64, 3)
    assert not (tmp_path / "download.bin").exists()


def test_download_restarts_a_complete_corrupt_partial(tmp_path):
    from scripts.download_modelscope import fetch_file
    source = tmp_path / "source.bin"
    source.write_bytes(b"good")
    destination = tmp_path / "cache/model.bin"
    destination.parent.mkdir()
    destination.with_name("model.bin.partial").write_bytes(b"bad!")
    fetch_file(source.as_uri(), destination, hashlib.sha256(b"good").hexdigest(), 4)
    assert destination.read_bytes() == b"good"


@pytest.mark.parametrize('provider', ['modelscope', 'huggingface'])
@pytest.mark.parametrize('is_archive', [False, True])
def test_download_distinguishes_asset_archive_from_torch_checkpoint(tmp_path, monkeypatch, provider, is_archive):
    import io
    import json
    import sys
    import tarfile
    from types import SimpleNamespace
    import torch
    from backend.storage import sha256
    from scripts import download_modelscope, download_models
    filename = 'LAM_audio2exp_streaming.tar'
    destination = tmp_path/'models/a2e'
    destination.mkdir(parents=True)
    checkpoint = destination/filename
    payload = io.BytesIO()
    torch.save({'state_dict': {'weight': torch.ones(2)}}, payload)
    data = payload.getvalue()
    if is_archive:
        with tarfile.open(checkpoint, 'w:gz') as archive:
            member = tarfile.TarInfo('pretrained_models/lam_audio2exp_streaming.tar')
            member.size = len(data)
            archive.addfile(member, io.BytesIO(data))
    else:
        checkpoint.write_bytes(data)
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/models.json').write_text(json.dumps({'a2e': {
        'modelscope_repo': 'fixture/model', 'hf_repo': 'fixture/model',
        'destination': 'models/a2e', 'allow': ['*.tar']}}))
    if provider == 'modelscope':
        monkeypatch.setattr(download_modelscope, 'list_files', lambda _: [{
            'Path': filename, 'Type': 'blob', 'Size': checkpoint.stat().st_size,
            'Sha256': sha256(checkpoint), 'Revision': 'fixture-revision'}])
        download_modelscope.download(tmp_path, ['a2e'])
    else:
        class Api:
            def model_info(self, *args, **kwargs): return SimpleNamespace(sha='fixture-revision')
        monkeypatch.setitem(sys.modules, 'huggingface_hub', SimpleNamespace(
            HfApi=Api, snapshot_download=lambda *args, **kwargs: None))
        download_models.download(tmp_path, ['a2e'])
    if is_archive:
        assert (destination/'pretrained_models/lam_audio2exp_streaming.tar').read_bytes() == data
    else:
        assert {path.name for path in destination.iterdir()} == {filename}, 'A Torch checkpoint is not an asset archive'


def locked_model(tmp_path, monkeypatch, replacement):
    import json
    from scripts import download_modelscope
    from backend.storage import sha256, atomic_json
    target = tmp_path/'models/example/model.bin'
    target.parent.mkdir(parents=True)
    target.write_bytes(b'accepted v1')
    config = tmp_path/'configs/models.json'
    config.parent.mkdir()
    config.write_text(json.dumps({'example': {'modelscope_repo': 'fixture/repo', 'destination': 'models/example'}}))
    atomic_json(tmp_path/'models/lock.json', {'example': {'repo': 'fixture/repo', 'provider': 'modelscope',
        'files': {'model.bin': sha256(target)}, 'revision': {'model.bin': 'v1'}}})
    source = tmp_path/'incoming.bin'
    source.write_bytes(replacement)
    monkeypatch.setattr(download_modelscope, 'list_files', lambda _: [{'Path': 'model.bin', 'Type': 'blob',
        'Size': source.stat().st_size, 'Sha256': sha256(source), 'Revision': 'v2'}])
    real_fetch = download_modelscope.fetch_file
    monkeypatch.setattr(download_modelscope, 'fetch_file',
                        lambda url, path, digest, size: real_fetch(source.as_uri(), path, digest, size))
    return target


def test_modelscope_does_not_replace_locked_weights_when_master_changes(tmp_path, monkeypatch):
    from scripts.download_modelscope import download
    target = locked_model(tmp_path, monkeypatch, b'different v2')
    with pytest.raises(RuntimeError, match='固定|locked'):
        download(tmp_path, ['example'])
    assert target.read_bytes() == b'accepted v1'


def test_modelscope_keeps_locked_revision_for_identical_bytes(tmp_path, monkeypatch):
    from scripts.download_modelscope import download
    locked_model(tmp_path, monkeypatch, b'accepted v1')
    lock = download(tmp_path, ['example'])
    assert lock['example']['revision']['model.bin'] == 'v1'
