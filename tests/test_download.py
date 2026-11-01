import hashlib
import pytest


def test_download_checks_publisher_hash_before_accepting(tmp_path):
    from scripts.fetch_artifact import fetch_file
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
    from scripts.fetch_artifact import fetch_file
    source = tmp_path / "bad.bin"
    source.write_bytes(b"bad")
    with pytest.raises(ValueError, match="SHA"):
        fetch_file(source.as_uri(), tmp_path / "download.bin", "0"*64, 3)
    assert not (tmp_path / "download.bin").exists()


def test_download_restarts_a_complete_corrupt_partial(tmp_path):
    from scripts.fetch_artifact import fetch_file
    source = tmp_path / "source.bin"
    source.write_bytes(b"good")
    destination = tmp_path / "cache/model.bin"
    destination.parent.mkdir()
    destination.with_name("model.bin.partial").write_bytes(b"bad!")
    fetch_file(source.as_uri(), destination, hashlib.sha256(b"good").hexdigest(), 4)
    assert destination.read_bytes() == b"good"
