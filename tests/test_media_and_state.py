import io
import tarfile
import wave
from pathlib import Path
import numpy as np
import pytest


def test_reference_selection_avoids_silence_and_clipping():
    from backend.pipeline.media import choose_reference
    sr = 16000
    t = np.arange(sr * 6) / sr
    speech = (.15 * np.sin(2 * np.pi * 180 * t)).astype(np.float32)
    pcm = np.concatenate([np.zeros(sr * 7), speech, np.ones(sr * 7)])
    start, duration = choose_reference(pcm, sr, 6)
    assert 6.5 <= start <= 7.5
    assert duration == 6


def test_reference_selection_rejects_silence():
    from backend.pipeline.media import choose_reference
    from backend.errors import PipelineError
    with pytest.raises(PipelineError, match="语音"):
        choose_reference(np.zeros(16000 * 12), 16000)


def test_audio_decode_resamples_without_mutating_source(tmp_path):
    from backend.pipeline.media import read_audio, write_wav
    src = tmp_path / "voice.wav"
    with wave.open(str(src), "wb") as f:
        f.setnchannels(1)
        f.setsampwidth(2)
        f.setframerate(8000)
        f.writeframes(np.full(8000, 1000, dtype="<i2").tobytes())
    original = src.read_bytes()
    pcm, sr = read_audio(src, 16000)
    assert sr == 16000
    assert abs(len(pcm) - 16000) <= 2
    assert src.read_bytes() == original
    write_wav(tmp_path / "copy.wav", pcm, sr)
    with wave.open(str(tmp_path / "copy.wav")) as f:
        assert f.getframerate() == 16000


def test_tensor_package_roundtrip_preserves_nested_types(tmp_path):
    import torch
    from backend.avatar.state import save_state, load_state
    state = {"query": {"points": torch.arange(9).reshape(3, 3)},
             "latent": [torch.tensor([.2, .4]), None], "shape": (1, 2), "flag": True}
    save_state(tmp_path, state)
    restored = load_state(tmp_path)
    assert torch.equal(restored["query"]["points"], torch.arange(9).reshape(3, 3))
    assert restored["shape"] == (1, 2)
    assert restored["latent"][1] is None
    assert restored["flag"] is True


def test_state_loader_rejects_executable_object_tags(tmp_path):
    from backend.avatar.state import decode_tree
    with pytest.raises(ValueError):
        decode_tree({"__object__": "os.system", "args": "ignored"}, {})


def test_camera_orbit_changes_actual_transform_and_keeps_orthonormal_basis():
    from backend.avatar.camera import orbit_camera
    a, intr = orbit_camera(0, 0, 3, [0, 0, 0], 512, 512)
    b, _ = orbit_camera(90, 0, 3, [0, 0, 0], 512, 512)
    np.testing.assert_allclose(a[:3, 3], [0, 0, 3], atol=1e-5)
    np.testing.assert_allclose(b[:3, 3], [3, 0, 0], atol=1e-5)
    np.testing.assert_allclose(b[:3, :3].T @ b[:3, :3], np.eye(3), atol=1e-5)
    assert intr[0, 2] == 256


def test_archive_extraction_rejects_path_traversal(tmp_path):
    from scripts.download_models import safe_extract
    path = tmp_path / "bad.tar"
    with tarfile.open(path, "w") as tar:
        item = tarfile.TarInfo("../escape.txt")
        item.size = 4
        tar.addfile(item, io.BytesIO(b"oops"))
    with pytest.raises(ValueError):
        safe_extract(path, tmp_path / "out")
    assert not (tmp_path / "escape.txt").exists()
