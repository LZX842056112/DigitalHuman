from pathlib import Path
import hashlib
import numpy as np
import pytest


def test_import_preserves_source_and_roles(tmp_path):
    from backend.storage import Store
    source = tmp_path / "source"
    source.mkdir()
    (source / "clip.mp4").write_bytes(b"original video bytes")
    (source / "body.png").write_bytes(b"body appearance")
    (source / "face.jpg").write_bytes(b"different clothes")
    store = Store(tmp_path / "project")
    avatar = store.import_files("person", source / "clip.mp4", [source / "body.png"], [source / "face.jpg"])
    assert avatar["status"] == "imported"
    assert avatar["body_photos"] == ["body.png"]
    assert avatar["face_photos"] == ["face.jpg"]
    copy = store.avatar_dir(avatar["id"]) / "source" / "clip.mp4"
    assert copy.read_bytes() == b"original video bytes"
    assert avatar["sources"]["clip.mp4"] == hashlib.sha256(b"original video bytes").hexdigest()
    assert (source / "clip.mp4").read_bytes() == b"original video bytes"


def test_job_failure_survives_store_restart(tmp_path):
    from backend.storage import Store
    store = Store(tmp_path)
    job = store.new_job("a", "build")
    store.update_job(job["id"], status="failed", stage="reconstruct", error="CUDA memory unavailable")
    assert Store(tmp_path).job(job["id"])["error"] == "CUDA memory unavailable"
    assert Store(tmp_path).job(job["id"])["status"] == "failed"


def test_unbuilt_avatar_never_reports_ready(tmp_path):
    from backend.storage import Store
    from backend.errors import NotReady
    store = Store(tmp_path)
    with pytest.raises(NotReady, match="人物"):
        store.require_ready("missing")


def test_ready_requires_verified_state_and_voice(tmp_path):
    from backend.storage import Store
    from backend.errors import NotReady
    store = Store(tmp_path)
    folder = store.avatar_dir("a")
    folder.mkdir(parents=True)
    store.write_manifest("a", {"id": "a", "status": "ready", "state": "state.safetensors", "voice": {"confirmed": False}})
    with pytest.raises(NotReady):
        store.require_ready("a")


def test_path_traversal_is_rejected(tmp_path):
    from backend.storage import Store
    with pytest.raises(ValueError):
        Store(tmp_path).avatar_dir("../../outside")


def test_expression_mapping_fits_known_vertices_and_roundtrips(tmp_path):
    from backend.avatar.expressions import ExpressionMap
    flame = np.array([[1., 0.], [0., 2.], [1., 1.]])
    arkit = np.array([[1., 2.], [2., 0.], [2., 2.]])
    mapping = ExpressionMap.fit(flame, arkit, ridge=0)
    expression, jaw = mapping.apply(np.array([1., 0.]))
    np.testing.assert_allclose(expression, [1., 1.], atol=1e-6)
    np.testing.assert_allclose(jaw, [0., 0., 0.])
    mapping.save(tmp_path / "map.npz")
    restored = ExpressionMap.load(tmp_path / "map.npz")
    np.testing.assert_allclose(restored.apply(np.array([0., 1.]))[0], [2., 0.], atol=1e-6)


def test_expression_mapping_rejects_unmatched_templates():
    from backend.avatar.expressions import ExpressionMap
    with pytest.raises(ValueError, match="顶点"):
        ExpressionMap.fit(np.zeros((9, 100)), np.zeros((12, 52)))


def test_actions_move_limbs_and_keep_idle_root_stationary():
    from backend.avatar.motion import sample_motion
    idle = sample_motion("idle", 0.5)
    assert idle["body_pose"].shape == (21, 3)
    np.testing.assert_allclose(idle["trans"], [0, 0, 0])
    wave0 = sample_motion("wave", 0)
    wave1 = sample_motion("wave", 0.7)
    assert not np.allclose(wave0["body_pose"], wave1["body_pose"])
    walk = sample_motion("walk", 0.3)
    assert abs(walk["body_pose"][0, 0]) > .05
    with pytest.raises(ValueError):
        sample_motion("unknown", 0)


def test_interrupt_discards_old_audio_and_resets_expression():
    from backend.transport.timeline import Timeline
    clock = Timeline()
    token = clock.begin_turn()
    clock.append(token, np.ones(1600, dtype=np.float32), 16000, np.ones((3, 100)), np.zeros((3, 3)))
    assert clock.read_audio(160)[0].max() > 0
    clock.interrupt()
    assert clock.read_audio(160)[0].max() == 0
    assert clock.expression()[0].max() == 0
    assert not clock.append(token, np.ones(100), 16000, np.ones((1, 100)), np.zeros((1, 3)))


def test_timeline_uses_audio_samples_for_expression_time():
    from backend.transport.timeline import Timeline
    clock = Timeline()
    token = clock.begin_turn()
    exp = np.zeros((30, 100))
    exp[:, 0] = np.arange(30)
    clock.append(token, np.ones(16000, np.float32), 16000, exp, np.zeros((30, 3)))
    clock.read_audio(24000)  # Half a second at the WebRTC track's 48 kHz.
    assert clock.expression()[0][0] == pytest.approx(15, abs=1)
