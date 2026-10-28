from fastapi.testclient import TestClient


def test_api_never_connects_an_unbuilt_person(tmp_path):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/sessions", json={"avatar_id": "not-built"})
        assert response.status_code == 409
        assert "人物" in response.json()["detail"]


def test_import_api_preserves_clothing_roles_and_returns_persistent_id(tmp_path):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/avatars/import", data={"name": "人物"}, files=[
            ("video", ("video.mp4", b"source", "video/mp4")),
            ("body_photos", ("body.png", b"body", "image/png")),
            ("face_photos", ("face.jpg", b"face", "image/jpeg"))])
        assert response.status_code == 201
        avatar_id = response.json()["avatar_id"]
    with TestClient(create_app(tmp_path)) as client:
        avatar = client.get(f"/api/avatars/{avatar_id}").json()
        assert avatar["body_photos"] == ["body.png"]
        assert avatar["face_photos"] == ["face.jpg"]
        assert avatar["status"] == "imported"


def test_import_api_rejects_parent_paths(tmp_path):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        response = client.post("/api/avatars/import", data={"name": "人物"}, files=[
            ("video", ("../../video.mp4", b"source", "video/mp4")),
            ("body_photos", ("body.png", b"body", "image/png"))])
        assert response.status_code == 400


def test_reference_confirmation_does_not_forge_voice_validation(tmp_path):
    from backend.api.app import create_app
    with TestClient(create_app(tmp_path)) as client:
        imported = client.post("/api/avatars/import", data={"name": "人物"}, files=[
            ("video", ("video.mp4", b"source", "video/mp4")),
            ("body_photos", ("body.png", b"body", "image/png"))]).json()
        avatar_id = imported["avatar_id"]
        response = client.put(f"/api/avatars/{avatar_id}/voice", json={"text": "参考录音的真实文本", "confirmed": True})
        assert response.status_code == 200
        manifest = client.get(f"/api/avatars/{avatar_id}").json()
        assert manifest["voice"]["confirmed"] is True
        assert manifest["validation"]["voice"] is False
