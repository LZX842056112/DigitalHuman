import time
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError
from backend.api.app import SessionRequest,create_app


def test_unsupported_native_size_is_rejected_before_gpu_allocation():
    with pytest.raises(ValidationError):
        SessionRequest(avatar_id='fixture',render_size=8192)


def test_requested_native_size_reaches_the_owned_gpu_loader(tmp_path,monkeypatch):
    from backend.storage import Store
    from backend.avatar.lhmpp import LHMRenderer
    from backend.transport import session
    sizes=[]
    camera={'yaw':0.,'pitch':0.,'distance':3.,'target':[0.,0.,0.]}
    class Renderer:
        default_camera=camera
        render_size=768
    def load(root,folder,render_size=512):
        sizes.append(render_size);return Renderer()
    class FakeSession:
        def __init__(self,root,manifest,renderer,config,executor):
            self.id='detail';self.closed=False;self.last_active=time.monotonic();self.executor=executor
        async def close(self):
            self.closed=True;self.executor.shutdown();self.gpu_lease.release()
    monkeypatch.setattr(Store,'require_ready',lambda *_:{'id':'fixture'})
    monkeypatch.setattr(LHMRenderer,'load',load)
    monkeypatch.setattr(session,'Session',FakeSession)
    with TestClient(create_app(tmp_path)) as client:
        response=client.post('/api/sessions',json={'avatar_id':'fixture','render_size':768})
        assert response.status_code==201,response.text
        assert response.json()['render_size']==768
    assert sizes==[768]
