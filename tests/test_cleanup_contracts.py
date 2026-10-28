import io
import json
import zipfile
import numpy as np
import pytest


@pytest.mark.parametrize('legacy_suffix,marker_name',[(False,'assets.zip.extracted'),(True,'assets.extracted')])
def test_shared_archive_extraction_keeps_each_providers_existing_marker(tmp_path,legacy_suffix,marker_name):
    from backend.storage import sha256
    from scripts.model_archives import extract_archives
    archive=tmp_path/'assets.zip'
    with zipfile.ZipFile(archive,'w') as output:output.writestr('asset.bin',b'published')
    digest=sha256(archive)
    extract_archives(tmp_path,{'assets.zip':digest},legacy_suffix=legacy_suffix)
    assert (tmp_path/marker_name).read_text()==digest
    assert (tmp_path/'asset.bin').read_bytes()==b'published'
    (tmp_path/'asset.bin').write_bytes(b'already-extracted')
    extract_archives(tmp_path,{'assets.zip':digest},legacy_suffix=legacy_suffix)
    assert (tmp_path/'asset.bin').read_bytes()==b'already-extracted'


@pytest.mark.parametrize('timeout,headers',[(180,{'Content-Type':'audio/wav'}),(300,None)])
async def test_shared_expression_client_preserves_payload_timeout_and_weights(monkeypatch,timeout,headers):
    import httpx
    from backend.avatar import a2e_client
    expected=np.arange(104,dtype=np.float32).reshape(2,52)
    result=io.BytesIO();np.savez_compressed(result,weights=expected)
    def respond(request):
        assert request.url==httpx.URL('http://127.0.0.1:9881/expression')
        assert request.content==b'original-wav'
        assert request.extensions['timeout']['read']==timeout
        assert request.headers.get('Content-Type')==(headers or {}).get('Content-Type')
        return httpx.Response(200,content=result.getvalue())
    original=httpx.AsyncClient
    monkeypatch.setattr(a2e_client.httpx,'AsyncClient',lambda **kwargs:original(transport=httpx.MockTransport(respond),**kwargs))
    actual=await a2e_client.request_weights(b'original-wav',timeout=timeout,headers=headers)
    np.testing.assert_array_equal(actual,expected)


@pytest.mark.parametrize('status,pickled',[(503,False),(200,True)])
async def test_shared_expression_client_propagates_http_and_unsafe_array_errors(monkeypatch,status,pickled):
    import httpx
    from backend.avatar import a2e_client
    result=io.BytesIO();np.savez(result,weights=np.array([{'unsafe':'object'}],dtype=object))
    original=httpx.AsyncClient
    transport=httpx.MockTransport(lambda request:httpx.Response(status,content=result.getvalue()))
    monkeypatch.setattr(a2e_client.httpx,'AsyncClient',lambda **kwargs:original(transport=transport,**kwargs))
    with pytest.raises(ValueError if pickled else httpx.HTTPStatusError):
        await a2e_client.request_weights(b'wav',timeout=180)


def test_state_loader_keeps_one_descriptor_snapshot_during_version_change(tmp_path,monkeypatch):
    import torch
    from backend.avatar import state
    state.save_state(tmp_path,{'original':torch.tensor([1.])})
    original=json.loads((tmp_path/'state.json').read_text())
    state.save_state(tmp_path,{'replacement':torch.tensor([2.])})
    original_read=state.read_json
    def descriptor(path):
        nonlocal original
        if original is not None:
            snapshot,original=original,None
            return snapshot
        return original_read(path)
    monkeypatch.setattr(state,'read_json',descriptor)
    loaded=state.load_state(tmp_path)
    assert set(loaded)=={'original'}
    torch.testing.assert_close(loaded['original'],torch.tensor([1.]))
