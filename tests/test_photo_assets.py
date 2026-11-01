import json
import hashlib
from pathlib import Path
import pytest


def test_photo_setup_uses_only_the_selected_inference_models():
    from scripts.photo_assets import load_catalog
    catalog = load_catalog(Path(__file__).resolve().parents[1])
    assert set(catalog['models']) == {'ditto','cosyvoice','text_normalizer','llm','asr'}
    for model in catalog['models'].values():
        assert len(model['revision']) == 40
        assert model['files']
        assert all(len(row['sha256']) == 64 and row['size'] > 0 for row in model['files'])
    ditto = [row['name'] for row in catalog['models']['ditto']['files']]
    assert all(name.startswith('ditto_pytorch/') or name == 'ditto_cfg/v0.4_hubert_cfg_pytorch.pkl' for name in ditto)
    assert not any(name.endswith(('.engine','llm.rl.pt')) for model in catalog['models'].values() for name in [row['name'] for row in model['files']])


def test_photo_source_inventory_rejects_a_changed_commit(tmp_path):
    from scripts.photo_assets import check_source_lock
    folder=tmp_path/'source';folder.mkdir()
    (folder/'.source-lock.json').write_text(json.dumps({'commit':'different'}))
    with pytest.raises(ValueError,match='版本'):
        check_source_lock(folder,'a'*40)


def source_fixture(tmp_path):
    data=b'publisher source code'
    catalog={'sources':{'fixture':{'repo':'fixture/source','commit':'a'*40,
        'destination':'third_party/fixture','files':[{'name':'module.py','size':len(data),
        'sha1':hashlib.sha1(b'blob '+str(len(data)).encode()+b'\0'+data).hexdigest()}]}}}
    return catalog,data,tmp_path/'third_party/fixture'


def test_interrupted_source_write_leaves_no_final_file_and_can_retry(tmp_path,monkeypatch):
    from scripts import photo_assets
    catalog,data,folder=source_fixture(tmp_path)
    monkeypatch.setattr(photo_assets,'get',lambda _:data)
    original_write=Path.write_bytes
    def interrupted_write(path,payload):
        original_write(path,payload[:4])
        raise OSError('simulated full disk')
    monkeypatch.setattr(Path,'write_bytes',interrupted_write)
    with pytest.raises(OSError,match='full disk'):
        photo_assets.fetch_sources(tmp_path,catalog)
    assert not (folder/'module.py').exists()
    assert not (folder/'.source-lock.json').exists()
    assert list(folder.iterdir())==[]
    monkeypatch.setattr(Path,'write_bytes',original_write)
    photo_assets.fetch_sources(tmp_path,catalog)
    assert (folder/'module.py').read_bytes()==data
    assert json.loads((folder/'.source-lock.json').read_text())['commit']=='a'*40


def test_changed_existing_source_is_not_overwritten(tmp_path,monkeypatch):
    from scripts import photo_assets
    catalog,_,folder=source_fixture(tmp_path);folder.mkdir(parents=True)
    target=folder/'module.py';target.write_bytes(b'local edits')
    monkeypatch.setattr(photo_assets,'get',lambda _:pytest.fail('requested changed source'))
    with pytest.raises(ValueError,match='已修改'):
        photo_assets.fetch_sources(tmp_path,catalog)
    assert target.read_bytes()==b'local edits'
    assert not (folder/'.source-lock.json').exists()
