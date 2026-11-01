"""The photo runtime must start safely without the removed 3D toolchain."""
import hashlib
import json
from pathlib import Path

import httpx
import pytest

from scripts import services


def test_photo_services_use_independent_environments(tmp_path, monkeypatch):
    model=tmp_path/'models/llm/qwen2.5-3b-instruct-q4_k_m.gguf'
    model.parent.mkdir(parents=True);model.write_bytes(b'fixture')
    monkeypatch.setattr(Path,'is_file',lambda self:True)
    plan=services.service_plan(tmp_path)
    by_name={item['name']:item for item in plan}
    assert set(by_name)=={'dialogue','voice','portrait','studio'}
    assert by_name['voice']['command'][:4]==[
        '/opt/digitalhuman/emotional-voice/bin/python','-m','uvicorn','backend.voice.service:app']
    assert by_name['portrait']['command'][:4]==[
        '/opt/digitalhuman/portrait/bin/python','-m','uvicorn','backend.portrait.service:app']
    assert by_name['studio']['command'][0]=='/opt/digitalhuman/portrait/bin/python'


@pytest.mark.parametrize('old_name',['voice','expression','studio'])
def test_start_rejects_running_old_services_before_launch(tmp_path,monkeypatch,old_name):
    folder=tmp_path/'.runtime';folder.mkdir()
    (folder/'services.json').write_text(json.dumps({old_name:{
        'pid':1234,'start_ticks':'1','port':9880,'command':['old-runtime','old-entry']}}))
    monkeypatch.setattr(services,'alive',lambda record:record.get('pid')==1234)
    monkeypatch.setattr(services,'service_plan',lambda *_:[{
        'name':'studio','command':['new-runtime'],'port':7860}])
    monkeypatch.setattr(services.subprocess,'Popen',lambda *a,**k:pytest.fail('launched before migration check'))
    with pytest.raises(RuntimeError,match='stop.ps1'):
        services.start(tmp_path)


def distribution(folder,name='torch',version='2.3.0+cu121',extra_record=''):
    package=folder/name;package.mkdir(parents=True)
    (package/'__init__.py').write_text('version=1')
    metadata=folder/f'{name}-{version}.dist-info';metadata.mkdir()
    (metadata/'METADATA').write_text(f'Metadata-Version: 2.1\nName: {name}\nVersion: {version}\n')
    entries=[f'{name}/__init__.py',f'{metadata.name}/METADATA',f'{metadata.name}/RECORD']
    (metadata/'RECORD').write_text(''.join(f'{item},,\n' for item in entries)+extra_record)


def test_reuse_copies_identical_pinned_distribution_inside_target_only(tmp_path):
    from scripts.reuse_environment import reuse
    source=tmp_path/'source';target=tmp_path/'target';target.mkdir()
    outside=tmp_path/'outside.py';outside.write_text('private')
    distribution(source,extra_record='../outside.py,,\n')
    result=reuse(target,[source],['torch==2.3.0+cu121'])
    assert result['reused']==['torch']
    assert (target/'torch/__init__.py').read_text()=='version=1'
    assert outside.read_text()=='private'
    assert not (target.parent/'outside.py').samefile(target/'torch/__init__.py')
    assert (target/'torch/__init__.py').stat().st_nlink>=2


@pytest.mark.parametrize('conflict',['package','version'])
def test_reuse_never_overwrites_existing_different_package(tmp_path,conflict):
    from scripts.reuse_environment import reuse
    source=tmp_path/'source';target=tmp_path/'target';target.mkdir()
    distribution(source)
    if conflict=='version':distribution(target,version='2.5.1')
    else:
        (target/'torch').mkdir();(target/'torch/__init__.py').write_text('different')
    before={p.relative_to(target):p.read_bytes() for p in target.rglob('*') if p.is_file()}
    assert reuse(target,[source],['torch==2.3.0+cu121'])['reused']==[]
    assert {p.relative_to(target):p.read_bytes() for p in target.rglob('*') if p.is_file()}==before


def test_reuse_rejects_source_symlinks(tmp_path):
    from scripts.reuse_environment import reuse
    source=tmp_path/'source';target=tmp_path/'target';target.mkdir()
    distribution(source)
    package=source/'torch/__init__.py';package.unlink()
    secret=tmp_path/'secret';secret.write_text('private')
    try:package.symlink_to(secret)
    except OSError:pytest.skip('symlink permission unavailable')
    assert reuse(target,[source],['torch==2.3.0+cu121'])['reused']==[]
    assert not (target/'torch').exists()


def test_default_download_uses_pinned_photo_catalog(tmp_path,monkeypatch):
    from scripts import download_models,photo_assets
    catalog={'sources':{},'models':{'ditto':{'revision':'fixed'}}}
    (tmp_path/'configs').mkdir()
    (tmp_path/'configs/photo-models.json').write_text(json.dumps(catalog))
    downloaded=[]
    monkeypatch.setattr(photo_assets,'fetch_models',lambda root,data,selected=None:downloaded.append(data))
    assert download_models.download(tmp_path)==catalog['models']
    assert downloaded==[catalog]


def photo_download_fixture(tmp_path,monkeypatch,installed):
    from scripts import photo_assets
    payloads={'ditto':b'portrait weights','llm':b'dialogue weights'}
    catalog={'sources':{},'models':{name:{'repo':'fixture/'+name,'revision':'fixed',
        'destination':'models/'+name,'files':[{'name':'weights.bin','size':len(data),
            'sha256':hashlib.sha256(data).hexdigest(),'url':'fixture://'+name}]}
        for name,data in payloads.items()}}
    (tmp_path/'configs').mkdir();(tmp_path/'models').mkdir()
    (tmp_path/'configs/photo-models.json').write_text(json.dumps(catalog))
    if installed:
        for name,data in payloads.items():
            path=tmp_path/f'models/{name}/weights.bin';path.parent.mkdir();path.write_bytes(data)
        (tmp_path/'models/photo-lock.json').write_text(json.dumps(catalog))
    def fetch(url,path,*_):
        # This local fixture catches accidental downloads of unselected weights.
        assert url=='fixture://ditto'
        path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes(payloads['ditto'])
    monkeypatch.setattr(photo_assets,'fetch_file',fetch)
    return catalog


def test_selective_download_preserves_complete_existing_model_lock(tmp_path,monkeypatch):
    from scripts.download_models import download
    catalog=photo_download_fixture(tmp_path,monkeypatch,installed=True)
    assert download(tmp_path,['ditto'])=={'ditto':catalog['models']['ditto']}
    assert json.loads((tmp_path/'models/photo-lock.json').read_text())==catalog
    assert (tmp_path/'models/llm/weights.bin').read_bytes()==b'dialogue weights'


def test_selective_fresh_download_cannot_hide_missing_model_from_doctor(tmp_path,monkeypatch):
    from scripts import doctor,check_frontend
    from scripts.download_models import download
    catalog=photo_download_fixture(tmp_path,monkeypatch,installed=False)
    download(tmp_path,['ditto'])
    assert not (tmp_path/'models/photo-lock.json').exists()
    monkeypatch.setattr(doctor,'probe_environment',lambda *_:{'ok':True,'detail':'fixture'})
    monkeypatch.setattr(doctor,'port_checks',lambda *_:[])
    monkeypatch.setattr(check_frontend,'check',lambda *_:True)
    result=doctor.inspect(tmp_path)
    assert next(item for item in result['checks'] if item['name']=='hash_llm')['ok'] is False
    assert result['ready'] is False


def test_selective_download_rejects_unknown_model_without_changing_lock(tmp_path,monkeypatch):
    from scripts.download_models import download
    catalog=photo_download_fixture(tmp_path,monkeypatch,installed=True)
    with pytest.raises(ValueError,match='unknown'):
        download(tmp_path,['unknown'])
    assert json.loads((tmp_path/'models/photo-lock.json').read_text())==catalog


@pytest.mark.parametrize('invalid_payload',[b'short',b'X'*16])
def test_selective_download_does_not_publish_lock_for_corrupt_unselected_weights(tmp_path,monkeypatch,invalid_payload):
    from scripts.download_models import download
    photo_download_fixture(tmp_path,monkeypatch,installed=True)
    lock=tmp_path/'models/photo-lock.json';lock.unlink()
    (tmp_path/'models/llm/weights.bin').write_bytes(invalid_payload)
    download(tmp_path,['ditto'])
    assert not lock.exists()


def test_build_reuses_existing_profile_without_import(tmp_path):
    from scripts.build_avatar import prepare
    folder=tmp_path/'.runtime';folder.mkdir();(folder/'default_avatar.txt').write_text('person_1')
    requests=[]
    def handle(request):
        requests.append((request.method,request.url.path))
        responses={
            ('GET','/api/avatars/person_1'):{'id':'person_1'},
            ('POST','/api/avatars/person_1/build'):{'job_id':'job_1'},
            ('GET','/api/jobs/job_1'):{'status':'waiting','stage':'reference_confirmation'}}
        return httpx.Response(200,json=responses[(request.method,request.url.path)])
    with httpx.Client(base_url='http://127.0.0.1:7860',transport=httpx.MockTransport(handle)) as client:
        result=prepare(tmp_path,client)
    assert result['status']=='waiting'
    assert requests==[('GET','/api/avatars/person_1'),('POST','/api/avatars/person_1/build'),('GET','/api/jobs/job_1')]


def test_build_requires_explicit_inputs_when_no_imported_profile(tmp_path):
    from scripts.build_avatar import prepare
    with httpx.Client(transport=httpx.MockTransport(lambda _:pytest.fail('sent request'))) as client:
        with pytest.raises(ValueError,match='photo|照片'):
            prepare(tmp_path,client)


def test_build_imports_only_explicit_photo_and_voice_video(tmp_path):
    from scripts.build_avatar import prepare
    photo=tmp_path/'photo.png';photo.write_bytes(b'photo')
    video=tmp_path/'voice.mp4';video.write_bytes(b'voice')
    def handle(request):
        if request.url.path=='/api/avatars/import':
            assert b'name="photo"' in request.content and b'name="voice_video"' in request.content
            return httpx.Response(201,json={'avatar_id':'person_2'})
        if request.url.path.endswith('/build'):return httpx.Response(202,json={'job_id':'job_2'})
        return httpx.Response(200,json={'status':'complete'})
    with httpx.Client(base_url='http://127.0.0.1:7860',transport=httpx.MockTransport(handle)) as client:
        assert prepare(tmp_path,client,photo=photo,voice_video=video)['status']=='complete'
    assert (tmp_path/'.runtime/default_avatar.txt').read_text()=='person_2'


def test_doctor_detects_modified_pinned_photo_weights(tmp_path,monkeypatch):
    from scripts import doctor,check_frontend
    (tmp_path/'configs').mkdir();(tmp_path/'models').mkdir()
    catalog={'sources':{},'models':{'ditto':{'destination':'models','revision':'fixed',
        'files':[{'name':'weight.bin','size':4,'sha256':hashlib.sha256(b'good').hexdigest()}]}}}
    (tmp_path/'models/weight.bin').write_bytes(b'evil')
    (tmp_path/'configs/photo-models.json').write_text(json.dumps(catalog))
    (tmp_path/'models/photo-lock.json').write_text(json.dumps(catalog))
    monkeypatch.setattr(doctor,'probe_environment',lambda *_:{'ok':True,'detail':'fixture'})
    monkeypatch.setattr(doctor,'port_checks',lambda *_:[])
    monkeypatch.setattr(check_frontend,'check',lambda *_:True)
    result=doctor.inspect(tmp_path)
    assert not result['ready']
    assert next(item for item in result['checks'] if item['name']=='hash_ditto')['ok'] is False


def test_doctor_environment_probe_reports_import_or_cuda_failure(tmp_path,monkeypatch):
    from scripts import doctor
    from types import SimpleNamespace
    def run(command,**kwargs):
        assert command[0]=='/opt/digitalhuman/emotional-voice/bin/python'
        assert 'torch.cuda.is_available()' in command[-1]
        assert "'cosyvoice.cli.cosyvoice'" in command[-1]
        return SimpleNamespace(returncode=1,stdout='',stderr='CUDA unavailable')
    monkeypatch.setattr(doctor.subprocess,'run',run)
    assert doctor.probe_environment(tmp_path,'emotional-voice')=={
        'ok':False,'detail':'CUDA unavailable'}


def test_studio_only_does_not_reject_current_photo_workers(tmp_path,monkeypatch):
    folder=tmp_path/'.runtime';folder.mkdir()
    studio=['/opt/digitalhuman/portrait/bin/python','-m','uvicorn','backend.api.app:app','--host','127.0.0.1','--port','7860']
    voice=['/opt/digitalhuman/emotional-voice/bin/python','-m','uvicorn','backend.voice.service:app','--host','127.0.0.1','--port','9880']
    (folder/'services.json').write_text(json.dumps({
        'studio':{'pid':1,'command':studio},'voice':{'pid':2,'command':voice}}))
    monkeypatch.setattr(services,'alive',lambda record:record.get('pid') in {1,2})
    monkeypatch.setattr(services,'service_plan',lambda *_:[{'name':'studio','command':studio,'port':7860}])
    monkeypatch.setattr(services.subprocess,'Popen',lambda *a,**k:pytest.fail('already running'))
    services.start(tmp_path,studio_only=True)
