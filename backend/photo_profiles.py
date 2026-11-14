"""Photo/voice sources have independent identities and immutable raw copies.

One avatar can hold several appearance variants. Each variant is just another
source photo with its own portrait cache, because the renderer only repaints
the head and cannot change clothes. The top-level photo/portrait/validation
fields stay as a mirror of the active variant so every existing reader keeps
working; ``sync_active_variant`` is the only writer of that mirror.
"""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid
from backend.storage import sha256, read_json
from backend import style
from backend.errors import NotReady
from backend.portrait.assets import portrait_version


VOICE_ARTIFACTS=('voice/reference.wav','voice/sample.wav')
DEFAULT_VARIANT_LIMIT=8


def source_path(folder, relative):
    target=(folder/relative).resolve()
    if folder.resolve() not in target.parents:raise ValueError('素材路径越界')
    return target


def copy_source(folder, role, path):
    path=Path(path)
    allowed={'.png','.jpg','.jpeg'} if role=='photo' else {'.mp4','.mov','.mkv','.webm'}
    if path.suffix.lower() not in allowed or not path.is_file():raise ValueError('素材格式无效或文件不存在')
    relative=f'source/{role}/{uuid.uuid4().hex}/{path.name}'
    target=source_path(folder,relative);target.parent.mkdir(parents=True,exist_ok=True)
    shutil.copy2(path,target)
    digest=sha256(target)
    if digest!=sha256(path):raise OSError('素材复制校验失败')
    return relative,digest


def _variant_id():
    return 'v'+uuid.uuid4().hex[:11]


def _variant(manifest, variant_id):
    for item in manifest.get('variants') or []:
        if item.get('id')==variant_id:return item
    return None


def active_variant(manifest):
    variants=manifest.get('variants') or []
    if not variants:return None
    return _variant(manifest,manifest.get('active_variant')) or variants[0]


def resolve_variant(manifest, variant_id=None):
    variants=manifest.get('variants') or []
    if not variants:return None
    if variant_id is None:return active_variant(manifest)
    found=_variant(manifest,variant_id)
    if found is None:raise ValueError('造型不存在')
    return found


def sync_active_variant(manifest):
    """Mirror the active variant into the legacy top-level fields."""
    variant=active_variant(manifest)
    if variant is None:
        manifest['portrait']=None
        manifest['validation']['photo']=False
        return manifest
    manifest['photo']=variant['photo']
    manifest['portrait']=deepcopy(variant.get('portrait'))
    manifest['validation']['photo']=bool((variant.get('validation') or {}).get('photo'))
    return manifest


def _prepared(variant):
    return bool(variant and variant.get('portrait') and (variant.get('validation') or {}).get('photo'))


def status_for(manifest):
    variant=active_variant(manifest)
    prepared=_prepared(variant)
    if manifest.get('voice',{}).get('confirmed'):
        return 'ready' if prepared else 'imported'
    return 'voice_pending' if prepared else 'imported'


def import_photo(store, name, photo, voice_video):
    avatar_id=uuid.uuid4().hex;folder=store.avatar_dir(avatar_id)
    image,image_hash=copy_source(folder,'photo',photo)
    video,video_hash=copy_source(folder,'voice_video',voice_video)
    variant={'id':_variant_id(),'label':'默认','photo':image,
             'added_at':datetime.now(timezone.utc).isoformat(),
             'portrait':None,'validation':{'photo':False}}
    manifest={'schema_version':3,'mode':'photo','id':avatar_id,'name':name[:80],
        'created_at':datetime.now(timezone.utc).isoformat(),'status':'imported',
        'settings':dict(style.DEFAULT_SETTINGS),
        'variants':[variant],'active_variant':variant['id'],
        'photo':image,'voice_video':video,'sources':{image:image_hash,video:video_hash},
        'voice':{'confirmed':False,'language':'zh'},'validation':{'photo':False,'voice':False},
        'portrait':None,'last_job_id':None}
    store.write_manifest(avatar_id,manifest)
    return manifest


def _v2_to_v3(manifest):
    photo=manifest.get('photo');portrait=manifest.get('portrait')
    variant={'id':_variant_id(),'label':'默认','photo':photo,
             'added_at':manifest.get('created_at') or datetime.now(timezone.utc).isoformat(),
             'portrait':deepcopy(portrait),
             'validation':{'photo':bool((manifest.get('validation') or {}).get('photo'))}}
    manifest.update(schema_version=3,
        settings={**style.DEFAULT_SETTINGS,**(manifest.get('settings') or {})},
        variants=[variant],active_variant=variant['id'],portrait=deepcopy(portrait))
    return manifest


def migrate_profile(manifest):
    if manifest.get('schema_version')==3 and manifest.get('mode')=='photo':return deepcopy(manifest)
    if manifest.get('schema_version')==2 and manifest.get('mode')=='photo':
        return _v2_to_v3(deepcopy(manifest))
    photo=manifest['body_photos'][0];video=manifest['video']
    result={key:manifest[key] for key in ('id','name','created_at')}
    result.update(schema_version=2,mode='photo',status='imported',photo='source/'+photo,voice_video='source/'+video,
        sources={'source/'+name:digest for name,digest in manifest['sources'].items()},
        voice=deepcopy(manifest.get('voice',{'confirmed':False,'language':'zh'})),
        validation={'photo':False,'voice':False},portrait=None,last_job_id=None)
    return _v2_to_v3(result)


def persist_photo_profile(store, avatar_id):
    """Keep the exact previous JSON before the first v3 write."""
    original=store.manifest(avatar_id)
    manifest=migrate_profile(original)
    current=(original.get('schema_version')==3 and original.get('mode')=='photo')
    if not current:
        folder=store.avatar_dir(avatar_id)
        name='avatar.legacy.json' if original.get('schema_version')!=2 else 'avatar.v2.json'
        backup=folder/name
        if not backup.exists():shutil.copy2(folder/'avatar.json',backup)
        store.write_manifest(avatar_id,manifest)
    return manifest


def pruned_hashes(manifest):
    """Keep only hashes that still belong to a variant or to the voice."""
    hashes=manifest.get('artifact_hashes') or {}
    folders=[item['portrait']['folder'] for item in manifest.get('variants') or []
             if item.get('portrait')]
    def keep(path):
        if path in VOICE_ARTIFACTS:return True
        return any(path.startswith(folder+'/') for folder in folders)
    return {path:digest for path,digest in hashes.items() if keep(path)}


def _add_source(store, avatar_id, manifest, path):
    folder=store.avatar_dir(avatar_id)
    relative,digest=copy_source(folder,'photo',path)
    manifest['sources'][relative]=digest
    variant={'id':_variant_id(),'label':'新造型','photo':relative,
             'added_at':datetime.now(timezone.utc).isoformat(),
             'portrait':None,'validation':{'photo':False}}
    manifest.setdefault('variants',[]).append(variant)
    store.write_manifest(avatar_id,manifest)
    return variant


def add_variant(store, avatar_id, path, label=None, limit=DEFAULT_VARIANT_LIMIT):
    manifest=persist_photo_profile(store,avatar_id)
    variants=manifest.setdefault('variants',[])
    if len(variants)>=max(1,int(limit)):
        raise ValueError(f'最多保存 {int(limit)} 套造型，请先删除不用的造型')
    variant=_add_source(store,avatar_id,manifest,path)
    if label:variant['label']=str(label)[:20]
    store.write_manifest(avatar_id,manifest)
    return manifest


def replace_variant(store, avatar_id, variant_id, path):
    manifest=persist_photo_profile(store,avatar_id)
    variant=resolve_variant(manifest,variant_id)
    folder=store.avatar_dir(avatar_id)
    incoming=sha256(Path(path))
    current=source_path(folder,variant['photo'])
    if (incoming==manifest['sources'].get(variant['photo'])
            and current.is_file() and sha256(current)==incoming):return manifest
    relative,digest=copy_source(folder,'photo',path)
    variant['photo']=relative;variant['portrait']=None;variant['validation']={'photo':False}
    manifest['sources'][relative]=digest
    manifest['artifact_hashes']=pruned_hashes(manifest)
    if variant['id']==manifest.get('active_variant'):
        sync_active_variant(manifest)
        manifest['status']='imported'
    store.write_manifest(avatar_id,manifest)
    return manifest


def delete_variant(store, avatar_id, variant_id):
    manifest=persist_photo_profile(store,avatar_id)
    variants=manifest.get('variants') or []
    if len(variants)<=1:raise ValueError('至少保留一套造型')
    variant=resolve_variant(manifest,variant_id)
    if variant['id']==manifest.get('active_variant'):raise ValueError('请先把其他造型设为使用中')
    manifest['variants']=[item for item in variants if item['id']!=variant['id']]
    manifest['artifact_hashes']=pruned_hashes(manifest)
    store.write_manifest(avatar_id,manifest)
    return manifest


def set_active_variant(store, avatar_id, variant_id):
    manifest=persist_photo_profile(store,avatar_id)
    variant=resolve_variant(manifest,variant_id)
    if not _prepared(variant):raise ValueError('请先准备该造型再切换')
    manifest['active_variant']=variant['id']
    sync_active_variant(manifest)
    manifest['status']=status_for(manifest)
    store.write_manifest(avatar_id,manifest)
    return manifest


def replace_source(store, avatar_id, role, path):
    if role=='voice_video':
        manifest=persist_photo_profile(store,avatar_id);folder=store.avatar_dir(avatar_id)
        incoming_hash=sha256(Path(path))
        current=source_path(folder,manifest['voice_video'])
        if (incoming_hash==manifest['sources'].get(manifest['voice_video'])
                and current.is_file() and sha256(current)==incoming_hash):return manifest
        relative,digest=copy_source(folder,'voice_video',path)
        manifest['voice_video']=relative;manifest['sources'][relative]=digest
        manifest['status']='imported'
        manifest['validation']['voice']=False
        manifest['voice']={'confirmed':False,'language':manifest['voice'].get('language','zh')}
        store.write_manifest(avatar_id,manifest)
        return manifest
    if role!='photo':raise ValueError('未知素材类型')
    return replace_variant(store,avatar_id,None,path)


def validate_photo_sources(folder, manifest):
    voice_video=manifest['voice_video'];path=source_path(folder,voice_video)
    if not path.is_file() or sha256(path)!=manifest['sources'][voice_video]:
        raise ValueError('照片或声音来源校验失败，请重新导入素材')
    for variant in manifest.get('variants') or []:
        photo=variant['photo'];path=source_path(folder,photo)
        if not path.is_file() or sha256(path)!=manifest['sources'][photo]:
            raise ValueError('照片或声音来源校验失败，请重新导入素材')
    variant=active_variant(manifest)
    if variant is None or manifest.get('photo')!=variant['photo']:
        raise ValueError('人物造型索引不一致，请重新选择造型')


def portrait_artifacts(folder, manifest, *, root, variant=None):
    variant=variant if variant is not None else active_variant(manifest)
    if variant is None:
        # Tolerate a legacy manifest that still carries photo/portrait on top.
        variant={'photo':manifest.get('photo'),'portrait':manifest.get('portrait')}
    portrait=variant.get('portrait')
    if not portrait:raise ValueError('该造型尚未准备')
    relative=portrait['folder']
    parts=Path(relative).parts
    if len(parts)!=2 or parts[0]!='portrait' or parts[1] in {'.','..'}:
        raise ValueError('照片缓存目录无效')
    cache=source_path(folder,relative)
    descriptor=read_json(source_path(cache,'source.json'))
    state=source_path(cache,descriptor['file'])
    if state.suffix!='.npz' or not state.is_file() or sha256(state)!=descriptor['sha256']:
        raise ValueError('照片缓存文件校验失败')
    ready=read_json(source_path(cache,'ready.json'))
    metadata=descriptor['metadata']
    expected={'photo_sha256':manifest['sources'][variant['photo']],
        **portrait_version(Path(root))}
    if (metadata!=expected
            or ready['metadata']!=metadata or descriptor['schema_version']!=1):
        raise ValueError('照片缓存与当前照片或模型版本不一致')
    idle=source_path(cache,'idle.mkv')
    if not idle.is_file() or sha256(idle)!=ready['idle_sha256']:
        raise ValueError('照片待机画面校验失败')
    if (portrait['fps']!=25 or portrait['width']<=0 or portrait['height']<=0
            or ready['width']!=portrait['width'] or ready['height']!=portrait['height']):
        raise ValueError('照片画面尺寸或帧率无效')
    return [f'{relative}/source.json',state.relative_to(folder.resolve()).as_posix(),
            f'{relative}/ready.json',f'{relative}/idle.mkv']


def required_photo_artifacts(folder, manifest, *, root, variant=None):
    return [*portrait_artifacts(folder,manifest,root=root,variant=variant),*VOICE_ARTIFACTS]


def verify_artifact_hashes(folder, manifest, required):
    hashes=manifest.get('artifact_hashes',{})
    for relative in required:
        path=source_path(folder,relative)
        if not path.is_file() or relative not in hashes or sha256(path)!=hashes[relative]:
            raise ValueError(f'人物资产校验失败：{relative}')


def require_photo_ready(store, avatar_id):
    try:
        manifest=migrate_profile(store.manifest(avatar_id))
        folder=store.avatar_dir(avatar_id)
        if (manifest.get('status')!='ready' or not manifest.get('voice',{}).get('confirmed')
                or not all(manifest.get('validation',{}).get(key) for key in ('photo','voice'))):
            raise ValueError('请先准备照片并确认声音')
        variant=active_variant(manifest)
        if variant is None:raise ValueError('人物没有可用造型')
        validate_photo_sources(folder,manifest)
        verify_artifact_hashes(folder,manifest,
            required_photo_artifacts(folder,manifest,root=store.root,variant=variant))
        voice=manifest['voice']
        if (voice['source_sha256']!=manifest['sources'][manifest['voice_video']]
                or voice['reference_sha256']!=sha256(folder/'voice/reference.wav')):
            raise ValueError('声音参考与当前素材不一致')
        return manifest
    except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
        raise NotReady(f'人物检查失败，请重新准备：{error}') from error


def _folder_bytes(path):
    total=0
    for item in path.rglob('*'):
        if item.is_file():total+=item.stat().st_size
    return total


def portrait_gc(store, avatar_id):
    """Remove portrait caches that no variant references any more."""
    manifest=migrate_profile(store.manifest(avatar_id));folder=store.avatar_dir(avatar_id)
    base=folder/'portrait'
    if not base.is_dir():return {'removed':[],'freed_bytes':0}
    referenced={item['portrait']['folder'] for item in manifest.get('variants') or []
                if item.get('portrait')}
    photos={manifest['sources'].get(item['photo']) for item in manifest.get('variants') or []}
    removed=[];freed=0
    for entry in sorted(base.iterdir()):
        if not entry.is_dir() or entry.is_symlink():continue
        relative='portrait/'+entry.name
        if relative in referenced:continue
        try:metadata=read_json(entry/'source.json').get('metadata') or {}
        except (OSError,ValueError):metadata={}
        if metadata.get('photo_sha256') in photos:continue
        if entry.resolve().parent!=base.resolve():continue
        size=_folder_bytes(entry)
        shutil.rmtree(entry)
        removed.append(relative);freed+=size
    return {'removed':removed,'freed_bytes':freed}
