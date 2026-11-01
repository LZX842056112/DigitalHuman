"""Photo/voice sources have independent identities and immutable raw copies."""
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path
import shutil
import uuid
from backend.storage import sha256, read_json
from backend.errors import NotReady
from backend.portrait.assets import portrait_version


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


def import_photo(store, name, photo, voice_video):
    avatar_id=uuid.uuid4().hex;folder=store.avatar_dir(avatar_id)
    image,image_hash=copy_source(folder,'photo',photo)
    video,video_hash=copy_source(folder,'voice_video',voice_video)
    manifest={'schema_version':2,'mode':'photo','id':avatar_id,'name':name[:80],
        'created_at':datetime.now(timezone.utc).isoformat(),'status':'imported',
        'photo':image,'voice_video':video,'sources':{image:image_hash,video:video_hash},
        'voice':{'confirmed':False,'language':'zh'},'validation':{'photo':False,'voice':False},
        'portrait':None,'last_job_id':None}
    store.write_manifest(avatar_id,manifest)
    return manifest


def migrate_profile(manifest):
    if manifest.get('schema_version')==2 and manifest.get('mode')=='photo':return deepcopy(manifest)
    photo=manifest['body_photos'][0];video=manifest['video']
    result={key:manifest[key] for key in ('id','name','created_at')}
    result.update(schema_version=2,mode='photo',status='imported',photo='source/'+photo,voice_video='source/'+video,
        sources={'source/'+name:digest for name,digest in manifest['sources'].items()},
        voice=deepcopy(manifest.get('voice',{'confirmed':False,'language':'zh'})),
        validation={'photo':False,'voice':False},portrait=None,last_job_id=None)
    return result


def persist_photo_profile(store, avatar_id):
    """Keep the exact legacy JSON before the first write in the photo path."""
    original=store.manifest(avatar_id)
    manifest=migrate_profile(original)
    if original.get('schema_version')!=2 or original.get('mode')!='photo':
        folder=store.avatar_dir(avatar_id);backup=folder/'avatar.legacy.json'
        if not backup.exists():shutil.copy2(folder/'avatar.json',backup)
        store.write_manifest(avatar_id,manifest)
    return manifest


def replace_source(store, avatar_id, role, path):
    if role not in {'photo','voice_video'}:raise ValueError('未知素材类型')
    manifest=persist_photo_profile(store,avatar_id);folder=store.avatar_dir(avatar_id)
    incoming_hash=sha256(Path(path))
    current=source_path(folder,manifest[role])
    if (incoming_hash==manifest['sources'].get(manifest[role])
            and current.is_file() and sha256(current)==incoming_hash):return manifest
    relative,digest=copy_source(folder,role,path)
    manifest[role]=relative;manifest['sources'][relative]=digest;manifest['status']='imported'
    if role=='photo':manifest['validation']['photo']=False
    else:
        manifest['validation']['voice']=False
        manifest['voice']={'confirmed':False,'language':manifest['voice'].get('language','zh')}
    store.write_manifest(avatar_id,manifest)
    return manifest


def validate_photo_sources(folder, manifest):
    for role in ('photo','voice_video'):
        relative=manifest[role];path=source_path(folder,relative)
        if not path.is_file() or sha256(path)!=manifest['sources'][relative]:
            raise ValueError('照片或声音来源校验失败，请重新导入素材')


def portrait_artifacts(folder, manifest, *, root):
    portrait=manifest['portrait'];relative=portrait['folder']
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
    expected={'photo_sha256':manifest['sources'][manifest['photo']],
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


def required_photo_artifacts(folder, manifest, *, root):
    return [*portrait_artifacts(folder,manifest,root=root),'voice/reference.wav','voice/sample.wav']


def verify_artifact_hashes(folder, manifest, required):
    hashes=manifest.get('artifact_hashes',{})
    for relative in required:
        path=source_path(folder,relative)
        if not path.is_file() or relative not in hashes or sha256(path)!=hashes[relative]:
            raise ValueError(f'人物资产校验失败：{relative}')


def require_photo_ready(store, avatar_id):
    try:
        manifest=migrate_profile(store.manifest(avatar_id));folder=store.avatar_dir(avatar_id)
        if (manifest.get('status')!='ready' or not manifest.get('voice',{}).get('confirmed')
                or not all(manifest.get('validation',{}).get(key) for key in ('photo','voice'))):
            raise ValueError('请先准备照片并确认声音')
        validate_photo_sources(folder,manifest)
        verify_artifact_hashes(folder,manifest,required_photo_artifacts(folder,manifest,root=store.root))
        voice=manifest['voice']
        if (voice['source_sha256']!=manifest['sources'][manifest['voice_video']]
                or voice['reference_sha256']!=sha256(folder/'voice/reference.wav')):
            raise ValueError('声音参考与当前素材不一致')
        return manifest
    except (OSError,ValueError,KeyError,TypeError,IndexError) as error:
        raise NotReady(f'人物检查失败，请重新准备：{error}') from error
