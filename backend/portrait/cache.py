"""A photo's appearance cache contains arrays, never executable objects."""
import uuid
import numpy as np
from backend.storage import atomic_json,read_json,sha256

LISTS=('f_s','M_c2o','eye_open','eye_ball','img_rgb')


def save_cache(folder, source, metadata):
    folder.mkdir(parents=True,exist_ok=True)
    arrays={name:source[name+'_lst'][0] for name in LISTS}
    arrays.update({'motion_'+name:value for name,value in source['x_s_info_lst'][0].items()})
    arrays['sc']=source['sc']
    if any(not np.isfinite(value).all() for value in arrays.values()):raise ValueError('照片特征含无效数值')
    name=uuid.uuid4().hex+'.npz';path=folder/name
    with path.open('wb') as stream:np.savez_compressed(stream,**arrays)
    atomic_json(folder/'source.json',{'schema_version':1,'metadata':metadata,'file':name,'sha256':sha256(path)})


def load_cache(folder, expected):
    record=read_json(folder/'source.json')
    if record.get('schema_version')!=1 or any(record['metadata'].get(k)!=v for k,v in expected.items()):
        raise ValueError('照片或模型已变化，请重新准备照片')
    path=(folder/record['file']).resolve()
    if folder.resolve() not in path.parents or sha256(path)!=record['sha256']:raise ValueError('照片特征校验失败')
    with np.load(path,allow_pickle=False) as data:arrays={key:data[key] for key in data.files}
    if any(not np.isfinite(value).all() for value in arrays.values()):raise ValueError('照片特征含无效数值')
    source={name+'_lst':[arrays[name]] for name in LISTS}
    source.update(x_s_info_lst=[{key[7:]:value for key,value in arrays.items() if key.startswith('motion_')}],
                  sc=arrays['sc'],is_image_flag=True)
    return source
