"""Pinned model checks and compatibility for the published NumPy configuration."""
import pickle
from backend.storage import read_json, sha256
from backend.errors import NotReady


class _ConfigReader(pickle.Unpickler):
    def find_class(self,module,name):
        if module.startswith('numpy._core'):module=module.replace('numpy._core','numpy.core',1)
        return super().find_class(module,name)


def read_published_config(path):
    # Only the checksum-verified upstream configuration reaches this loader.
    with open(path,'rb') as stream:return _ConfigReader(stream).load()


def verified_model(root,key):
    model=read_json(root/'configs/photo-models.json')['models'][key]
    folder=root/model['destination']
    for row in model['files']:
        path=folder/row['name']
        if not path.is_file():raise NotReady(f'模型文件尚未下载：{key}/{row["name"]}')
        if path.stat().st_size!=row['size'] or sha256(path)!=row['sha256']:
            raise NotReady(f'模型文件校验失败：{key}/{row["name"]}')
    return folder
