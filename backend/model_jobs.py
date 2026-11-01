"""Local inference jobs share a generation marker, not a live GPU session."""
from pathlib import Path
from backend.storage import safe_id,read_json


def session_folder(root,session_id):
    base=(Path(root)/'data/sessions').resolve()
    folder=base/safe_id(session_id)
    if folder.resolve()!=folder or base not in folder.resolve().parents:raise ValueError('会话路径越界')
    return folder


def expired(root,session_id,generation):
    if session_id is None:return False
    if generation is None:raise ValueError('缺少回答编号')
    folder=session_folder(root,session_id)
    try:record=read_json(folder/'generation.json')
    except FileNotFoundError:return True
    return bool(record.get('closed')) or record.get('generation')!=generation


def sentence_paths(root,session_id,generation,index):
    if generation<0 or index<0:raise ValueError('回答编号无效')
    folder=session_folder(root,session_id)
    return folder/f'{generation}_{index}.wav',folder/f'{generation}_{index}.mkv'
