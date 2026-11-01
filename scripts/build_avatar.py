"""Import explicit media or prepare an existing photo profile through the studio."""
import argparse
import json
import time
from pathlib import Path
import httpx


def prepare(root,client,avatar_id=None,photo=None,voice_video=None,name='新数字人',timeout=900):
    root=Path(root);default=root/'.runtime/default_avatar.txt'
    if photo is not None or voice_video is not None:
        if avatar_id or photo is None or voice_video is None:
            raise ValueError('导入需要 --photo 和 --voice-video。不要同时指定 --avatar-id。')
        with Path(photo).open('rb') as image,Path(voice_video).open('rb') as video:
            response=client.post('/api/avatars/import',data={'name':name},files={
                'photo':(Path(photo).name,image),'voice_video':(Path(voice_video).name,video)})
            response.raise_for_status();avatar_id=response.json()['avatar_id']
        default.parent.mkdir(parents=True,exist_ok=True);default.write_text(avatar_id,encoding='utf-8')
    else:
        avatar_id=avatar_id or (default.read_text(encoding='utf-8').strip() if default.exists() else None)
        if not avatar_id:raise ValueError('请指定已导入的 --avatar-id，或指定 --photo 和 --voice-video。')
        # A stale pointer must fail here. Never import old local media automatically.
        response=client.get(f'/api/avatars/{avatar_id}');response.raise_for_status()
    response=client.post(f'/api/avatars/{avatar_id}/build');response.raise_for_status()
    job_id=response.json()['job_id'];deadline=time.monotonic()+timeout
    while True:
        response=client.get(f'/api/jobs/{job_id}');response.raise_for_status();job=response.json()
        if job['status']=='failed':raise RuntimeError(job.get('error') or '准备失败。请查看任务记录。')
        if job['status'] in {'complete','waiting'}:return {'avatar_id':avatar_id,'job_id':job_id,**job}
        if time.monotonic()>=deadline:raise TimeoutError(f'任务仍在运行：{job_id}。请在界面查看。')
        time.sleep(.5)


def main():
    parser=argparse.ArgumentParser(description='准备照片和参考声音。请先启动服务。')
    parser.add_argument('--avatar-id')
    parser.add_argument('--photo',type=Path)
    parser.add_argument('--voice-video',type=Path)
    parser.add_argument('--name',default='新数字人')
    args=parser.parse_args()
    try:
        with httpx.Client(base_url='http://127.0.0.1:7860',timeout=60,trust_env=False) as client:
            result=prepare(Path(__file__).resolve().parents[1],client,**vars(args))
        print(json.dumps(result,ensure_ascii=False,indent=2))
        if result['status']=='waiting':print('请在界面试听并确认参考声音，然后再次准备。')
    except (ValueError,OSError,httpx.HTTPError,RuntimeError) as error:
        parser.exit(2,f'准备失败：{error}\n请检查服务及素材，然后重试。\n')


if __name__=='__main__':main()
