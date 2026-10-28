import argparse
from pathlib import Path
import yaml
from backend.storage import Store
from backend.pipeline.build import build_avatar


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--avatar-id')
    parser.add_argument('--stage', choices=['all','prepare','reconstruct','validate'], default='all')
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    store = Store(root)
    default = root/'.runtime/default_avatar.txt'
    avatar_id = args.avatar_id or (default.read_text().strip() if default.exists() else None)
    if not avatar_id:
        subject = yaml.safe_load((root/'configs/subject.yaml').read_text(encoding='utf-8'))
        from scripts.common_paths import linux_path
        source = root/subject['source_directory']
        avatar = store.import_files(subject.get('name','数字人'),source/subject['video'],
                   [source/p for p in subject['body_photos']], [source/p for p in subject['face_photos']])
        avatar_id = avatar['id']
        default.write_text(avatar_id,encoding='ascii')
    job = store.new_job(avatar_id, args.stage)
    print(f"Avatar: {avatar_id}\nJob: {job['id']}",flush=True)
    build_avatar(store, avatar_id, job['id'], args.stage)
    print(store.job(job['id']), flush=True)


if __name__ == '__main__':
    main()
