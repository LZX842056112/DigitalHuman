"""Progressive preview: cadence, file names, job record and degradation."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from backend.outfit import PreviewWriter
from backend.storage import Store


class FakeVae:
    config = SimpleNamespace(scaling_factor=.18215)

    def __init__(self):
        self.calls = 0

    def decode(self, latents, return_dict=False):
        self.calls += 1
        return ('decoded',)


class FakeProcessor:
    def postprocess(self, image, output_type=None):
        return [Image.new('RGB', (64, 64), (20, 30, 40))]


class FakePipe:
    def __init__(self):
        self.vae = FakeVae()
        self.image_processor = FakeProcessor()


def latents():
    return np.zeros((1, 4, 8, 8), np.float32)


def writer(tmp_path, **kwargs):
    folder = tmp_path/'previews'
    options = {'interval': 4, 'skip': 2, 'long_edge': 512, 'mode': 'decode', 'total': 12}
    options.update(kwargs)
    return PreviewWriter(folder, **options)


def drive(preview, steps=12, pipe=None):
    pipe = pipe or FakePipe()
    for step in range(steps):
        preview.callback(pipe, step, None, {'latents': latents()})
    return pipe


def test_writes_every_interval_after_the_skip(tmp_path):
    preview = writer(tmp_path)
    drive(preview)
    assert preview.writes == 3
    # The first picture was already pruned; only the newest two stay on disk.
    names = sorted(path.name for path in preview.folder.glob('*.png'))
    assert names == ['step-0007.png', 'step-0011.png']


def test_keeps_only_the_two_newest_pictures(tmp_path):
    preview = writer(tmp_path)
    drive(preview, steps=24)
    remaining = sorted(path.name for path in preview.folder.glob('*.png'))
    assert len(remaining) == 2
    assert remaining[-1] == 'step-0023.png'


def test_version_increases_with_every_write(tmp_path):
    preview = writer(tmp_path)
    drive(preview)
    assert preview.version == preview.writes == 3


def test_reports_the_preview_in_the_job_record(tmp_path):
    store = Store(tmp_path)
    job = store.new_job('avatar', 'extend')
    preview = writer(tmp_path)
    preview.job_id = job['id']
    preview.root = tmp_path
    drive(preview)
    record = store.job(job['id'])
    assert record['stage'] == 'diffuse'
    assert 0. < record['progress'] <= .90
    assert record['preview']['file'] == 'step-0011.png'
    assert record['preview']['version'] == 3
    assert record['preview']['step'] == 11


def test_progress_never_passes_the_ceiling(tmp_path):
    store = Store(tmp_path)
    job = store.new_job('avatar', 'extend')
    preview = writer(tmp_path, progress_from=.10, progress_to=.60)
    preview.job_id = job['id']
    preview.root = tmp_path
    drive(preview, steps=40)
    assert store.job(job['id'])['progress'] <= .60


def test_latent_mode_never_decodes(tmp_path):
    preview = writer(tmp_path, mode='latent')
    pipe = drive(preview)
    assert pipe.vae.calls == 0
    assert preview.writes == 3


def test_off_mode_writes_nothing(tmp_path):
    preview = writer(tmp_path, mode='off')
    drive(preview)
    assert preview.writes == 0
    assert not list(preview.folder.glob('*.png'))


def test_decode_failure_falls_back_to_the_latent_picture(tmp_path):
    preview = writer(tmp_path)

    class Broken(FakePipe):
        def __init__(self):
            super().__init__()
            self.vae = SimpleNamespace(config=SimpleNamespace(scaling_factor=.18),
                                       decode=lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))

    drive(preview, pipe=Broken())
    assert preview.degraded is True
    assert preview.writes == 3


def test_cpu_offload_forces_the_latent_path(tmp_path):
    from backend import outfit
    preview = writer(tmp_path)
    pipe = FakePipe()
    outfit._cache['offloaded'] = True
    try:
        drive(preview, pipe=pipe)
    finally:
        outfit._cache.pop('offloaded', None)
    assert pipe.vae.calls == 0
    assert preview.writes == 3


def test_nothing_is_written_before_the_skip(tmp_path):
    preview = writer(tmp_path, interval=1, skip=5)
    drive(preview, steps=6)
    assert preview.writes == 1
    assert (preview.folder/'step-0005.png').is_file()


def test_a_missing_latent_keeps_the_step_running(tmp_path):
    preview = writer(tmp_path, interval=1, skip=0)
    for step in range(3):
        preview.callback(FakePipe(), step, None, {})
    assert preview.writes == 0
