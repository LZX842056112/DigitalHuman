"""The extend pass: geometry wiring, retry and the report contract."""
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from backend import edit_geometry as geo
from backend import outfit


class FakePipe:
    def __init__(self):
        self.scale = None
        self.calls = []

    def set_ip_adapter_scale(self, value):
        self.scale = float(value)


def source_image(size=(400, 600), color=(205, 180, 158)):
    return Image.new('RGB', size, color)


def source_labels(size=(400, 600), box=(120, 20, 280, 590), head=90):
    labels = np.zeros((size[1], size[0]), np.uint8)
    x0, y0, x1, y1 = box
    labels[y0:y1, x0:x1] = 4
    labels[y0:y0 + head, x0:x1] = 11
    return labels


def leg_labels(size):
    labels = np.zeros((size[1], size[0]), np.uint8)
    labels[:, :] = 6
    return labels


def install(monkeypatch, *, labels=None, legs=True, render_calls=None):
    """Replace segmentation, the pipeline and the sampler."""
    state = {'label_calls': 0, 'renders': 0}

    def fake_label_map(image):
        state['label_calls'] += 1
        if tuple(image.size) == (400, 600):
            return source_labels() if labels is None else labels
        return leg_labels(image.size) if legs else source_labels(image.size)

    def fake_pipe(*args, **kwargs):
        return FakePipe()

    def fake_render(pipe, canvas, mask, reference, positive, negative, steps, spec, seed, writer):
        state['renders'] += 1
        if render_calls is not None:
            render_calls.append({'steps': steps, 'seed': seed, 'prompt': positive,
                                 'negative': negative, 'size': canvas.size})
        return canvas.copy()

    monkeypatch.setattr(outfit, 'label_map', fake_label_map)
    monkeypatch.setattr(outfit, '_load_pipe', fake_pipe)
    monkeypatch.setattr(outfit, '_render', fake_render)
    monkeypatch.setattr(outfit.prompting, 'translate_cached', lambda *a, **k: '')
    monkeypatch.setattr(outfit.prompting, 'garment_hint', lambda *a, **k: 'wearing dark trousers')
    return SimpleNamespace(**state)


def run_extend(tmp_path, spec=None):
    out = tmp_path/'out.png'
    report = outfit.run(source_image(), {'mode': 'extend', 'seed': 3, **(spec or {})}, out)
    return out, report


def test_extend_writes_a_taller_picture(tmp_path, monkeypatch):
    install(monkeypatch)
    out, report = run_extend(tmp_path)
    assert out.is_file()
    saved = Image.open(out)
    assert saved.height > saved.width
    assert report['canvas'][1] > 600
    assert report['face_diff_ratio'] == 0.
    assert report['protected_diff_ratio'] == 0.


def test_extend_reports_quality_and_legs(tmp_path, monkeypatch):
    install(monkeypatch, legs=True)
    _, report = run_extend(tmp_path)
    assert report['leg_ratio'] == pytest.approx(1.)
    assert report['quality'] == 'good'
    assert 'warnings' not in report


def test_extend_flags_a_result_without_legs(tmp_path, monkeypatch):
    install(monkeypatch, legs=False)
    _, report = run_extend(tmp_path)
    assert report['leg_ratio'] == pytest.approx(0.)
    assert report['quality'] == 'low'
    assert report['warnings']


def test_extend_retries_once_without_legs(tmp_path, monkeypatch):
    calls = []
    install(monkeypatch, legs=False, render_calls=calls)
    run_extend(tmp_path)
    assert len(calls) == 2
    assert calls[0]['seed'] != calls[1]['seed']
    assert 'both feet visible' in calls[1]['prompt']


def test_extend_skips_a_picture_that_is_already_full_body(tmp_path, monkeypatch):
    full = source_labels(size=(400, 600), box=(120, 20, 280, 600), head=80)
    install(monkeypatch, labels=full)
    out, report = run_extend(tmp_path)
    assert report['skipped'] == geo.SKIPPED
    assert report['output'] is None
    assert not out.exists()


def test_extend_rejects_a_picture_without_a_person(tmp_path, monkeypatch):
    install(monkeypatch, labels=np.zeros((600, 400), np.uint8))
    with pytest.raises(ValueError, match='没有识别到完整的人物'):
        run_extend(tmp_path)


def test_extend_uses_the_adapter_scale_from_the_spec(tmp_path, monkeypatch):
    install(monkeypatch)
    pipe = FakePipe()
    monkeypatch.setattr(outfit, '_load_pipe', lambda *a, **k: pipe)
    report = outfit.run(source_image(), {'mode': 'extend', 'scale': .42}, tmp_path/'a.png')
    assert pipe.scale == pytest.approx(.42)
    assert report['adapter'] is True


def test_extend_falls_back_when_the_adapter_is_missing(tmp_path, monkeypatch):
    install(monkeypatch)

    def missing(*args, **kwargs):
        if kwargs.get('with_adapter'):
            raise FileNotFoundError('没有参考模型')
        return FakePipe()

    monkeypatch.setattr(outfit, '_load_pipe', missing)
    report = outfit.run(source_image(), {'mode': 'extend'}, tmp_path/'b.png')
    assert report['adapter'] is False


def test_extend_splits_a_nine_by_sixteen_target(tmp_path, monkeypatch):
    calls = []
    install(monkeypatch, render_calls=calls)
    report = outfit.run(source_image(), {'mode': 'extend', 'aspect': 9 / 16},
                        tmp_path/'c.png')
    assert report['passes'] == 2
    assert len(calls) == 2


def test_extend_keeps_a_single_pass_for_three_by_four(tmp_path, monkeypatch):
    calls = []
    install(monkeypatch, render_calls=calls)
    report = outfit.run(source_image(), {'mode': 'extend', 'aspect': .75},
                        tmp_path/'d.png')
    assert report['passes'] == 1
    assert len(calls) == 1


def test_extend_passes_a_negative_prompt_with_anti_duplicate_terms(tmp_path, monkeypatch):
    calls = []
    install(monkeypatch, render_calls=calls)
    run_extend(tmp_path)
    assert 'two people' in calls[0]['negative']
    assert 'head to toe' in calls[0]['prompt']
