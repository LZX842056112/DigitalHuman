from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def load_split():
    """Import split_first_clause without importing torch or the transport stack."""
    source = (ROOT / 'backend/transport/photo_session.py').read_text(encoding='utf-8')
    start = source.index('def split_first_clause')
    end = source.index('class PhotoSession:')
    namespace = {}
    exec(source[start:end], namespace)
    return namespace['split_first_clause']


def test_split_keeps_every_character():
    split = load_split()
    sentences = ['今天天气不错，适合出门散步，你也要注意保暖。', '第二句在这里。']
    assert ''.join(split(sentences, 14)) == ''.join(sentences)


def test_split_cuts_first_sentence_at_comma():
    split = load_split()
    assert split(['今天天气不错，适合出门散步，你也要注意保暖。'], 14) == [
        '今天天气不错，', '适合出门散步，你也要注意保暖。']


def test_split_leaves_short_first_sentence():
    split = load_split()
    assert split(['你好，欢迎观看！', '第二句在这里。'], 14) == ['你好，欢迎观看！', '第二句在这里。']


def test_split_disabled_with_zero_limit():
    split = load_split()
    sentences = ['今天天气不错，适合出门散步。']
    assert split(sentences, 0) == sentences


def test_split_ignores_clause_that_would_be_too_short():
    split = load_split()
    sentences = ['好，今天天气不错，适合出门散步。']
    assert split(sentences, 14) == sentences


def test_runtime_quality_reads_knobs(monkeypatch):
    from backend.portrait import engine
    engine._RUNTIME_CACHE.clear()
    monkeypatch.setattr(engine, 'load_config', lambda root: {'runtime': {
        'portrait_sampling_steps': 12, 'portrait_frame_batch': 4, 'portrait_frame_stride': 2}})
    quality = engine.runtime_quality(ROOT)
    assert quality == {'sampling_steps': 12, 'frame_batch': 4, 'frame_stride': 2}


def test_runtime_quality_falls_back_when_missing(monkeypatch):
    from backend.portrait import engine
    engine._RUNTIME_CACHE.clear()
    monkeypatch.setattr(engine, 'load_config', lambda root: {})
    assert engine.runtime_quality(ROOT) == {'sampling_steps': None, 'frame_batch': 1, 'frame_stride': 1}


def test_photo_options_applies_steps_only_when_asked():
    from backend.portrait.engine import photo_options
    defaults = {'sampling_timesteps': 50, 'delta_eye_arr': None}
    assert photo_options(defaults, 'neutral')['sampling_timesteps'] == 50
    assert photo_options(defaults, 'neutral', {'sampling_steps': 20})['sampling_timesteps'] == 20
    assert photo_options(defaults, 'neutral', {'sampling_steps': None})['sampling_timesteps'] == 50
