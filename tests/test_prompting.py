"""Prompt translation, caching and prompt assembly."""
import json
import urllib.request

import numpy as np
import pytest
from PIL import Image

from backend import prompting


class Response:
    def __init__(self, text):
        self.text = text

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def read(self, *args):
        return json.dumps({'choices': [{'message': {'content': self.text}}]}).encode('utf-8')


def fake_opener(seen, text='Standing, arms relaxed'):
    class Opener:
        def open(self, request, timeout=None):
            seen.append(json.loads(request.data.decode('utf-8')))
            return Response(text)
    return lambda handlers=None: Opener()


def test_ascii_text_passes_through():
    assert prompting.translate('black denim jacket', 'garment') == 'black denim jacket'


def test_empty_text_stays_empty():
    assert prompting.translate('   ', 'action') == ''


def test_translate_cleans_and_lowercases(monkeypatch):
    seen = []
    monkeypatch.setattr(urllib.request, 'build_opener', fake_opener(seen, ' Standing,Arms relaxed '))
    assert prompting.translate('站好', 'action') == 'standing, arms relaxed'


def test_translate_uses_the_token_budget_of_each_kind(monkeypatch):
    seen = []
    monkeypatch.setattr(urllib.request, 'build_opener', fake_opener(seen))
    for kind in prompting.KINDS:
        prompting.translate('测试', kind)
    assert [item['max_tokens'] for item in seen] == [prompting.MAX_TOKENS[kind]
                                                     for kind in prompting.KINDS]


def test_translate_uses_the_system_prompt_of_each_kind(monkeypatch):
    seen = []
    monkeypatch.setattr(urllib.request, 'build_opener', fake_opener(seen))
    prompting.translate('温柔一点', 'tone')
    assert seen[0]['messages'][0]['content'] == prompting.SYSTEM['tone']


def test_translate_returns_empty_when_the_server_is_down(monkeypatch):
    def boom(handlers=None):
        raise OSError('connection refused')
    monkeypatch.setattr(urllib.request, 'build_opener', boom)
    assert prompting.translate('轻轻微笑', 'expression') == ''


def test_unknown_kind_falls_back_to_garment(monkeypatch):
    seen = []
    monkeypatch.setattr(urllib.request, 'build_opener', fake_opener(seen))
    prompting.translate('外套', 'nonsense')
    assert seen[0]['messages'][0]['content'] == prompting.SYSTEM['garment']


def test_cache_avoids_a_second_request(tmp_path, monkeypatch):
    calls = []

    def counted(text, kind='garment', **kwargs):
        calls.append((text, kind))
        return 'soft smile'

    monkeypatch.setattr(prompting, 'translate', counted)
    assert prompting.translate_cached('微笑', 'expression', root=tmp_path) == 'soft smile'
    assert prompting.translate_cached('微笑', 'expression', root=tmp_path) == 'soft smile'
    assert len(calls) == 1


def test_cache_is_written_to_disk(tmp_path, monkeypatch):
    monkeypatch.setattr(prompting, 'translate', lambda *a, **k: 'warm and gentle')
    prompting.translate_cached('温和', 'tone', root=tmp_path)
    stored = json.loads((tmp_path/prompting.CACHE_FILE).read_text(encoding='utf-8'))
    assert list(stored.values()) == ['warm and gentle']


def test_cache_ignores_a_failed_translation(tmp_path, monkeypatch):
    monkeypatch.setattr(prompting, 'translate', lambda *a, **k: '')
    prompting.translate_cached('无法翻译', 'action', root=tmp_path)
    assert prompting.read_cache(tmp_path) == {}


def test_cache_drops_the_oldest_entries(tmp_path, monkeypatch):
    monkeypatch.setattr(prompting, 'translate', lambda text, *a, **k: text)
    for index in range(5):
        prompting.translate_cached(f'词{index}', 'action', root=tmp_path, size=3)
    assert len(prompting.read_cache(tmp_path)) == 3


def test_extend_prompt_has_the_full_body_clause_and_the_anti_duplicate_terms():
    positive, negative = prompting.extend_prompt(action='walking towards the camera',
                                                 tone='warm and calm',
                                                 expression='soft smile',
                                                 garment='wearing navy trousers')
    assert 'walking towards the camera' in positive
    assert 'head to toe' in positive
    assert 'warm and calm' in positive
    assert 'soft smile' in positive
    assert 'wearing navy trousers' in positive
    for term in ('two people', 'duplicate', 'extra legs', 'tiny head'):
        assert term in negative


def test_extend_prompt_falls_back_to_neutral_defaults():
    positive, _ = prompting.extend_prompt()
    assert prompting.DEFAULT_ACTION in positive
    assert prompting.DEFAULT_EXPRESSION in positive


def test_wear_prompt_uses_the_translated_garment_phrase():
    positive, negative = prompting.wear_prompt(garment='black leather jacket',
                                               action='standing', expression='calm')
    assert positive.startswith('a photo of a person wearing a black leather jacket')
    assert 'standing' in positive and 'calm' in positive
    assert 'two people' in negative


def test_wear_prompt_without_a_phrase_still_reads_as_a_sentence():
    positive, _ = prompting.wear_prompt()
    assert positive.startswith('a photo of a person wearing the garment')


def test_nearest_color_name_matches_the_palette():
    assert prompting.nearest_color_name((30, 42, 68)) == 'navy'
    assert prompting.nearest_color_name((107, 31, 43)) == 'wine'


def test_palette_matches_the_outfit_colours():
    from backend import outfit
    for name, code in outfit.COLORS.items():
        expected = tuple(int(code.lstrip('#')[index:index + 2], 16) for index in (0, 2, 4))
        assert prompting.PALETTE[name] == expected


def test_garment_hint_names_the_trousers_it_can_see():
    from backend.edit_geometry import LOWER_LABELS
    labels = np.zeros((100, 100), np.uint8)
    labels[60:100, 30:70] = LOWER_LABELS[0]
    image = Image.new('RGB', (100, 100), (30, 42, 68))
    assert prompting.garment_hint(labels, image) == 'wearing navy blue trousers and matching shoes'


def test_garment_hint_without_visible_legs_stays_generic():
    labels = np.zeros((100, 100), np.uint8)
    labels[10:50, 30:70] = 4
    image = Image.new('RGB', (100, 100), (30, 42, 68))
    assert prompting.garment_hint(labels, image) == 'wearing dark trousers'


def test_describe_wraps_the_garment_translation(tmp_path, monkeypatch):
    monkeypatch.setattr(prompting, 'translate', lambda *a, **k: 'denim jacket')
    assert prompting.describe('牛仔外套', root=tmp_path) == 'denim jacket'


def test_outfit_negative_is_shared_with_prompting():
    from backend import outfit
    assert outfit.NEGATIVE == prompting.BASE_NEGATIVE
