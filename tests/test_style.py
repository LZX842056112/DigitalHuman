import numpy as np
import pytest


def test_enumerations_are_closed():
    from backend import style
    assert style.TONES == ('natural', 'warm', 'professional', 'lively')
    assert style.EXPRESSIONS == ('natural', 'smile', 'serious', 'surprise')
    assert style.DEFAULT_SETTINGS == {'tone': 'natural', 'expression': 'natural',
                                      'tone_prompt': ''}


def test_unknown_values_fall_back_for_runtime_paths():
    from backend.style import normalize_expression, normalize_tone
    assert normalize_tone('warm') == 'warm'
    assert normalize_tone('louder') == 'natural'
    assert normalize_expression('smile') == 'smile'
    assert normalize_expression(None) == 'natural'


def test_settings_read_tolerates_missing_and_broken_values():
    from backend.style import settings_of
    empty = {'tone': 'natural', 'expression': 'natural', 'tone_prompt': ''}
    assert settings_of({}) == empty
    assert settings_of({'settings': None}) == empty
    assert settings_of({'settings': {'tone': 'lively', 'expression': 'x'}}) == {
        'tone': 'lively', 'expression': 'natural', 'tone_prompt': ''}


def test_merge_settings_updates_one_field_and_keeps_the_other():
    from backend.style import merge_settings
    current = {'settings': {'tone': 'warm', 'expression': 'smile'}}
    assert merge_settings(current, {'tone': 'professional'}) == {
        'tone': 'professional', 'expression': 'smile', 'tone_prompt': ''}
    assert merge_settings(current, {}) == {
        'tone': 'warm', 'expression': 'smile', 'tone_prompt': ''}
    assert current['settings'] == {'tone': 'warm', 'expression': 'smile'}


@pytest.mark.parametrize('update', [{'tone': 'angry'}, {'expression': 'grin'}, {'speed': 2}])
def test_merge_settings_rejects_unknown_values(update):
    from backend.style import merge_settings
    with pytest.raises(ValueError):
        merge_settings({}, update)


def test_expression_presets_map_to_ditto_labels():
    from backend.style import expression_emo, expression_exp_gain
    assert expression_emo('natural') == 4
    assert expression_emo('smile') == (3, 4)
    assert expression_emo('surprise') == 6
    serious = expression_emo('serious')
    assert isinstance(serious, np.ndarray) and serious.shape == (1, 8)
    assert serious[0, 4] > .8 and serious[0, 5] == pytest.approx(1 - serious[0, 4])
    assert float(serious.sum()) == pytest.approx(1.)
    assert expression_exp_gain('serious') < 1.
    assert expression_exp_gain('natural') == 1.
    assert expression_emo('anything') == 4
