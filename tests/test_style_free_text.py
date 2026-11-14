"""The free text tone: validation, prompt assembly and the voice cache key."""
import pytest

from backend import style


def manifest(**settings):
    return {'settings': settings}


def test_settings_of_fills_the_new_field_for_old_manifests():
    assert style.settings_of(manifest(tone='warm', expression='smile'))['tone_prompt'] == ''


def test_merge_settings_stores_and_normalizes_the_free_text():
    merged = style.merge_settings({}, {'tone_prompt': '  温和   一点  '})
    assert merged['tone_prompt'] == '温和 一点'


def test_merge_settings_keeps_the_other_fields():
    current = manifest(tone='warm', expression='smile', tone_prompt='亲切')
    merged = style.merge_settings(current, {'expression': 'serious'})
    assert merged == {'tone': 'warm', 'expression': 'serious', 'tone_prompt': '亲切'}


def test_merge_settings_clears_the_free_text():
    current = manifest(tone_prompt='亲切')
    assert style.merge_settings(current, {'tone_prompt': ''})['tone_prompt'] == ''


def test_merge_settings_rejects_text_over_the_limit():
    with pytest.raises(ValueError, match='最多 60 个字'):
        style.merge_settings({}, {'tone_prompt': '温' * (style.MAX_TONE_PROMPT + 1)})


def test_merge_settings_accepts_text_at_the_limit():
    value = '温' * style.MAX_TONE_PROMPT
    assert style.merge_settings({}, {'tone_prompt': value})['tone_prompt'] == value


def test_merge_settings_rejects_a_non_string():
    with pytest.raises(ValueError, match='必须是文字'):
        style.merge_settings({}, {'tone_prompt': 42})


def test_normalize_strips_control_characters():
    assert style.normalize_tone_prompt('温和\n一点\x07') == '温和 一点'


def test_normalize_truncates_at_the_limit():
    assert len(style.normalize_tone_prompt('温' * 200)) == style.MAX_TONE_PROMPT


def test_tone_clause_appends_the_refinement_to_the_preset():
    clause = style.tone_clause('professional', '不要用术语')
    assert style.TONE_PROMPTS['professional'] in clause
    assert clause.endswith('另外要求：不要用术语。')


def test_tone_clause_without_a_preset_uses_only_the_refinement():
    assert style.tone_clause('natural', '温柔一点') == '另外要求：温柔一点。'


def test_tone_clause_without_free_text_is_the_plain_preset():
    assert style.tone_clause('warm') == style.TONE_PROMPTS['warm']


def test_voice_instruction_uses_the_free_text():
    from backend.emotion import voice_instruction
    instruction = voice_instruction('neutral', 'natural', '温柔而坚定')
    assert '请用温柔而坚定的语气说一句话。' in instruction


def test_voice_instruction_keeps_the_preset_when_no_text_is_given():
    from backend.emotion import voice_instruction
    assert voice_instruction('neutral', 'warm') == \
        voice_instruction('neutral', 'warm', '')


def test_a_real_emotion_ignores_the_free_text():
    from backend.emotion import voice_instruction
    assert voice_instruction('happy', 'warm', '温柔') == voice_instruction('happy', 'warm')


def test_system_prompt_carries_the_free_text():
    from backend.dialogue.client import system_prompt
    prompt = system_prompt('warm', '多用短句')
    assert '另外要求：多用短句。' in prompt
    assert prompt.endswith('不要照搬用户的情绪。')


def test_system_prompt_without_text_is_the_shared_default():
    from backend.dialogue.client import SYSTEM_PROMPT, system_prompt
    assert system_prompt('natural') == SYSTEM_PROMPT
    assert system_prompt() == SYSTEM_PROMPT


def test_voice_cache_key_changes_with_the_free_text():
    from backend.voice.engine import voice_cache_key
    base = voice_cache_key('digest', 'neutral', 'rev', '', 'instruct', 'natural')
    other = voice_cache_key('digest', 'neutral', 'rev', '', 'instruct', 'natural', '温柔')
    assert base != other


def test_clone_settings_moves_to_instruct_when_only_the_text_is_set():
    from backend.voice.service import clone_settings
    assert clone_settings('neutral', '这是一句参考文本', 'natural') == \
        ('zero_shot', '这是一句参考文本')
    assert clone_settings('neutral', '这是一句参考文本', 'natural',
                          tone_prompt='温柔') == ('instruct', '')


def test_clone_settings_ignores_an_empty_free_text():
    from backend.voice.service import clone_settings
    assert clone_settings('neutral', '参考文本', 'natural', tone_prompt='  ') == \
        ('zero_shot', '参考文本')
