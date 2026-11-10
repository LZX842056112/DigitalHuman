import pytest


HARD_RULES = ('最多两句', '不要虚构照片人物的经历、身份或记忆',
              '{"emotion":"neutral","text":"回答文字"}')


def test_natural_prompt_is_byte_identical_to_the_original():
    from backend.dialogue.client import SYSTEM_PROMPT, system_prompt
    assert system_prompt('natural') == SYSTEM_PROMPT
    assert system_prompt(None) == SYSTEM_PROMPT


@pytest.mark.parametrize('tone', ['natural', 'warm', 'professional', 'lively'])
def test_every_tone_keeps_the_hard_rules(tone):
    from backend.dialogue.client import system_prompt
    prompt = system_prompt(tone)
    for rule in HARD_RULES:
        assert rule in prompt
    assert prompt.index('最多两句') < prompt.index('不要虚构')


@pytest.mark.parametrize('tone,marker', [
    ('warm', '亲切'), ('professional', '专业'), ('lively', '活泼')])
def test_tone_clause_is_present(tone, marker):
    from backend.dialogue.client import system_prompt
    assert marker in system_prompt(tone)
    assert marker not in system_prompt('natural')


def test_non_neutral_voice_instruction_ignores_tone():
    from backend.emotion import voice_instruction
    for emotion in ('happy', 'sad', 'angry'):
        assert voice_instruction(emotion, 'warm') == voice_instruction(emotion)
        assert voice_instruction(emotion, 'lively') == voice_instruction(emotion)


def test_neutral_voice_instruction_follows_tone():
    from backend.emotion import voice_instruction
    assert voice_instruction('neutral', 'natural') == voice_instruction('neutral')
    assert '亲切' in voice_instruction('neutral', 'warm')
    assert voice_instruction('neutral', 'professional').endswith('<|endofprompt|>')
    assert voice_instruction('neutral', 'unknown') == voice_instruction('neutral')


def test_payload_carries_the_tone_into_the_system_message(tmp_path):
    from backend.dialogue.client import Dialogue
    dialogue = Dialogue(tmp_path, {'dialogue': {'base_url': 'http://127.0.0.1:1/v1'}})
    _, payload = dialogue._payload('你好', [], 'professional')
    assert payload['messages'][0]['role'] == 'system'
    assert '专业' in payload['messages'][0]['content']
    _, default = dialogue._payload('你好', [])
    assert '专业' not in default['messages'][0]['content']
