"""One bounded emotion vocabulary shared by speech and photo animation."""
from dataclasses import dataclass
import json

EMOTIONS = {'neutral':4, 'happy':3, 'sad':5, 'angry':0}
INSTRUCTIONS = {'neutral':'', 'happy':'请非常开心地说一句话。',
                'sad':'请非常伤心地说一句话。', 'angry':'请非常生气地说一句话。'}


@dataclass(frozen=True)
class Reply:
    text: str
    emotion: str = 'neutral'


def normalize_emotion(value):
    return value if isinstance(value,str) and value in EMOTIONS else 'neutral'


def parse_reply(value):
    if isinstance(value,Reply):return value
    raw=str(value).strip()
    if raw.startswith('```'):
        raw=raw.split('\n',1)[-1].rsplit('```',1)[0].strip()
    try:
        data=json.loads(raw)
    except json.JSONDecodeError:
        if raw.startswith(('{','[')):raise ValueError('对话模型返回的回答格式无效')
        data={'text':raw}
    if not isinstance(data,dict) or not isinstance(data.get('text'),str) or not data['text'].strip():
        raise ValueError('对话模型没有返回有效回答')
    return Reply(data['text'].strip()[:300],normalize_emotion(data.get('emotion')))


def portrait_emotion(emotion):
    return EMOTIONS[normalize_emotion(emotion)]


def voice_instruction(emotion, tone='natural'):
    """A real emotion wins; the tone only fills the neutral instruction slot.

    Keeping non-neutral strings byte-identical means existing speaker caches
    stay valid when a user picks a tone.
    """
    from backend.style import voice_instruction_clause
    emotion=normalize_emotion(emotion)
    if emotion!='neutral':
        return 'You are a helpful assistant. '+INSTRUCTIONS[emotion]+'<|endofprompt|>'
    return 'You are a helpful assistant. '+voice_instruction_clause(tone)+'<|endofprompt|>'
