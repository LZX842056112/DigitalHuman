"""One source of truth for the user-facing tone and expression presets.

Tone changes how the character speaks (wording plus, on neutral turns, the
CosyVoice instruction). Expression changes the face only on neutral turns -
a turn with a real emotion always wins. Neither dimension touches the photo
cache identity, so switching them never rebuilds portrait assets.
"""
import numpy as np

TONES=('natural','warm','professional','lively')
EXPRESSIONS=('natural','smile','serious','surprise')
DEFAULT_SETTINGS={'tone':'natural','expression':'natural','tone_prompt':''}

# A free sentence is a refinement, never a replacement. The cap keeps it from
# overriding the hard rules a preset already wrote.
MAX_TONE_PROMPT=60

# Spoken delivery. Empty for natural so the default neutral path keeps using
# zero-shot cloning (the closest timbre) instead of an instruction.
TONE_INSTRUCTIONS={
    'natural':'',
    'warm':'请用亲切温和的语气说一句话。',
    'professional':'请用专业、平稳、克制的语气说一句话。',
    'lively':'请用活泼明快的语气说一句话。',
}

# Wording. Inserted after the opening sentence in the dialogue system prompt.
TONE_PROMPTS={
    'natural':'',
    'warm':'语气亲切温和，像熟悉的朋友：多用“你”“我们”，可以有一句轻声的关心，不要撒娇、不要网络流行语。',
    'professional':'语气专业克制：先给结论，再给一句理由或步骤；不用口语填充词，不用感叹号，不寒暄。',
    'lively':'语气活泼明快：句子短、节奏快，可以用一个轻快的语气词，不要夸张、不要卖萌。',
}

# Ditto's eight classes are Angry0, Disgust1, Fear2, Happy3, Neutral4,
# Sad5, Surprise6, Contempt7. `_get_emo_avg` takes the softmax of the listed
# labels, so a tuple gives equal weights and an ndarray gives exact weights.
_EXPRESSIONS={
    'natural':{'emo':4,'exp':1.0},
    'smile':{'emo':(3,4),'exp':1.0},
    'serious':{'emo':(4,5),'exp':0.75},
    'surprise':{'emo':6,'exp':0.85},
}
# Neutral 88% + Sad 12% reads as a settled, unsmiling face; an even mix reads
# as sadness. The exact split is an initial value pending calibration.
_SERIOUS=0.12


def normalize_tone(value):
    return value if isinstance(value,str) and value in TONES else 'natural'


def normalize_expression(value):
    return value if isinstance(value,str) and value in EXPRESSIONS else 'natural'


def normalize_tone_prompt(value):
    """Keep one printable line, collapse runs of spaces, cut at the limit."""
    if not isinstance(value,str):return ''
    # A newline becomes a space; other control characters are dropped.
    text=''.join(' ' if char.isspace() else char for char in value
                 if char.isspace() or char.isprintable())
    return ' '.join(text.split())[:MAX_TONE_PROMPT]


def settings_of(manifest):
    raw=(manifest or {}).get('settings') or {}
    return {'tone':normalize_tone(raw.get('tone')),
            'expression':normalize_expression(raw.get('expression')),
            'tone_prompt':normalize_tone_prompt(raw.get('tone_prompt'))}


def merge_settings(current, update):
    """Validate one partial update; unknown keys or values are rejected."""
    if not isinstance(update,dict):raise ValueError('设置格式无效')
    unknown=set(update)-{'tone','expression','tone_prompt'}
    if unknown:raise ValueError('未知的设置项：'+'、'.join(sorted(unknown)))
    result=settings_of(current)
    if 'tone' in update:
        if update['tone'] not in TONES:raise ValueError('未知的语气风格')
        result['tone']=update['tone']
    if 'expression' in update:
        if update['expression'] not in EXPRESSIONS:raise ValueError('未知的表情')
        result['expression']=update['expression']
    if 'tone_prompt' in update:
        value=update['tone_prompt']
        if not isinstance(value,str):raise ValueError('自定义语气必须是文字')
        collapsed=' '.join(value.split())
        if len(collapsed)>MAX_TONE_PROMPT:
            raise ValueError(f'自定义语气最多 {MAX_TONE_PROMPT} 个字')
        result['tone_prompt']=collapsed
    return result


def voice_instruction_clause(tone, tone_prompt=''):
    extra=normalize_tone_prompt(tone_prompt)
    if extra:return f'请用{extra}的语气说一句话。'
    return TONE_INSTRUCTIONS[normalize_tone(tone)]


def tone_clause(tone, tone_prompt=''):
    base=TONE_PROMPTS[normalize_tone(tone)]
    extra=normalize_tone_prompt(tone_prompt)
    if not extra:return base
    refinement=f'另外要求：{extra}。'
    return base+refinement if base else refinement


def expression_emo(value):
    """Return the Ditto emotion condition for one preset.

    natural and surprise stay plain integers so the default path is exactly
    what it was before this feature existed.
    """
    preset=normalize_expression(value)
    if preset=='serious':
        row=np.zeros((1,8),np.float32)
        row[0,4]=1.-_SERIOUS;row[0,5]=_SERIOUS
        return row
    return _EXPRESSIONS[preset]['emo']


def expression_exp_gain(value):
    return _EXPRESSIONS[normalize_expression(value)]['exp']
