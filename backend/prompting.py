"""Turn short Chinese prompt text into the English that SD 1.5 expects.

The inpainting model was trained on English captions. A Chinese phrase either
does nothing or steers the picture at random, so the local dialogue model is
used as a tiny translator, exactly as ``outfit.describe`` already did for
garment names.

A translation failure must never block an edit. Every function here returns an
empty string instead of raising, and the caller falls back to a neutral prompt.

The cache lives on disk: each edit runs in a new subprocess, so an in-memory
cache would be empty every time and would pay the round trip again.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import urllib.request

ROOT = Path(__file__).resolve().parents[1]
CACHE_FILE = 'work/cache/prompts.json'
CACHE_SIZE = 500

KINDS = ('action', 'tone', 'expression', 'garment')

SYSTEM = {
    'action': '你是动作描述翻译器。把中文动作描述翻译成简短的英文动作短语，'
              '只输出英文，不要解释，不要包含主语。',
    'tone': '你是神态描述翻译器。把中文语气或神态描述翻译成简短的英文气质与身体语言短语，'
            '例如 gentle and warm, relaxed shoulders。只输出英文，不要解释。',
    'expression': '你是表情描述翻译器。把中文表情描述翻译成一个英文表情短语，只输出英文，不要解释。',
    'garment': '你是服装描述翻译器。把中文服装描述翻译成简短的英文名词短语，只输出英文，不要解释。',
}

MAX_TOKENS = {'action': 48, 'tone': 40, 'expression': 24, 'garment': 32}

# Shared with ``outfit``. Kept here so the prompt strings have one home.
BASE_NEGATIVE = ('plaid, checkered, striped, pattern, print, sheer, translucent, transparent, '
                 'see-through, lace, satin, text, watermark, logo, extra limbs, deformed')

# SD 1.5 has a strong habit of drawing a second person when a tall canvas is
# mostly empty. Every one of these terms is a countermeasure that was needed
# during calibration; do not trim the list without re-testing 9:16.
EXTEND_NEGATIVE = BASE_NEGATIVE + (
    ', two people, multiple persons, duplicate, cloned person, second head, extra head, '
    'extra legs, extra arms, fused legs, deformed feet, cropped feet, cut off, out of frame, '
    'long neck, tiny head, cartoon, 3d render, border, frame')

DEFAULT_ACTION = 'standing still, arms relaxed'
DEFAULT_EXPRESSION = 'calm neutral expression'
FULL_BODY = 'full body photo of the same person, head to toe visible, both feet in frame'
EXTEND_TAIL = ('photorealistic, natural skin tone, natural lighting, '
               'plain continuous background, sharp focus')

# Same eight names as ``outfit.COLORS``. A test keeps the two in sync.
PALETTE = {
    'black': (28, 28, 30), 'white': (244, 241, 234), 'navy': (30, 42, 68),
    'beige': (216, 198, 164), 'wine': (107, 31, 43), 'green': (36, 58, 47),
    'grey': (138, 143, 148), 'blue': (47, 93, 138),
}

COLOR_WORDS = {
    'black': 'black', 'white': 'off white', 'navy': 'navy blue', 'beige': 'beige',
    'wine': 'deep wine', 'green': 'dark green', 'grey': 'grey', 'blue': 'muted blue',
}


def _clean(text):
    phrase = str(text or '').strip().strip('.\"\n ')
    # Small models sometimes glue the words together.
    phrase = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', phrase).lower()
    phrase = re.sub(r'\s*,\s*', ', ', phrase)
    return ' '.join(phrase.split())[:80]


def _is_ascii(text):
    return all(ord(char) < 128 for char in text)


def _dialogue(config=None):
    if config is None:
        try:
            from backend.config import load_config
            config = load_config(ROOT) or {}
        except Exception:
            config = {}
    return config.get('dialogue') or {}


def translate(text, kind='garment', *, timeout=6., config=None):
    """One Chinese phrase to English. Returns '' on any failure."""
    text = (text or '').strip()
    if not text:
        return ''
    if _is_ascii(text):
        return text[:80]
    if kind not in SYSTEM:
        kind = 'garment'
    dialogue = _dialogue(config)
    payload = {'model': dialogue.get('model', 'qwen2.5-3b-instruct'),
               'messages': [{'role': 'system', 'content': SYSTEM[kind]},
                            {'role': 'user', 'content': text}],
               'max_tokens': MAX_TOKENS[kind], 'temperature': 0}
    request = urllib.request.Request(
        dialogue.get('base_url', 'http://127.0.0.1:8081/v1') + '/chat/completions',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    # The dialogue server is local. A system proxy would break the loopback call.
    try:
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
        with opener.open(request, timeout=timeout) as response:
            data = json.load(response)
        return _clean(data['choices'][0]['message']['content'])
    except Exception:
        return ''


def _cache_path(root=ROOT):
    return Path(root) / CACHE_FILE


def read_cache(root=ROOT):
    try:
        data = json.loads(_cache_path(root).read_text(encoding='utf-8'))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError):
        return {}


def write_cache(cache, root=ROOT, size=CACHE_SIZE):
    """Drop the oldest entries, then replace the file in one step."""
    try:
        if len(cache) > size:
            cache = dict(list(cache.items())[len(cache) - size:])
        path = _cache_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(json.dumps(cache, ensure_ascii=False), encoding='utf-8')
        os.replace(temporary, path)
    except OSError:
        pass
    return cache


def translate_cached(text, kind='garment', *, root=ROOT, size=CACHE_SIZE, **kwargs):
    text = (text or '').strip()
    if not text:
        return ''
    if _is_ascii(text):
        return text[:80]
    key = hashlib.sha256(f'{kind}\0{text}'.encode('utf-8')).hexdigest()[:32]
    cache = read_cache(root)
    if key in cache:
        return cache[key]
    phrase = translate(text, kind, root=root, **kwargs)
    if phrase:
        cache[key] = phrase
        write_cache(cache, root, size)
    return phrase


def nearest_color_name(rgb):
    """Closest palette name by Euclidean RGB distance."""
    values = [float(v) for v in rgb[:3]]
    best, distance = 'grey', None
    for name, reference in PALETTE.items():
        current = sum((values[i] - reference[i]) ** 2 for i in range(3)) ** 0.5
        if distance is None or current < distance:
            best, distance = name, current
    return best


def garment_hint(labels, image):
    """A one clause description of what the lower body already wears.

    Matching the trousers the person already has is the cheapest way to stop
    the model from inventing a second outfit below the waist.
    """
    import numpy as np
    from backend.edit_geometry import LOWER_LABELS

    rgb = np.asarray(image.convert('RGB'), np.float32)
    zone = np.isin(labels, list(LOWER_LABELS))
    if float(zone.mean()) <= 0.004:
        return 'wearing dark trousers'
    name = nearest_color_name(np.median(rgb[zone], axis=0))
    return f'wearing {COLOR_WORDS[name]} trousers and matching shoes'


def extend_prompt(*, action='', tone='', expression='', garment=''):
    """Positive and negative prompts for one outpainting pass."""
    parts = [action or DEFAULT_ACTION, FULL_BODY, expression or DEFAULT_EXPRESSION,
             tone, garment, EXTEND_TAIL]
    positive = ', '.join(part.strip() for part in parts if part and part.strip())
    return positive, EXTEND_NEGATIVE


def wear_prompt(*, garment='', action='', expression=''):
    """Positive and negative prompts for one outfit pass."""
    head = (f'a photo of a person wearing a {garment}' if garment
            else 'a photo of a person wearing the garment')
    parts = [head, action, expression,
             'front view, natural fabric folds, studio portrait, photorealistic']
    positive = ', '.join(part.strip() for part in parts if part and part.strip())
    return positive, EXTEND_NEGATIVE


def describe(text, timeout=8., root=ROOT):
    """Backward compatible garment translation used by ``outfit.describe``."""
    return translate_cached(text, 'garment', root=root, timeout=timeout)
