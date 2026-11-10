"""Change the clothes on the source photo instead of asking for another photo.

The renderer only repaints a 512 px head crop, so clothing, background and
lighting stay the source pixels. That means an outfit change has to happen in
the photo itself, before ``prepare`` builds the portrait cache. This module
does exactly that, offline, on one image:

1. segment the garment (Segformer trained on clothes, labels 4 / 7);
2. either recolour it in Lab space, which keeps every fold and highlight, or
   regenerate it with the local inpainting model when one is installed.

Both paths leave every pixel outside the garment untouched, including the face,
so identity cannot drift.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SEGMENTER = 'models/outfit/clothes-seg'
INPAINT = 'models/outfit/inpaint'
ADAPTER = 'models/outfit/ip-adapter'
ADAPTER_WEIGHT = 'ip-adapter-plus_sd15.safetensors'

# Segformer clothes labels used here. 4 upper-clothes, 7 dress; 5/6 are kept
# optional because a full-body photo may show trousers that should follow the
# shirt into the new outfit.
UPPER_LABELS = (4, 7)
LOWER_LABELS = (5, 6, 12, 13)

COLORS = {
    'black': '#1c1c1e', 'white': '#f4f1ea', 'navy': '#1e2a44',
    'beige': '#d8c6a4', 'wine': '#6b1f2b', 'green': '#243a2f',
    'grey': '#8a8f94', 'blue': '#2f5d8a',
}

COLOR_LABELS = {
    'black': '黑色', 'white': '米白', 'navy': '藏青', 'beige': '米色',
    'wine': '酒红', 'green': '墨绿', 'grey': '浅灰', 'blue': '雾蓝',
}

STYLES = {
    'shirt': 'a crisp white cotton button-up shirt, collar and cuffs visible',
    'sweater': 'a soft oatmeal knit sweater with a round neck',
    'blazer': 'a navy tailored blazer over a white tee',
    'hoodie': 'a light grey hoodie with drawstrings',
    'coat': 'a camel wool coat with a lapel collar',
    'dress': 'a simple dark green dress with short sleeves',
}

STYLE_FILL = {
    'shirt': '#f2efe9', 'sweater': '#d8cbb4', 'blazer': '#1e2a44',
    'hoodie': '#b9bcc0', 'coat': '#c2a377', 'dress': '#243a2f',
}

STYLE_LABELS = {
    'shirt': '白衬衫', 'sweater': '针织衫', 'blazer': '西装外套',
    'hoodie': '卫衣', 'coat': '大衣', 'dress': '连衣裙',
}


def _color_hex(spec):
    value = spec.get('color') or COLORS['navy']
    for name, code in COLORS.items():
        if value == name:
            return code
    return value


def prompt_for(spec):
    style = (spec.get('style') or '').strip()
    base = STYLES.get(style, style or STYLES['shirt'])
    return (spec.get('prompt') or
            f'a portrait photo of the person wearing {base}, photorealistic, '
            'natural light, same pose and background')


def label_for(spec):
    if spec.get('label'):
        return str(spec['label'])[:20]
    if spec.get('mode') == 'wear':
        text = (spec.get('describe') or '').strip()
        head = text.replace('，', ',').split(',')[0][:8]
        return ('换装·' + head) if head else '换装·上传'
    if spec.get('mode') == 'generate':
        name = STYLE_LABELS.get(spec.get('style') or '', '新衣')
        return f'换装·{name}'[:20]
    for name, code in COLORS.items():
        if _color_hex(spec).lower() == code.lower():
            return f'换装·{COLOR_LABELS[name]}'[:20]
    return '换装·自定义'


_cache: dict[str, object] = {}


def _torch():
    import torch
    return torch


def _device():
    torch = _torch()
    return 'cuda' if torch.cuda.is_available() else 'cpu'


def _segmenter():
    """Load the clothes segmenter once per process (about 430 MB)."""
    net = _cache.get('seg')
    if net is None:
        from transformers import SegformerForSemanticSegmentation
        net = SegformerForSemanticSegmentation.from_pretrained(
            str(ROOT / SEGMENTER)).eval().to(_device())
        _cache['seg'] = net
    return net


def _processor():
    proc = _cache.get('proc')
    if proc is None:
        from transformers import SegformerImageProcessor
        proc = SegformerImageProcessor.from_pretrained(str(ROOT / SEGMENTER))
        _cache['proc'] = proc
    return proc


def label_map(image):
    """Per-pixel clothes labels at the photo resolution."""
    torch = _torch()
    proc, net = _processor(), _segmenter()
    inputs = {k: v.to(_device()) for k, v in proc(images=image, return_tensors='pt').items()}
    with torch.no_grad():
        logits = net(**inputs).logits
    logits = torch.nn.functional.interpolate(logits, size=(image.height, image.width),
                                             mode='bilinear', align_corners=False)
    return logits.argmax(1)[0].cpu().numpy().astype(np.uint8)


def garment_mask(labels, image=None, parts=None, feather=1.6, grow=2):
    """Soft 0..1 mask for the requested labels.

    The segmenter stops a pixel or two inside the real silhouette, so a plain
    mask leaves the garment's own dark rim behind as an outline. The ring
    around the mask is therefore grown back only where the pixel looks like the
    garment rather than like skin or background.
    """
    import cv2
    wanted = set(parts or UPPER_LABELS)
    base = np.isin(labels, list(wanted)).astype(np.float32)
    if not base.any():
        raise ValueError('没有找到可以换装的衣物区域')
    if image is not None and grow > 0:
        rgb = np.asarray(image.convert('RGB'), np.float32)
        ring = (cv2.dilate(base, np.ones((3, 3), np.uint8), iterations=int(grow)) - base) > 0
        if ring.any():
            core = base > 0
            garment = np.median(rgb[core], axis=0)
            skin_pixels = rgb[np.isin(labels, [3, 11])]
            skin = np.median(skin_pixels, axis=0) if skin_pixels.size else garment
            near = np.linalg.norm(rgb[ring] - garment, axis=1)
            far = np.linalg.norm(rgb[ring] - skin, axis=1)
            keep = near <= far
            grown = base.copy()
            rows, cols = np.nonzero(ring)
            grown[rows[keep], cols[keep]] = 1.
            base = grown
    mask = cv2.GaussianBlur(base, (0, 0), feather)
    return np.clip(mask, 0., 1.)


def _rgb_to_lab(rgb):
    import cv2
    return cv2.cvtColor(np.asarray(rgb, np.float32), cv2.COLOR_RGB2LAB)


def _lab_to_rgb(lab):
    import cv2
    return np.clip(cv2.cvtColor(np.asarray(lab, np.float32), cv2.COLOR_LAB2RGB), 0., 1.)


def _hex_to_lab(value):
    value = value.strip().lstrip('#')
    if len(value) != 6:
        raise ValueError('颜色要写成 #RRGGBB')
    rgb = np.array([[[int(value[i:i + 2], 16) / 255. for i in (0, 2, 4)]]], np.float32)
    return _rgb_to_lab(rgb)[0, 0]


def recolor(image, mask, color, strength=1., chroma=1.45, floor=32.):
    """Repaint the garment in a new colour while keeping its shading.

    Lightness keeps the source's relative variation, so folds, seams and the
    shadow under the chin survive. Chroma is replaced outright, which is what
    makes a black tee read as navy instead of dark grey.
    """
    strength = float(np.clip(strength, 0., 1.))
    rgb = np.asarray(image.convert('RGB'), np.float32) / 255.
    lab = _rgb_to_lab(rgb)
    target = _hex_to_lab(color)
    region = mask > .02
    source_l = lab[..., 0][region]
    shift = float(target[0] - np.median(source_l))
    # A black tee sits around L 25. Copying navy's L 16 straight onto it makes
    # the result read as charcoal, so very dark targets get lifted towards the
    # darkest level at which the hue is still visible.
    if target[0] < floor:
        shift += (floor - target[0]) * .55
    out = lab.copy()
    out[..., 0] = np.clip(lab[..., 0] + shift * .92, 0., 100.)
    # Dark fabrics lose most of their chroma on screen, so a straight copy of
    # the target's a/b reads charcoal. Boost it so "navy" still looks navy.
    out[..., 1] = target[1] * chroma
    out[..., 2] = target[2] * chroma
    # Only L is bounded: clipping a/b at zero would flatten every blue and
    # green target into grey, which is exactly what navy must not do.
    out[..., 0] = np.clip(out[..., 0], 0., 100.)
    edited = _lab_to_rgb(out)
    weight = (mask * strength)[..., None]
    mixed = rgb * (1. - weight) + edited * weight
    return (np.clip(mixed, 0., 1.) * 255.).round().astype(np.uint8)


NEGATIVE = ('plaid, checkered, striped, pattern, print, sheer, translucent, transparent, '
            'see-through, lace, satin, text, watermark, logo, extra limbs, deformed')


def _load_pipe(with_adapter=False):
    """Load SD 1.5 inpainting once per process, optionally with IP-Adapter."""
    key = 'pipe_adapter' if with_adapter else 'pipe'
    pipe = _cache.get(key)
    if pipe is not None:
        return pipe
    folder = ROOT / INPAINT
    if not (folder / 'model_index.json').is_file():
        raise FileNotFoundError(
            '未安装换装模型（models/outfit/inpaint）。可先使用换色，或运行 '
            'python scripts/fetch_outfit_model.py')
    if with_adapter:
        weight = ROOT / ADAPTER / 'models' / ADAPTER_WEIGHT
        encoder = ROOT / ADAPTER / 'models/image_encoder/model.safetensors'
        if not weight.is_file() or not encoder.is_file():
            raise FileNotFoundError(
                '未安装衣服参考模型（models/outfit/ip-adapter）。运行 '
                'python scripts/fetch_outfit_model.py --only adapter')
    import torch
    from diffusers import AutoPipelineForInpainting
    kwargs = {'torch_dtype': torch.float16}
    try:
        pipe = AutoPipelineForInpainting.from_pretrained(
            str(folder), variant='fp16', **kwargs)
    except (ValueError, OSError):
        pipe = AutoPipelineForInpainting.from_pretrained(str(folder), **kwargs)
    pipe.set_progress_bar_config(disable=True)
    if with_adapter:
        # The adapter reads the reference garment through CLIP ViT-H, so the
        # prompt only describes the scene; the fabric comes from the image.
        pipe.load_ip_adapter(str(ROOT / ADAPTER), subfolder='models',
                             weight_name=ADAPTER_WEIGHT,
                             image_encoder_folder=str((ROOT / ADAPTER / 'models/image_encoder').resolve()))
    if _device() == 'cuda':
        try:
            pipe.to('cuda')
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            pipe.enable_model_cpu_offload()
    _cache[key] = pipe
    return pipe


def _canvas(image, mask, size, fill):
    """Resize the photo, then flat-fill the hole so colour survives diffusion."""
    from PIL import Image
    width, height = image.size
    scale = size / max(width, height)
    small = image.resize((int(width * scale) // 8 * 8, int(height * scale) // 8 * 8),
                         Image.LANCZOS)
    mask_img = Image.fromarray((np.clip(mask, 0., 1.) * 255).astype(np.uint8)).resize(
        small.size, Image.LANCZOS)
    hole = np.asarray(mask_img, np.float32)[..., None] / 255.
    fill = (fill or '#cfcfcf').lstrip('#')
    fill_rgb = np.array([int(fill[i:i + 2], 16) for i in (0, 2, 4)], np.float32)
    base = np.asarray(small, np.float32)
    small = Image.fromarray((base * (1. - hole) + fill_rgb * hole).astype(np.uint8))
    return small, mask_img


def _compose(image, mask, result):
    """Paste the generated garment back through the feathered mask."""
    from PIL import Image
    result = result.resize(image.size, Image.LANCZOS)
    edited = np.asarray(result, np.float32) / 255.
    base = np.asarray(image.convert('RGB'), np.float32) / 255.
    weight = mask[..., None]
    mixed = base * (1. - weight) + edited * weight
    return (np.clip(mixed, 0., 1.) * 255.).round().astype(np.uint8)


def inpaint(image, mask, prompt, *, steps=36, size=768, seed=0, guidance=7.5, fill=None):
    """Regenerate the garment from a text description."""
    import torch
    pipe = _load_pipe()
    small, mask_img = _canvas(image, mask, size, fill)
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    result = pipe(prompt=prompt, negative_prompt=NEGATIVE, image=small,
                  mask_image=mask_img, num_inference_steps=int(steps),
                  guidance_scale=float(guidance), strength=1., generator=generator).images[0]
    return _compose(image, mask, result)


def describe(text, timeout=8.):
    """Turn a Chinese garment description into an English noun phrase.

    SD 1.5 was trained on English captions, and the noun decides which
    silhouette the model draws (jacket vs shirt vs sweater), so the local
    dialogue model is used as a tiny translator. Any failure returns an empty
    string instead of blocking the edit.
    """
    text = (text or '').strip()
    if not text:
        return ''
    if all(ord(char) < 128 for char in text):
        return text[:80]
    import json as _json
    import urllib.request
    try:
        from backend.config import load_config
        config = load_config(ROOT) or {}
    except Exception:
        config = {}
    dialogue = config.get('dialogue') or {}
    payload = {'model': dialogue.get('model', 'qwen2.5-3b-instruct'),
               'messages': [
                   {'role': 'system', 'content': '你是服装描述翻译器。把中文服装描述翻译成简短的'
                                                 '英文名词短语，只输出英文，不要解释。'},
                   {'role': 'user', 'content': text}],
               'max_tokens': 32, 'temperature': 0}
    request = urllib.request.Request(
        dialogue.get('base_url', 'http://127.0.0.1:8081/v1') + '/chat/completions',
        data=_json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(request, timeout=timeout) as response:
            data = _json.load(response)
        phrase = str(data['choices'][0]['message']['content']).strip().strip('.\"\n ')
        import re
        # Small models sometimes glue the words together.
        phrase = re.sub(r'(?<=[a-z])(?=[A-Z])', ' ', phrase).lower()
        return ' '.join(phrase.split())[:80]
    except Exception:
        return ''


def _garment_crop(garment):
    """Trim the white product background so CLIP sees mostly fabric."""
    from PIL import Image
    rgb = np.asarray(garment.convert('RGB'), np.float32)
    solid = (rgb.min(axis=2) < 244)
    if solid.mean() < .02:
        return garment
    rows, cols = np.nonzero(solid)
    pad_y = int((rows.max() - rows.min()) * .04) + 2
    pad_x = int((cols.max() - cols.min()) * .04) + 2
    box = (max(0, cols.min() - pad_x), max(0, rows.min() - pad_y),
           min(rgb.shape[1], cols.max() + pad_x), min(rgb.shape[0], rows.max() + pad_y))
    return garment.crop(box)


def garment_fill(garment):
    """Dominant fabric colour, ignoring the near-white background."""
    rgb = np.asarray(garment.convert('RGB'), np.float32)
    solid = rgb[rgb.min(axis=2) < 244]
    if not solid.size:
        solid = rgb.reshape(-1, 3)
    return '#%02x%02x%02x' % tuple(int(v) for v in np.median(solid, axis=0))


def wear(image, mask, garment, prompt=None, *, steps=36, size=768, seed=0,
         guidance=6.5, scale=.7, grow=.03):
    """Put the garment from a reference photo onto the person in ``image``.

    The cut comes from the person's own silhouette, the fabric and colour come
    from the reference image through IP-Adapter. Nothing outside the mask is
    touched, so the face cannot drift.
    """
    import torch
    from PIL import Image
    if not Path(garment).is_file():
        raise FileNotFoundError('找不到衣服图片')
    reference = _garment_crop(Image.open(garment))
    if grow:
        import cv2
        px = max(1, int(min(image.size) * float(grow)))
        grown = cv2.dilate(mask, np.ones((3, 3), np.uint8), iterations=int(px))
        mask = np.clip(np.maximum(mask, grown * .9), 0., 1.)
    pipe = _load_pipe(with_adapter=True)
    pipe.set_ip_adapter_scale(float(scale))
    small, mask_img = _canvas(image, mask, size, garment_fill(reference))
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    result = pipe(prompt=prompt or ('a photo of a person wearing the garment, front view, '
                                    'natural fabric folds, studio portrait, photorealistic'),
                  negative_prompt=('text, watermark, logo, deformed, extra limbs, low quality, '
                                   'cartoon, illustration'),
                  image=small, mask_image=mask_img,
                  ip_adapter_image=reference, num_inference_steps=int(steps),
                  guidance_scale=float(guidance), strength=1.,
                  generator=generator).images[0]
    return _compose(image, mask, result)


def run(photo, spec, out):
    """Apply one outfit edit and write a PNG. Returns a small report."""
    from PIL import Image
    image = Image.open(photo).convert('RGB')
    labels = label_map(image)
    parts = UPPER_LABELS if spec.get('scope', 'upper') != 'full' else UPPER_LABELS + LOWER_LABELS
    mask = garment_mask(labels, image, parts)
    mode = spec.get('mode', 'recolor')
    if mode == 'recolor':
        rgb = recolor(image, mask, _color_hex(spec), float(spec.get('strength', 1.)))
    elif mode == 'generate':
        rgb = inpaint(image, mask, prompt_for(spec),
                      steps=int(spec.get('steps', 36)), size=int(spec.get('size', 768)),
                      seed=int(spec.get('seed', 0)), guidance=float(spec.get('guidance', 7.5)),
                      fill=STYLE_FILL.get(spec.get('style') or '', '#cfcfcf'))
    elif mode == 'wear':
        garment = spec.get('garment')
        if not garment:
            raise ValueError('缺少衣服图片')
        phrase = describe(spec.get('describe'))
        prompt = (f'a photo of a person wearing a {phrase}, front view, natural folds, '
                  'studio portrait, photorealistic') if phrase else spec.get('prompt')
        rgb = wear(image, mask, garment, prompt,
                   steps=int(spec.get('steps', 36)), size=int(spec.get('size', 768)),
                   seed=int(spec.get('seed', 0)), guidance=float(spec.get('guidance', 6.5)),
                   scale=float(spec.get('scale', .85)))
    else:
        raise ValueError(f'未知的换装方式：{mode}')
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out, format='PNG')
    covered = float(mask.mean())
    changed = float((np.abs(rgb.astype(np.int16)
                            - np.asarray(image, np.int16)).max(axis=2) > 2).mean())
    return {'mode': mode, 'scope': spec.get('scope', 'upper'), 'label': label_for(spec),
            'mask_ratio': round(covered, 4),
            'changed_ratio': round(changed, 4), 'output': str(out),
            'size': list(image.size)}


def main(argv=None):
    parser = argparse.ArgumentParser(description='给照片里的人物换装')
    parser.add_argument('--photo', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--garment', help='衣服图片路径（mode=wear 时必填）')
    parser.add_argument('--spec', default='{}', help='JSON: mode/color/strength/prompt/scope')
    parser.add_argument('--spec-file')
    options = parser.parse_args(argv)
    spec = json.loads(Path(options.spec_file).read_text(encoding='utf-8')
                      if options.spec_file else options.spec)
    if options.garment:
        spec['garment'] = options.garment
    try:
        report = run(options.photo, spec, options.out)
    except (ValueError, FileNotFoundError) as error:
        print(json.dumps({'error': str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
