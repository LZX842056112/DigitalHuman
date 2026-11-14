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
import os
import sys
from pathlib import Path

import numpy as np

from backend import edit_geometry, image_rules, prompting

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


NEGATIVE = prompting.BASE_NEGATIVE


def _load_pipe(with_adapter=False, safety=True):
    """Load SD 1.5 inpainting once per process, optionally with IP-Adapter."""
    key = ('pipe_adapter' if with_adapter else 'pipe') + ('' if safety else '_nosafety')
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
    if not safety:
        # Offline, own photos, and the preview decode bypasses the checker
        # anyway. Dropping it frees about 600 MB of VRAM.
        pipe.safety_checker = None
        pipe.requires_safety_checker = False
    if with_adapter:
        # The adapter reads the reference garment through CLIP ViT-H, so the
        # prompt only describes the scene; the fabric comes from the image.
        pipe.load_ip_adapter(str(ROOT / ADAPTER), subfolder='models',
                             weight_name=ADAPTER_WEIGHT,
                             image_encoder_folder=str((ROOT / ADAPTER / 'models/image_encoder').resolve()))
    _cache['offloaded'] = False
    if _device() == 'cuda':
        try:
            pipe.to('cuda')
        except (RuntimeError, torch.cuda.OutOfMemoryError):
            pipe.enable_model_cpu_offload()
            _cache['offloaded'] = True
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


def inpaint(image, mask, prompt, *, steps=36, size=768, seed=0, guidance=7.5, fill=None,
            negative=None):
    """Regenerate the garment from a text description."""
    import torch
    pipe = _load_pipe()
    small, mask_img = _canvas(image, mask, size, fill)
    generator = torch.Generator(device='cpu').manual_seed(int(seed))
    result = pipe(prompt=prompt, negative_prompt=negative or NEGATIVE, image=small,
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
    return prompting.describe(text, timeout=timeout)


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
         guidance=6.5, scale=.7, grow=.03, negative=None):
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
                  negative_prompt=negative or ('text, watermark, logo, deformed, extra limbs, '
                                               'low quality, cartoon, illustration'),
                  image=small, mask_image=mask_img,
                  ip_adapter_image=reference, num_inference_steps=int(steps),
                  guidance_scale=float(guidance), strength=1.,
                  generator=generator).images[0]
    return _compose(image, mask, result)


# ---------------------------------------------------------------------------
# Outpainting: extend the photo downward until the whole person is visible.
# ---------------------------------------------------------------------------

PREVIEW_KEEP = 2
LEGS_MIN = 0.03


class PreviewWriter:
    """Write one intermediate picture every few diffusion steps.

    Files are named by step number and never overwritten, so the reader can
    never catch a half written PNG. The job record carries the current name and
    a version number the UI uses to bust its cache.
    """

    def __init__(self, folder=None, job_id=None, root=None, *, interval=4, skip=2,
                 long_edge=512, mode='decode', total=36,
                 progress_from=.10, progress_to=.90):
        self.folder = Path(folder) if folder else None
        self.job_id = job_id
        self.root = Path(root) if root else ROOT
        self.interval = max(1, int(interval))
        self.skip = max(0, int(skip))
        self.long_edge = int(long_edge)
        self.mode = (mode or 'decode').lower()
        self.total = max(1, int(total))
        self.progress_from = float(progress_from)
        self.progress_to = float(progress_to)
        self.version = 0
        self.writes = 0
        self.degraded = False
        if self.folder is not None:
            self.folder.mkdir(parents=True, exist_ok=True)

    @property
    def enabled(self):
        return self.folder is not None and self.mode != 'off'

    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        return self.callback(pipe, step_index, timestep, callback_kwargs)

    def callback(self, pipe, step_index, timestep, callback_kwargs):
        if not self.enabled:
            return callback_kwargs
        step = int(step_index)
        self._progress(step)
        if step < self.skip or (step + 1) % self.interval:
            return callback_kwargs
        latents = callback_kwargs.get('latents')
        if latents is None:
            return callback_kwargs
        picture = None
        if self.mode == 'decode' and not _cache.get('offloaded'):
            try:
                picture = self._decode(pipe, latents)
            except Exception:
                self.degraded = True
        if picture is None:
            try:
                picture = self._latent(latents)
            except Exception:
                return callback_kwargs
        try:
            self.write(step, picture)
        except OSError:
            pass
        return callback_kwargs

    def _decode(self, pipe, latents):
        import torch
        with torch.no_grad():
            image = pipe.vae.decode(
                latents / pipe.vae.config.scaling_factor, return_dict=False)[0]
        return pipe.image_processor.postprocess(image, output_type='pil')[0]

    def _latent(self, latents):
        """Cheap fallback: project the first three channels. Layout only."""
        from PIL import Image
        data = latents[0, :3]
        if hasattr(data, 'float'):
            data = data.float() / .18215
            data = data - data.min()
            data = data / (data.max() + 1e-6)
            array = data.permute(1, 2, 0).cpu().numpy()
        else:
            data = np.asarray(data, np.float32) / .18215
            data = data - data.min()
            data = data / (data.max() + 1e-6)
            array = np.transpose(data, (1, 2, 0))
        return Image.fromarray((array * 255).astype(np.uint8)).resize(
            (max(8, array.shape[1] * 8), max(8, array.shape[0] * 8)), Image.NEAREST)

    def _progress(self, step):
        share = (step + 1) / float(self.total)
        value = self.progress_from + (self.progress_to - self.progress_from) * share
        self._update(stage='diffuse',
                     progress=round(min(self.progress_to, max(0., value)), 3))

    def _update(self, **fields):
        if not self.job_id:
            return
        try:
            from backend.storage import Store
            Store(self.root).update_job(self.job_id, **fields)
        except (OSError, ValueError, KeyError):
            pass

    def write(self, step, image):
        from PIL import Image
        width, height = image.size
        if max(width, height) > self.long_edge:
            scale = self.long_edge / float(max(width, height))
            image = image.resize((max(1, round(width * scale)), max(1, round(height * scale))),
                                 Image.LANCZOS)
        name = f'step-{int(step):04d}.png'
        target = self.folder / name
        temporary = target.with_suffix('.tmp')
        image.convert('RGB').save(temporary, format='PNG')
        os.replace(temporary, target)
        self.version += 1
        self.writes += 1
        files = sorted(self.folder.glob('step-*.png'))
        for stale in files[:-PREVIEW_KEEP]:
            try:
                stale.unlink()
            except OSError:
                pass
        self._update(preview={'version': self.version, 'step': int(step),
                              'steps': self.total, 'file': name,
                              'long_edge': self.long_edge, 'mode': self.mode})


def _resize_pair(canvas, mask, size):
    from PIL import Image
    width, height = size
    small = Image.fromarray(np.asarray(canvas, np.uint8)).resize((width, height), Image.LANCZOS)
    mask_img = Image.fromarray((np.clip(mask, 0., 1.) * 255).astype(np.uint8)).resize(
        (width, height), Image.LANCZOS)
    return small, np.asarray(mask_img, np.float32) / 255.


def _upper_body(image, labels):
    """Tight crop of the visible torso, used as the adapter reference.

    A high adapter scale would copy this clothing into the generated legs, so
    the caller keeps the scale low. The crop still carries skin tone, lighting
    and fabric texture across the seam.
    """
    zone = edit_geometry.labels_of((4, 7, 11, 14, 15), labels)
    if not zone.any():
        return image
    rows = np.nonzero(zone.any(axis=1))[0]
    cols = np.nonzero(zone.any(axis=0))[0]
    if not rows.size or not cols.size:
        return image
    width, height = image.size
    pad_y = int((rows[-1] - rows[0]) * .08) + 4
    pad_x = int((cols[-1] - cols[0]) * .08) + 4
    box = (max(0, int(cols[0]) - pad_x), max(0, int(rows[0]) - pad_y),
           min(width, int(cols[-1]) + pad_x), min(height, int(rows[-1]) + pad_y))
    return image.crop(box)


def _quality(legs, capped=False):
    """Gate on what can actually be measured.

    A cross seam colour step is not a defect: a black top over light trousers
    legitimately changes colour at the hem, so colour is not used here. What
    can be measured is whether legs exist at all, and whether the request had
    to be capped into a three-quarter shot.
    """
    if legs < LEGS_MIN:
        return 'low'
    return 'fair' if capped else 'good'


def extend(photo, spec, out):
    """Extend one photo into a taller, fuller shot. Returns a report dict.

    Two passes are used when the target is taller than SD 1.5 handles well in
    one go. The second pass re-segments the intermediate, so its body estimate
    includes the legs the first pass just drew.
    """
    from PIL import Image
    source = photo if isinstance(photo, Image.Image) else image_rules.open_normalized(photo)
    target = float(spec.get('aspect', .75))
    limit = float(spec.get('max_ratio_per_pass', 1.55))
    ratios = [target]
    if 1. / target > limit and int(spec.get('passes', 2)) > 1:
        ratios = [1. / limit, target]

    writer = spec.get('_writer')
    current = source
    last = None
    for index, aspect in enumerate(ratios):
        final = index == len(ratios) - 1
        current, last = _extend_once(current, spec, aspect,
                                     writer=writer if final or len(ratios) == 1 else None)
        if last.get('skipped'):
            break
    if last.get('skipped'):
        return {'mode': 'extend', 'skipped': last['skipped'],
                'label': spec.get('label') or '全身·已完整',
                'canvas': list(current.size), 'size': list(current.size),
                'passes': len(ratios), 'quality': 'good', 'output': None}

    rgb, scale = edit_geometry.fit_output(np.asarray(current), int(spec.get('output_edge', 1536)))
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(rgb).save(out, format='PNG')
    last.update({'mode': 'extend', 'output': str(out), 'size': list(current.size),
                 'output_scale': round(scale, 3), 'passes': len(ratios),
                 'label': spec.get('label') or '全身'})
    return last


def _extend_once(image, spec, aspect, writer=None):
    from PIL import Image

    labels = label_map(image)
    box = edit_geometry.person_bbox(labels)
    head = edit_geometry.head_height(labels, box)
    plan = edit_geometry.plan_canvas(
        image.width, image.height, box, head, aspect=aspect,
        headroom=float(spec.get('headroom', .05)),
        min_gain=float(spec.get('min_gain', .06)),
        max_gain=float(spec.get('max_gain', 2.2)))
    if plan['skipped']:
        return image, {'skipped': plan['skipped'], 'plan': plan}

    person = edit_geometry.labels_of(edit_geometry.PERSON_LABELS, labels)
    protected = edit_geometry.protect_mask(plan['canvas'], plan['offset'], labels)
    mask, _ = edit_geometry.build_mask(plan['canvas'], plan['offset'], box,
                                       plan['source'], protected=protected)
    colors = edit_geometry.body_prior(labels, image)
    seed = int(spec.get('seed', 0))
    canvas, mask = edit_geometry.fill_canvas(image, plan, mask, colors,
                                             body_prior_fill=bool(spec.get('body_prior', True)),
                                             seed=seed)

    action = prompting.translate_cached(spec.get('action'), 'action')
    tone = prompting.translate_cached(spec.get('tone'), 'tone')
    expression = prompting.translate_cached(spec.get('expression'), 'expression')
    positive, negative = prompting.extend_prompt(
        action=action, tone=tone, expression=expression,
        garment=prompting.garment_hint(labels, image))

    scale = float(spec.get('scale', .35))
    try:
        pipe = _load_pipe(with_adapter=True, safety=False)
        reference = _upper_body(image, labels)
        adapted = True
    except FileNotFoundError:
        pipe = _load_pipe(with_adapter=False, safety=False)
        reference = None
        adapted = False
    if adapted:
        # The pipe is process cached; the scale is stateful on it.
        pipe.set_ip_adapter_scale(scale)

    steps = int(spec.get('steps', 36))
    work = edit_geometry.preview_size(plan, int(spec.get('working_edge', 768)))
    degraded = False
    legs = 0.
    result = None
    for attempt in range(2):
        canvas_small, mask_small = _resize_pair(canvas, mask, work)
        if writer is not None:
            writer.total = steps
        try:
            result = _render(pipe, canvas_small, mask_small, reference, positive,
                             negative, steps, spec, seed + attempt, writer)
        except Exception as error:  # noqa: BLE001 - only OOM is retried
            if 'out of memory' not in str(error).lower() or attempt:
                raise
            degraded = True
            _free_memory()
            work = (max(256, work[0] * 2 // 3 // 8 * 8), max(256, work[1] * 2 // 3 // 8 * 8))
            steps = max(20, steps * 2 // 3)
            continue
        small_plan = edit_geometry.scale_plan(plan, result.size)
        legs = edit_geometry.leg_ratio(label_map(result), small_plan)
        if legs >= LEGS_MIN or attempt:
            break
        # No legs usually means the model stopped at the hip. One retry with a
        # different seed and an explicit cue is cheaper than failing the edit.
        positive = positive + ', both feet visible, standing on the floor'

    native = result.resize((plan['canvas'][0], plan['canvas'][1]), Image.LANCZOS)
    out = edit_geometry.compose_extend(canvas, mask, np.asarray(native))
    identity = edit_geometry.check_identity(np.asarray(image), out, plan, labels)
    quality = _quality(legs, bool(plan.get('capped')))
    report = {'plan': plan, 'face_diff_ratio': identity['face_diff_ratio'],
              'protected_diff_ratio': identity['protected_diff_ratio'],
              'leg_ratio': round(float(legs), 3),
              'quality': quality, 'adapter': adapted, 'degraded': degraded,
              'canvas': plan['canvas'], 'gain': round(plan['gain'], 1),
              'capped': bool(plan.get('capped')), 'prompt': positive}
    if quality == 'low':
        report['warnings'] = ['没有检测到完整的腿部，建议换一张照片或重试']
    elif plan.get('capped'):
        report['warnings'] = ['照片里的人物偏大，这次只扩到七分身']
    return Image.fromarray(out), report


def _free_memory():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _render(pipe, canvas, mask, reference, positive, negative, steps, spec, seed, writer):
    import torch
    from PIL import Image
    kwargs = {'prompt': positive, 'negative_prompt': negative,
              'image': canvas,
              'mask_image': Image.fromarray((np.clip(mask, 0., 1.) * 255).astype(np.uint8)),
              'num_inference_steps': int(steps),
              'guidance_scale': float(spec.get('guidance', 6.5)),
              'strength': 1.,
              'generator': torch.Generator(device='cpu').manual_seed(int(seed))}
    if reference is not None:
        kwargs['ip_adapter_image'] = reference
    if writer is not None and writer.enabled:
        kwargs['callback_on_step_end'] = writer.callback
        kwargs['callback_on_step_end_tensor_inputs'] = ['latents']
    return pipe(**kwargs).images[0]


def run(photo, spec, out):
    """Apply one edit and write a PNG. Returns a small report."""
    from PIL import Image
    mode = spec.get('mode', 'recolor')
    if mode == 'extend':
        return extend(photo, spec, out)
    image = image_rules.open_normalized(photo)
    labels = label_map(image)
    parts = UPPER_LABELS if spec.get('scope', 'upper') != 'full' else UPPER_LABELS + LOWER_LABELS
    mask = garment_mask(labels, image, parts)
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
        phrase = prompting.translate_cached(spec.get('describe'))
        # A Chinese prompt reaches SD 1.5 as noise. Use it only when it is the
        # English phrase the caller already translated.
        given = (spec.get('prompt') or '').strip()
        prompt = given if given and all(ord(char) < 128 for char in given) else \
            prompting.wear_prompt(garment=phrase)[0]
        rgb = wear(image, mask, garment, prompt,
                   steps=int(spec.get('steps', 36)), size=int(spec.get('size', 768)),
                   seed=int(spec.get('seed', 0)), guidance=float(spec.get('guidance', 6.5)),
                   scale=float(spec.get('scale', .85)),
                   negative=prompting.EXTEND_NEGATIVE)
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


def analyze(photo):
    """Read the person geometry once, so the UI can preview a plan without torch."""
    image = image_rules.open_normalized(photo)
    labels = label_map(image)
    box = edit_geometry.person_bbox(labels)
    head = edit_geometry.head_height(labels, box)
    colors = edit_geometry.body_prior(labels, image)
    return {'size': [image.width, image.height], 'box': [int(value) for value in box],
            'head': round(float(head), 2),
            'colors': {name: [round(float(value), 1) for value in values]
                       for name, values in colors.items()}}


def main(argv=None):
    parser = argparse.ArgumentParser(description='给照片里的人物换装，或把照片扩成全身照')
    parser.add_argument('--photo', required=True)
    parser.add_argument('--out', help='结果图片路径')
    parser.add_argument('--garment', help='衣服图片路径（mode=wear 时必填）')
    parser.add_argument('--spec', default='{}', help='JSON: mode/color/strength/prompt/scope')
    parser.add_argument('--spec-file')
    parser.add_argument('--job-id', help='任务编号，用于写进度与中间图')
    parser.add_argument('--preview-dir', help='中间图目录，默认 work/previews/<job-id>')
    parser.add_argument('--analyze', action='store_true', help='只输出人物几何，不生成图片')
    options = parser.parse_args(argv)
    if options.analyze:
        try:
            print(json.dumps(analyze(options.photo), ensure_ascii=False))
        except (ValueError, FileNotFoundError) as error:
            print(json.dumps({'error': str(error)}, ensure_ascii=False))
            return 2
        return 0
    if not options.out:
        parser.error('--out 是必需的（--analyze 除外）')
    spec = json.loads(Path(options.spec_file).read_text(encoding='utf-8')
                      if options.spec_file else options.spec)
    if options.garment:
        spec['garment'] = options.garment
    if options.job_id and spec.get('mode') == 'extend':
        # Objects cannot travel through --spec, so the writer is built here.
        root = Path(os.environ.get('DIGITAL_HUMAN_ROOT') or ROOT)
        preview = spec.get('preview') or {}
        folder = Path(options.preview_dir) if options.preview_dir else \
            root / 'work/previews' / options.job_id
        spec['_writer'] = PreviewWriter(
            folder, job_id=options.job_id, root=root,
            interval=preview.get('steps_interval', preview.get('interval', 4)),
            skip=preview.get('skip_steps', preview.get('skip', 2)),
            long_edge=preview.get('long_edge', 512), mode=preview.get('mode', 'decode'),
            total=int(spec.get('steps', 36)))
    try:
        report = run(options.photo, spec, options.out)
    except (ValueError, FileNotFoundError) as error:
        print(json.dumps({'error': str(error)}, ensure_ascii=False))
        return 2
    print(json.dumps(report, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    sys.exit(main())
