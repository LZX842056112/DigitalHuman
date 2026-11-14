"""Geometry and masks for extending a photo into a full body shot.

The inpainting pipeline can only repaint inside a mask. Extending a headshot
therefore has to answer three questions before any diffusion runs:

1. How tall is the person, so how much canvas is missing below the waist?
2. Which part of the new canvas may the model touch?
3. What should that part contain before diffusion starts?

This module answers all three with numpy and PIL only. It must never import
torch: the API process imports it to draw the instant preview, and that process
is not allowed to load torch.

The safety property is the same as ``outfit``: every pixel outside the mask is
copied, not blended, so the face cannot drift.
"""
from __future__ import annotations

import numpy as np

# Segformer clothes labels. 0 is background, 16 is a bag. A bag is not part of
# the body, so it must not decide the body height, but it must not be repainted
# either: it belongs to the person's silhouette.
PERSON_LABELS = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 17)
HEAD_LABELS = (1, 2, 3, 11)
LOWER_LABELS = (5, 6, 9, 10, 12, 13)
FACE_LABELS = (3, 11)
PROTECT_LABELS = PERSON_LABELS + (16,)

# An adult is about 7.2 heads tall. 7.5 over-shoots for a seated or cropped
# shot, which makes the model invent a longer torso than the person has.
HEADS_PER_BODY = 7.2

# Corridor width as a multiple of the visible body width. Legs splay a little
# wider than the shoulders, so the corridor is slightly wider than the bbox.
CORRIDOR_WIDTH = 1.18
CORRIDOR_SPLAY = 0.18
SEAM_BAND_RATIO = 0.012
SEAM_BAND_BOUNDS = (8, 24)

SKIPPED = 'already_full_body'
NO_PERSON = '没有识别到完整的人物，请换一张能看到头和肩的照片'


def labels_of(classes, labels):
    """Boolean map of the wanted label classes."""
    return np.isin(labels, list(classes))


def person_bbox(labels, minimum=0.02):
    """Tight box around every body label, or raise when nothing was found."""
    mask = labels_of(PERSON_LABELS, labels)
    if mask.mean() < minimum:
        raise ValueError(NO_PERSON)
    rows = np.nonzero(mask.any(axis=1))[0]
    cols = np.nonzero(mask.any(axis=0))[0]
    if not rows.size or not cols.size:
        raise ValueError(NO_PERSON)
    return int(cols[0]), int(rows[0]), int(cols[-1]) + 1, int(rows[-1]) + 1


def head_height(labels, box):
    """Pixel height of the head, used as the body ruler.

    When the segmenter finds no head label at all the photo probably shows an
    unsupported crop. Falling back to a quarter of the visible body keeps the
    estimate finite instead of failing the whole edit.
    """
    x0, y0, x1, y1 = box
    head = labels_of(HEAD_LABELS, labels)[y0:y1, x0:x1]
    rows = np.nonzero(head.any(axis=1))[0]
    if rows.size:
        return float(rows[-1] - rows[0] + 1)
    return float(max(1., (y1 - y0) * 0.24))


def body_prior(labels, image):
    """Colours that stand in for the parts the photo does not show.

    ``lower`` is the best guess for trousers and shoes: the visible lower body
    when there is one, else the top, else skin. The corridor is filled with it
    so the model continues the outfit instead of inventing a second one.
    """
    rgb = np.asarray(image.convert('RGB'), np.float32)
    lower = None
    for classes, floor in ((LOWER_LABELS, 0.004), (PERSON_LABELS[:8], 0.01),
                           (HEAD_LABELS, 0.002)):
        zone = labels_of(classes, labels)
        if zone.mean() > floor:
            lower = np.median(rgb[zone], axis=0)
            break
    if lower is None:
        lower = np.array([62., 62., 66.], np.float32)
    face = labels_of(FACE_LABELS, labels)
    skin = (np.median(rgb[face] if face.mean() > 0.002 else rgb.reshape(-1, 3)[::7],
                      axis=0))
    return {'lower': lower.astype(np.float32), 'skin': np.asarray(skin, np.float32)}


def seam_band(canvas_width, canvas_height, ratio=SEAM_BAND_RATIO, bounds=SEAM_BAND_BOUNDS):
    band = int(round(ratio * max(canvas_width, canvas_height)))
    return int(np.clip(band, bounds[0], bounds[1]))


def plan_canvas(width, height, box, head, *, aspect=0.75, headroom=0.05,
                min_gain=0.06, max_gain=2.2):
    """Place the source photo on a taller canvas and say how much is new.

    Returns a dict. ``skipped`` is set when the photo already reaches the
    bottom edge and the body estimate needs no extra room, in which case the
    caller must not generate anything.
    """
    if not 0. < float(aspect) <= 1.:
        raise ValueError(f'不支持的画幅比例：{aspect}')
    x0, y0, x1, y1 = box
    floor_gain = float(clip_scalar(min_gain * height, 4., max_gain * height))
    predicted = (y0 + HEADS_PER_BODY * float(head)) - height
    already = y1 >= 0.97 * height and predicted <= min_gain * height

    # A photo already cropped at the crown must not gain a synthetic scalp:
    # the model would repaint hair over the face's own hairline.
    pad = 0. if y0 <= 0.01 * height else float(
        np.clip(headroom * height, 0.02 * height, 0.12 * height))

    if already:
        return {'source': [width, height], 'canvas': [width, height],
                'offset': [0, 0], 'gain': 0., 'head_pad': 0., 'band': 8,
                'aspect': float(aspect), 'skipped': SKIPPED,
                'body_box': [x0, y0, x1, y1], 'head': float(head),
                'predicted_gain': float(predicted), 'capped': False}

    gain = float(np.clip(predicted, floor_gain, max_gain * height))
    canvas_h = height + pad + gain
    canvas_w = round(canvas_h * float(aspect))
    if canvas_w < width:
        # A wide source cannot shrink. Grow the canvas height instead so the
        # requested aspect still holds after the source is placed.
        canvas_w = width
        canvas_h = round(width / float(aspect))
        gain = canvas_h - height - pad

    offset_x = int(np.clip(round(canvas_w / 2. - (x0 + x1) / 2.), 0, canvas_w - width))
    offset_y = int(round(pad))
    canvas_w = int(canvas_w)
    canvas_h = int(canvas_h)
    return {'source': [int(width), int(height)],
            'canvas': [canvas_w, canvas_h],
            'offset': [offset_x, offset_y],
            'gain': float(gain), 'head_pad': float(pad),
            'band': seam_band(canvas_w, canvas_h),
            'aspect': float(aspect), 'skipped': None,
            'body_box': [int(x0), int(y0), int(x1), int(y1)],
            'head': float(head),
            # A close-up needs more room than one pass may add. The caller
            # must tell the user the result is a three-quarter shot, not full.
            'predicted_gain': float(predicted),
            'capped': bool(predicted > gain + 1.)}


def clip_scalar(value, low, high):
    return max(low, min(high, value))


def corridor(canvas_size, offset, box, *, width=CORRIDOR_WIDTH, splay=CORRIDOR_SPLAY):
    """Vertical band under the person where the model is allowed to draw legs.

    Everything outside this band stays background. Handing the whole padding
    area to the model is what produces a second person in the background.
    """
    canvas_w, canvas_h = canvas_size
    offset_x, offset_y = offset
    x0, _, x1, y1 = box
    axis = offset_x + (x0 + x1) / 2.
    half = max(2., (x1 - x0) * width / 2.)
    top = offset_y + y1
    top = int(min(max(0, top), canvas_h - 1))
    rows = np.arange(canvas_h)
    depth = np.clip((rows - top) / max(1., canvas_h - 1 - top), 0., 1.)
    half_w = half * (1. + splay * depth)
    cols = np.arange(canvas_w)
    return ((rows[:, None] >= top) &
            (np.abs(cols[None, :] - axis) <= half_w[:, None]))


def build_mask(canvas_size, offset, box, source_size, *, protected=None):
    """Mask = the person corridor, starting at the visible hem.

    The corridor starts at the hem, not at the source edge. A photo cropped
    above the hem therefore gets continuous legs instead of a stripe of
    background between the hem and the generated body. The price is that a few
    rows of *background* inside the corridor may be repainted; person pixels
    never are, because the corridor starts below them.

    ``protected`` is cut out afterwards. It is normally a no-op, because the
    face sits well above the hem, but it keeps the guarantee independent from
    the segmentation.
    """
    import cv2

    canvas_w, canvas_h = canvas_size
    offset_x, offset_y = offset
    source_w, source_h = source_size
    body = corridor(canvas_size, offset, box)
    mask = body.astype(np.float32)
    if protected is not None and np.asarray(protected).any():
        edge = cv2.GaussianBlur(protected.astype(np.float32), (0, 0), 1.)
        mask = mask * (1. - np.clip(edge, 0., 1.))
    new = np.ones((canvas_h, canvas_w), bool)
    new[offset_y:offset_y + source_h, offset_x:offset_x + source_w] = False
    return np.clip(mask, 0., 1.), new


def protect_mask(canvas_size, offset, labels, classes=FACE_LABELS, grow=2):
    """Zone that must stay byte-identical, in canvas coordinates."""
    import cv2
    canvas_w, canvas_h = canvas_size
    offset_x, offset_y = offset
    source_h, source_w = labels.shape
    zone = np.zeros((canvas_h, canvas_w), bool)
    zone[offset_y:offset_y + source_h,
         offset_x:offset_x + source_w] = labels_of(classes, labels)
    if not zone.any():
        return zone
    grown = cv2.dilate(zone.astype(np.uint8), np.ones((3, 3), np.uint8),
                       iterations=int(grow))
    return grown.astype(bool)


def fill_canvas(image, plan, mask, colors, *, body_prior_fill=True, seed=0):
    """Prepare the canvas the model sees before diffusion.

    Padding must not copy the person. A plain mirror duplicates the face into
    the corners and under the body, and the model happily follows those
    silhouettes into a second person. The padding therefore extends the border
    colour and fades it towards a single backdrop tone, which reads as the
    studio wall that most portraits already have.
    """
    import cv2

    canvas_w, canvas_h = plan['canvas']
    offset_x, offset_y = plan['offset']
    source = np.asarray(image.convert('RGB'), np.uint8)
    source_h, source_w = source.shape[:2]
    top, bottom = offset_y, canvas_h - offset_y - source_h
    left, right = offset_x, canvas_w - offset_x - source_w
    band = int(plan.get('band') or 12)

    canvas = cv2.copyMakeBorder(source, top, bottom, left, right, cv2.BORDER_REPLICATE)
    if top or bottom or left or right:
        inside = np.zeros((canvas_h, canvas_w), bool)
        inside[offset_y:offset_y + source_h, offset_x:offset_x + source_w] = True
        distance = cv2.distanceTransform((~inside).astype(np.uint8), cv2.DIST_L2, 3)
        reach = max(1., float(distance.max()))
        # Weight grows with the distance from the photo: near the seam the
        # replicated pixels keep their structure, far away it is one flat tone.
        weight = np.clip(distance / reach, 0., 1.) ** .6 * .92
        blurred = cv2.GaussianBlur(canvas, (0, 0), max(3., band * .8))
        ring = np.concatenate([source[:4].reshape(-1, 3), source[-4:].reshape(-1, 3),
                               source[:, :4].reshape(-1, 3), source[:, -4:].reshape(-1, 3)])
        backdrop = np.median(ring, axis=0).astype(np.float32)
        canvas = canvas.astype(np.float32) * (1. - weight[..., None]) + \
            backdrop * weight[..., None]

    if body_prior_fill:
        rng = np.random.default_rng(int(seed))
        corridor_map = corridor(plan['canvas'], plan['offset'], plan['body_box'])
        lower = np.asarray(colors['lower'], np.float32)
        rows = np.arange(canvas_h)[:, None, None]
        start = offset_y + plan['body_box'][3]
        depth = np.clip((rows - start) / max(1., canvas_h - 1 - start), 0., 1.)
        tint = lower * (1. - 0.12 * depth)
        noise = rng.normal(0., 3., (canvas_h, canvas_w, 1)).astype(np.float32)
        body = np.clip(tint + noise, 0., 255.)
        soft = cv2.GaussianBlur(corridor_map.astype(np.float32), (0, 0), band * 0.7)[..., None]
        canvas = canvas * (1. - soft) + body * soft

    canvas = np.clip(canvas, 0., 255.).round().astype(np.uint8)
    # The source pixels go back exactly: this is the byte-identical promise.
    canvas[offset_y:offset_y + source_h, offset_x:offset_x + source_w] = source
    mask = np.clip(mask, 0., 1.).astype(np.float32)
    return canvas, mask


def compose_extend(canvas, mask, generated):
    """Blend the generated canvas through the mask, copying the rest.

    ``np.where`` is deliberate: an arithmetic blend would leave a rounding
    residue outside the mask and break the identity check.
    """
    canvas = np.asarray(canvas, np.float32)
    generated = np.asarray(generated, np.float32)
    if generated.shape[:2] != canvas.shape[:2]:
        raise ValueError('生成结果与画布尺寸不一致')
    weight = np.clip(mask, 0., 1.)[..., None]
    mixed = canvas * (1. - weight) + generated * weight
    out = np.where(weight > 0., mixed, canvas)
    return np.clip(out, 0., 255.).round().astype(np.uint8)


def leg_ratio(labels, plan):
    """Share of the new area covered by legs, trousers or shoes."""
    offset_x, offset_y = plan['offset']
    canvas_w, canvas_h = plan['canvas']
    source_w, source_h = plan['source']
    zone = np.ones((canvas_h, canvas_w), bool)
    zone[offset_y:offset_y + source_h, offset_x:offset_x + source_w] = False
    if not zone.any():
        return 0.
    return float(labels_of(LOWER_LABELS, labels)[zone].mean())


def check_identity(source, result, plan, labels, *, tolerance=2, minimum=0.):
    """Compare the protected pixels. Any change is a hard failure.

    ``source`` is the working source (the same array that was pasted on the
    canvas), so a resized input still compares like for like.
    """
    offset_x, offset_y = plan['offset']
    source_h, source_w = source.shape[:2]
    crop = result[offset_y:offset_y + source_h, offset_x:offset_x + source_w]
    if crop.shape[:2] != source.shape[:2]:
        raise ValueError('结果与源图尺寸不一致，无法校验身份')
    diff = (np.abs(crop.astype(np.int16) - source.astype(np.int16))
            .max(axis=2) > tolerance)

    face = labels_of(FACE_LABELS, labels)
    if _any(face):
        face = _erode(face, 3)
    protected = labels_of(PROTECT_LABELS, labels)
    report = {'face_diff_ratio': float(diff[face].mean()) if _any(face) else 0.,
              'protected_diff_ratio': float(diff[protected].mean()) if _any(protected) else 0.}
    for key, value in report.items():
        if value > minimum:
            raise ValueError(f'扩图改动了{"人脸" if key == "face_diff_ratio" else "人物"}像素，已中止')
    return report


def _any(zone):
    return bool(zone is not None and np.asarray(zone).any())


def _erode(zone, pixels):
    import cv2
    kernel = np.ones((3, 3), np.uint8)
    return cv2.erode(zone.astype(np.uint8), kernel, iterations=int(pixels)).astype(bool)


def fit_output(rgb, limit=1536):
    """Cap the saved long edge. Native canvas memory is not capped here."""
    from PIL import Image
    height, width = rgb.shape[:2]
    longest = max(width, height)
    if longest <= limit:
        return rgb, 1.
    scale = limit / longest
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return np.asarray(Image.fromarray(rgb).resize(size, Image.LANCZOS)), scale


def preview_size(plan, long_edge=768):
    """Working diffusion size: long edge capped, both sides multiple of 8."""
    canvas_w, canvas_h = plan['canvas']
    scale = min(1., long_edge / max(canvas_w, canvas_h))
    width = max(256, int(round(canvas_w * scale)) // 8 * 8)
    height = max(256, int(round(canvas_h * scale)) // 8 * 8)
    return width, height


def scale_plan(plan, size):
    """The same plan expressed at the working diffusion size."""
    canvas_w, canvas_h = plan['canvas']
    width, height = size
    scale_x, scale_y = width / canvas_w, height / canvas_h
    box = plan['body_box']
    return {**plan,
            'canvas': [int(width), int(height)],
            'offset': [int(round(plan['offset'][0] * scale_x)),
                       int(round(plan['offset'][1] * scale_y))],
            'source': [int(round(plan['source'][0] * scale_x)),
                       int(round(plan['source'][1] * scale_y))],
            'body_box': [int(round(box[0] * scale_x)), int(round(box[1] * scale_y)),
                         int(round(box[2] * scale_x)), int(round(box[3] * scale_y))],
            'band': max(4, int(round(plan['band'] * min(scale_x, scale_y))))}
