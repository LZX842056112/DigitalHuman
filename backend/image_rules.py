"""Basic upload checks for person photos and garment pictures.

The API already rejects bad file names and oversized bodies. This module adds
what the extension check cannot see: the real pixel format, the real pixel
size, and the shape of the picture. Every failure carries a Chinese message
that names the exact limit, so the UI can show it without translating codes.

Nothing here rewrites the file. The source copy must stay byte-identical for
the SHA-256 checks in ``photo_profiles``. Orientation is therefore read, not
applied; ``open_normalized`` applies it at use time.
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageOps

# A 40 MP picture at 8192 px long edge is already far beyond what the renderer
# and the editor can use. Reject it instead of silently burning memory.
MAX_PIXELS = 40_000_000
Image.MAX_IMAGE_PIXELS = MAX_PIXELS

# Person photos accept JPG and PNG only. Garment pictures accept WEBP too,
# because product shots are often saved that way.
RULES = {
    'person': {
        'formats': ('JPEG', 'PNG'),
        'suffixes': ('.jpg', '.jpeg', '.png'),
        'min_short': 320,
        'max_edge': 8192,
        'aspect': (0.33, 3.0),
        'format_error': '只支持 JPG 或 PNG 照片',
        'short_error': '照片短边至少 {value} 像素，请换一张更清晰的照片',
        'edge_error': '照片长边不能超过 {value} 像素',
        'aspect_error': '照片长宽比异常，请上传单人照片',
    },
    'garment': {
        'formats': ('JPEG', 'PNG', 'WEBP'),
        'suffixes': ('.jpg', '.jpeg', '.png', '.webp'),
        'min_short': 256,
        'max_edge': 4096,
        'aspect': (0.5, 2.0),
        'format_error': '服装图只支持 JPG、PNG 或 WEBP',
        'short_error': '服装图短边至少 {value} 像素',
        'edge_error': '服装图长边不能超过 {value} 像素',
        'aspect_error': '服装图长宽比异常，请上传单件衣服的正面图',
    },
}

# The API passes role='photo' today. Keep the old name working.
ALIASES = {'photo': 'person'}

BROKEN = '图片无法读取，可能已损坏'
MISMATCH = '文件内容与扩展名不一致，请重新选择'
TOO_MANY_PIXELS = '图片像素过多，请压缩后再上传'


class ImageRuleError(ValueError):
    """One rejected upload. ``status`` is the HTTP code the API should use."""

    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = int(status)


def _rule(role):
    name = ALIASES.get(role, role)
    if name not in RULES:
        raise ValueError(f'未知的图片角色：{role}')
    return RULES[name]


def orientation(image):
    """EXIF orientation tag, 1 when absent. Read only, never written back."""
    try:
        value = image.getexif().get(0x0112, 1)
    except (AttributeError, OSError):
        return 1
    try:
        return int(value) if int(value) in range(1, 9) else 1
    except (TypeError, ValueError):
        return 1


def read_size(path):
    """(width, height) after EXIF orientation, without decoding the pixels."""
    with Image.open(path) as image:
        width, height = image.size
        if orientation(image) >= 5:
            width, height = height, width
    return width, height


def verify_image(path, role='person'):
    """Validate one upload. Return a small report or raise ImageRuleError.

    ``path`` is the temporary copy that was just written by the API. The
    caller deletes it after the request, so reading it twice is safe.
    """
    path = Path(path)
    rule = _rule(role)
    suffix = path.suffix.lower()
    if suffix not in rule['suffixes']:
        raise ImageRuleError(rule['format_error'])

    # PngImageFile.verify() refuses to run once the pixels were touched, so the
    # format and the integrity check must come before any size or EXIF read.
    try:
        with Image.open(path) as image:
            image_format = (image.format or '').upper()
            image.verify()
    except Image.DecompressionBombError as error:
        raise ImageRuleError(f'{TOO_MANY_PIXELS}（当前上限 {MAX_PIXELS // 1_000_000} 百万像素）',
                             status=413) from error
    except (OSError, SyntaxError, ValueError, RuntimeError) as error:
        raise ImageRuleError(BROKEN) from error

    if image_format not in rule['formats']:
        raise ImageRuleError(rule['format_error'])
    # A .png that is really a JPEG would pass the extension check above but
    # break every later reader that trusts the suffix.
    if image_format == 'JPEG' and suffix == '.png':
        raise ImageRuleError(MISMATCH)
    if image_format == 'PNG' and suffix in {'.jpg', '.jpeg'}:
        raise ImageRuleError(MISMATCH)

    # verify() consumed the file object. Re-open to read the header again.
    try:
        with Image.open(path) as image:
            width, height = image.size
            tag = orientation(image)
    except (OSError, SyntaxError, ValueError) as error:
        raise ImageRuleError(BROKEN) from error

    if width * height > MAX_PIXELS:
        raise ImageRuleError(f'{TOO_MANY_PIXELS}（当前上限 {MAX_PIXELS // 1_000_000} 百万像素）',
                             status=413)
    if tag >= 5:
        width, height = height, width

    long_edge, short_edge = max(width, height), min(width, height)
    if long_edge > rule['max_edge']:
        raise ImageRuleError(rule['edge_error'].format(value=rule['max_edge']))
    if short_edge < rule['min_short']:
        raise ImageRuleError(rule['short_error'].format(value=rule['min_short']))

    low, high = rule['aspect']
    ratio = width / height if height else 0.
    if not low <= ratio <= high:
        raise ImageRuleError(rule['aspect_error'])

    return {'format': image_format, 'width': width, 'height': height,
            'orientation': tag, 'role': ALIASES.get(role, role)}


def open_normalized(path):
    """Open for processing: orientation applied, RGB, never the raw file."""
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert('RGB')
