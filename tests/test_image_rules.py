"""Upload checks for person photos and garment pictures."""
import pytest
from PIL import Image

from backend import image_rules
from backend.image_rules import ImageRuleError, open_normalized, read_size, verify_image


def save(path, size, image_format=None, color=(120, 130, 140), exif=None):
    image_format = image_format or ('PNG' if path.suffix == '.png' else 'JPEG')
    tags = None
    if exif:
        tags = Image.Exif()
        for key, value in exif.items():
            tags[key] = value
    options = {'exif': tags} if tags is not None else {}
    Image.new('RGB', size, color).save(path, format=image_format, **options)
    return path


def message(call):
    with pytest.raises(ImageRuleError) as caught:
        call()
    return str(caught.value)


def test_accepts_png_person_photo(tmp_path):
    path = save(tmp_path/'photo.png', (768, 1024))
    report = verify_image(path, 'person')
    assert report['format'] == 'PNG'
    assert (report['width'], report['height']) == (768, 1024)
    assert report['role'] == 'person'


def test_accepts_legacy_photo_role(tmp_path):
    path = save(tmp_path/'photo.jpg', (600, 800))
    assert verify_image(path, 'photo')['role'] == 'person'


def test_rejects_jpeg_content_behind_png_suffix(tmp_path):
    path = save(tmp_path/'photo.png', (600, 800), image_format='JPEG')
    assert '内容与扩展名不一致' in message(lambda: verify_image(path, 'person'))


def test_rejects_unreadable_bytes(tmp_path):
    path = tmp_path/'photo.png'
    path.write_bytes(b'not an image at all')
    assert image_rules.BROKEN in message(lambda: verify_image(path, 'person'))


def test_rejects_unsupported_suffix(tmp_path):
    path = save(tmp_path/'photo.gif', (600, 800), image_format='GIF')
    assert '只支持 JPG 或 PNG' in message(lambda: verify_image(path, 'person'))


def test_rejects_short_edge(tmp_path):
    path = save(tmp_path/'photo.png', (3000, 200))
    assert '短边至少 320' in message(lambda: verify_image(path, 'person'))


def test_rejects_long_edge(tmp_path):
    path = save(tmp_path/'photo.png', (9000, 400))
    assert '长边不能超过 8192' in message(lambda: verify_image(path, 'person'))


def test_rejects_extreme_aspect(tmp_path):
    path = save(tmp_path/'photo.png', (1600, 500))
    assert '长宽比异常' in message(lambda: verify_image(path, 'person'))


def test_pixel_bomb_uses_413(tmp_path, monkeypatch):
    monkeypatch.setattr(image_rules, 'MAX_PIXELS', 1000)
    path = save(tmp_path/'photo.png', (200, 200))
    with pytest.raises(ImageRuleError) as caught:
        verify_image(path, 'person')
    assert caught.value.status == 413
    assert '像素过多' in str(caught.value)


def test_garment_accepts_webp(tmp_path):
    path = save(tmp_path/'shirt.webp', (600, 800), image_format='WEBP')
    assert verify_image(path, 'garment')['format'] == 'WEBP'


def test_garment_rejects_person_sized_aspect(tmp_path):
    path = save(tmp_path/'shirt.png', (1200, 500))
    assert '单件衣服的正面图' in message(lambda: verify_image(path, 'garment'))


def test_garment_uses_its_own_short_edge(tmp_path):
    path = save(tmp_path/'shirt.png', (300, 200))
    assert '服装图短边至少 256' in message(lambda: verify_image(path, 'garment'))


def test_read_size_applies_orientation(tmp_path):
    path = save(tmp_path/'photo.jpg', (1000, 340), exif={0x0112: 6})
    assert read_size(path) == (340, 1000)


def test_orientation_defaults_to_one(tmp_path):
    path = save(tmp_path/'photo.jpg', (600, 800))
    assert image_rules.orientation(Image.open(path)) == 1


def test_open_normalized_returns_rgb_without_touching_source(tmp_path):
    path = save(tmp_path/'photo.jpg', (1000, 340), exif={0x0112: 6})
    before = path.read_bytes()
    image = open_normalized(path)
    assert image.mode == 'RGB'
    assert image.size == (340, 1000)
    assert path.read_bytes() == before


def test_unknown_role_is_a_programming_error():
    with pytest.raises(ValueError, match='未知的图片角色'):
        image_rules._rule('nobody')
