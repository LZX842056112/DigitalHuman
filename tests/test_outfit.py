"""Outfit edits must change the garment and nothing else."""
import numpy as np
import pytest
from PIL import Image

from backend import outfit


def _scene(tmp_path, *, gap=False):
    """A stand-in photo: skin on top, black garment below, wall on the right.

    ``gap`` leaves the rows just under the skin unlabelled while keeping their
    garment colour, which is what the mask has to grow back into.
    """
    rgb = np.zeros((96, 96, 3), np.uint8)
    rgb[:, :] = (120, 110, 100)
    rgb[:40, :64] = (232, 190, 168)
    rgb[40:, :64] = (30, 30, 32)
    labels = np.zeros((96, 96), np.uint8)
    labels[:40, :64] = 11
    labels[(43 if gap else 40):, :64] = 4
    path = tmp_path / 'photo.png'
    Image.fromarray(rgb).save(path)
    return path, rgb, labels


def test_mask_covers_the_garment_and_never_the_skin(tmp_path):
    path, rgb, labels = _scene(tmp_path)
    mask = outfit.garment_mask(labels, Image.open(path))
    assert mask[70, 30] > .9
    assert mask[30, 30] < .05
    assert mask[80, 80] < .05


def test_mask_grows_back_into_garment_coloured_rows(tmp_path):
    path, rgb, labels = _scene(tmp_path, gap=True)
    mask = outfit.garment_mask(labels, Image.open(path))
    assert mask[44, 30] > .8, '标签外的衣服像素应被收进掩膜，否则会留下深色描边'
    assert mask[41, 30] > .3
    assert mask[38, 30] < .2, '颈部皮肤不能被收进掩膜'


def test_recolor_leaves_every_pixel_outside_the_mask(tmp_path):
    path, rgb, labels = _scene(tmp_path)
    image = Image.open(path)
    mask = outfit.garment_mask(labels, image)
    out = outfit.recolor(image, mask, outfit.COLORS['navy'])
    diff = np.abs(out.astype(np.int16) - rgb.astype(np.int16)).max(axis=2)
    assert diff[30, 30] == 0
    assert diff[80, 80] == 0
    assert diff[70, 30] > 20
    assert int(out[70, 30, 2]) > int(out[70, 30, 0]) + 10


def test_navy_keeps_its_blue_chroma(tmp_path):
    """A clip of the a/b channels at zero would turn navy into charcoal."""
    path, _, labels = _scene(tmp_path)
    image = Image.open(path)
    mask = outfit.garment_mask(labels, image)
    out = outfit.recolor(image, mask, outfit.COLORS['navy'])
    lab = outfit._rgb_to_lab(np.asarray(out, np.float32) / 255.)
    assert lab[70, 30, 2] < -8
    assert lab[70, 30, 0] > 15


def test_white_leaves_the_garment_light(tmp_path):
    path, _, labels = _scene(tmp_path)
    image = Image.open(path)
    mask = outfit.garment_mask(labels, image)
    out = outfit.recolor(image, mask, outfit.COLORS['white'])
    lab = outfit._rgb_to_lab(np.asarray(out, np.float32) / 255.)
    assert lab[70, 30, 0] > 70


def test_strength_blends_back_towards_the_original(tmp_path):
    path, rgb, labels = _scene(tmp_path)
    image = Image.open(path)
    mask = outfit.garment_mask(labels, image)
    half = outfit.recolor(image, mask, outfit.COLORS['white'], strength=.5)
    full = outfit.recolor(image, mask, outfit.COLORS['white'])
    assert rgb[70, 30].mean() < half[70, 30].mean() < full[70, 30].mean()


def test_unknown_colour_is_rejected(tmp_path):
    path, _, labels = _scene(tmp_path)
    image = Image.open(path)
    mask = outfit.garment_mask(labels, image)
    with pytest.raises(ValueError, match='RRGGBB'):
        outfit.recolor(image, mask, 'navy')


def test_labels_and_prompts_follow_the_choice():
    assert outfit.label_for({'mode': 'recolor', 'color': '#1e2a44'}) == '换装·藏青'
    assert outfit.label_for({'mode': 'generate', 'style': 'shirt'}) == '换装·白衬衫'
    assert outfit.label_for({'mode': 'recolor', 'color': '#123456'}) == '换装·自定义'
    assert 'button-up' in outfit.prompt_for({'style': 'shirt'})
    assert outfit.prompt_for({'prompt': 'a red tuxedo'}).startswith('a red tuxedo')


def test_generate_without_a_model_says_so(tmp_path, monkeypatch):
    path, _, labels = _scene(tmp_path)
    image = Image.open(path)
    mask = outfit.garment_mask(labels, image)
    monkeypatch.setattr(outfit, 'ROOT', tmp_path)
    with pytest.raises(FileNotFoundError, match='换装模型'):
        outfit.inpaint(image, mask, 'a white shirt')


def test_garment_crop_drops_the_product_background(tmp_path):
    canvas = np.full((200, 200, 3), 255, np.uint8)
    canvas[40:160, 50:150] = (30, 60, 120)
    path = tmp_path / 'garment.png'
    Image.fromarray(canvas).save(path)
    crop = outfit._garment_crop(Image.open(path))
    assert crop.width < 130 and crop.height < 150
    assert crop.width >= 100 and crop.height >= 120


def test_garment_fill_ignores_the_white_background(tmp_path):
    canvas = np.full((120, 120, 3), 255, np.uint8)
    canvas[20:100, 30:90] = (30, 60, 120)
    path = tmp_path / 'garment.png'
    Image.fromarray(canvas).save(path)
    colour = outfit.garment_fill(Image.open(path))
    red, green, blue = (int(colour[i:i + 2], 16) for i in (1, 3, 5))
    assert blue > red + 40 and blue > green + 20


def test_wear_needs_the_reference_model(tmp_path, monkeypatch):
    path, _, labels = _scene(tmp_path)
    image = Image.open(path)
    mask = outfit.garment_mask(labels, image)
    garment = tmp_path / 'garment.png'
    Image.fromarray(np.full((64, 64, 3), 90, np.uint8)).save(garment)
    monkeypatch.setattr(outfit, 'ROOT', tmp_path)
    with pytest.raises(FileNotFoundError):
        outfit.wear(image, mask, garment)


def test_wear_label_is_upload():
    assert outfit.label_for({'mode': 'wear'}) == '换装·上传'
