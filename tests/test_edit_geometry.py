"""Outpainting geometry: canvas, mask, fill and the identity invariant."""
import numpy as np
import pytest
from PIL import Image

from backend import edit_geometry as geo

SKIN = (205, 180, 158)
CLOTH = (46, 50, 62)


def labels_array(width=400, height=600, box=(120, 20, 280, 590), head=90):
    labels = np.zeros((height, width), np.uint8)
    x0, y0, x1, y1 = box
    labels[y0:y1, x0:x1] = 4
    labels[y0:y0 + head, x0:x1] = 11
    return labels


def scene_image(labels):
    """Background in skin tone, the body block in dark clothing."""
    rgb = np.zeros(labels.shape + (3,), np.uint8)
    rgb[:, :] = SKIN
    rgb[labels > 0] = CLOTH
    return Image.fromarray(rgb)


def test_person_bbox_is_tight_around_the_body():
    labels = labels_array()
    assert geo.person_bbox(labels) == (120, 20, 280, 590)


def test_person_bbox_rejects_an_empty_picture():
    with pytest.raises(ValueError, match='没有识别到完整的人物'):
        geo.person_bbox(np.zeros((300, 300), np.uint8))


def test_head_height_reads_the_head_labels():
    labels = labels_array(head=90)
    assert geo.head_height(labels, geo.person_bbox(labels)) == 90


def test_head_height_falls_back_when_no_head_label_exists():
    labels = labels_array(head=0)
    head = geo.head_height(labels, geo.person_bbox(labels))
    assert head == pytest.approx((590 - 20) * 0.24)


def test_body_prior_prefers_the_visible_lower_body():
    labels = labels_array()
    labels[400:590, 120:280] = 6
    colors = geo.body_prior(labels, scene_image(labels))
    assert np.allclose(colors['lower'], CLOTH)


def test_body_prior_falls_back_to_the_top_when_no_lower_body_exists():
    labels = labels_array()
    colors = geo.body_prior(labels, scene_image(labels))
    assert np.allclose(colors['lower'], CLOTH)


def test_plan_canvas_extends_a_headshot():
    labels = labels_array()
    box = geo.person_bbox(labels)
    plan = geo.plan_canvas(400, 600, box, geo.head_height(labels, box), aspect=0.75)
    width, height = plan['canvas']
    assert plan['skipped'] is None
    assert height > 600
    assert width / height == pytest.approx(0.75, abs=0.01)
    assert plan['offset'][1] == round(plan['head_pad'])
    assert height == round(600 + plan['head_pad'] + plan['gain'])
    assert 8 <= plan['band'] <= 24


def test_plan_canvas_skips_a_photo_that_already_reaches_the_feet():
    labels = labels_array(height=600, box=(120, 20, 280, 600), head=80)
    box = geo.person_bbox(labels)
    plan = geo.plan_canvas(400, 600, box, geo.head_height(labels, box), aspect=0.75)
    assert plan['skipped'] == geo.SKIPPED
    assert plan['canvas'] == [400, 600]


def test_plan_canvas_adds_headroom_only_when_the_crown_is_visible():
    labels = labels_array(box=(120, 80, 280, 600), head=80)
    box = geo.person_bbox(labels)
    plan = geo.plan_canvas(400, 620, box, geo.head_height(labels, box), aspect=0.75)
    assert plan['head_pad'] > 0
    assert plan['offset'][1] == round(plan['head_pad'])


def test_plan_canvas_never_adds_headroom_above_a_cropped_crown():
    labels = labels_array(box=(120, 0, 280, 590), head=80)
    box = geo.person_bbox(labels)
    plan = geo.plan_canvas(400, 600, box, geo.head_height(labels, box), aspect=0.75)
    assert plan['head_pad'] == 0
    assert plan['offset'][1] == 0


def test_plan_canvas_keeps_a_wide_source_at_full_width():
    labels = labels_array(width=900, height=400, box=(300, 20, 600, 400), head=70)
    box = geo.person_bbox(labels)
    plan = geo.plan_canvas(900, 400, box, geo.head_height(labels, box), aspect=0.75)
    assert plan['canvas'][0] >= 900
    assert plan['gain'] > 0


def test_plan_canvas_rejects_a_landscape_aspect():
    with pytest.raises(ValueError, match='不支持的画幅比例'):
        geo.plan_canvas(400, 600, (120, 20, 280, 590), 90, aspect=1.5)


def test_plan_canvas_honours_the_maximum_gain():
    labels = labels_array(head=30)
    box = geo.person_bbox(labels)
    plan = geo.plan_canvas(400, 600, box, geo.head_height(labels, box),
                           aspect=0.75, max_gain=0.5)
    assert plan['gain'] <= 0.5 * 600 + 1


def test_plan_canvas_flags_a_close_up_it_cannot_fully_extend():
    labels = labels_array(box=(60, 10, 340, 600), head=380)
    box = geo.person_bbox(labels)
    plan = geo.plan_canvas(400, 600, box, geo.head_height(labels, box), aspect=0.75)
    assert plan['skipped'] is None
    assert plan['predicted_gain'] > plan['gain']
    assert plan['capped'] is True


def test_plan_canvas_does_not_flag_a_normal_half_body():
    labels = labels_array(box=(120, 20, 300, 430), head=110)
    box = geo.person_bbox(labels)
    plan = geo.plan_canvas(400, 500, box, geo.head_height(labels, box), aspect=0.75)
    assert plan['capped'] is False
    assert plan['predicted_gain'] == pytest.approx(plan['gain'])


def build(seed=0):
    labels = labels_array()
    box = geo.person_bbox(labels)
    head = geo.head_height(labels, box)
    plan = geo.plan_canvas(400, 600, box, head, aspect=0.75)
    person = np.isin(labels, list(geo.PERSON_LABELS))
    protected = geo.protect_mask(plan['canvas'], plan['offset'], labels)
    mask, new = geo.build_mask(plan['canvas'], plan['offset'], box, plan['source'],
                               protected=protected)
    image = scene_image(labels)
    colors = geo.body_prior(labels, image)
    canvas, mask = geo.fill_canvas(image, plan, mask, colors, seed=seed)
    return labels, box, plan, person, mask, new, canvas


def test_mask_is_exactly_the_person_corridor():
    _, box, plan, _, mask, _, _ = build()
    corridor = geo.corridor(plan['canvas'], plan['offset'], box)
    assert np.array_equal(mask > 0, corridor)


def test_corridor_starts_at_the_hem_so_the_body_never_enters_the_mask():
    _, box, plan, _, mask, _, _ = build()
    offset_y = plan['offset'][1]
    assert mask[:offset_y + box[3]].max() == 0


def test_mask_cuts_the_protected_zone():
    labels = labels_array()
    box = geo.person_bbox(labels)
    plan = geo.plan_canvas(400, 600, box, geo.head_height(labels, box), aspect=0.75)
    protected = geo.protect_mask(plan['canvas'], plan['offset'], labels)
    mask, _ = geo.build_mask(plan['canvas'], plan['offset'], box, plan['source'],
                             protected=protected)
    assert mask[protected].max() < 0.05


def test_fill_canvas_copies_the_source_pixels_exactly():
    labels, _, plan, _, _, _, canvas = build()
    offset_x, offset_y = plan['offset']
    source = np.asarray(scene_image(labels))
    crop = canvas[offset_y:offset_y + source.shape[0], offset_x:offset_x + source.shape[1]]
    assert np.array_equal(crop, source)


def test_fill_canvas_paints_the_corridor_with_the_outfit_colour():
    labels, box, plan, _, _, _, canvas = build()
    corridor = geo.corridor(plan['canvas'], plan['offset'], box)
    offset_x, offset_y = plan['offset']
    new = np.ones(corridor.shape, bool)
    new[offset_y:offset_y + 600, offset_x:offset_x + 400] = False
    pixels = canvas[corridor & new].astype(np.float32)
    lower = geo.body_prior(labels, scene_image(labels))['lower']
    assert np.abs(pixels.mean(axis=0) - lower).max() < 45


def test_fill_canvas_does_not_mirror_the_body_into_the_corridor():
    _, box, plan, _, _, _, canvas = build()
    corridor = geo.corridor(plan['canvas'], plan['offset'], box)
    offset_x, offset_y = plan['offset']
    new = np.ones(corridor.shape, bool)
    new[offset_y:offset_y + 600, offset_x:offset_x + 400] = False
    skin = np.asarray(SKIN, np.float32)
    pixels = canvas[corridor & new].astype(np.float32)
    assert np.abs(pixels.mean(axis=0) - skin).max() > 90


def test_fill_canvas_fades_the_padding_to_one_backdrop():
    labels = labels_array()
    box = geo.person_bbox(labels)
    plan = geo.plan_canvas(400, 600, box, geo.head_height(labels, box), aspect=.75)
    image = scene_image(labels)
    mask = np.zeros((plan['canvas'][1], plan['canvas'][0]), np.float32)
    canvas, _ = geo.fill_canvas(image, plan, mask, geo.body_prior(labels, image),
                                body_prior_fill=False)
    height = canvas.shape[0]
    corner = canvas[height - 40:height - 5, 5:40].astype(np.float32)
    # One flat tone, not a copy of the person: low spread, close to the backdrop.
    assert float(corner.std(axis=(0, 1)).max()) < 6.
    assert abs(float(corner.mean()) - float(np.mean(SKIN))) < 12.


def test_compose_extend_keeps_every_pixel_outside_the_mask():
    _, _, plan, _, mask, _, canvas = build()
    rng = np.random.default_rng(7)
    generated = rng.integers(0, 255, canvas.shape, dtype=np.uint8)
    out = geo.compose_extend(canvas, mask, generated)
    assert np.array_equal(out[mask <= 0], canvas[mask <= 0])
    assert not np.array_equal(out[mask >= 1], canvas[mask >= 1])


def test_compose_extend_rejects_a_size_mismatch():
    with pytest.raises(ValueError, match='尺寸不一致'):
        geo.compose_extend(np.zeros((10, 10, 3), np.uint8), np.zeros((10, 10), np.float32),
                           np.zeros((9, 9, 3), np.uint8))


def test_check_identity_passes_on_the_untouched_canvas():
    labels, _, plan, _, _, _, canvas = build()
    offset_x, offset_y = plan['offset']
    source = canvas[offset_y:offset_y + 600, offset_x:offset_x + 400]
    report = geo.check_identity(source, canvas, plan, labels)
    assert report == {'face_diff_ratio': 0., 'protected_diff_ratio': 0.}


def test_check_identity_rejects_a_changed_face():
    labels, _, plan, _, _, _, canvas = build()
    offset_x, offset_y = plan['offset']
    source = canvas[offset_y:offset_y + 600, offset_x:offset_x + 400].copy()
    tampered = canvas.copy()
    tampered[offset_y + 40:offset_y + 60, offset_x + 150:offset_x + 250] = 0
    with pytest.raises(ValueError, match='人脸'):
        geo.check_identity(source, tampered, plan, labels)


def test_leg_ratio_counts_the_new_area_only():
    labels, _, plan, _, _, _, _ = build()
    offset_x, offset_y = plan['offset']
    canvas_w, canvas_h = plan['canvas']
    canvas_labels = np.zeros((canvas_h, canvas_w), np.uint8)
    canvas_labels[offset_y:offset_y + 600, offset_x:offset_x + 400] = labels
    assert geo.leg_ratio(canvas_labels, plan) == pytest.approx(0.)
    canvas_labels[:, :] = 6
    canvas_labels[offset_y:offset_y + 600, offset_x:offset_x + 400] = labels
    assert geo.leg_ratio(canvas_labels, plan) == pytest.approx(1.)


def test_fit_output_caps_the_long_edge():
    rgb = np.zeros((2400, 1600, 3), np.uint8)
    out, scale = geo.fit_output(rgb, limit=1200)
    assert max(out.shape[:2]) == 1200
    assert scale == pytest.approx(0.5)


def test_fit_output_keeps_a_small_picture_untouched():
    rgb = np.zeros((900, 600, 3), np.uint8)
    out, scale = geo.fit_output(rgb, limit=1200)
    assert out.shape == rgb.shape and scale == 1.


def test_preview_size_rounds_to_eight_and_caps_the_long_edge():
    width, height = geo.preview_size({'canvas': [1620, 2160]}, long_edge=768)
    assert width % 8 == 0 and height % 8 == 0
    assert max(width, height) <= 768
