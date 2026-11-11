import numpy as np
import pytest

from fpyolo.svgparse import parse_tile
from svgfixtures import tile_box, tile_with, write_svg


def test_semicircle_bbox_uses_arc_extremum(tmp_path):  # endpoints alone would give height 0
    t = tile_with(tmp_path, '<path d="M 40,50 A 10,10 0 0 0 60,50" semantic-id="3" instance-id="0"/>')
    assert t.primitives[0].bbox == pytest.approx((40, 50, 60, 60), abs=1e-3)
    t = tile_with(tmp_path, '<path d="M 40,50 A 10,10 0 0 1 60,50" semantic-id="3" instance-id="0"/>')
    assert t.primitives[0].bbox == pytest.approx((40, 40, 60, 50), abs=1e-3)


def test_rotated_ellipse_bbox(tmp_path):
    t = tile_with(tmp_path, '<ellipse cx="50" cy="50" rx="20" ry="10" transform="rotate(45 50 50)" semantic-id="9" instance-id="1"/>')
    r = 250 ** 0.5
    assert t.primitives[0].bbox == pytest.approx((50 - r, 50 - r, 50 + r, 50 + r), abs=1e-3)
    assert t.primitives[0].closed


def test_rotated_ellipse_polyline_is_rotated(tmp_path):
    # Real data (train-01/1099-0032, sid 23): reify cannot fold a rotation into an
    # <ellipse>, so the residual transform must still be applied to the samples.
    t = tile_with(tmp_path, '<ellipse cx="50" cy="50" rx="20" ry="10" transform="rotate(30 50 50)" semantic-id="9" instance-id="1"/>')
    p = t.primitives[0]
    d = p.polyline - 50
    c, s = np.cos(np.radians(30)), np.sin(np.radians(30))
    u, v = d[:, 0] * c + d[:, 1] * s, -d[:, 0] * s + d[:, 1] * c  # back to the ellipse frame
    assert (u / 20) ** 2 + (v / 10) ** 2 == pytest.approx(1.0, abs=1e-6)
    assert (*p.polyline.min(axis=0), *p.polyline.max(axis=0)) == pytest.approx(p.bbox, abs=1e-2)


def test_scaled_circle_polyline_stays_dense(tmp_path):
    t = tile_with(tmp_path, '<circle cx="10" cy="10" r="5" transform="scale(4,2)" semantic-id="9" instance-id="1"/>')
    p = t.primitives[0]
    assert p.bbox == pytest.approx((20, 10, 60, 30), abs=1e-3)
    assert (*p.polyline.min(axis=0), *p.polyline.max(axis=0)) == pytest.approx(p.bbox, abs=1e-2)
    assert np.hypot(*np.diff(p.polyline, axis=0).T).max() <= 0.05 + 1e-9


def test_group_transform_applied(tmp_path):
    t = tile_with(tmp_path, '<g transform="translate(10,0)"><path d="M 0,0 L 5,5" semantic-id="3" instance-id="0"/></g>')
    assert t.primitives[0].bbox == pytest.approx((10, 0, 15, 5))
    assert not t.primitives[0].closed


def test_polyline_dense_and_on_curve(tmp_path):
    t = tile_with(tmp_path, '<path d="M 40,50 A 10,10 0 0 0 60,50" semantic-id="3" instance-id="0"/>')
    poly = t.primitives[0].polyline
    assert poly.ndim == 2 and poly.shape[1] == 2 and poly.dtype == np.float64
    steps = np.hypot(*np.diff(poly, axis=0).T)
    assert steps.max() <= 0.05 + 1e-9
    dist = np.hypot(poly[:, 0] - 50, poly[:, 1] - 50)
    assert dist == pytest.approx(10.0, abs=1e-6)


def test_polyline_dense_on_long_line(tmp_path):
    t = tile_with(tmp_path, '<path d="M 0,0 L 30,40" semantic-id="3" instance-id="0"/>')
    poly = t.primitives[0].polyline
    assert poly[0] == pytest.approx((0, 0)) and poly[-1] == pytest.approx((30, 40))
    assert np.hypot(*np.diff(poly, axis=0).T).max() <= 0.05 + 1e-9


def test_unlabeled_ignored_malformed_counted(tmp_path):
    body = (
        '<path d="M 0,0 L 5,5"/>'
        '<path d="M 0,0 L 6,6" semantic-id="3"/>'
        '<path d="M 0,0 L 7,7" semantic-id="3" instance-id="0"/>'
    )
    t = tile_with(tmp_path, body)
    assert len(t.primitives) == 1 and t.skipped == 0
    t = tile_with(tmp_path, body + '<path d="M 0,0 L 8,8" semantic-id="abc" instance-id="0"/>')
    assert len(t.primitives) == 1 and t.skipped == 1


def test_viewbox_parsed(tmp_path):
    t = tile_with(tmp_path, "")
    assert t.viewbox == (0, 0, 100, 100)
    assert (t.archive, t.stem) == ("train-00", "0000-0000")


def test_circle_bbox(tmp_path):
    t = tile_with(tmp_path, '<circle cx="30" cy="40" r="5" semantic-id="3" instance-id="0"/>')
    p = t.primitives[0]
    assert p.bbox == pytest.approx((25, 35, 35, 45), abs=1e-3)
    assert p.closed and (p.semantic_id, p.instance_id) == (3, 0)


def test_closed_path_flag_and_negative_instance(tmp_path):
    t = tile_with(tmp_path, '<path d="M 0,0 L 4,0 L 4,4 Z" semantic-id="1" instance-id="-1"/>')
    p = t.primitives[0]
    assert p.closed and p.instance_id == -1


def test_spaced_path(spaced_tmp):
    t = parse_tile(write_svg(spaced_tmp / "0001-0001.svg", '<path d="M 0,0 L 5,5" semantic-id="3" instance-id="0"/>'), "test-00")
    assert t.stem == "0001-0001" and len(t.primitives) == 1


def test_tile_box_helper():
    t = tile_box(10, 20, 30, 40, sid=5, iid=2)
    p = t.primitives[0]
    assert p.bbox == (10, 20, 30, 40) and (p.semantic_id, p.instance_id) == (5, 2) and p.closed


def test_polyline_dense_on_scaled_ellipse_arc(tmp_path):
    t = tile_with(tmp_path, '<g transform="scale(2,3)"><path d="M 10,10 A 10,5 0 0 0 20,10" semantic-id="3" instance-id="0"/></g>')
    poly = t.primitives[0].polyline
    assert np.hypot(*np.diff(poly, axis=0).T).max() <= 0.05 + 1e-9
    assert poly[0] == pytest.approx((20, 30)) and poly[-1] == pytest.approx((40, 30))
