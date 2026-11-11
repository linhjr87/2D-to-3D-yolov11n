import cv2
import numpy as np
import pytest
import shapely
from shapely.geometry import Polygon, box

from fpyolo.walls import format_yolo_seg, keyhole_ring, wall_regions
from svgfixtures import tile_with

W = H = 1000  # image px; viewBox is 0 0 100 100, so 1 unit = 10 px


def _path(d, sid=1, iid=-1):
    return f'<path d="{d}" fill="none" semantic-id="{sid}" instance-id="{iid}"/>'


def _rect(x, y, w, h, sid=1):
    return _path(f"M {x} {y} L {x + w} {y} L {x + w} {y + h} L {x} {y + h} Z", sid)


def _walls(tmp_path, *elements, **kw):
    return wall_regions(tile_with(tmp_path, "\n".join(elements)), {1}, W, H, **kw)


def test_rect_outline_is_one_body(tmp_path):
    res = _walls(tmp_path, _rect(10, 20, 50, 2))
    assert len(res.polygons) == 1
    p = res.polygons[0]
    assert p.area == pytest.approx(0.01, rel=1e-3)
    assert p.bounds == pytest.approx((0.10, 0.20, 0.60, 0.22), abs=1e-6)
    assert res.coverage >= 0.99
    assert res.stroke_len_px == pytest.approx(2 * (500 + 20), rel=1e-3)


def test_room_interior_excluded(tmp_path):
    # 30x30 room (22..52) enclosed by four 2-unit walls, each a closed outline
    res = _walls(
        tmp_path,
        _rect(20, 20, 34, 2),
        _rect(20, 52, 34, 2),
        _rect(20, 22, 2, 30),
        _rect(52, 22, 2, 30),
    )
    assert len(res.polygons) == 1
    p = res.polygons[0]
    assert len(p.interiors) == 1
    assert p.area == pytest.approx((34**2 - 30**2) / 100**2, rel=1e-3)


def test_thick_wall_kept(tmp_path):
    res = _walls(tmp_path, _rect(10, 20, 50, 6))  # 60 px thick: inscribed radius 30 = 60 / 2
    assert len(res.polygons) == 1
    assert res.polygons[0].area == pytest.approx(0.5 * 0.06, rel=1e-3)
    assert _walls(tmp_path, _rect(10, 20, 50, 7)).polygons == []  # 70 px: a room, not a wall


def test_border_cut_wall_closed_by_border(tmp_path):
    res = _walls(
        tmp_path,
        _path("M -5 40 L 20 40"),
        _path("M -5 42 L 20 42"),
        _path("M 20 40 L 20 42"),
    )
    assert len(res.polygons) == 1
    assert res.polygons[0].bounds == pytest.approx((0.0, 0.40, 0.20, 0.42), abs=1e-6)
    assert res.coverage >= 0.99  # only the part inside the image counts


def test_border_sliver_excluded(tmp_path):
    # wall body y 5..7 units across the whole tile; the band y 0..5 above it is
    # thin enough (radius 25 px) to pass the thickness test but lies on the frame
    res = _walls(tmp_path, _path("M -5 5 L 105 5"), _path("M -5 7 L 105 7"))
    assert len(res.polygons) == 1
    assert res.polygons[0].bounds == pytest.approx((0.0, 0.05, 1.0, 0.07), abs=1e-6)
    band = box(0.0, 0.0, 1.0, 0.05)
    assert all(p.intersection(band).area < 1e-9 for p in res.polygons)


def test_hatch_lines_merged(tmp_path):
    hatch = [_path(f"M {x} 20 L {x + 3} 23") for x in (15, 23, 31, 39, 47)]
    res = _walls(tmp_path, _rect(10, 20, 50, 3), *hatch)
    assert len(res.polygons) == 1
    assert len(res.polygons[0].interiors) == 0
    assert res.polygons[0].area == pytest.approx(0.5 * 0.03, rel=1e-3)
    assert res.coverage >= 0.99  # hatching inside the body counts as covered


def test_gap_closing(tmp_path):
    # outline 10..60 x 20..22 drawn as two strokes, 2 px gap at the corner (60, 22)
    strokes = (_path("M 10 22 L 10 20 L 60 20 L 60 22"), _path("M 10 22 L 59.8 22"))
    res = _walls(tmp_path, *strokes, gap_px=3.0)
    assert len(res.polygons) == 1
    assert res.polygons[0].area == pytest.approx(0.01, rel=1e-3)
    res = _walls(tmp_path, *strokes, gap_px=1.0)
    assert res.polygons == []
    assert res.coverage < 0.5


def _wall_lines(x0, x1, closed_left=True, closed_right=True):
    """A wall x0..x1 by 50..52 units drawn as separate lines, like the real data."""
    lines = [_path(f"M {x0} 50 L {x1} 50"), _path(f"M {x0} 52 L {x1} 52")]
    if closed_left:
        lines.append(_path(f"M {x0} 50 L {x0} 52"))
    if closed_right:
        lines.append(_path(f"M {x1} 50 L {x1} 52"))
    return lines


def test_end_cap_closes_open_wall_end(tmp_path):
    # two parallel strokes 20 px apart end at x=40 (a door), the other end is closed
    strokes = _wall_lines(10, 40, closed_right=False)
    res = _walls(tmp_path, *strokes)
    assert len(res.polygons) == 1
    assert res.polygons[0].bounds == pytest.approx((0.10, 0.50, 0.40, 0.52), abs=1e-6)
    assert _walls(tmp_path, *strokes, cap_px=0).polygons == []  # the cap closes it


@pytest.mark.parametrize("closed_ends", [True, False])
def test_door_opening_between_walls_not_bridged(tmp_path, closed_ends):
    # collinear walls 10..40 and 49..80 face each other across a 90 px opening
    res = _walls(
        tmp_path,
        *_wall_lines(10, 40, closed_right=closed_ends),
        *_wall_lines(49, 80, closed_left=closed_ends),
    )
    assert len(res.polygons) == 2
    opening = box(0.40, 0.50, 0.49, 0.52)
    assert all(p.intersection(opening).area < 1e-9 for p in res.polygons)


def test_end_not_capped_along_stroke_direction(tmp_path):
    # the top line of wall 10..60 x 50..52 has a 20 px break (x 30..32): its two
    # ends are 20 px apart but the joining segment is parallel to the strokes
    res = _walls(
        tmp_path,
        _path("M 10 50 L 30 50"),
        _path("M 32 50 L 60 50"),
        _path("M 10 52 L 60 52"),
        _path("M 10 50 L 10 52"),
        _path("M 60 50 L 60 52"),
    )
    assert res.polygons == []


def test_no_wall_strokes(tmp_path):
    res = _walls(tmp_path, _rect(10, 20, 50, 2, sid=3))
    assert res.coverage is None
    assert res.polygons == []
    assert res.stroke_len_px == 0.0


def _fill(ring, size):
    canvas = np.zeros((size, size), np.uint8)
    cv2.fillPoly(canvas, [np.round(ring).astype(np.int32)], 1)
    return canvas.astype(bool)


def _iou(a, b):
    return (a & b).sum() / (a | b).sum()


def test_keyhole_roundtrip_iou():
    poly = Polygon(
        [(0, 0), (100, 0), (100, 100), (0, 100)],
        [[(30, 30), (70, 30), (70, 70), (30, 70)]],
    )
    ring = keyhole_ring(poly)
    assert ring.ndim == 2 and ring.shape[1] == 2
    got = _fill(ring, 128)
    ys, xs = np.mgrid[0:128, 0:128]
    ref = shapely.intersects_xy(poly, xs, ys)
    assert _iou(got, ref) >= 0.99
    assert not got[50, 50]


def test_keyhole_bridges_avoid_other_holes():
    # the diamond's shortest way to the exterior (straight up from its apex at
    # x=50) passes through the small square hole: a bridge must not cross it
    square = [(45, 10), (55, 10), (55, 20), (45, 20)]
    diamond = [(50, 25), (68, 45), (50, 65), (32, 45)]
    poly = Polygon([(0, 0), (100, 0), (100, 100), (0, 100)], [square, diamond])
    ring = keyhole_ring(poly)
    got = _fill(ring, 128)
    ys, xs = np.mgrid[0:128, 0:128]
    assert _iou(got, shapely.intersects_xy(poly, xs, ys)) >= 0.99
    assert not got[11:20, 46:55].any()  # square hole interior stays empty
    assert not got[45, 50]  # diamond center


def test_format_yolo_seg():
    ring = np.array([[0.1, 0.2], [0.6, 0.2], [0.6, 0.22], [1.0000009, 0.22]])
    line = format_yolo_seg(0, ring)
    assert line.startswith("0 ")
    tokens = line.split()[1:]
    assert len(tokens) >= 6 and len(tokens) == ring.size
    assert tokens[0] == "0.100000"
    assert all(0.0 <= float(v) <= 1.0 for v in tokens)
