import numpy as np
import pytest

from fpyolo.boxes import YoloBox, format_yolo_det, thing_boxes
from fpyolo.svgparse import Primitive, Tile
from svgfixtures import tile_box, tile_with

MAP = {3: 0}


def _prim(sid, iid, bbox):
    x0, y0, x1, y1 = bbox
    poly = np.array([[x0, y0], [x1, y1]], dtype=np.float64)
    return Primitive(sid, iid, bbox, poly, False)


def _tile(prims, viewbox=(0.0, 0.0, 100.0, 100.0)):
    return Tile("train-00", "0000-0000", viewbox, prims, 0)


def test_union_per_instance():
    t = _tile([_prim(3, 7, (10, 10, 20, 20)), _prim(3, 7, (15, 15, 30, 25))])
    boxes, st = thing_boxes(t, MAP, 1000, 1000)
    assert len(boxes) == 1 and st["n_instances"] == 1
    b = boxes[0]
    assert b.yolo_index == 0
    assert b.cx == pytest.approx(0.20) and b.cy == pytest.approx(0.175)
    assert b.w == pytest.approx(0.20) and b.h == pytest.approx(0.15)


def test_distinct_instances_are_separate_objects():
    t = _tile([_prim(3, 1, (10, 10, 20, 20)), _prim(3, 2, (50, 50, 60, 60))])
    boxes, st = thing_boxes(t, MAP, 1000, 1000)
    assert len(boxes) == 2 and st["n_instances"] == 2


def test_stuff_and_negative_instance_ignored():
    t = _tile(
        [
            _prim(1, -1, (0, 0, 10, 10)),
            _prim(34, 0, (0, 0, 10, 10)),
            _prim(3, -1, (0, 0, 10, 10)),
        ]
    )
    boxes, st = thing_boxes(t, MAP, 1000, 1000)
    assert boxes == [] and st["n_instances"] == 0


def test_cut_object_dropped_below_30pct():
    boxes, st = thing_boxes(tile_box(-50, 10, 10, 20), MAP, 1000, 1000)  # 10/60 visible
    assert boxes == [] and st["dropped_cut"] == 1 and st["n_instances"] == 1
    boxes, st = thing_boxes(tile_box(-10, 10, 30, 20), MAP, 1000, 1000)  # 0.75 visible
    assert st["dropped_cut"] == 0
    assert boxes[0].cx == pytest.approx(0.15) and boxes[0].w == pytest.approx(0.30)


def test_object_fully_outside_dropped_without_error():
    boxes, st = thing_boxes(tile_box(150, 10, 160, 20), MAP, 1000, 1000)
    assert boxes == [] and st["dropped_cut"] == 1


def test_thin_box_padded_to_2px():
    t = _tile([_prim(3, 0, (10, 50, 20, 50))])  # zero-height horizontal line
    boxes, st = thing_boxes(t, MAP, 1000, 1000)
    assert boxes[0].h == pytest.approx(0.002) and boxes[0].cy == pytest.approx(0.5)
    assert boxes[0].w == pytest.approx(0.10)
    assert st["padded"] == 1


def test_zero_area_point_at_border_does_not_crash():
    # padded to 2 px: x -1..1 px, half visible -> kept and clipped to x 0..1 px
    t = _tile([_prim(3, 0, (0, 50, 0, 50))])
    boxes, st = thing_boxes(t, MAP, 1000, 1000)
    assert st["padded"] == 1 and len(boxes) == 1
    assert boxes[0].w == pytest.approx(0.001) and boxes[0].cx == pytest.approx(0.0005)
    # at the corner only a quarter of the padded box is visible -> dropped
    boxes, st = thing_boxes(_tile([_prim(3, 0, (0, 0, 0, 0))]), MAP, 1000, 1000)
    assert boxes == [] and st["dropped_cut"] == 1


def test_999px_uses_viewbox_and_true_width():
    t = _tile([_prim(3, 0, (10, 10, 10, 20))])  # zero-width vertical line
    b1000, _ = thing_boxes(t, MAP, 1000, 1000)
    b999, st = thing_boxes(t, MAP, 999, 1000)
    assert b999[0].cx == pytest.approx(b1000[0].cx)
    assert b999[0].w == pytest.approx(2 / 999)
    assert b1000[0].w == pytest.approx(2 / 1000)
    assert st["padded"] == 1


def test_viewbox_offset_and_size_normalization():
    # viewbox as stored by parse_tile: (min_x, min_y, max_x, max_y)
    t = _tile([_prim(3, 0, (20, 30, 60, 50))], viewbox=(10.0, 20.0, 110.0, 70.0))
    boxes, _ = thing_boxes(t, MAP, 1000, 500)
    b = boxes[0]
    assert b.cx == pytest.approx(0.30) and b.w == pytest.approx(0.40)
    assert b.cy == pytest.approx(0.40) and b.h == pytest.approx(0.40)


def test_small_lt_32px_counts_area_below_32_squared():
    t = _tile(
        [
            _prim(3, 1, (10, 10, 13, 13)),  # 30x30 px: small
            _prim(3, 2, (40, 10, 41, 90)),  # 10x800 px: area 8000 > 1024
            _prim(3, 3, (60, 10, 63.5, 13.5)),  # 35x35 px: not small
        ]
    )
    _, st = thing_boxes(t, MAP, 1000, 1000)
    assert st["small_lt_32px"] == 1 and st["n_instances"] == 3


def test_parse_tile_viewbox_roundtrip(tmp_path):
    t = tile_with(
        tmp_path,
        '<rect x="10" y="10" width="10" height="10" semantic-id="3" instance-id="0"/>',
    )
    boxes, _ = thing_boxes(t, MAP, 1000, 1000)
    assert boxes[0].cx == pytest.approx(0.15) and boxes[0].w == pytest.approx(0.10)


def test_format_yolo_det():
    s = format_yolo_det([YoloBox(0, 0.2, 0.175, 0.2, 0.15)])
    assert s == "0 0.200000 0.175000 0.200000 0.150000"
    assert format_yolo_det([]) == ""
    two = format_yolo_det([YoloBox(0, 0.5, 0.5, 0.1, 0.1), YoloBox(12, 0.5, 0.5, 0.1, 0.1)])
    assert two.split("\n")[1].startswith("12 ")
