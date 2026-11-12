import json
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import torch
from PIL import Image

from conftest import CONFIGS, REPO
from fpyolo import predict
from fpyolo.predict import mask_to_polygons, to_json
from fpyolo.schema import load_schema

SCHEMA = load_schema(CONFIGS / "classes_v0.yaml")
IDS = {"D": "d.pt@aaaaaaaaaaaa", "W": "w.pt@bbbbbbbbbbbb"}


def _annulus(n=60):
    m = np.zeros((n, n), np.uint8)
    cv2.circle(m, (30, 30), 25, 1, -1)
    cv2.circle(m, (30, 30), 10, 0, -1)
    return m


def test_mask_ring_gives_one_hole():
    [(ext, holes)] = mask_to_polygons(_annulus())
    assert len(ext) >= 3 and len(holes) == 1 and len(holes[0]) >= 3
    xs = [p[0] for p in ext]
    assert min(xs) < 10 and max(xs) > 50          # exterior spans the disc
    assert all(abs(p[0] - 30) < 12 and abs(p[1] - 30) < 12 for p in holes[0])


def test_mask_components_and_empty():
    m = np.zeros((40, 40), np.uint8)
    m[2:10, 2:10] = 1
    m[20:30, 20:35] = 1
    assert len(mask_to_polygons(m)) == 2
    assert mask_to_polygons(np.zeros((5, 5), np.uint8)) == []


def test_schema_ids_not_yolo_indices():
    d_out = [(0, 0.93, (0.0, 0.0, 10.004, 10.0))]
    w_out = [(0, 0.88, _annulus()), (2, 0.5, _annulus())]
    doc = to_json("x.png", 60, 60, d_out, w_out, SCHEMA, IDS)
    assert doc["objects"] == [{"schema_id": 1, "name": "single_door", "score": 0.93,
                               "bbox_xyxy": [0.0, 0.0, 10.0, 10.0]}]
    assert [(w["schema_id"], w["name"]) for w in doc["walls"]] == [(31, "wall"), (33, "railing")]
    assert len(doc["walls"][0]["holes"]) == 1
    assert doc["schema_version"] == SCHEMA.version and doc["models"] == IDS
    assert "yolo_index" not in json.dumps(doc)
    assert list(doc)[:5] == ["image", "width", "height", "schema_version", "models"]


def test_empty_outputs():
    doc = to_json("x.png", 5, 7, [], [], SCHEMA, IDS)
    assert doc["objects"] == [] and doc["walls"] == [] and (doc["width"], doc["height"]) == (5, 7)


def test_degenerate_rings_dropped():
    m = np.zeros((20, 20), np.uint8)
    m[5, 5:12] = 1                                 # a 1 px line gives a degenerate ring
    doc = to_json("x.png", 20, 20, [], [(0, 0.9, m)], SCHEMA, IDS)
    assert doc["walls"] == []


class FakeModel:
    """Records every predict call; returns one Result-like object per call."""

    def __init__(self, h, w, boxes=(), masks=None):
        self.calls, self.h, self.w, self.boxes, self.masks = [], h, w, boxes, masks

    def predict(self, source, **kw):
        self.calls.append((source, kw))
        n = len(self.boxes)
        b = _Boxes(
            xyxy=torch.tensor([bx[2] for bx in self.boxes], dtype=torch.float32).reshape(n, 4),
            cls=torch.tensor([bx[0] for bx in self.boxes], dtype=torch.float32),
            conf=torch.tensor([bx[1] for bx in self.boxes], dtype=torch.float32))
        masks = None if self.masks is None else SimpleNamespace(data=torch.as_tensor(self.masks))
        path = source if isinstance(source, str) else "image0.jpg"
        return [SimpleNamespace(boxes=b, masks=masks, orig_shape=(self.h, self.w), path=path)]


class _Boxes:
    def __init__(self, xyxy, cls, conf):
        self.xyxy, self.cls, self.conf = xyxy, cls, conf

    def __len__(self):
        return len(self.conf)


def _png(path, rgba):
    Image.fromarray(rgba, "RGBA").save(path)


def _models(h, w):
    d = FakeModel(h, w, boxes=[(0, 0.9, (1.0, 2.0, 11.0, 12.0))])
    wm = FakeModel(h, w, boxes=[(0, 0.8, (0.0, 0.0, 1.0, 1.0))], masks=np.stack([_annulus(h)])[:, :h, :w])
    return d, wm


def test_from_floorplancad_png(tmp_path):
    rgba = np.zeros((60, 60, 4), np.uint8)
    rgba[10:20, 10:50, 3] = 255                    # ink only in alpha
    p = tmp_path / "tile.png"
    _png(p, rgba)
    d, wm = _models(60, 60)
    doc = predict.predict_image(d, wm, p, SCHEMA, IDS, d_conf=0.25, w_conf=0.4, device="cpu",
                                from_floorplancad_png=True)
    for m in (d, wm):
        [(src, kw)] = m.calls
        assert isinstance(src, np.ndarray) and src.shape == (60, 60, 3)   # one image per call
        assert src[15, 20].tolist() == [0, 0, 0] and src[0, 0].tolist() == [255, 255, 255]
        assert kw["imgsz"] == 1024 and kw["iou"] == 0.7 and kw["max_det"] == 300
    assert d.calls[0][1]["conf"] == 0.25 and wm.calls[0][1]["conf"] == 0.4
    assert wm.calls[0][1]["retina_masks"] is True
    assert doc["image"] == "tile.png" and doc["thresholds"] == {"D": 0.25, "W": 0.4}
    assert doc["objects"][0]["schema_id"] == 1 and doc["walls"][0]["schema_id"] == 31


def test_path_input_passes_one_path_string(tmp_path):
    p = tmp_path / "a.png"
    Image.fromarray(np.full((60, 60, 3), 255, np.uint8)).save(p)
    d, wm = _models(60, 60)
    predict.predict_image(d, wm, p, SCHEMA, IDS, d_conf=0.25, w_conf=0.4, device="cpu",
                          from_floorplancad_png=False)
    assert d.calls[0][0] == str(p) and wm.calls[0][0] == str(p)


def test_collect_inputs(tmp_path):
    for n in ("b.png", "a.JPG", "c.txt", "d.jpeg"):
        (tmp_path / n).write_bytes(b"")
    (tmp_path / "sub").mkdir()
    assert [p.name for p in predict.collect_inputs(tmp_path)] == ["a.JPG", "b.png", "d.jpeg"]
    assert predict.collect_inputs(tmp_path / "b.png") == [tmp_path / "b.png"]
    (tmp_path / "a.png").write_bytes(b"")
    with pytest.raises(ValueError, match="share a stem"):
        predict.collect_inputs(tmp_path)


def test_refuses_overwrite_before_loading_models(tmp_path):
    img = tmp_path / "in" / "a.png"
    img.parent.mkdir()
    img.write_bytes(b"")
    out = tmp_path / "out"
    out.mkdir()
    (out / "a.json").write_text("{}")
    with pytest.raises(FileExistsError, match="--overwrite"):
        predict.run(Path("d.pt"), Path("w.pt"), 0.3, CONFIGS / "classes_v0.yaml", img, out, "cpu")
    assert (out / "a.json").read_text() == "{}"


SMOKE_D, SMOKE_W = REPO / "runs/smoke/d/weights/last.pt", REPO / "runs/smoke/w/weights/last.pt"


@pytest.mark.skipif(not (SMOKE_D.is_file() and SMOKE_W.is_file()), reason="smoke weights absent")
def test_smoke_weights_end_to_end(tmp_path):
    img = next((REPO / "data/fpcad_v0/D/images/val").glob("*.png"), None)
    if img is None:
        pytest.skip("no val images")
    [out] = predict.run(SMOKE_D, SMOKE_W, 0.25, CONFIGS / "classes_v0.yaml", img, tmp_path, "cpu")
    doc = json.loads(out.read_text())
    assert doc["image"] == img.name and doc["models"]["D"].startswith("last.pt@")
    assert all(o["schema_id"] in range(1, 31) for o in doc["objects"])
