import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
import shapely
import torch
import yaml
from PIL import Image

from conftest import CONFIGS, REPO
from fpyolo.compare import main as compare_main
from fpyolo.compare import real_difference
from fpyolo.evaluate_d import load_split_gt, predict_split, read_preds, write_json, write_preds
from fpyolo.evaluate_w import (
    THRESHOLDS, gt_union_mask, iou_curve, max_score_map, predict_stream, select_threshold,
)
from fpyolo.schema import load_schema
from fpyolo.walls import format_yolo_seg, keyhole_ring

SMOKE_D = REPO / "runs" / "smoke" / "d" / "weights" / "last.pt"
SMOKE_W = REPO / "runs" / "smoke" / "w" / "weights" / "last.pt"


def _dataset(root, split, images):
    """images: {image_id: ((width, height), label_text or None)}."""
    for kind in ("images", "labels"):
        (root / kind / split).mkdir(parents=True, exist_ok=True)
    for key, (size, label) in images.items():
        Image.new("RGB", size, "white").save(root / "images" / split / f"{key}.png")
        if label is not None:
            (root / "labels" / split / f"{key}.txt").write_text(label, encoding="utf-8")
    return root


def test_load_split_gt_999px(spaced_tmp):
    ds = _dataset(spaced_tmp / "D", "test", {
        "test-00_0001-0002": ((1000, 999), "0 0.5 0.5 0.1 0.2\n"),
        "train-00_0000-0001": ((1000, 1000), None),  # no label file: no objects
        "train-01_0003-0004": ((1000, 1000), ""),  # empty label file: no objects
    })
    gts = load_split_gt(ds, "test")
    assert sorted(gts) == ["test-00_0001-0002", "train-00_0000-0001", "train-01_0003-0004"]
    boxes, cls = gts["test-00_0001-0002"]
    np.testing.assert_allclose(boxes, [[450, 399.6, 550, 599.4]])
    assert cls.tolist() == [0] and cls.dtype == np.int64
    for key in ("train-00_0000-0001", "train-01_0003-0004"):
        assert gts[key][0].shape == (0, 4) and gts[key][1].shape == (0,)
    assert list(load_split_gt(ds, "test", limit=1)) == ["test-00_0001-0002"]  # first id in sorted order


def test_preds_npz_roundtrip(spaced_tmp):
    rng = np.random.default_rng(0)
    preds = {
        "b_0001-0001": (rng.random((3, 4), dtype=np.float32), np.array([0, 5, 29]),
                        rng.random(3, dtype=np.float32)),
        "a_0000-0000": (np.zeros((0, 4), np.float32), np.zeros(0, np.int64), np.zeros(0, np.float32)),
        "c_0002-0002": (rng.random((1, 4), dtype=np.float32), np.array([7]), np.array([0.5], np.float32)),
    }
    path = spaced_tmp / "preds.npz"
    write_preds(path, preds)
    back = read_preds(path)
    assert sorted(back) == sorted(preds)
    for key, (b, c, s) in preds.items():
        rb, rc, rs = back[key]
        assert np.array_equal(rb, b) and rb.dtype == b.dtype and rb.shape == (len(b), 4)
        assert np.array_equal(rc, c) and np.array_equal(rs, s) and rs.dtype == s.dtype
    write_preds(path, {})
    assert read_preds(path) == {}


def test_write_json_nan_as_null(spaced_tmp):
    path = spaced_tmp / "m.json"
    write_json(path, {"a": float("nan"), "b": [np.float64(0.5), np.nan], "c": {1: np.int64(3)},
                      "d": np.bool_(True), "e": float("inf")})
    assert json.loads(path.read_text(encoding="utf-8")) == {
        "a": None, "b": [0.5, None], "c": {"1": 3}, "d": True, "e": None}


def test_iou_curve_perfect_and_empty():
    assert THRESHOLDS.tolist() == [round(0.05 * i, 2) for i in range(1, 20)]
    gt = np.zeros((20, 30), bool)
    gt[5:10, 4:20] = True
    score = np.where(gt, np.float32(0.8), np.float32(0)).astype(np.float32)
    curve = iou_curve([(gt, score)], THRESHOLDS)
    np.testing.assert_array_equal(curve, np.where(THRESHOLDS <= 0.8, 1.0, 0.0))
    empty = iou_curve([(gt, np.zeros_like(score))], THRESHOLDS)
    np.testing.assert_array_equal(empty, np.zeros(len(THRESHOLDS)))


def test_iou_curve_is_pooled_over_images():
    # image 1: IoU 1 (10 px); image 2: the prediction is 3x the label (IoU 1/3).
    # Pooled: (10 + 10) / (10 + 30) = 0.5, not the per-image mean 2/3.
    gt1 = np.zeros((10, 10), bool)
    gt1[0, :] = True
    gt2 = np.zeros((10, 10), bool)
    gt2[0, :] = True
    s2 = np.zeros((10, 10), np.float32)
    s2[:3, :] = 0.6
    pairs = ((g, s) for g, s in [(gt1, gt1.astype(np.float32) * 0.9), (gt2, s2)])  # a one-shot generator
    curve = iou_curve(pairs, np.array([0.5, 0.7]))
    # at 0.7 only image 1's 10 px are predicted: 10 / 20
    np.testing.assert_allclose(curve, [0.5, 0.5])
    nothing = iou_curve([(np.zeros((4, 4), bool), np.zeros((4, 4), np.float32))], np.array([0.5]))
    assert np.isnan(nothing).all()  # no label and no prediction anywhere: undefined, not 0


def test_threshold_selection_picks_max():
    t, iou = select_threshold(np.array([0.2, 0.6, 0.5]), np.array([0.05, 0.10, 0.15]))
    assert t == 0.10 and iou == 0.6
    t, _ = select_threshold(np.array([0.2, 0.6, 0.6]), np.array([0.05, 0.10, 0.15]))
    assert t == 0.10  # ties: the first (lowest) threshold


def test_gt_mask_keeps_hole(spaced_tmp):
    annulus = shapely.Polygon([(0.2037, 0.2013), (0.7962, 0.2), (0.7987, 0.8), (0.2, 0.7981)],
                              [[(0.4071, 0.4), (0.5911, 0.4031), (0.6, 0.5969), (0.4, 0.6)]])
    square = shapely.box(0.85, 0.85, 0.95, 0.95)
    path = spaced_tmp / "x.txt"
    path.write_text(format_yolo_seg(0, keyhole_ring(annulus)) + "\n"
                    + format_yolo_seg(2, keyhole_ring(square)) + "\n", encoding="utf-8")
    w, h = 200, 160
    mask = gt_union_mask(path, w, h)
    assert mask.shape == (h, w) and mask.dtype == bool
    assert not mask[h // 2, w // 2]  # inside the hole
    assert mask[int(0.3 * h), w // 2] and mask[int(0.9 * h), int(0.9 * w)]
    # cv2.fillPoly also fills boundary pixels: the mask is the region of pixel centres
    # ((j + 0.5) / w, (i + 0.5) / h) inside a polygon, grown by at most 1 px (a few corner
    # pixels of tolerance)
    ys, xs = np.mgrid[0:h, 0:w]
    ref = shapely.intersects_xy(shapely.union(annulus, square), (xs + 0.5) / w, (ys + 0.5) / h)
    grown = cv2.dilate(ref.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    assert (ref & ~mask).sum() + (mask & ~grown).sum() <= 4
    only_wall = gt_union_mask(path, w, h, cls=0)
    assert only_wall[int(0.3 * h), w // 2] and not only_wall[int(0.9 * h), int(0.9 * w)]
    only_railing = gt_union_mask(path, w, h, cls=2)
    assert only_railing.sum() == (mask & ~only_wall).sum()
    assert not gt_union_mask(path, w, h, cls=1).any()
    assert not gt_union_mask(spaced_tmp / "missing.txt", w, h).any()


def _fake_result(h, w, boxes):
    """boxes: [(conf, cls, (y0, y1, x0, x1))] -> object shaped like an Ultralytics seg Result."""
    if not boxes:
        return SimpleNamespace(masks=None, boxes=SimpleNamespace(conf=torch.zeros(0), cls=torch.zeros(0)))
    data = torch.zeros((len(boxes), h, w), dtype=torch.uint8)
    for i, (_, _, (y0, y1, x0, x1)) in enumerate(boxes):
        data[i, y0:y1, x0:x1] = 1
    return SimpleNamespace(masks=SimpleNamespace(data=data),
                           boxes=SimpleNamespace(conf=torch.tensor([b[0] for b in boxes]),
                                                 cls=torch.tensor([float(b[1]) for b in boxes])))


def test_max_score_map_takes_max_and_filters_class():
    h, w = 8, 10
    r = _fake_result(h, w, [(0.3, 0, (0, 4, 0, 4)), (0.9, 0, (2, 6, 2, 6)), (0.5, 2, (0, 2, 4, 8))])
    s = max_score_map(r, h, w)
    assert s.shape == (h, w) and s.dtype == np.float32
    assert s[0, 0] == np.float32(0.3) and s[3, 3] == np.float32(0.9) and s[0, 5] == np.float32(0.5)
    assert s[7, 9] == 0
    walls = max_score_map(r, h, w, cls=0)
    assert walls[0, 5] == 0 and walls[3, 3] == np.float32(0.9)
    railing = max_score_map(r, h, w, cls=2)
    assert railing[0, 5] == np.float32(0.5) and railing[3, 3] == 0
    assert not max_score_map(_fake_result(h, w, []), h, w).any()
    with pytest.raises(ValueError, match="retina_masks"):
        max_score_map(r, h // 2, w // 2)  # masks at another resolution than the image


class _RecordingModel:
    """Stands in for ultralytics.YOLO: records every predict call, returns one fake Result."""

    def __init__(self):
        self.calls = []

    def predict(self, source, **kw):
        self.calls.append((source, kw))
        data = torch.tensor([[1.0, 2.0, 30.0, 40.0, 0.9, 4.0], [5.0, 6.0, 7.0, 8.0, 0.2, 0.0]])
        boxes = SimpleNamespace(xyxy=data[:, :4], conf=data[:, 4], cls=data[:, 5])
        return [SimpleNamespace(path=str(source), boxes=boxes, masks=None, orig_shape=(10, 10))]


def test_predict_one_image_per_call(spaced_tmp, monkeypatch):
    # Ultralytics 8.4 turns a list source into ONE in-memory batch of the whole list
    # (check_source -> autocast_list -> LoadPilAndNumpy), which took ~48 GB on the val split
    import ultralytics

    model = _RecordingModel()
    monkeypatch.setattr(ultralytics, "YOLO", lambda weights: model)
    imgs = [spaced_tmp / f"test-00_000{i}-0000.png" for i in range(3)]
    preds = predict_split(Path("w.pt"), imgs, device="cpu")
    assert [src for src, _ in model.calls] == [str(p) for p in imgs]  # a str path per call, never a list
    assert all(kw["device"] == "cpu" and kw["conf"] == 0.001 and kw["max_det"] == 300 for _, kw in model.calls)
    for boxes, cls, scores in preds.values():  # plain numpy arrays, no torch storage kept alive
        assert all(isinstance(a, np.ndarray) and a.flags.owndata for a in (boxes, cls, scores))
        assert cls.dtype == np.int64 and boxes.shape == (2, 4)
    model.calls.clear()
    pairs = list(predict_stream(Path("w.pt"), imgs, device="cpu"))
    assert [p for p, _ in pairs] == imgs
    assert [src for src, _ in model.calls] == [str(p) for p in imgs]
    assert all(kw["retina_masks"] is True for _, kw in model.calls)


def test_ultralytics_val_trims_heap_and_pins_loader(spaced_tmp, monkeypatch):
    # glibc keeps freed heap pages: without a trim after each val batch, RSS drifted up
    # ~6 MB/image on W val (trimmed: flat). batch/workers are explicit to bound host RAM.
    import ultralytics

    from fpyolo import evaluate_d

    class _Val:
        def __init__(self):
            self.callbacks, self.kwargs = {}, None

        def add_callback(self, event, fn):
            self.callbacks.setdefault(event, []).append(fn)

        def val(self, **kw):
            self.kwargs = kw
            for fn in self.callbacks.get("on_val_batch_end", []):
                fn(None)  # must accept the validator argument
            return "metrics"

    model = _Val()
    monkeypatch.setattr(ultralytics, "YOLO", lambda weights: model)
    out = spaced_tmp / "out"
    assert evaluate_d.ultralytics_val(Path("w.pt"), spaced_tmp, "test", out, "cpu") == "metrics"
    assert evaluate_d.trim_heap in model.callbacks["on_val_batch_end"]
    kw = model.kwargs
    assert kw["batch"] == 16 and kw["workers"] == 2 and kw["split"] == "test" and kw["imgsz"] == 1024
    assert kw["project"] == str(spaced_tmp) and kw["name"] == "out" and kw["exist_ok"] is True


SCHEMA_V0 = CONFIGS / "classes_v0.yaml"


def _write_data_yaml(root, model="D"):
    names = load_schema(SCHEMA_V0).yolo_names(model)
    (root / "data.yaml").write_text(yaml.safe_dump({"path": str(root), "names": names}), encoding="utf-8")


def _write_run(run_dir, preds, dataset, manifests, split="test"):
    run_dir.mkdir(parents=True)
    write_preds(run_dir / "preds.npz", preds)
    write_json(run_dir / "metrics.json", {"dataset": str(dataset), "split": split,
                                          "manifest": str(manifests), "limit": None,
                                          "schema": str(SCHEMA_V0.resolve())})


def _compare_setup(root):
    """10 test images in 5 clusters, each with one object of class 0 and one of class 1."""
    gt_boxes = np.array([[10, 10, 40, 40], [50, 50, 90, 90]], float)
    ids = [f"test-00_{i:04d}-0000" for i in range(10)]
    label = "".join(f"{c} {(x0 + x1) / 200} {(y0 + y1) / 200} {(x1 - x0) / 100} {(y1 - y0) / 100}\n"
                    for c, (x0, y0, x1, y1) in enumerate(gt_boxes))
    ds = _dataset(root / "data" / "D", "test", {k: ((100, 100), label) for k in ids})
    _write_data_yaml(ds)
    man = root / "data" / "manifests"
    man.mkdir(parents=True)
    with open(man / "tiles.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["archive", "stem", "keep", "cluster"])
        for i, k in enumerate(ids):
            wr.writerow(["test-00", k.split("_", 1)[1], "True", i // 2])
        wr.writerow(["test-00", "9999-9999", "False", ""])  # a dropped duplicate has no cluster
    yolo = load_schema(CONFIGS / "classes_v0.yaml").yolo_index("D")
    core_sids = sorted(s for s, i in yolo.items() if i in (0, 1))
    (man / "core_classes.json").write_text(json.dumps({"semantic_ids": core_sids}), encoding="utf-8")

    def run(name, shift):
        boxes = gt_boxes + shift
        preds = {k: (boxes.astype(np.float32), np.array([0, 1]), np.array([0.9, 0.8], np.float32))
                 for k in ids}
        _write_run(root / "runs" / name, preds, ds, man)
        return str(root / "runs" / name)

    return man, run


def test_compare_real_flag(spaced_tmp):
    man, run = _compare_setup(spaced_tmp)
    good = [run(f"good_s{s}", 0.3 * s) for s in range(3)]  # near-perfect boxes, small seed spread
    poor = [run(f"poor_s{s}", 3 + 0.3 * s) for s in range(3)]  # boxes 3+ px off: IoU 0.5-0.75
    out = spaced_tmp / "res" / "compare.json"
    assert compare_main(["--a", *good, "--b", *poor, "--manifest", str(man), "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["a"]["runs"] == [str(Path(g).resolve()) for g in good]
    assert len(res["a"]["headlines"]) == 3
    assert res["a"]["mean"] == pytest.approx(np.mean(res["a"]["headlines"]))
    assert res["a"]["std"] == pytest.approx(np.std(res["a"]["headlines"], ddof=1))
    assert res["bootstrap"]["delta"] == pytest.approx(res["a"]["mean"] - res["b"]["mean"])
    assert res["bootstrap"]["delta"] > 0.3 and res["bootstrap"]["ci_low"] > 0
    assert res["bootstrap"]["n"] == 1000
    assert res["real"] is True

    # the same three models on both sides, in another order: delta ~ 0
    same = good[1:] + good[:1]
    assert compare_main(["--a", *good, "--b", *same, "--manifest", str(man), "--out", str(out)]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert abs(res["bootstrap"]["delta"]) < 1e-12
    assert res["real"] is False


def test_compare_rejects_mixed_splits(spaced_tmp):
    man, run = _compare_setup(spaced_tmp)
    a = run("a", 0.0)
    b = run("b", 1.0)
    meta = json.loads((spaced_tmp / "runs" / "b" / "metrics.json").read_text(encoding="utf-8"))
    write_json(spaced_tmp / "runs" / "b" / "metrics.json", dict(meta, split="val"))
    with pytest.raises(ValueError, match="split"):
        compare_main(["--a", a, "--b", b, "--manifest", str(man), "--out", str(spaced_tmp / "c.json")])


def _set_meta(run_dir, **kw):
    meta = json.loads((Path(run_dir) / "metrics.json").read_text(encoding="utf-8"))
    write_json(Path(run_dir) / "metrics.json", dict(meta, **kw))


def test_compare_requires_one_schema(spaced_tmp):
    man, run = _compare_setup(spaced_tmp)
    a, b = run("a", 0.0), run("b", 1.0)
    out = str(spaced_tmp / "c.json")
    other = spaced_tmp / "classes_other.yaml"
    other.write_text(SCHEMA_V0.read_text(encoding="utf-8"), encoding="utf-8")
    _set_meta(b, schema=str(other))
    with pytest.raises(ValueError, match="disagree on schema"):
        compare_main(["--a", a, "--b", b, "--manifest", str(man), "--out", out])
    _set_meta(a, schema=str(other))  # both runs agree, but not with --schema (the default)
    with pytest.raises(ValueError, match="--schema .* is not the schema of the runs"):
        compare_main(["--a", a, "--b", b, "--manifest", str(man), "--out", out])
    assert compare_main(["--a", a, "--b", b, "--manifest", str(man), "--out", out,
                         "--schema", str(other)]) == 0
    _set_meta(a, schema=None)
    with pytest.raises(ValueError, match="records no schema"):
        compare_main(["--a", a, "--b", b, "--manifest", str(man), "--out", out, "--schema", str(other)])


def test_compare_checks_dataset_class_names(spaced_tmp):
    man, run = _compare_setup(spaced_tmp)
    a, b = run("a", 0.0), run("b", 1.0)
    data_yaml = spaced_tmp / "data" / "D" / "data.yaml"
    doc = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    doc["names"][0], doc["names"][1] = doc["names"][1], doc["names"][0]
    data_yaml.write_text(yaml.safe_dump(doc), encoding="utf-8")
    with pytest.raises(ValueError, match="class names differ from the D classes"):
        compare_main(["--a", a, "--b", b, "--manifest", str(man), "--out", str(spaced_tmp / "c.json")])


class _NamedModel:
    def __init__(self, names):
        self.names = names


@pytest.mark.parametrize("key", ["D", "W"])
def test_check_weights_against_schema(monkeypatch, key):
    import ultralytics

    from fpyolo.evaluate_d import check_weights

    schema = load_schema(SCHEMA_V0)
    names = schema.yolo_names(key)
    monkeypatch.setattr(ultralytics, "YOLO", lambda weights: _NamedModel(dict(names)))
    check_weights(Path("best.pt"), schema, key)
    wrong = dict(names)
    wrong[0] = "not_a_class"
    monkeypatch.setattr(ultralytics, "YOLO", lambda weights: _NamedModel(wrong))
    with pytest.raises(ValueError, match=f"best.pt: class names differ from the {key} classes"):
        check_weights(Path("best.pt"), schema, key)


def test_evaluate_d_refuses_mismatched_weights_before_predicting(spaced_tmp, monkeypatch):
    import ultralytics

    from fpyolo import evaluate_d

    man, _ = _compare_setup(spaced_tmp)
    w_names = load_schema(SCHEMA_V0).yolo_names("W")  # W weights passed to the D evaluation
    monkeypatch.setattr(ultralytics, "YOLO", lambda weights: _NamedModel(w_names))
    monkeypatch.setattr(evaluate_d, "predict_split", lambda *a, **k: pytest.fail("predicted"))
    with pytest.raises(ValueError, match="class names differ from the D classes"):
        evaluate_d.evaluate(spaced_tmp / "w.pt", spaced_tmp / "data" / "D", "test", man,
                            spaced_tmp / "out", "cpu", SCHEMA_V0)


def test_evaluate_w_refuses_mismatched_weights_before_predicting(spaced_tmp, monkeypatch):
    import ultralytics

    from fpyolo import evaluate_w

    ds = _dataset(spaced_tmp / "W", "val", {"val-00_0000-0000": ((100, 100), None)})
    _write_data_yaml(ds, "W")
    d_names = load_schema(SCHEMA_V0).yolo_names("D")
    monkeypatch.setattr(ultralytics, "YOLO", lambda weights: _NamedModel(d_names))
    monkeypatch.setattr(evaluate_w, "predict_stream", lambda *a, **k: pytest.fail("predicted"))
    with pytest.raises(ValueError, match="class names differ from the W classes"):
        evaluate_w.evaluate(spaced_tmp / "d.pt", ds, spaced_tmp / "out", "cpu", schema_path=SCHEMA_V0)


@pytest.mark.parametrize("module", ["evaluate_d", "evaluate_w"])
def test_limit_only_on_cpu(module, spaced_tmp, monkeypatch, capsys):
    import importlib

    mod = importlib.import_module(f"fpyolo.{module}")
    monkeypatch.setattr(mod, "evaluate", lambda *a, **k: pytest.fail("evaluated"))
    monkeypatch.setattr(mod, "apply_device_env", lambda d: pytest.fail("device env set"))
    args = ["--weights", "w.pt", "--dataset", str(spaced_tmp), "--out", str(spaced_tmp / "o"),
            "--device", "0", "--limit", "5"]
    if module == "evaluate_d":
        args += ["--manifest", str(spaced_tmp)]
    with pytest.raises(SystemExit) as e:
        mod.main(args)
    assert e.value.code == 2 and "--limit is for CPU smoke runs only" in capsys.readouterr().err


def test_evaluate_w_main_passes_scale_conclusion(spaced_tmp, monkeypatch):
    from fpyolo import evaluate_w

    seen = {}

    def fake_evaluate(*a, **k):
        seen["args"] = a
        return {"t_star": 0.5, "val_iou_at_t_star": 0.5, "test_pixel_iou": 0.5, "mask_map50_95": 0.5}

    monkeypatch.setattr(evaluate_w, "evaluate", fake_evaluate)
    monkeypatch.setattr(evaluate_w, "apply_device_env", lambda d: None)
    base = ["--weights", "w.pt", "--dataset", str(spaced_tmp), "--out", str(spaced_tmp / "o"), "--device", "cpu"]
    evaluate_w.main(base + ["--mm-per-px", "12.5"])
    assert seen["args"][5:7] == (12.5, "user-supplied")
    monkeypatch.setattr(evaluate_w, "default_scale", lambda: (10.0, "constant"))
    evaluate_w.main(base)
    assert seen["args"][5:7] == (10.0, "constant")


def test_real_difference_rule():
    assert real_difference(0.10, 0.05, 0.15, 0.01, 0.01)["real"] is True
    assert real_difference(-0.10, -0.15, -0.05, 0.01, 0.01)["real"] is True
    assert real_difference(0.10, -0.01, 0.20, 0.01, 0.01)["real"] is False  # CI contains 0
    r = real_difference(0.10, 0.05, 0.15, 0.06, 0.06)  # 2 * pooled std = 0.12 > |delta|
    assert r["pooled_std"] == pytest.approx(0.06) and r["real"] is False
    assert real_difference(0.10, 0.05, 0.15, math.nan, 0.01)["real"] is False  # one seed: no std


# Runs in a child process: Ultralytics leaves threads behind that make later fork()-based
# tests warn, and CUDA_VISIBLE_DEVICES="" is only reliable when set before the process starts.
_SMOKE_SCRIPT = """
import json, sys
from pathlib import Path
import numpy as np
from fpyolo.evaluate_d import predict_split
from fpyolo.evaluate_w import max_score_map, predict_stream
d, w, img = map(Path, sys.argv[1:4])
boxes, cls, scores = predict_split(d, [img], device="cpu")[img.stem]
[(path, r)] = list(predict_stream(w, [img], device="cpu"))
s = max_score_map(r, 999, 1000)
print(json.dumps({"boxes": boxes.shape, "cls": cls.shape, "scores": scores.shape,
                  "path": str(path), "orig_shape": r.orig_shape, "map_shape": s.shape,
                  "map_dtype": str(s.dtype), "min": float(s.min()), "max": float(s.max())}))
"""


@pytest.mark.skipif(not (SMOKE_D.is_file() and SMOKE_W.is_file()),
                    reason="needs the Task 11 CPU smoke weights in runs/smoke")
def test_smoke_weights_predict_on_cpu(spaced_tmp):
    rgb = np.full((999, 1000, 3), 255, np.uint8)
    rgb[100:900:50, 50:950] = 0  # a few horizontal lines
    img = spaced_tmp / "test-00_0000-0001.png"
    Image.fromarray(rgb).save(img)
    proc = subprocess.run([sys.executable, "-c", _SMOKE_SCRIPT, str(SMOKE_D), str(SMOKE_W), str(img)],
                          env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True,
                          text=True, check=True)
    r = json.loads(proc.stdout.strip().splitlines()[-1])
    n = r["cls"][0]
    assert r["boxes"] == [n, 4] and r["scores"] == [n] and n <= 300
    assert r["path"] == str(img) and r["orig_shape"] == [999, 1000]
    assert r["map_shape"] == [999, 1000] and r["map_dtype"] == "float32"
    assert 0 <= r["min"] <= r["max"] <= 1


_RSS_SCRIPT = """
import json, resource, shutil, sys
from pathlib import Path
from fpyolo.evaluate_d import predict_split
weights, img, k, tmp = Path(sys.argv[1]), Path(sys.argv[2]), int(sys.argv[3]), Path(sys.argv[4])
imgs = []
for i in range(k):
    imgs.append(tmp / f"test-00_{i:04d}-0000.png")
    shutil.copy(img, imgs[-1])
predict_split(weights, imgs, device="cpu")
print(json.dumps({"maxrss_mib": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024}))
"""


@pytest.mark.skipif(os.environ.get("FPYOLO_MEMCHECK") != "1" or not SMOKE_D.is_file(),
                    reason="opt-in (FPYOLO_MEMCHECK=1), needs the Task 11 smoke weights")
def test_predict_peak_rss_does_not_grow_with_images(spaced_tmp):
    # The one-batch-per-list bug added ~57 MB of peak RSS per listed image on CPU.
    rgb = np.full((1000, 1000, 3), 255, np.uint8)
    rgb[100:900:50, 50:950] = 0
    Image.fromarray(rgb).save(spaced_tmp / "src.png")
    peaks = []
    for k in (2, 24):
        tmp = spaced_tmp / f"k{k}"
        tmp.mkdir()
        proc = subprocess.run([sys.executable, "-c", _RSS_SCRIPT, str(SMOKE_D), str(spaced_tmp / "src.png"),
                               str(k), str(tmp)], env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
                              capture_output=True, text=True, check=True)
        peaks.append(json.loads(proc.stdout.strip().splitlines()[-1])["maxrss_mib"])
    assert peaks[1] - peaks[0] < 300, peaks
