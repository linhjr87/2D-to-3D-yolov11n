"""Tests for the COCO-style AP evaluator and the cluster bootstrap (fpyolo.apeval)."""
import contextlib
import io
import os
import time

import numpy as np
import pytest

from fpyolo.apeval import ap_table, bootstrap_delta, build_cache, summarize

AREAS = ("all", "small", "medium", "large")


def gt(boxes, labels):
    return np.asarray(boxes, float).reshape(-1, 4), np.asarray(labels, int)


def pred(boxes, labels, scores):
    return np.asarray(boxes, float).reshape(-1, 4), np.asarray(labels, int), np.asarray(scores, float)


def random_boxes(rng, n, size=640.0):
    wh = np.exp(rng.uniform(np.log(4), np.log(200), (n, 2)))  # small, medium and large boxes
    xy = rng.uniform(0, size - 200, (n, 2))
    return np.c_[xy, xy + wh]


def random_data(seed=0, n_images=20, n_classes=3):
    """Random labels and noisy detections: jittered copies (some relabelled, some
    duplicated), false positives, an image without predictions, images without
    labels, and one image with more than 300 detections of a class."""
    rng = np.random.default_rng(seed)
    gts, preds = {}, {}
    for i in range(n_images):
        key = f"img{i:02d}"
        n = int(rng.integers(0, 8))
        boxes, labels = random_boxes(rng, n), rng.integers(0, n_classes, n)
        gts[key] = (boxes, labels)
        src = np.repeat(np.arange(n), rng.integers(0, 3, n))
        wh = boxes[src, 2:] - boxes[src, :2]
        jit = boxes[src] + rng.normal(0, 0.12, (len(src), 4)) * np.tile(wh, 2)
        jit = np.c_[np.minimum(jit[:, :2], jit[:, 2:]), np.maximum(jit[:, :2], jit[:, 2:])]
        relabel = rng.random(len(src)) < 0.15
        dl = np.where(relabel, rng.integers(0, n_classes, len(src)), labels[src])
        m = int(rng.integers(0, 6))
        db = np.r_[jit, random_boxes(rng, m)]
        dl = np.r_[dl, rng.integers(0, n_classes, m)]
        preds[key] = (db, dl, rng.random(len(dl)))
    del preds["img03"]  # an image with labels but no prediction entry at all
    crowded = random_boxes(rng, 320)
    b, l, s = preds["img19"]
    preds["img19"] = (np.r_[b, crowded], np.r_[l, np.zeros(320, int)], np.r_[s, rng.random(320)])
    return gts, preds


def coco_eval(gts, preds, n_classes=3):
    """Run pycocotools COCOeval (bbox, maxDets [1, 10, 300]) on the same data."""
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    ids = {k: i + 1 for i, k in enumerate(sorted(set(gts) | set(preds)))}
    anns = []
    for k, (boxes, labels) in gts.items():
        for b, c in zip(boxes, labels):
            w, h = b[2] - b[0], b[3] - b[1]
            anns.append({"id": len(anns) + 1, "image_id": ids[k], "category_id": int(c) + 1,
                         "bbox": [float(b[0]), float(b[1]), float(w), float(h)],
                         "area": float(w * h), "iscrowd": 0})
    dts = [{"image_id": ids[k], "category_id": int(c) + 1,
            "bbox": [float(b[0]), float(b[1]), float(b[2] - b[0]), float(b[3] - b[1])],
            "score": float(s)}
           for k, (boxes, labels, scores) in preds.items()
           for b, c, s in zip(boxes, labels, scores)]
    with contextlib.redirect_stdout(io.StringIO()):  # pycocotools prints progress
        coco_gt = COCO()
        coco_gt.dataset = {"images": [{"id": i} for i in ids.values()],
                           "categories": [{"id": c + 1} for c in range(n_classes)],
                           "annotations": anns}
        coco_gt.createIndex()
        ev = COCOeval(coco_gt, coco_gt.loadRes(dts), "bbox")
        ev.params.maxDets = [1, 10, 300]
        ev.evaluate()
        ev.accumulate()
        ev.summarize()
    return ev


def coco_ap(ev, area="all", iou=None, cls=None):
    """Mean of COCOeval's precision array at maxDets=300, the way COCOeval.summarize
    averages it. Needed for mAP50-95: pycocotools pins stats[0] to maxDets=100,
    which is absent from [1, 10, 300], so stats[0] is -1 there."""
    s = ev.eval["precision"][:, :, :, AREAS.index(area), -1]
    if iou is not None:
        s = s[np.flatnonzero(ev.params.iouThrs == iou)]
    if cls is not None:
        s = s[:, :, cls]
    return float(np.mean(s[s > -1]))


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_pycocotools(seed):
    gts, preds = random_data(seed)
    coco = coco_eval(gts, preds)
    assert coco.params.maxDets[-1] == 300
    s = summarize(build_cache(gts, preds), core=[0, 1, 2], all_classes=[0, 1, 2])
    assert 0 < s["map50_95_all"] < 1
    assert coco.stats[3] > 0  # the data has small boxes, so the size check is not vacuous
    assert s["map50_95_all"] == pytest.approx(coco_ap(coco), abs=1e-6)
    assert s["map50_all"] == pytest.approx(coco.stats[1], abs=1e-6)
    assert s["map75_all"] == pytest.approx(coco.stats[2], abs=1e-6)
    assert s["by_size"]["small"] == pytest.approx(coco.stats[3], abs=1e-6)
    assert s["by_size"]["medium"] == pytest.approx(coco.stats[4], abs=1e-6)
    assert s["by_size"]["large"] == pytest.approx(coco.stats[5], abs=1e-6)
    for c in range(3):
        assert s["per_class"][c]["ap50_95"] == pytest.approx(coco_ap(coco, cls=c), abs=1e-6)
        assert s["per_class"][c]["ap50"] == pytest.approx(coco_ap(coco, iou=0.5, cls=c), abs=1e-6)
        assert s["per_class"][c]["n_gt"] == sum(int((l == c).sum()) for _, l in gts.values())


def tie_data(seed, n_images=25, n_classes=3):
    """Integer boxes on a grid (IoU exactly at thresholds, areas exactly 32**2 and
    96**2), duplicated label boxes (IoU ties) and one-decimal scores (score ties)."""
    rng = np.random.default_rng(seed)
    gts, preds = {}, {}
    for i in range(n_images):
        n = int(rng.integers(2, 10))
        xy = rng.integers(0, 20, (n, 2)) * 10.0
        boxes = np.c_[xy, xy + rng.choice([8, 16, 32, 40, 64, 96, 100, 128], (n, 2))]
        boxes[1] = boxes[0]
        labels = rng.integers(0, n_classes, n)
        labels[1] = labels[0]
        gts[f"t{i:02d}"] = (boxes, labels)
        src = np.repeat(np.arange(n), rng.integers(0, 4, n))
        jit = boxes[src] + rng.integers(-4, 5, (len(src), 4)) * 2.0
        jit = np.c_[np.minimum(jit[:, :2], jit[:, 2:] - 1), np.maximum(jit[:, 2:], jit[:, :2] + 1)]
        m = int(rng.integers(0, 5))
        fxy = rng.integers(0, 20, (m, 2)) * 10.0
        db = np.r_[jit, np.c_[fxy, fxy + rng.choice([8, 32, 96, 120], (m, 2))]]
        dl = np.r_[labels[src], rng.integers(0, n_classes, m)]
        preds[f"t{i:02d}"] = (db, dl, np.round(rng.random(len(dl)), 1))
    return gts, preds


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matches_pycocotools_with_ties(seed):
    gts, preds = tie_data(seed)
    coco, cache = coco_eval(gts, preds), build_cache(gts, preds)
    for a, area in enumerate(AREAS):
        prec = coco.eval["precision"][:, :, :, a, -1]  # (T, R, K)
        ref = np.where((prec > -1).all(axis=1), prec.mean(axis=1), np.nan).T
        np.testing.assert_allclose(ap_table(cache, [0, 1, 2], area=area), ref, rtol=0, atol=1e-12)


def test_iou_tie_goes_to_the_later_label():
    # d1 overlaps both labels with IoU 0.6; COCOeval gives it the later label, so d2
    # (an exact copy of the first label) is a true positive too. First-label-wins
    # would leave d2 unmatched (IoU 1/3 with the second label).
    gts = {"a": gt([[0, 0, 10, 10], [5, 0, 15, 10]], [0, 0])}
    preds = {"a": pred([[2.5, 0, 12.5, 10], [0, 0, 10, 10]], [0, 0], [0.9, 0.8])}
    table = ap_table(build_cache(gts, preds), [0])
    assert table[0, :3] == pytest.approx([1, 1, 1])  # IoU 0.50, 0.55, 0.60
    # above 0.6, d1 is a FP ranked before the TP d2: precision 1/2 up to recall 1/2
    assert table[0, 3:] == pytest.approx(np.full(7, 51 * 0.5 / 101))
    prec = coco_eval(gts, preds, n_classes=1).eval["precision"][:, :, 0, 0, -1]
    np.testing.assert_allclose(table[0], prec.mean(axis=1), rtol=0, atol=1e-12)


def test_area_range_prefers_labels_in_range():
    # For "small", the 34x34 label is out of range (ignored). The 33x33 detection has a
    # higher IoU with it (0.94) than with the 30x30 small label (0.83) but must match the
    # small label while IoU 0.83 clears the threshold: a TP although its own area is
    # outside the range.
    gts = {"a": gt([[0, 0, 30, 30], [0, 0, 34, 34]], [0, 0])}
    preds = {"a": pred([[0, 0, 33, 33]], [0], [0.9])}
    table = ap_table(build_cache(gts, preds), [0], area="small")
    assert table[0] == pytest.approx([1] * 7 + [0] * 3)  # IoU 0.50 ... 0.80 vs 0.85+
    prec = coco_eval(gts, preds, n_classes=1).eval["precision"][:, :, 0, 1, -1]
    np.testing.assert_allclose(table[0], prec.mean(axis=1), rtol=0, atol=1e-12)


def test_perfect_predictions():
    gts, _ = random_data(0)
    rng = np.random.default_rng(1)
    preds = {k: (b, l, rng.random(len(l))) for k, (b, l) in gts.items()}
    cache = build_cache(gts, preds)
    for area in AREAS:
        table = ap_table(cache, [0, 1, 2], area=area)
        assert table.shape == (3, 10)
        # COCO's precision tp / (tp + fp + eps) is 1 - 2e-16 at a single true positive
        assert table[~np.isnan(table)] == pytest.approx(1.0, abs=1e-12)
    assert not np.isnan(ap_table(cache, [0, 1, 2], area="small")).all()
    s = summarize(cache, core=[0, 1, 2], all_classes=[0, 1, 2])
    assert s["headline_map50_95"] == pytest.approx(1.0, abs=1e-12)
    assert s["map50_95_all"] == pytest.approx(1.0, abs=1e-12)


def test_image_without_predictions():
    gts = {"a": gt([[10, 10, 60, 60]], [0])}
    preds = {"a": pred([[10, 10, 60, 60]], [0], [0.9])}
    assert ap_table(build_cache(gts, preds), [0]) == pytest.approx(np.ones((1, 10)))
    gts["b"] = gt([[0, 0, 40, 40], [100, 100, 150, 150], [200, 200, 230, 260]], [0, 0, 0])
    table = ap_table(build_cache(gts, preds), [0])
    # recall tops out at 1/4 with precision 1: recall thresholds 0.00..0.25 score 1, the rest 0
    assert np.all(np.isfinite(table))
    assert table == pytest.approx(np.full((1, 10), 26 / 101))


def test_predictions_without_gt():
    gts = {"a": gt([[10, 10, 60, 60]], [0]), "b": gt(np.zeros((0, 4)), [])}
    preds = {"a": pred([[10, 10, 60, 60]], [0], [0.5]),
             "b": pred([[300, 300, 340, 340]], [0], [0.9])}
    # the higher-scored detection on the unlabelled image is a false positive: precision 1/2
    table = ap_table(build_cache(gts, preds), [0])
    assert table == pytest.approx(np.full((1, 10), 0.5))
    # also when the image is missing from gts altogether
    del gts["b"]
    assert ap_table(build_cache(gts, preds), [0]) == pytest.approx(np.full((1, 10), 0.5))


def test_missing_class_listed_not_averaged():
    gts = {"a": gt([[10, 10, 60, 60], [100, 100, 150, 150]], [0, 1]),
           "b": gt([[10, 10, 60, 60]], [1])}
    preds = {"a": pred([[10, 10, 60, 60], [100, 100, 150, 150], [400, 400, 450, 450]],
                       [0, 1, 2], [0.9, 0.5, 0.7]),
             "b": pred([[300, 300, 350, 350], [10, 10, 60, 60]], [1, 1], [0.8, 0.4])}
    cache = build_cache(gts, preds)
    s = summarize(cache, core=[0, 1], all_classes=[0, 1, 2])
    assert s["missing_classes"] == [2]
    assert np.isnan(s["per_class"][2]["ap50_95"]) and s["per_class"][2]["n_gt"] == 0
    # class 0: AP 1; class 1: FP(0.8), TP(0.5), TP(0.4) -> precision 1/2 then 2/3 at recall 1
    assert s["per_class"][1]["ap50_95"] == pytest.approx(2 / 3)
    assert s["map50_95_all"] == pytest.approx((1 + 2 / 3) / 2)  # class 2 is not a zero in the mean
    assert s["headline_map50_95"] == pytest.approx(s["map50_95_core"])
    with pytest.raises(ValueError, match="2"):
        summarize(cache, core=[0, 2], all_classes=[0, 1, 2])


def test_weights_equal_duplication():
    """Tie-free by construction: random float scores never tie within an image, so
    weighting equals literal duplication through build_cache."""
    gts, preds = random_data(0)
    cache = build_cache(gts, preds)
    weights = {k: 1 for k in gts}
    weights["img00"] = 2  # counted twice
    weights["img02"] = 3  # counted three times
    weights["img01"] = 0  # dropped
    del weights["img04"]  # absent from weights -> dropped
    gts_dup, preds_dup = dict(gts), dict(preds)
    for k, copies in (("img00", 1), ("img02", 2)):
        for j in range(copies):
            gts_dup[f"{k}~{j}"] = gts[k]
            preds_dup[f"{k}~{j}"] = preds[k]
    for k in ("img01", "img04"):
        del gts_dup[k], preds_dup[k]
    dup = build_cache(gts_dup, preds_dup)
    for area in AREAS:
        np.testing.assert_allclose(ap_table(cache, [0, 1, 2], area=area, weights=weights),
                                   ap_table(dup, [0, 1, 2], area=area), rtol=0, atol=1e-12)


def test_weights_convention_at_within_image_score_ties():
    """Weight k keeps each detection's k copies adjacent in score order; literal
    duplication keeps image blocks instead, so the two differ when detections of one
    image and class tie on score."""
    gts = {"a": gt([[0, 0, 50, 50], [100, 100, 150, 150]], [0, 0]), "b": gt([[0, 0, 50, 50]], [0])}
    preds = {"a": pred([[0, 0, 50, 50], [300, 300, 350, 350]], [0, 0], [0.5, 0.5]),  # TP, FP tied
             "b": pred([[0, 0, 50, 50]], [0], [0.4])}
    weighted = ap_table(build_cache(gts, preds), [0], weights={"a": 2, "b": 1})
    # copies adjacent: TP TP FP FP TP -> precision 1 up to recall 2/5, then 3/5 at recall 3/5:
    # AP50 = (41 * 1 + 20 * 3/5) / 101 = 53/101 = 0.5248
    assert weighted[0, 0] == pytest.approx(53 / 101, abs=1e-12)
    # literal duplication ("a~" sorts right after "a"): TP FP TP FP TP ->
    # AP50 = (21 * 1 + 20 * 2/3 + 20 * 3/5) / 101 = 139/303 = 0.4587
    dup = build_cache({**gts, "a~": gts["a"]}, {**preds, "a~": preds["a"]})
    assert ap_table(dup, [0])[0, 0] == pytest.approx(139 / 303, abs=1e-12)


def test_bootstrap_identical_models():
    gts, preds = random_data(0)
    cache = build_cache(gts, preds)
    cluster_of = {k: i % 6 for i, k in enumerate(sorted(gts))}
    r = bootstrap_delta([cache] * 3, [cache] * 3, core=[0, 1, 2], cluster_of=cluster_of, n=200)
    assert (r["delta"], r["ci_low"], r["ci_high"]) == (0.0, 0.0, 0.0)


def test_bootstrap_sign_and_degenerate_resamples():
    gts = {f"im{i}": gt([[10, 10, 60, 60]], [0]) for i in range(8)}
    gts["im0"] = gt([[10, 10, 60, 60], [100, 100, 150, 150]], [0, 1])  # class 1 in one cluster only
    perfect = {k: (b, l, np.ones(len(l))) for k, (b, l) in gts.items()}
    cluster_of = {f"im{i}": i // 2 for i in range(8)}
    r = bootstrap_delta([build_cache(gts, perfect)], [build_cache(gts, {})], core=[0, 1],
                        cluster_of=cluster_of, n=200, seed=0)
    # A is perfect, B predicts nothing: every resample gives A - B = 1, even those without class 1
    assert (r["delta"], r["ci_low"], r["ci_high"]) == (1.0, 1.0, 1.0)
    assert 0 < r["n_degenerate"] < 200


def test_bootstrap_resamples_without_any_core_class():
    # the only core class (1) lives in cluster 0 of 4: resamples that miss cluster 0 have
    # no headline, are counted as degenerate and left out of the percentiles
    gts = {f"im{i}": gt([[10, 10, 60, 60]], [0]) for i in range(8)}
    gts["im0"] = gt([[10, 10, 60, 60], [100, 100, 150, 150]], [0, 1])
    perfect = {k: (b, l, np.ones(len(l))) for k, (b, l) in gts.items()}
    a, b = build_cache(gts, perfect), build_cache(gts, {})
    cluster_of = {f"im{i}": i // 2 for i in range(8)}
    r = bootstrap_delta([a], [b], core=[1], cluster_of=cluster_of, n=200, seed=0)
    # 1 - 2e-16: a single true positive has COCO precision 1 / (1 + eps)
    assert (r["delta"], r["ci_low"], r["ci_high"]) == pytest.approx((1.0, 1.0, 1.0), abs=1e-12)
    assert 0 < r["n_degenerate"] < 200
    # with one resample per seed, the seeds that miss cluster 0 leave no finite delta
    raised = 0
    for seed in range(10):
        try:
            bootstrap_delta([a], [b], core=[1], cluster_of=cluster_of, n=1, seed=seed)
        except ValueError as e:
            assert "finite" in str(e)
            raised += 1
    assert 0 < raised < 10


def test_bootstrap_is_seeded():
    gts, preds = random_data(0)
    gts_b, preds_b = random_data(1)
    preds_b = {k: v for k, v in preds_b.items() if k in gts}
    a, b = build_cache(gts, preds), build_cache(gts, preds_b)
    cluster_of = {k: i % 6 for i, k in enumerate(sorted(gts))}
    r1 = bootstrap_delta([a], [b], core=[0, 1, 2], cluster_of=cluster_of, n=100, seed=3)
    r2 = bootstrap_delta([a], [b], core=[0, 1, 2], cluster_of=cluster_of, n=100, seed=3)
    assert r1 == r2
    assert r1["ci_low"] < r1["ci_high"]
    point = (summarize(a, [0, 1, 2], [0, 1, 2])["headline_map50_95"]
             - summarize(b, [0, 1, 2], [0, 1, 2])["headline_map50_95"])
    assert r1["delta"] == pytest.approx(point, abs=1e-12)


def full_size_gts(n_images=2700, n_classes=30, seed=0):
    """Test-split-sized labels: ~7.5 boxes per image over skewed classes."""
    rng = np.random.default_rng(seed)
    p = rng.dirichlet(np.full(n_classes, 0.7))
    gts = {}
    for i in range(n_images):
        n = int(rng.poisson(7.5))
        gts[f"test-00_{i:04d}"] = (random_boxes(rng, n, size=1000.0), rng.choice(n_classes, n, p=p))
    return gts, p


def full_size_preds(gts, p, seed, dets_per_image=300):
    """1-3 jittered detections per label plus low-score false positives, 300 per image."""
    rng = np.random.default_rng(seed)
    preds = {}
    for key, (boxes, labels) in gts.items():
        src = np.repeat(np.arange(len(labels)), rng.integers(1, 4, len(labels)))
        wh = boxes[src, 2:] - boxes[src, :2]
        jit = boxes[src] + rng.normal(0, 0.08, (len(src), 4)) * np.tile(wh, 2)
        jit = np.c_[np.minimum(jit[:, :2], jit[:, 2:]), np.maximum(jit[:, :2], jit[:, 2:])]
        m = dets_per_image - len(src)
        preds[key] = (np.r_[jit, random_boxes(rng, m, size=1000.0)],
                      np.r_[labels[src], rng.choice(len(p), m, p=p)],
                      np.r_[rng.uniform(0.2, 1.0, len(src)), rng.uniform(0.001, 0.4, m)])
    return preds


@pytest.mark.skipif(os.environ.get("FPYOLO_TIMING") != "1",
                    reason="full-size bootstrap timing; run with FPYOLO_TIMING=1")
def test_bootstrap_full_size_timing():
    gts, p = full_size_gts()
    caches = []
    for seed in range(6):
        preds = full_size_preds(gts, p, seed)
        t0 = time.perf_counter()
        caches.append(build_cache(gts, preds))
        print(f"build_cache seed {seed}: {time.perf_counter() - t0:.1f} s")
    cluster_of = {k: i % 1000 for i, k in enumerate(sorted(gts))}
    t0 = time.perf_counter()
    r = bootstrap_delta(caches[:3], caches[3:], core=range(27), cluster_of=cluster_of, n=1000)
    elapsed = time.perf_counter() - t0
    print(f"bootstrap_delta 3+3 caches, n=1000: {elapsed:.1f} s; {r}")
    assert elapsed < 120
