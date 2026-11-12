import json
import math

import numpy as np
import pytest

from fpyolo.evaluate_w import (
    THICKNESS_METHOD, ThicknessErrors, default_mm_per_px, mask_thickness_px,
)


def _bar(h, w, y0, y1, x0, x1):
    m = np.zeros((h, w), bool)
    m[y0:y1, x0:x1] = True
    return m


def test_bar_thickness_is_close_to_its_width():
    t = mask_thickness_px(_bar(60, 500, 20, 30, 20, 420))  # 10 px x 400 px
    assert t == pytest.approx(10.0, rel=0.05)
    assert mask_thickness_px(_bar(500, 60, 20, 420, 20, 30)) == pytest.approx(t)  # rotation by 90 deg


def test_thickness_scales_with_width():
    t10 = mask_thickness_px(_bar(80, 500, 20, 30, 20, 420))
    t20 = mask_thickness_px(_bar(80, 500, 20, 40, 20, 420))
    assert t20 / t10 == pytest.approx(2.0, rel=0.05)


def test_thickness_counts_hole_contours_in_the_perimeter():
    solid = _bar(100, 100, 10, 90, 10, 90)
    ring = solid.copy()
    ring[40:60, 40:60] = False
    # a hole removes area and adds perimeter: a thinner object than the solid square
    assert mask_thickness_px(ring) < mask_thickness_px(solid)


def test_thickness_of_empty_or_degenerate_mask_is_none():
    assert mask_thickness_px(np.zeros((10, 10), bool)) is None
    one = np.zeros((10, 10), bool)
    one[5, 5] = True  # zero-length contour
    assert mask_thickness_px(one) is None


def test_identical_masks_have_zero_error():
    m = _bar(60, 500, 20, 30, 20, 420)
    te = ThicknessErrors()
    te.add(m, m.astype(np.float32) * 0.9, t_star=0.5)
    s = te.summary(mm_per_px=10.0)
    assert s["mae"] == 0 and s["median_abs"] == 0 and s["bias"] == 0 and s["n_images"] == 1


def test_error_is_pred_minus_gt_in_mm():
    gt = _bar(80, 500, 20, 30, 20, 420)       # ~10 px
    pred = _bar(80, 500, 20, 40, 20, 420)     # ~20 px, thicker
    te = ThicknessErrors()
    te.add(gt, pred.astype(np.float32), t_star=0.5)
    s = te.summary(mm_per_px=10.0)
    assert s["n_images"] == 1 and s["mm_per_px"] == 10.0
    assert s["bias"] == pytest.approx(10.0 * 10.0, rel=0.1)  # +10 px * 10 mm/px
    assert s["mae"] == pytest.approx(s["bias"]) and s["median_abs"] == pytest.approx(s["bias"])
    assert isinstance(s["method"], str) and s["method"] == THICKNESS_METHOD


def test_threshold_applies_to_score_map_and_empty_sides_are_skipped():
    gt = _bar(60, 500, 20, 30, 20, 420)
    te = ThicknessErrors()
    te.add(gt, gt.astype(np.float32) * 0.3, t_star=0.5)            # prediction empty at t*
    te.add(np.zeros_like(gt), gt.astype(np.float32), t_star=0.5)   # GT empty
    s = te.summary(mm_per_px=10.0)
    assert s["n_images"] == 0
    assert all(math.isnan(s[k]) for k in ("mae", "median_abs", "bias"))


def test_summary_aggregates_over_images():
    te = ThicknessErrors()
    for w_pred in (10, 14, 6):
        te.add(_bar(80, 500, 20, 30, 20, 420), _bar(80, 500, 20, 20 + w_pred, 20, 420).astype(np.float32), 0.5)
    s = te.summary(mm_per_px=2.0)
    assert s["n_images"] == 3
    assert s["bias"] == pytest.approx(0.0, abs=2.0)
    assert s["mae"] > 0 and s["mae"] >= s["median_abs"] * 0.5


def test_unknown_scale_gives_no_mm_values():
    m = _bar(60, 500, 20, 30, 20, 420)
    te = ThicknessErrors()
    te.add(m, m.astype(np.float32), 0.5)
    s = te.summary(mm_per_px=None)
    assert s["mm_per_px"] is None and s["n_images"] == 1
    assert all(math.isnan(s[k]) for k in ("mae", "median_abs", "bias"))


def test_default_mm_per_px_reads_scale_json(tmp_path):
    p = tmp_path / "scale.json"
    p.write_text(json.dumps({"overall_median": 10.0, "conclusion": "constant"}), encoding="utf-8")
    assert default_mm_per_px(p) == 10.0
    assert default_mm_per_px(tmp_path / "missing.json") is None
