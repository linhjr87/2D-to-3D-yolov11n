import json
import math
from pathlib import Path

import pytest

from conftest import CONFIGS
from fpyolo import report
from fpyolo.report import ReportError, build_report, fmt_mean_std
from fpyolo.schema import load_schema

SCHEMA = load_schema(CONFIGS / "classes_v0.yaml")
D_DEFS = SCHEMA.for_model("D")
CORE_SIDS = [s for s in range(3, 33) if s not in (6, 7, 8)]  # 6, 7, 8: not core in the fixture
ABSENT = "rolling_door"  # no test instances in the fixture


def _per_class(base: float, jitter: float) -> dict:
    out = {}
    for i, c in enumerate(D_DEFS):
        absent = c.name == ABSENT
        ap = None if absent else max(0.0, base - 0.01 * i + jitter)
        out[f"{i}_{c.name}"] = {"name": c.name, "schema_id": c.schema_id, "core": c.semantic_id in CORE_SIDS,
                                "ap50_95": ap, "ap50": None if absent else (ap or 0) + 0.2,
                                "n_gt": 0 if absent else 100 + i}
    return out


def _d_metrics(headline, jitter=0.0, **extra):
    m = {"headline_map50_95": headline, "map50_95_core": headline, "map50_core": headline + 0.2,
         "map75_core": headline + 0.05, "map50_95_all": headline - 0.03, "map50_all": headline + 0.15,
         "map75_all": headline + 0.01, "per_class": _per_class(0.7, jitter),
         "by_size": {"small": 0.3 + jitter, "medium": 0.5 + jitter, "large": 0.7 + jitter},
         "missing_classes": [f"5_{ABSENT}"], "core_reduced": False, "limit": None, "split": "test",
         "ultralytics_val": {"speed_ms": {"preprocess": 1.0, "inference": 4.0, "postprocess": 2.0},
                             "precision": headline + 0.1, "recall": headline - 0.1, "device_name": "GPU X"}}
    m.update(extra)
    return m


def _w_metrics(iou, **extra):
    m = {"t_star": 0.35, "val_iou_at_t_star": iou - 0.01, "test_pixel_iou": iou, "mask_map50_95": 0.4 + iou / 10,
         "mask_map50": 0.7, "per_class_pixel_iou": {"wall": iou, "curtain_wall": 0.5, "railing": None},
         "limit": None, "wall_thickness_mm": {"mae": 20.0 + iou, "median_abs": 15.0, "bias": -2.0,
                                              "n_images": 2000, "mm_per_px": 10.0, "scale_conclusion": "constant",
                                              "method": "2*area/perimeter"}}
    m.update(extra)
    return m


def _write(results: Path, name: str, metrics: dict):
    (results / name).mkdir(parents=True)
    (results / name / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")


STATS = {
    "dedup": {"n_tiles": 15663, "n_sets": 1163, "n_tiles_in_sets": 3474, "kept": 13352, "dropped": 2311,
              "agree_things_frac": 0.5993, "agree_walls_frac": 0.4557, "missing_instance_frac": 0.2162},
    "clusters": {"n": 1108, "largest": 187, "largest_frac": 0.014, "top10_sizes": [187, 173],
                 "clusters_per_set": {"train": 952, "val": 40, "test": 116}, "n_pairs": 9166},
    "split_report": {"tiles": {"train": 9341, "val": 1341, "test": 2670},
                     "tile_ratio": {"train": 0.7, "val": 0.1, "test": 0.2}},
    "box_stats": {"n_boxes": 99755, "padded": 59, "small_lt_32px": 8646, "negative_tiles": 3875},
    "w_coverage": {"coverage_min": 0.8, "n_wall_tiles": 12220, "n_polygons_in_W": 56317,
                   "excluded": {"train": {"wall_tiles": 8612, "excluded": 899, "frac": 0.1044},
                                "val": {"wall_tiles": 1189, "excluded": 128, "frac": 0.1077},
                                "test": {"wall_tiles": 2419, "excluded": 239, "frac": 0.0988},
                                "all": {"wall_tiles": 12220, "excluded": 1266, "frac": 0.1036}}},
    "w_candidates": {"2": {"n_tiles": 1843, "median_coverage": 1.0, "enabled": True},
                     "33": {"n_tiles": 2376, "median_coverage": 1.0, "enabled": True}},
}


@pytest.fixture
def tree(tmp_path):
    results, man = tmp_path / "results", tmp_path / "manifests"
    results.mkdir()
    man.mkdir()
    (man / "stats.json").write_text(json.dumps(STATS), encoding="utf-8")
    (man / "core_classes.json").write_text(json.dumps({"min_train": 50, "min_test": 15,
                                                       "semantic_ids": CORE_SIDS}), encoding="utf-8")
    return results, man


def _full(results):
    for s, (a, b, w) in enumerate([(0.610, 0.570, 0.80), (0.616, 0.572, 0.81), (0.610, 0.568, 0.79)]):
        _write(results, f"d11n_s{s}", _d_metrics(a, jitter=0.01))
        _write(results, f"d8n_s{s}", _d_metrics(b, jitter=-0.01))
        _write(results, f"w11n_s{s}", _w_metrics(w))
    (results / "compare.json").write_text(json.dumps({
        "a": {"runs": [str(results / f"d11n_s{s}") for s in range(3)], "headlines": [0.61, 0.616, 0.61],
              "mean": 0.612, "std": 0.0035},
        "b": {"runs": [str(results / f"d8n_s{s}") for s in range(3)], "headlines": [0.57, 0.572, 0.568],
              "mean": 0.57, "std": 0.002},
        "bootstrap": {"delta": 0.042, "ci_low": 0.03, "ci_high": 0.055, "n_degenerate": 0, "n": 1000, "seed": 0},
        "pooled_std": 0.003, "ci_excludes_zero": True, "exceeds_2x_pooled_std": True, "real": True,
        "limit": None, "split": "test"}), encoding="utf-8")


def test_fmt_mean_std():
    assert fmt_mean_std([0.610, 0.616, 0.610]) == "0.612 ± 0.003"
    assert fmt_mean_std([0.5]) == "0.500 ± n/a"
    assert fmt_mean_std([]) == "missing"
    assert fmt_mean_std([None, math.nan]) == "n/a"
    assert fmt_mean_std([20.0, 22.0], digits=1) == "21.0 ± 1.4"


def test_headline_shows_mean_and_sample_std(tree):
    results, man = tree
    for s, v in enumerate((0.608, 0.612, 0.616)):  # mean 0.612, ddof=1 std 0.004
        _write(results, f"d11n_s{s}", _d_metrics(v))
    text = build_report(results, man)
    assert "0.612 ± 0.004" in text


def test_compare_result_is_reported(tree):
    results, man = tree
    _full(results)
    text = build_report(results, man)
    assert "+0.042" in text and "[+0.030, +0.055]" in text
    assert "real" in text.lower() and "yes" in text


def test_precision_recall_from_ultralytics_val(tree):
    results, man = tree
    for s, v in enumerate((0.608, 0.612, 0.616)):
        _write(results, f"d11n_s{s}", _d_metrics(v))
    text = build_report(results, man)
    assert "Precision (Ultralytics averaging)" in text and "Recall (Ultralytics averaging)" in text
    row = next(line for line in text.splitlines() if line.startswith("| YOLO11n |"))
    assert "0.712 ± 0.004" in row and "0.512 ± 0.004" in row  # precision, recall
    assert "| YOLOv8n | 0/3 |" in text and "missing" in text


@pytest.mark.parametrize("side,runs", [("a", ["d11n_s0", "d11n_s1"]), ("b", ["d8n_s0", "d8n_s1", "d8n_s9"]),
                                       ("a", ["d8n_s0", "d8n_s1", "d8n_s2"])])
def test_compare_runs_must_match_loaded_runs(tree, side, runs):
    results, man = tree
    _full(results)
    c = json.loads((results / "compare.json").read_text())
    c[side]["runs"] = [str(Path("/elsewhere") / r) for r in runs]
    (results / "compare.json").write_text(json.dumps(c))
    with pytest.raises(ReportError, match=f"side {side.upper()}"):
        build_report(results, man)


def test_compare_runs_match_by_basename(tree):
    results, man = tree
    _full(results)
    c = json.loads((results / "compare.json").read_text())
    c["a"]["runs"] = [f"/other/machine/results/d11n_s{s}" for s in (2, 0, 1)]  # other root, other order
    (results / "compare.json").write_text(json.dumps(c))
    assert "+0.042" in build_report(results, man)


def test_compare_not_real_says_no(tree):
    results, man = tree
    _full(results)
    c = json.loads((results / "compare.json").read_text())
    c["real"] = False
    (results / "compare.json").write_text(json.dumps(c))
    assert "| no |" in build_report(results, man)


def test_flags_and_bold_airconditioner(tree):
    results, man = tree
    _full(results)
    text = build_report(results, man)
    rows = {line.split("|")[2].strip().strip("*"): line for line in text.splitlines()
            if line.startswith("|") and any(f" {c.name} " in line or f"**{c.name}**" in line for c in D_DEFS)}
    assert "absent from test" in rows[ABSENT]
    assert "not a core class" in rows["folding_door"]
    assert "not a core class" not in rows["single_door"] and "absent from test" not in rows["single_door"]
    air = rows["airconditioner"]
    assert "**airconditioner**" in air and air.count("**") >= 4 * 2
    assert "**" not in rows["sofa"]


def test_per_class_sorted_by_ap_descending_absent_last(tree):
    results, man = tree
    _full(results)
    text = build_report(results, man)
    order = [line.split("|")[2].strip().strip("*") for line in text.splitlines()
             if line.startswith("|") and any(f" {c.name} " in line or f"**{c.name}**" in line for c in D_DEFS)]
    assert len(order) == 30
    assert order[0] == "single_door" and order[-1] == ABSENT


def test_per_class_has_yolov8n_alongside(tree):
    results, man = tree
    _full(results)
    text = build_report(results, man)
    assert "YOLOv8n" in text and "YOLO11n" in text


def test_by_size_table(tree):
    results, man = tree
    _full(results)
    text = build_report(results, man)
    assert "small" in text and "medium" in text and "large" in text
    assert "0.310 ± 0.000" in text and "0.510 ± 0.000" in text and "0.710 ± 0.000" in text  # d11n, jitter +0.01
    assert "0.290 ± 0.000" in text  # d8n small, jitter -0.01


def test_w_table_and_thickness(tree):
    results, man = tree
    _full(results)
    text = build_report(results, man)
    assert "0.800 ± 0.010" in text           # pixel IoU
    assert "0.35 ± 0.00" in text             # t*
    assert "wall-thickness" in text.lower() and "mm" in text
    assert "20.8 ± 0.0" in text or "20.8" in text
    assert "n/a" in text                     # railing has no test pixels
    assert "10 mm/px (constant)." in text


def test_w_scale_conclusion_is_printed(tree):
    results, man = tree
    m = _w_metrics(0.8)
    m["wall_thickness_mm"]["scale_conclusion"] = "user-supplied"
    m["wall_thickness_mm"]["mm_per_px"] = 12.5
    _write(results, "w11n_s0", m)
    text = build_report(results, man)
    assert "12.5 mm/px (user-supplied)." in text and "(constant)" not in text


def test_w_without_thickness_key_still_renders(tree):
    results, man = tree
    for s in range(2):
        m = _w_metrics(0.8)
        del m["wall_thickness_mm"]
        _write(results, f"w11n_s{s}", m)
    text = build_report(results, man)
    assert "0.800 ± 0.000" in text


def test_data_statistics(tree):
    results, man = tree
    text = build_report(results, man)
    assert "15663" in text or "15,663" in text
    assert "1163" in text or "1,163" in text
    assert "59.9%" in text and "45.6%" in text and "21.6%" in text   # label disagreement
    assert "label-noise ceiling" in text
    assert "187" in text                                             # largest cluster
    assert "10.4%" in text                                           # W exclusion, train
    assert "curtain_wall" in text and "railing" in text


def test_missing_runs_reported_not_crash(tree):
    results, man = tree
    _write(results, "d11n_s0", _d_metrics(0.6))
    text = build_report(results, man)           # no d8n, no w11n, no compare.json
    assert "missing" in text
    assert "d11n_s1" in text and "d8n_s0" in text and "w11n_s2" in text
    assert "compare.json" in text


def test_no_runs_at_all(tree):
    results, man = tree
    text = build_report(results, man)
    assert "missing" in text and "Data statistics" in text


@pytest.mark.parametrize("extra", [{"limit": 20}, {"core_reduced": True}, {"split": "val"}])
def test_d_runs_from_smoke_are_refused(tree, extra):
    results, man = tree
    _write(results, "d11n_s0", _d_metrics(0.6, **extra))
    with pytest.raises(ReportError, match="d11n_s0"):
        build_report(results, man)


def test_w_limit_run_is_refused(tree):
    results, man = tree
    _write(results, "w11n_s0", _w_metrics(0.8, limit=20))
    with pytest.raises(ReportError, match="limit"):
        build_report(results, man)


def test_compare_with_limit_is_refused(tree):
    results, man = tree
    _full(results)
    c = json.loads((results / "compare.json").read_text())
    c["limit"] = 5
    (results / "compare.json").write_text(json.dumps(c))
    with pytest.raises(ReportError, match="compare"):
        build_report(results, man)


def test_cli_writes_file_and_refusal_exits_nonzero(tree, tmp_path, capsys):
    results, man = tree
    _full(results)
    out = tmp_path / "reports" / "tables.md"
    assert report.main(["--results", str(results), "--manifests", str(man), "--out", str(out)]) == 0
    assert "0.612 ± 0.003" in out.read_text(encoding="utf-8")
    _write(results, "d11n_s5", _d_metrics(0.6, limit=3))
    out2 = tmp_path / "t2.md"
    assert report.main(["--results", str(results), "--manifests", str(man), "--out", str(out2),
                        "--seeds", "0", "1", "2", "5"]) == 2
    assert "d11n_s5" in capsys.readouterr().err and not out2.exists()


def test_paths_with_spaces(spaced_tmp):
    results, man = spaced_tmp / "res ults", spaced_tmp / "man ifests"
    results.mkdir()
    man.mkdir()
    (man / "stats.json").write_text(json.dumps(STATS), encoding="utf-8")
    (man / "core_classes.json").write_text(json.dumps({"semantic_ids": CORE_SIDS}), encoding="utf-8")
    _write(results, "d11n_s0", _d_metrics(0.6))
    out = spaced_tmp / "o ut" / "tables.md"
    assert report.main(["--results", str(results), "--manifests", str(man), "--out", str(out)]) == 0
    assert out.read_text(encoding="utf-8")
