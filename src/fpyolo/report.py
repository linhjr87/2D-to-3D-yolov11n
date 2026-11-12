"""Markdown result tables from the evaluation outputs (results/<run>/metrics.json, compare.json,
manifests/stats.json).

    python -m fpyolo.report --results results --manifests data/fpcad_v0/manifests --out reports/tables.md

Runs are results/<name>/metrics.json with name d11n_s<seed> (YOLO11n D), d8n_s<seed> (YOLOv8n D) and
w11n_s<seed> (YOLO11n-seg W). A missing run is listed as "missing"; a run evaluated with --limit, on a
split other than test, or with a reduced core set is refused (ReportError, exit code 2).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np

from fpyolo.schema import load_schema
from fpyolo.train import SCHEMA_PATH

SEEDS = (0, 1, 2)
D_MODELS = {"d11n": "YOLO11n", "d8n": "YOLOv8n"}
W_RUN = "w11n"
AIRCONDITIONER_SEMANTIC_ID = 21
PER_CLASS_NOTE_CLASSES = ("not a core class", "absent from test")


class ReportError(ValueError):
    """The results cannot be reported as they are (e.g. a smoke run among them)."""


# ------------------------------------------------------------------ helpers ---


def _finite(values: Iterable) -> list[float]:
    return [float(v) for v in values if v is not None and math.isfinite(float(v))]


def fmt_mean_std(values: Iterable, digits: int = 3) -> str:
    """"0.612 ± 0.004" (sample std, ddof=1); "missing" without values, "n/a" if none is a number,
    "0.612 ± n/a" for a single value."""
    values = list(values)
    if not values:
        return "missing"
    v = _finite(values)
    if not v:
        return "n/a"
    mean = f"{np.mean(v):.{digits}f}"
    return f"{mean} ± {np.std(v, ddof=1):.{digits}f}" if len(v) > 1 else f"{mean} ± n/a"


def fmt_mean(values: Iterable, digits: int = 3) -> str:
    v = _finite(values)
    return f"{np.mean(v):.{digits}f}" if v else "n/a"


def _mean_or_nan(values: Iterable) -> float:
    v = _finite(values)
    return float(np.mean(v)) if v else math.nan


def _pct(x) -> str:
    return "n/a" if x is None else f"{100 * float(x):.1f}%"


def _int(x) -> str:
    return "n/a" if x is None else f"{int(x):,}"


def _table(header: Sequence[str], rows: Iterable[Sequence[str]]) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "|" + "|".join("---" for _ in header) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return out


def _read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ---------------------------------------------------------------- discovery ---


def load_runs(results: Path, prefix: str, seeds: Sequence[int], kind: str) -> tuple[list[dict], list[str]]:
    """(metrics of the runs found, names of the missing runs); refuses smoke-style runs.

    kind: "D" runs must be the test split with the full core set, "W" runs must not use --limit."""
    found, missing = [], []
    for s in seeds:
        name = f"{prefix}_s{s}"
        path = Path(results) / name / "metrics.json"
        if not path.is_file():
            missing.append(name)
            continue
        m = _read_json(path)
        if m.get("limit") is not None:
            raise ReportError(f"{name}: evaluated with --limit {m['limit']} (a smoke run), refusing to report it")
        if kind == "D":
            if m.get("core_reduced"):
                raise ReportError(f"{name}: evaluated with a reduced core class set (core_reduced), refusing to report it")
            if m.get("split") != "test":
                raise ReportError(f"{name}: evaluated on split {m.get('split')!r}, the tables report test")
        m["_name"] = name
        found.append(m)
    return found, missing


def _missing_line(missing: list[str]) -> list[str]:
    return [f"Missing runs: {', '.join(missing)}.", ""] if missing else []


# ------------------------------------------------------------------ D tables ---


def _ultra(m: dict, key: str):
    return (m.get("ultralytics_val") or {}).get(key)


def _speed(m: dict) -> float:
    sp = (m.get("ultralytics_val") or {}).get("speed_ms")
    return float(sum(sp.values())) if sp else math.nan


def headline_section(d_runs: dict[str, list[dict]], d_missing: dict[str, list[str]], compare: dict | None,
                     compare_missing: bool, n_seeds: int) -> list[str]:
    cols = [("headline_map50_95", "mAP50-95 (core, headline)"), ("map50_core", "mAP50 (core)"),
            ("map75_core", "mAP75 (core)"), ("map50_95_all", "mAP50-95 (all 30)"),
            ("map50_all", "mAP50 (all 30)"), ("map75_all", "mAP75 (all 30)")]
    rows = []
    for key, label in D_MODELS.items():
        runs = d_runs[key]
        cells = [fmt_mean_std([m.get(c) for m in runs]) for c, _ in cols]
        pr = [fmt_mean_std([_ultra(m, k) for m in runs]) for k in ("precision", "recall")]
        speed = fmt_mean_std([_speed(m) for m in runs], digits=1) if runs else "missing"
        rows.append([label, f"{len(runs)}/{n_seeds}", *cells, *pr, speed])
    out = ["## D: object detection (test split)", ""]
    out += _table(["Model", "Seeds", *[label for _, label in cols], "Precision (Ultralytics averaging)",
                   "Recall (Ultralytics averaging)", "Val speed ms/img (pre+inf+post)"], rows)
    out += ["", "Mean ± sample standard deviation (ddof=1) over seeds. \"core\" = classes with enough train and test "
            "instances (core_classes.json); \"all 30\" averages over the classes that have test labels. "
            "Precision and recall are Ultralytics' val numbers (mean over the classes with test labels, at "
            "Ultralytics' own operating point), not the fixed class sets of the AP columns.", ""]
    for key in D_MODELS:
        out += _missing_line(d_missing[key])
    out += ["### YOLO11n vs YOLOv8n", ""]
    if compare is None:
        out += ["compare.json: missing (run `python -m fpyolo.compare`).", ""] if compare_missing else \
               ["compare: n/a.", ""]
        return out
    boot = compare["bootstrap"]
    side = lambda s: ", ".join(Path(r).name for r in compare[s]["runs"])  # noqa: E731
    pooled = compare.get("pooled_std")
    out += _table(
        ["Δ mAP50-95 (core), A − B", "95% CI (cluster bootstrap)", "Pooled seed std", "CI excludes 0",
         "Δ exceeds 2 × pooled std (absolute)", "Real difference", "Degenerate resamples"],
        [[f"{boot['delta']:+.3f}", f"[{boot['ci_low']:+.3f}, {boot['ci_high']:+.3f}]",
          "n/a" if pooled is None else f"{pooled:.3f}",
          "yes" if compare.get("ci_excludes_zero") else "no",
          "yes" if compare.get("exceeds_2x_pooled_std") else "no",
          "yes" if compare["real"] else "no", f"{boot.get('n_degenerate', 0)} of {boot.get('n', 'n/a')}"]])
    out += ["", f"A = {side('a')}; B = {side('b')}. Resampling unit: drawing cluster of the test tiles "
            f"({boot.get('n', 'n/a')} resamples, seed {boot.get('seed', 'n/a')}). "
            "Real = the CI excludes 0 and |Δ| exceeds twice the pooled across-seed standard deviation.", ""]
    return out


def _aircon_name(schema_path: Path) -> str | None:
    for c in load_schema(schema_path).for_model("D"):
        if c.semantic_id == AIRCONDITIONER_SEMANTIC_ID:
            return c.name
    return None


def per_class_section(d_runs: dict[str, list[dict]], schema_path: Path) -> list[str]:
    a, b = d_runs["d11n"], d_runs["d8n"]
    out = ["### Per-class AP (test)", ""]
    ref = (a or b)
    if not ref:
        return out + ["missing (no D runs).", ""]
    aircon = _aircon_name(schema_path)
    classes = {}
    for key, info in ref[0]["per_class"].items():
        classes[key] = {"name": info["name"], "core": bool(info["core"]), "n_gt": int(info["n_gt"])}

    def mean_of(runs, key, field):
        return _mean_or_nan(m["per_class"][key][field] for m in runs if key in m["per_class"])

    rows = []
    for key, c in classes.items():
        rows.append({**c, "key": key,
                     "a50_95": mean_of(a, key, "ap50_95"), "a50": mean_of(a, key, "ap50"),
                     "b50_95": mean_of(b, key, "ap50_95"), "b50": mean_of(b, key, "ap50")})
    primary = "a50_95" if a else "b50_95"
    rows.sort(key=lambda r: (math.isnan(r[primary]), -(0 if math.isnan(r[primary]) else r[primary]), r["name"]))
    table = []
    for rank, r in enumerate(rows, 1):
        flags = []
        if not r["core"]:
            flags.append("not a core class")
        if r["n_gt"] == 0:
            flags.append("absent from test")
        cells = [str(rank), r["name"], str(r["n_gt"]),
                 fmt_mean([r["a50_95"]]) if a else "missing", fmt_mean([r["a50"]]) if a else "missing",
                 fmt_mean([r["b50_95"]]) if b else "missing", fmt_mean([r["b50"]]) if b else "missing",
                 "; ".join(flags)]
        if aircon is not None and r["name"] == aircon:
            cells = [f"**{c}**" if c else c for c in cells]
        table.append(cells)
    out += _table(["#", "Class", "Test n_gt", "AP50-95 YOLO11n", "AP50 YOLO11n", "AP50-95 YOLOv8n",
                   "AP50 YOLOv8n", "Flags"], table)
    absent = [r["name"] for r in rows if r["n_gt"] == 0]
    out += ["", "Mean over seeds (YOLO11n: " + f"{len(a)}, YOLOv8n: {len(b)} run(s)). Sorted by YOLO11n AP50-95, "
            "descending; classes without test labels have no AP (n/a) and come last. "
            + (f"Absent from test: {', '.join(absent)}. " if absent else "")
            + (f"The {aircon} row is in bold." if aircon else ""), ""]
    return out


def by_size_section(d_runs: dict[str, list[dict]]) -> list[str]:
    rows = []
    for key, label in D_MODELS.items():
        runs = d_runs[key]
        rows.append([label] + [fmt_mean_std([m["by_size"].get(s) for m in runs]) for s in ("small", "medium", "large")])
    return ["### mAP50-95 by object size (test, all classes with labels)", "",
            *_table(["Model", "small (< 32² px)", "medium (32²–96² px)", "large (> 96² px)"], rows), "",
            "Box area in image pixels, COCO limits; mean ± std over seeds.", ""]


# ------------------------------------------------------------------ W table ---


def w_section(w_runs: list[dict], missing: list[str], n_seeds: int) -> list[str]:
    out = ["## W: wall segmentation (YOLO11n-seg, test split)", ""]
    if not w_runs:
        return out + ["missing (no W runs).", "", *_missing_line(missing)]

    def col(f):
        return [f(m) for m in w_runs]

    def thick(field):
        return lambda m: (m.get("wall_thickness_mm") or {}).get(field)

    rows = [["Pixel IoU (pooled, class-agnostic, at t*)", fmt_mean_std(col(lambda m: m.get("test_pixel_iou")))],
            ["t* (chosen on val)", fmt_mean_std(col(lambda m: m.get("t_star")), 2)],
            ["Val pixel IoU at t*", fmt_mean_std(col(lambda m: m.get("val_iou_at_t_star")))],
            ["Mask mAP50-95", fmt_mean_std(col(lambda m: m.get("mask_map50_95")))],
            ["Mask mAP50", fmt_mean_std(col(lambda m: m.get("mask_map50")))]]
    for name in w_runs[0].get("per_class_pixel_iou", {}):
        rows.append([f"Pixel IoU: {name}",
                     fmt_mean_std(col(lambda m, n=name: (m.get("per_class_pixel_iou") or {}).get(n)))])
    has_t = any(m.get("wall_thickness_mm") for m in w_runs)
    for field, label in (("mae", "Wall-thickness error: MAE (mm)"), ("median_abs", "Wall-thickness error: median |error| (mm)"),
                         ("bias", "Wall-thickness error: bias, prediction − label (mm)")):
        rows.append([label, fmt_mean_std(col(thick(field)), 2) if has_t else "n/a (not recorded)"])
    rows.append(["Images in the thickness comparison", fmt_mean_std(col(thick("n_images")), 0) if has_t else "n/a"])
    out += _table(["Metric", f"Mean ± std over seeds ({len(w_runs)}/{n_seeds} runs)"], rows)
    info = next(((m["wall_thickness_mm"]) for m in w_runs if m.get("wall_thickness_mm")), None)
    out += [""]
    if info:
        scale = info.get("mm_per_px")
        conclusion = info.get("scale_conclusion") or "conclusion not recorded"
        out += [f"Wall thickness: {info.get('method', 'n/a')}. Scale: "
                + (f"{scale:g} mm/px ({conclusion})." if scale is not None else "unknown (mm values not available).")
                + " Only test images with a non-empty label mask and a non-empty prediction (score ≥ t*) count.", ""]
    out += ["Per-class pixel IoU is n/a for a class without label pixels in test. Mask mAP is Ultralytics' own "
            "averaging (classes with labels only).", ""]
    out += _missing_line(missing)
    return out


# ----------------------------------------------------------- data statistics ---


def stats_section(manifests: Path, schema_path: Path) -> list[str]:
    stats_path, core_path = Path(manifests) / "stats.json", Path(manifests) / "core_classes.json"
    for p in (stats_path, core_path):
        if not p.is_file():
            raise ReportError(f"{p} not found (run the dataset build first)")
    st, core = _read_json(stats_path), _read_json(core_path)
    schema = load_schema(schema_path)
    by_sem = {c.semantic_id: c.name for c in schema.classes}
    dd, cl, sp = st["dedup"], st["clusters"], st["split_report"]
    out = ["## Data statistics", "", "### Deduplication (byte-identical PNG copies)", ""]
    out += _table(["Tiles", "Identical sets", "Tiles in sets", "Kept", "Dropped"],
                  [[_int(dd["n_tiles"]), _int(dd["n_sets"]), _int(dd["n_tiles_in_sets"]), _int(dd["kept"]),
                    _int(dd["dropped"])]])
    out += ["", "### Label disagreement between identical copies (label-noise ceiling)", ""]
    out += _table(["Sets whose copies agree on every thing count", "Sets whose copies agree on the wall stroke count",
                   "Thing instances missing from at least one copy"],
                  [[_pct(dd.get("agree_things_frac")), _pct(dd.get("agree_walls_frac")),
                    _pct(dd.get("missing_instance_frac"))]])
    out += ["", "The same drawing, byte for byte, is labelled differently in different copies: this share of thing "
            "instances is absent from at least one copy, a label-noise ceiling that no model can be expected to "
            "beat.", "", "### Clusters and split", ""]
    out += _table(["Clusters", "Largest", "Largest share", "Top sizes", "Clusters train / val / test"],
                  [[_int(cl.get("n")), _int(cl.get("largest")), _pct(cl.get("largest_frac")),
                    ", ".join(str(x) for x in cl.get("top10_sizes", [])),
                    " / ".join(_int(cl.get("clusters_per_set", {}).get(s)) for s in ("train", "val", "test"))]])
    out += ["", ""]
    ratio = sp.get("tile_ratio", {})
    out += _table(["Split", "Tiles", "Share"], [[s, _int(sp["tiles"].get(s)), _pct(ratio.get(s))]
                                                 for s in ("train", "val", "test")])
    names = ", ".join(by_sem.get(int(s), str(s)) for s in core["semantic_ids"])
    out += ["", f"Core classes ({len(core['semantic_ids'])} of 30; at least {core.get('min_train', '?')} train and "
            f"{core.get('min_test', '?')} test instances): {names}.", ""]
    bs = st.get("box_stats")
    if bs:
        out += [f"Boxes: {_int(bs.get('n_boxes'))} in total, {_int(bs.get('padded'))} padded to 2 px, "
                f"{_int(bs.get('small_lt_32px'))} smaller than 32 px, {_int(bs.get('negative_tiles'))} tiles "
                "without objects.", ""]
    wc = st.get("w_coverage")
    if wc:
        out += ["### W coverage and exclusion", ""]
        rows = [[s, _int(e["wall_tiles"]), _int(e["excluded"]), _pct(e["frac"])]
                for s, e in wc["excluded"].items()]
        out += _table(["Split", "Tiles with walls", f"Excluded (coverage < {wc.get('coverage_min', '?')})",
                       "Excluded share"], rows)
        out += ["", f"{_int(wc.get('n_polygons_in_W'))} polygons in the W dataset. Excluded tiles are still used "
                "for D.", ""]
        cand = st.get("w_candidates", {})
        if cand:
            out += _table(["W class", "Tiles", "Median coverage", "Enabled"],
                          [[by_sem.get(int(k), k), _int(v.get("n_tiles")), f"{v.get('median_coverage', math.nan):.2f}",
                            "yes" if v.get("enabled") else "no"] for k, v in cand.items()])
            out += [""]
    return out


# --------------------------------------------------------------------- main ---


def check_compare_runs(compare: dict, d_runs: dict[str, list[dict]], path: Path) -> None:
    """compare.json's A runs must be the loaded d11n runs and its B runs the loaded d8n runs."""
    for side, key in (("a", "d11n"), ("b", "d8n")):
        got = sorted(Path(r).name for r in compare[side]["runs"])
        want = sorted(m["_name"] for m in d_runs[key])
        if got != want:
            raise ReportError(f"{path}: side {side.upper()} compares {got or 'no runs'}, but the {key} runs in "
                              f"the results folder are {want or 'none'}; re-run fpyolo.compare "
                              f"(--a results/d11n_s*, --b results/d8n_s*)")


def build_report(results: Path, manifests: Path, schema_path: Path = SCHEMA_PATH,
                 seeds: Sequence[int] = SEEDS) -> str:
    results, seeds = Path(results), tuple(seeds)
    d_runs, d_missing = {}, {}
    for key in D_MODELS:
        d_runs[key], d_missing[key] = load_runs(results, key, seeds, "D")
    w_runs, w_missing = load_runs(results, W_RUN, seeds, "W")
    compare, compare_missing = None, True
    cpath = results / "compare.json"
    if cpath.is_file():
        compare, compare_missing = _read_json(cpath), False
        if compare.get("limit") is not None or compare.get("split", "test") != "test":
            raise ReportError(f"{cpath}: the compare result comes from a smoke or non-test evaluation "
                              f"(limit {compare.get('limit')}, split {compare.get('split')}), refusing to report it")
        check_compare_runs(compare, d_runs, cpath)
    lines = ["# Result tables", "",
             f"Generated by `python -m fpyolo.report`; seeds {', '.join(map(str, seeds))}. All numbers are on the "
             "test split.", ""]
    lines += headline_section(d_runs, d_missing, compare, compare_missing, len(seeds))
    lines += per_class_section(d_runs, Path(schema_path))
    lines += by_size_section(d_runs)
    lines += w_section(w_runs, w_missing, len(seeds))
    lines += stats_section(Path(manifests), Path(schema_path))
    return "\n".join(lines).rstrip() + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fpyolo.report", description=__doc__.split("\n")[0])
    ap.add_argument("--results", required=True, type=Path, help="folder of <run>/metrics.json and compare.json")
    ap.add_argument("--manifests", required=True, type=Path, help="<out>/manifests (stats.json, core_classes.json)")
    ap.add_argument("--out", required=True, type=Path, help="markdown file to write")
    ap.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS))
    args = ap.parse_args(argv)
    try:
        text = build_report(args.results, args.manifests, args.schema, args.seeds)
    except ReportError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(text, encoding="utf-8")
    print(f"tables -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
