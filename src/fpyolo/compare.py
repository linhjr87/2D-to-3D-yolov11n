"""Seed-aware comparison of two D models (A - B) on the headline mAP50-95 over the core classes.

Each run directory is an evaluate_d output. The label set is rebuilt from the dataset and
split recorded in the runs' metrics.json, predictions come from each run's preds.npz, and
the headline of every run is recomputed with the core classes of --manifest. All runs must
record the same schema, equal to --schema, and the dataset's class names must be the D
classes of that schema. A difference is "real" when the 95% cluster-bootstrap CI excludes 0
and |delta| > 2 * pooled std, pooled std = sqrt((sa^2 + sb^2) / 2), with sa, sb the sample (ddof=1) std over seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import yaml

from fpyolo.apeval import bootstrap_delta, build_cache, summarize
from fpyolo.evaluate_d import class_key, core_indices, load_split_gt, read_preds, write_json
from fpyolo.schema import load_schema
from fpyolo.train import SCHEMA_PATH

N_BOOT, BOOT_SEED = 1000, 0


def cluster_map(manifest_dir: Path) -> dict[str, int]:
    """Image id `<archive>_<stem>` -> drawing cluster, for the kept tiles of tiles.csv."""
    with open(Path(manifest_dir) / "tiles.csv", newline="", encoding="utf-8") as f:
        return {f"{r['archive']}_{r['stem']}": int(r["cluster"])
                for r in csv.DictReader(f) if r["keep"] == "True"}


def real_difference(delta: float, ci_low: float, ci_high: float, std_a: float, std_b: float) -> dict:
    """The "real difference" rule; a NaN std (a single seed) never yields a real difference."""
    pooled = math.sqrt((std_a**2 + std_b**2) / 2)
    excludes_zero = bool(ci_low > 0 or ci_high < 0)
    exceeds = bool(abs(delta) > 2 * pooled)  # False when pooled is NaN
    return {"pooled_std": pooled, "ci_excludes_zero": excludes_zero,
            "exceeds_2x_pooled_std": exceeds, "real": excludes_zero and exceeds}


def _side(runs: Sequence[Path], heads: list[float]) -> dict:
    return {"runs": [str(r) for r in runs], "headlines": heads, "mean": float(np.mean(heads)),
            "std": float(np.std(heads, ddof=1)) if len(heads) > 1 else math.nan}


def _common(metas: list[dict], key: str):
    values = {json.dumps(m.get(key)) for m in metas}
    if len(values) != 1:
        raise ValueError(f"the runs disagree on {key}: {sorted(values)}")
    return metas[0].get(key)


def compare(a_dirs: Sequence[Path], b_dirs: Sequence[Path], manifest: Path,
            schema_path: Path = SCHEMA_PATH, n: int = N_BOOT, seed: int = BOOT_SEED) -> dict:
    if not a_dirs or not b_dirs:
        raise ValueError("need at least one run on each side")
    runs = [Path(d).resolve() for d in [*a_dirs, *b_dirs]]
    metas = [json.loads((d / "metrics.json").read_text(encoding="utf-8")) for d in runs]
    dataset, split, limit = (_common(metas, k) for k in ("dataset", "split", "limit"))
    manifest = Path(manifest).resolve()
    recorded = {str(Path(m["manifest"]).resolve()) for m in metas}
    if recorded != {str(manifest)}:
        raise ValueError(f"--manifest {manifest} is not the manifest of the runs: {sorted(recorded)}")

    schema_path = Path(schema_path).resolve()
    recorded = {m.get("schema") for m in metas}
    if None in recorded:
        raise ValueError("a run's metrics.json records no schema; re-run evaluate_d")
    recorded = {str(Path(s).resolve()) for s in recorded}
    if len(recorded) != 1:
        raise ValueError(f"the runs disagree on schema: {sorted(recorded)}")
    if recorded != {str(schema_path)}:
        raise ValueError(f"--schema {schema_path} is not the schema of the runs: {recorded.pop()}")
    schema = load_schema(schema_path)
    names = schema.yolo_names("D")
    data_yaml = Path(dataset) / "data.yaml"
    data_names = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))["names"]
    if {int(k): v for k, v in data_names.items()} != names:
        raise ValueError(f"{data_yaml} class names differ from the D classes of {schema_path}")
    core, all_classes = core_indices(manifest, schema), sorted(names)
    gts = load_split_gt(Path(dataset), split, limit)
    caches, heads = [], []
    for d in runs:
        preds = read_preds(d / "preds.npz")
        if set(preds) != set(gts):
            raise ValueError(f"{d}: predictions cover {len(preds)} images, the {split} split has {len(gts)}")
        caches.append(build_cache(gts, preds))
        heads.append(summarize(caches[-1], core, all_classes)["headline_map50_95"])
    na = len(a_dirs)
    boot = bootstrap_delta(caches[:na], caches[na:], core, cluster_map(manifest), n=n, seed=seed)
    a, b = _side(runs[:na], heads[:na]), _side(runs[na:], heads[na:])
    return {
        "a": a, "b": b,
        "bootstrap": {**boot, "n": n, "seed": seed},
        **real_difference(boot["delta"], boot["ci_low"], boot["ci_high"], a["std"], b["std"]),
        "core": [class_key(c, names) for c in core],
        "dataset": dataset, "split": split, "limit": limit, "manifest": str(manifest),
        "schema": str(schema_path),
    }


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fpyolo.compare", description=__doc__.split("\n")[0])
    ap.add_argument("--a", required=True, nargs="+", type=Path, help="evaluate_d dirs of model A (seeds)")
    ap.add_argument("--b", required=True, nargs="+", type=Path, help="evaluate_d dirs of model B (seeds)")
    ap.add_argument("--manifest", required=True, type=Path, help="<out>/manifests (tiles.csv, core_classes.json)")
    ap.add_argument("--out", required=True, type=Path, help="output JSON file")
    ap.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    args = ap.parse_args(argv)
    res = compare(args.a, args.b, args.manifest, args.schema)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.out, res)
    boot = res["bootstrap"]
    print(f"A {res['a']['mean']:.4f} ± {res['a']['std']:.4f}, B {res['b']['mean']:.4f} ± {res['b']['std']:.4f}; "
          f"delta {boot['delta']:+.4f} [{boot['ci_low']:+.4f}, {boot['ci_high']:+.4f}] real={res['real']} "
          f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
