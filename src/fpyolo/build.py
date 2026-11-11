"""Dataset build CLI: FloorPlanCAD tiles -> YOLO datasets D (thing boxes) and W (wall polygons).

    python -m fpyolo.build --src <FloorPlanCAD> --out <dir> --config configs/build.yaml \
        [--workers N] [--reuse-manifest <tiles.csv>]

Steps: per-tile analysis (parallel, cached in <out>/cache) -> exact dedup -> near-duplicate
drawing clusters -> stratified split -> manifests -> D/W images and labels.
Exit codes: 0 done; 3 too many wall tiles excluded from W; 4 SVG/PNG misaligned. On 3 or 4
the manifests are written as evidence but no dataset is.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import os
import pickle
import random
import shutil
import sys
from collections import Counter
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml

from fpyolo import split as splitmod
from fpyolo.boxes import YoloBox, format_yolo_det, thing_boxes
from fpyolo.dedup import (TileId, build_clusters, choose_keeper, drawing_group, exact_duplicate_sets,
                          near_duplicate_pairs, thumbnail)
from fpyolo.ink import ink_fraction, load_ink, write_ink_rgb
from fpyolo.qa import render_polylines
from fpyolo.schema import Schema, load_schema
from fpyolo.svgparse import Tile, parse_tile
from fpyolo.walls import format_yolo_seg, keyhole_ring, wall_regions

EXIT_W_EXCLUDED, EXIT_MISALIGNED = 3, 4
ALIGN_MIN = 0.9  # a tile is misaligned when < 90% of its drawn label pixels land on (dilated) ink
MISALIGNED_STOP_FRAC = 0.01
WALL_SID = 1
OFFICIAL_TEST_ARCHIVE, OFFICIAL_VAL_FRAC = "test-00", 0.10
CROSS_SPACE_RANGE, CROSS_SPACE_N = (0.85, 0.95), 30
CACHE_VERSION = 2  # bump when the per-tile analysis changes
SETS = splitmod.SETS
COLUMNS = ["archive", "stem", "md5", "keep", "drop_reason", "group", "cluster", "split", "n_things",
           "wall_strokes", "wall_coverage", "in_W", "align_frac", "width", "height"]
_REQUIRED = ("schema", "archives", "split_mode", "ratios", "seed", "near_dup", "boxes", "walls", "core",
             "manual_merges")


def log(*a) -> None:
    print(*a, file=sys.stderr, flush=True)


def image_id(t: TileId) -> str:
    return f"{t[0]}_{t[1]}"


def group_str(t: TileId) -> str:
    return "/".join(drawing_group(t))


# ------------------------------------------------------------------ config ---


def load_config(path: Path) -> dict:
    """build.yaml as a dict; `schema` resolved against the config's folder, merges as group tuples."""
    path = Path(path)
    cfg = yaml.safe_load(path.read_text(encoding="utf-8"))
    missing = [k for k in _REQUIRED if k not in cfg]
    if missing:
        raise ValueError(f"{path}: missing keys {missing}")
    if cfg["split_mode"] not in ("cluster", "official"):
        raise ValueError(f"{path}: split_mode must be 'cluster' or 'official'")
    cfg["schema"] = path.parent / cfg["schema"]  # an absolute schema path stays absolute
    cfg["ratios"] = tuple(float(r) for r in cfg["ratios"])
    merges = []
    for pair in cfg["manual_merges"]:
        if len(pair) != 2 or not all(len(g) == 2 and all(isinstance(v, str) for v in g) for g in pair):
            raise ValueError(f"{path}: manual_merges entries are [[space, number], [space, number]] "
                             f"with quoted strings, got {pair!r}")
        merges.append((tuple(pair[0]), tuple(pair[1])))
    cfg["manual_merges"] = merges
    return cfg


# ---------------------------------------------------------- per-tile work ---


@dataclass
class ClassWalls:
    coverage: float
    stroke_len_px: float
    rings: list[np.ndarray]  # keyhole rings, normalized


@dataclass
class TileResult:
    tile: TileId
    stamp: tuple[int, ...]  # sizes and mtimes of the SVG and PNG: a changed source invalidates the cache
    md5: str
    width: int
    height: int
    ink_frac: float
    thumb: np.ndarray
    skipped: int
    det_boxes: list[YoloBox]
    box_stats: dict[str, int]
    things: dict[int, int]  # semantic id -> written boxes
    wall_strokes: int
    walls: dict[int, ClassWalls]  # W-class semantic id -> regions; only classes with strokes in the image
    align_frac: float | None


def align_frac(tile: Tile, ink: np.ndarray) -> float | None:
    """Fraction of the 1-px raster of every labeled primitive on ink (alpha > 0) dilated by 1 px."""
    h, w = ink.shape
    x0, y0, x1, y1 = tile.viewbox
    scale = np.array([w / (x1 - x0), h / (y1 - y0)])
    drawn = render_polylines([(p.polyline - (x0, y0)) * scale for p in tile.primitives], h, w)
    n = int(drawn.sum())
    if n == 0:
        return None
    near = cv2.dilate((ink < 255).astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    return float((drawn & near).sum() / n)


def analyse_tile(src: Path, t: TileId, p: dict, stamp: tuple[int, ...]) -> TileResult:
    svg, png = src / t[0] / f"{t[1]}.svg", src / t[0] / f"{t[1]}.png"
    data = png.read_bytes()
    ink = load_ink(io.BytesIO(data))
    h, w = ink.shape
    tile = parse_tile(svg, t[0])
    boxes, bstats = thing_boxes(tile, p["sem_to_yolo"], w, h, **p["boxes"])
    yolo_to_sem = {v: k for k, v in p["sem_to_yolo"].items()}
    walls = {}
    for sid in p["w_sids"]:
        r = wall_regions(tile, {sid}, w, h, **p["walls"])
        if r.coverage is not None:
            walls[sid] = ClassWalls(r.coverage, r.stroke_len_px, [keyhole_ring(q) for q in r.polygons])
    return TileResult(
        t, stamp, hashlib.md5(data).hexdigest(), w, h, ink_fraction(ink), thumbnail(ink, p["thumb"]),
        tile.skipped, boxes, bstats, dict(Counter(yolo_to_sem[b.yolo_index] for b in boxes)),
        sum(1 for q in tile.primitives if q.semantic_id == WALL_SID), walls, align_frac(tile, ink),
    )


def _tile_worker(job) -> TileResult:
    src, t, p = job
    stats = [os.stat(src / t[0] / f"{t[1]}{ext}") for ext in (".svg", ".png")]
    stamp = tuple(v for st in stats for v in (st.st_size, st.st_mtime_ns))
    cache = p["cache_dir"] / f"{image_id(t)}.pkl"
    if cache.exists():
        try:
            res = pickle.loads(cache.read_bytes())
            if res.stamp == stamp:
                return res
        except Exception:  # a truncated or stale cache file is recomputed
            pass
    res = analyse_tile(src, t, p, stamp)
    tmp = cache.with_name(f"{cache.name}.{os.getpid()}.tmp")
    tmp.write_bytes(pickle.dumps(res))
    os.replace(tmp, cache)
    return res


def _write_image(job) -> None:
    png, out = job
    write_ink_rgb(load_ink(png), out)


def _pmap(fn, jobs: list, workers: int, chunksize: int = 16):
    if workers <= 1:
        yield from map(fn, jobs)
        return
    with Pool(workers) as pool:
        yield from pool.imap(fn, jobs, chunksize=chunksize)


def analyse_all(src: Path, tiles: list[TileId], cfg: dict, schema: Schema, cache_root: Path,
                workers: int) -> dict[TileId, TileResult]:
    w_sids = sorted({c.semantic_id for c in schema.classes if c.model == "W"} | set(cfg["walls"]["candidates"]))
    p = {
        "sem_to_yolo": schema.yolo_index("D"),
        "boxes": {k: float(cfg["boxes"][k]) for k in ("min_side_px", "min_visible")},
        "walls": {k: float(cfg["walls"][k]) for k in ("gap_px", "max_thickness_px", "cap_px")},
        "w_sids": w_sids,
        "thumb": int(cfg["near_dup"]["thumb"]),
    }
    key = json.dumps({"v": CACHE_VERSION, **p, "sem_to_yolo": sorted(p["sem_to_yolo"].items())}, sort_keys=True)
    p["cache_dir"] = cache_root / hashlib.sha1(key.encode()).hexdigest()[:12]
    p["cache_dir"].mkdir(parents=True, exist_ok=True)
    results = {}
    for n, res in enumerate(_pmap(_tile_worker, [(src, t, p) for t in tiles], workers), 1):
        results[res.tile] = res
        if n % 1000 == 0 or n == len(tiles):
            log(f"analysed {n}/{len(tiles)} tiles")
    return results


# ------------------------------------------------- dedup, clusters, split ---


@dataclass
class Placement:
    keep: bool
    drop_reason: str
    cluster: int
    split: str  # "" for dropped tiles


def dedup_step(results: dict[TileId, TileResult]) -> tuple[dict[TileId, str], dict]:
    """Drop reason per dropped tile, and the label disagreement between byte-identical copies.

    missing_instance_frac = sum over sets and classes of (max - min copy count) / sum of max:
    the share of thing instances absent from at least one copy (label-noise ceiling, spec §6.8).
    """
    sets = exact_duplicate_sets({t: r.md5 for t, r in results.items()})
    things = {t: len(r.det_boxes) for t, r in results.items()}
    walls = {t: r.wall_strokes for t, r in results.items()}
    drop, agree_things, agree_walls, missing, union = {}, 0, 0, 0, 0
    for s in sets:
        keeper = choose_keeper(s, things, walls)
        drop.update({t: f"exact_duplicate_of:{image_id(keeper)}" for t in s if t != keeper})
        counts = [Counter(results[t].things) for t in s]
        agree_things += all(c == counts[0] for c in counts)
        agree_walls += len({walls[t] for t in s}) == 1
        for c in set().union(*counts):
            hi, lo = max(x[c] for x in counts), min(x[c] for x in counts)
            missing, union = missing + hi - lo, union + hi
    n_sets = len(sets)
    return drop, {
        "n_tiles": len(results), "n_sets": n_sets, "n_tiles_in_sets": sum(map(len, sets)),
        "kept": len(results) - len(drop), "dropped": len(drop),
        "agree_things_frac": agree_things / n_sets if n_sets else None,
        "agree_walls_frac": agree_walls / n_sets if n_sets else None,
        "missing_instance_frac": missing / union if union else None,
    }


def cross_space_candidates(tiles, thumbs, valid, cluster_of, block: int = 1024) -> list[dict]:
    """Best train-00 <-> group-B tile pair per drawing-group pair with cosine in CROSS_SPACE_RANGE,
    for group pairs not already in one cluster; the CROSS_SPACE_N highest."""
    lo, hi = CROSS_SPACE_RANGE
    a = np.array([i for i, t in enumerate(tiles) if valid[i] and drawing_group(t)[0] == "A"], dtype=int)
    b = np.array([i for i, t in enumerate(tiles) if valid[i] and drawing_group(t)[0] == "B"], dtype=int)
    best: dict[tuple, tuple[float, int, int]] = {}
    for s in range(0, len(a), block):
        ia = a[s:s + block]
        sims = thumbs[ia] @ thumbs[b].T
        r, c = np.nonzero((sims >= lo) & (sims < hi))
        for rr, cc, v in zip(ia[r].tolist(), b[c].tolist(), sims[r, c].tolist()):
            key = (drawing_group(tiles[rr]), drawing_group(tiles[cc]))
            if v > best.get(key, (-1.0,))[0] and cluster_of[tiles[rr]] != cluster_of[tiles[cc]]:
                best[key] = (v, rr, cc)
    top = sorted(best.values(), reverse=True)[:CROSS_SPACE_N]
    return [{"a": image_id(tiles[i]), "b": image_id(tiles[j]), "group_a": group_str(tiles[i]),
             "group_b": group_str(tiles[j]), "cosine": round(v, 4)} for v, i, j in top]


def cluster_step(results, cfg) -> tuple[dict[TileId, int], dict, list[dict]]:
    """Drawing clusters over all tiles (dropped copies included), link statistics, cross-space pairs."""
    tiles = sorted(results)
    nd = cfg["near_dup"]
    thumbs = np.stack([results[t].thumb for t in tiles])
    valid = np.array([results[t].ink_frac >= nd["min_ink"] for t in tiles])
    pairs = near_duplicate_pairs(thumbs, valid, float(nd["threshold"]))
    groups = {drawing_group(t) for t in tiles}
    unknown = [g for pair in cfg["manual_merges"] for g in pair if g not in groups]
    if unknown:
        raise ValueError(f"manual_merges name unknown drawing groups: {unknown}")
    cluster_of = build_clusters(tiles, pairs, cfg["manual_merges"])
    linking = [(i, j) for i, j in pairs if drawing_group(tiles[i]) != drawing_group(tiles[j])]
    examples = random.Random(cfg["seed"]).sample(linking, min(10, len(linking)))
    stats = {
        "n_tiles_valid_for_linking": int(valid.sum()), "n_pairs": len(pairs),
        "n_pairs_linking_groups": len(linking),
        "n_pairs_cross_space": sum(drawing_group(tiles[i])[0] != drawing_group(tiles[j])[0] for i, j in linking),
        "manual_merges": [list(map(list, m)) for m in cfg["manual_merges"]],
        "example_pairs": [{"a": image_id(tiles[i]), "b": image_id(tiles[j]),
                           "cosine": round(float(thumbs[i] @ thumbs[j]), 4)} for i, j in sorted(examples)],
    }
    return cluster_of, stats, cross_space_candidates(tiles, thumbs, valid, cluster_of)


def official_split(cluster_of: dict[TileId, int], seed: int) -> dict[TileId, str]:
    """test = the official test archive; val = whole clusters of the rest, in seeded random order,
    until they hold OFFICIAL_VAL_FRAC of the rest's tiles; train = the remainder."""
    rest = [t for t in sorted(cluster_of) if t[0] != OFFICIAL_TEST_ARCHIVE]
    size = Counter(cluster_of[t] for t in rest)
    order = sorted(size)
    random.Random(seed).shuffle(order)
    val, n = set(), 0
    for k in order:
        if n >= OFFICIAL_VAL_FRAC * len(rest):
            break
        val.add(k)
        n += size[k]
    return {t: "test" if t[0] == OFFICIAL_TEST_ARCHIVE else "val" if cluster_of[t] in val else "train"
            for t in cluster_of}


def class_counts_of(results, kept) -> dict[TileId, Counter]:
    """D semantic-id instance counts per tile, plus pseudo-class 1 = 1 when the tile has wall strokes."""
    return {t: Counter(results[t].things) + Counter({WALL_SID: int(results[t].wall_strokes > 0)})
            for t in kept}


def place_tiles(results, cfg) -> tuple[dict[TileId, Placement], dict]:
    drop, dedup_stats = dedup_step(results)
    cluster_of, link_stats, cross = cluster_step(results, cfg)
    kept = {t: cluster_of[t] for t in sorted(results) if t not in drop}
    if cfg["split_mode"] == "cluster":
        split_of = splitmod.stratified_cluster_split(kept, class_counts_of(results, kept), cfg["ratios"], cfg["seed"])
    else:
        split_of = official_split(kept, cfg["seed"])
    placement = {t: Placement(t not in drop, drop.get(t, ""), cluster_of[t], split_of.get(t, ""))
                 for t in results}
    return placement, {"dedup": dedup_stats, "clusters": link_stats, "cross_space_candidates": cross}


def read_manifest(path: Path, tiles: list[TileId]) -> dict[TileId, Placement]:
    with open(path, newline="", encoding="utf-8") as f:
        rows = {(r["archive"], r["stem"]): r for r in csv.DictReader(f)}
    if set(rows) != set(tiles):
        raise ValueError(f"{path}: manifest tiles differ from the source "
                         f"({len(set(rows) - set(tiles))} only in manifest, {len(set(tiles) - set(rows))} only in source)")
    return {t: Placement(r["keep"] == "True", r["drop_reason"], int(r["cluster"]), r["split"])
            for t, r in rows.items()}


# ------------------------------------------------------------- statistics ---


def w_coverage(r: TileResult, w_sids: list[int]) -> float | None:
    """Stroke-length weighted coverage over the enabled W classes; None without such strokes."""
    parts = [r.walls[s] for s in w_sids if s in r.walls]
    total = sum(p.stroke_len_px for p in parts)
    return sum(p.coverage * p.stroke_len_px for p in parts) / total if total > 0 else None


def cluster_size_stats(placement) -> dict:
    kept = {t: p.cluster for t, p in placement.items() if p.keep}
    size = Counter(kept.values())
    big, largest = size.most_common(1)[0]
    return {"n": len(size), "n_all_tiles": len({p.cluster for p in placement.values()}),
            "largest": largest, "largest_frac": largest / len(kept), "largest_cluster": big,
            "largest_n_groups": len({group_str(t) for t, k in kept.items() if k == big}),
            "top10_sizes": [n for _, n in size.most_common(10)],
            "clusters_per_set": {s: len({p.cluster for p in placement.values() if p.split == s}) for s in SETS}}


def quantiles(values) -> dict[str, float]:
    if not values:
        return {}
    return {q: round(float(np.quantile(values, float(q))), 4) for q in ("0", "0.1", "0.25", "0.5", "0.75", "0.9", "1")}


def w_stats(results, placement, in_w, w_cov, cfg, w_sids) -> tuple[dict, dict]:
    wc = cfg["walls"]
    kept = [t for t, p in placement.items() if p.keep]
    excluded = {}
    for name, sets in [*((s, {s}) for s in SETS), ("all", set(SETS))]:
        wall = [t for t in kept if placement[t].split in sets and w_cov[t] is not None]
        n_ex = sum(not in_w[t] for t in wall)
        excluded[name] = {"wall_tiles": len(wall), "excluded": n_ex, "frac": n_ex / len(wall) if wall else 0.0}
    covs = [w_cov[t] for t in kept if w_cov[t] is not None]
    coverage = {
        "coverage_min": wc["coverage_min"], "stop_if_excluded_frac": wc["stop_if_excluded_frac"],
        "n_wall_tiles": len(covs), "quantiles": quantiles(covs),
        "histogram": np.histogram(covs, bins=10, range=(0, 1))[0].tolist(), "excluded": excluded,
        "n_polygons_in_W": sum(len(results[t].walls[s].rings) for t in kept if in_w[t]
                               for s in w_sids if s in results[t].walls),
    }
    candidates = {}
    for sid in wc["candidates"]:
        c = [results[t].walls[sid].coverage for t in kept if sid in results[t].walls]
        med = float(np.median(c)) if c else None
        candidates[str(sid)] = {"n_tiles": len(c), "median_coverage": med, "enabled": sid in w_sids,
                                "would_enable": med is not None and med >= wc["candidate_enable_median_cov"]}
    return coverage, candidates


def misaligned_stats(results) -> dict:
    vals = sorted((r.align_frac, image_id(t)) for t, r in results.items() if r.align_frac is not None)
    bad = [v for v in vals if v[0] < ALIGN_MIN]
    return {"threshold": ALIGN_MIN, "stop_frac": MISALIGNED_STOP_FRAC, "n_checked": len(vals),
            "n_misaligned": len(bad), "frac": len(bad) / len(vals) if vals else 0.0,
            "quantiles": quantiles([v for v, _ in vals]),
            "worst": [{"image_id": i, "align_frac": round(v, 4)} for v, i in vals[:20]]}


def box_stats(results, placement) -> dict:
    kept = [t for t, p in placement.items() if p.keep]
    total = Counter()
    for t in kept:
        total.update(results[t].box_stats)
    return {**total, "n_boxes": sum(len(results[t].det_boxes) for t in kept),
            "negative_tiles": sum(not results[t].det_boxes for t in kept),
            "skipped_svg_elements": sum(results[t].skipped for t in kept),
            "n_boxes_per_set": {s: sum(len(results[t].det_boxes) for t in kept if placement[t].split == s)
                                for s in SETS}}


def core_classes(report: dict, schema: Schema, core: dict) -> dict:
    d = schema.for_model("D")
    n = report["class_instances"]
    sids = [c.semantic_id for c in d if str(c.semantic_id) in n
            and n[str(c.semantic_id)]["train"] >= core["min_train"] and n[str(c.semantic_id)]["test"] >= core["min_test"]]
    schema_id = {c.semantic_id: c.schema_id for c in d}
    return {"min_train": core["min_train"], "min_test": core["min_test"], "semantic_ids": sids,
            "schema_ids": [schema_id[s] for s in sids]}


# ----------------------------------------------------------------- output ---


def write_manifest(path: Path, results, placement, in_w, w_cov) -> None:
    def num(v):
        return "" if v is None else round(v, 4)

    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(COLUMNS)
        for t in sorted(results):
            r, p = results[t], placement[t]
            wr.writerow([t[0], t[1], r.md5, p.keep, p.drop_reason, group_str(t), p.cluster, p.split,
                         len(r.det_boxes), r.wall_strokes, num(w_cov[t]), in_w[t], num(r.align_frac),
                         r.width, r.height])


def write_datasets(out: Path, src: Path, results, placement, in_w, schema: Schema, workers: int) -> None:
    d, w = out / "D", out / "W"
    for root in (d, w):
        for sub in ("images", "labels"):
            for s in SETS:
                (root / sub / s).mkdir(parents=True, exist_ok=True)
    kept = [t for t in sorted(results) if placement[t].keep]
    img = {t: d / "images" / placement[t].split / f"{image_id(t)}.png" for t in kept}
    jobs = [(src / t[0] / f"{t[1]}.png", img[t]) for t in kept]
    for n, _ in enumerate(_pmap(_write_image, jobs, workers), 1):
        if n % 2000 == 0:
            log(f"wrote {n}/{len(jobs)} images")
    w_yolo = schema.yolo_index("W")
    for t in kept:
        r, s, name = results[t], placement[t].split, f"{image_id(t)}.txt"
        det = format_yolo_det(r.det_boxes)
        (d / "labels" / s / name).write_text(det + "\n" if det else "", encoding="utf-8")
        if not in_w[t]:
            continue
        w_img = w / "images" / s / img[t].name
        try:
            os.link(img[t], w_img)
        except OSError:
            shutil.copyfile(img[t], w_img)
        lines = [format_yolo_seg(w_yolo[sid], ring) for sid in sorted(w_yolo) if sid in r.walls
                 for ring in r.walls[sid].rings]
        (w / "labels" / s / name).write_text("".join(f"{l}\n" for l in lines), encoding="utf-8")
    for root, model in ((d, "D"), (w, "W")):
        data = {"path": str(root), **{s: f"images/{s}" for s in SETS}, "names": schema.yolo_names(model)}
        (root / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")


def list_tiles(src: Path, archives: list[str]) -> list[TileId]:
    tiles = []
    for a in archives:
        svgs = sorted((src / a).glob("*.svg"))
        if not svgs:
            raise FileNotFoundError(f"no SVG tiles in {src / a}")
        missing = [p.name for p in svgs if not p.with_suffix(".png").exists()]
        if missing:
            log(f"{a}: {len(missing)} SVGs without a PNG skipped, e.g. {missing[:3]}")
        tiles += [(a, p.stem) for p in svgs if p.with_suffix(".png").exists()]
    return sorted(tiles)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fpyolo.build", description=__doc__.split("\n")[0])
    ap.add_argument("--src", type=Path, required=True, help="FloorPlanCAD root (holds the archive folders)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--reuse-manifest", type=Path, help="tiles.csv whose keep/cluster/split columns are reused")
    args = ap.parse_args(argv)
    cfg = load_config(args.config)
    schema = load_schema(cfg["schema"])
    out, src = args.out.resolve(), args.src.resolve()
    tiles = list_tiles(src, cfg["archives"])
    log(f"{len(tiles)} tiles in {cfg['archives']}")
    results = analyse_all(src, tiles, cfg, schema, out / "cache", args.workers)

    if args.reuse_manifest:
        placement = read_manifest(args.reuse_manifest, tiles)
        old = args.reuse_manifest.with_name("stats.json")
        prev = json.loads(old.read_text(encoding="utf-8")) if old.exists() else {}
        stats = {k: prev.get(k, {"reused_manifest": str(args.reuse_manifest)})
                 for k in ("dedup", "clusters", "cross_space_candidates")}
        stats["reused_manifest"] = str(args.reuse_manifest.resolve())
    else:
        placement, stats = place_tiles(results, cfg)
    stats["clusters"] = {**stats["clusters"], **cluster_size_stats(placement)}

    kept = [t for t in sorted(results) if placement[t].keep]
    counts = class_counts_of(results, kept)
    report = splitmod.split_report({t: placement[t].split for t in kept}, counts)
    w_sids = [c.semantic_id for c in schema.for_model("W")]
    w_cov = {t: w_coverage(r, w_sids) for t, r in results.items()}
    in_w = {t: placement[t].keep and (w_cov[t] is None or w_cov[t] >= cfg["walls"]["coverage_min"])
            for t in results}
    stats["split_report"] = report
    stats["box_stats"] = box_stats(results, placement)
    stats["w_coverage"], stats["w_candidates"] = w_stats(results, placement, in_w, w_cov, cfg, w_sids)
    stats["misaligned"] = misaligned_stats(results)

    for name in ("D", "W"):  # never leave a dataset that disagrees with the new manifest
        shutil.rmtree(out / name, ignore_errors=True)
    man = out / "manifests"
    man.mkdir(parents=True, exist_ok=True)
    write_manifest(man / "tiles.csv", results, placement, in_w, w_cov)
    (man / "stats.json").write_text(json.dumps(stats, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    core = core_classes(report, schema, cfg["core"])
    (man / "core_classes.json").write_text(json.dumps(core) + "\n", encoding="utf-8")

    c, ex, mis = stats["clusters"], stats["w_coverage"]["excluded"]["all"], stats["misaligned"]
    log(f"kept {len(kept)}/{len(tiles)}; {c['n']} clusters, largest {c['largest']} ({c['largest_frac']:.1%}); "
        f"tile ratios {report['tile_ratio']}")
    log(f"classes absent from some set: {report['classes_absent_from_some_set']}; core classes: {core['semantic_ids']}")
    log(f"W: {ex['excluded']}/{ex['wall_tiles']} wall tiles excluded ({ex['frac']:.1%}); "
        f"misaligned {mis['n_misaligned']}/{mis['n_checked']} ({mis['frac']:.2%})")
    if mis["frac"] > MISALIGNED_STOP_FRAC:
        log(f"STOP: more than {MISALIGNED_STOP_FRAC:.0%} of tiles have align_frac < {ALIGN_MIN}")
        return EXIT_MISALIGNED
    if ex["frac"] > cfg["walls"]["stop_if_excluded_frac"]:
        log(f"STOP: {ex['frac']:.1%} of wall tiles excluded from W > {cfg['walls']['stop_if_excluded_frac']:.0%}")
        return EXIT_W_EXCLUDED
    write_datasets(out, src, results, placement, in_w, schema, args.workers)
    log(f"datasets written to {out}")
    return 0


if __name__ == "__main__":
    # Run the importable module, not this __main__ copy: the cache pickles TileResult, and its
    # class must be fpyolo.build.TileResult for any later in-process reader.
    from fpyolo.build import main as _main

    sys.exit(_main())
