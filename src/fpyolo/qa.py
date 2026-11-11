"""Real-data QA: class contact sheets, the analytic-bbox vs. re-rasterization check,
label overlays of a built dataset and side-by-side sheets of cross-space candidate pairs.

    python -m fpyolo.qa contact-sheet --src <FloorPlanCAD> --out reports/qa/classes --per-class 12 --seed 0
    python -m fpyolo.qa bbox-raster --src <FloorPlanCAD> --n 500 --seed 0 --out reports/qa/bbox_raster.json
    python -m fpyolo.qa overlays --dataset <out>/D --n 50 --seed 0 --out reports/qa/overlays_D
    python -m fpyolo.qa pair-sheet --stats <out>/manifests/stats.json --src <FloorPlanCAD> --out reports/qa/cross_space

Pixel convention: image coordinates are viewBox coordinates scaled to the
true image size; pixel (i, j) is centered on integer coordinates (OpenCV
convention) and `snap_px` maps a coordinate to its pixel. Boxes are
(x0, y0, x1, y1).
"""
from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from multiprocessing import Pool
from pathlib import Path

import cv2
import numpy as np
import yaml
from PIL import Image

from fpyolo.ink import load_ink
from fpyolo.svgparse import Primitive, Tile, parse_tile

ARCHIVES = ("train-00", "train-01", "test-00")
THING_SIDS = tuple(range(3, 33))
_LABEL_ATTR = "{http://www.inkscape.org/namespaces/inkscape}label"
_CELL = 360  # side of one crop in the contact sheet, px
_CAPTION_H = 22
_GAP = 12
_PAIRS_PER_ROW = 2
_INK_BOX = (220, 0, 0)  # RGB
_VIS_BOX = (0, 0, 255)
_LABEL_W = 140.0  # image px a coco_vis label such as "airconditioner" spans


# ---------------------------------------------------------------- geometry ---


_SNAP_EPS = 1e-6  # px


def snap_px(v) -> np.ndarray:
    """Round half up to the pixel grid, tolerating float noise: CAD coordinates
    often land exactly on x.5 px, where 1e-13 of noise would otherwise flip the
    result by a whole pixel."""
    return np.floor(np.asarray(v, dtype=np.float64) + 0.5 + _SNAP_EPS)


def render_polylines(polylines_px: list[np.ndarray], h: int, w: int) -> np.ndarray:
    """Draw (N, 2) pixel polylines as 1-px, non-anti-aliased strokes; bool (h, w) mask."""
    canvas = np.zeros((h, w), dtype=np.uint8)
    pts = [snap_px(p).astype(np.int32).reshape(-1, 1, 2) for p in polylines_px if len(p)]
    if pts:
        cv2.polylines(canvas, pts, isClosed=False, color=1, thickness=1, lineType=cv2.LINE_8)
    return canvas.astype(bool)


def tight_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """(x0, y0, x1, y1) of the extreme set pixels (inclusive indices), None if empty."""
    xs = np.flatnonzero(mask.any(axis=0))
    if len(xs) == 0:
        return None
    ys = np.flatnonzero(mask.any(axis=1))
    return int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])


def box_iou(a, b) -> float:
    """IoU of two (x0, y0, x1, y1) boxes; 0.0 when the union has no area."""
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    inter = max(iw, 0.0) * max(ih, 0.0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return float(inter / union) if union > 0 else 0.0


def thing_instances(tile: Tile) -> dict[tuple[int, int], list[Primitive]]:
    """Primitives of every thing object, keyed by (semantic_id, instance_id)."""
    out: dict[tuple[int, int], list[Primitive]] = defaultdict(list)
    for p in tile.primitives:
        if p.semantic_id in THING_SIDS and p.instance_id >= 0:
            out[(p.semantic_id, p.instance_id)].append(p)
    return dict(out)


def _px_affine(tile: Tile, img_w: int, img_h: int) -> tuple[np.ndarray, np.ndarray]:
    vx0, vy0, vx1, vy1 = tile.viewbox
    return np.array([vx0, vy0]), np.array([img_w / (vx1 - vx0), img_h / (vy1 - vy0)])


def instance_box_px(prims: list[Primitive], origin: np.ndarray, scale: np.ndarray) -> np.ndarray:
    """Union of the primitives' analytic bboxes, in image pixels: [x0, y0, x1, y1]."""
    b = np.array([p.bbox for p in prims])
    lo = (b[:, :2].min(axis=0) - origin) * scale
    hi = (b[:, 2:].max(axis=0) - origin) * scale
    return np.concatenate([lo, hi])


def instance_ious(tile: Tile, img_w: int, img_h: int, min_side_px: float = 4.0) -> list[dict]:
    """Per thing object with both box sides >= `min_side_px`: IoU between its
    analytic box and the tight box of its re-rasterized polylines.

    Each object is drawn alone on a local canvas covering both its box and its
    polylines, so objects crossing the image border are not clipped. The
    analytic box is snapped with the same `snap_px` as the polyline vertices,
    so sub-pixel quantization is not counted as error.
    """
    origin, scale = _px_affine(tile, img_w, img_h)
    rows = []
    for (sid, iid), prims in sorted(thing_instances(tile).items()):
        box = instance_box_px(prims, origin, scale)
        if min(box[2] - box[0], box[3] - box[1]) < min_side_px:
            continue
        polys = [(p.polyline - origin) * scale for p in prims]
        pts = np.concatenate(polys)
        shift = np.floor(np.minimum(box[:2], pts.min(axis=0))) - 2
        far = np.ceil(np.maximum(box[2:], pts.max(axis=0)) - shift) + 3
        mask = render_polylines([q - shift for q in polys], int(far[1]), int(far[0]))
        snapped = snap_px(box) - np.tile(shift, 2)
        tb = tight_bbox(mask)
        rows.append(
            {
                "sid": sid,
                "iid": iid,
                "iou": 0.0 if tb is None else box_iou(snapped, tb),
                "analytic_px": [round(float(v), 2) for v in box],
                "raster_px": None if tb is None else [int(v) for v in np.array(tb) + np.tile(shift, 2)],
            }
        )
    return rows


# ---------------------------------------------------------------- coco_vis ---


def _outer_lines(counts: np.ndarray, rel: float = 0.5) -> tuple[int, int]:
    """First and last index among the lines with at least `rel` x the top count."""
    idx = np.flatnonzero(counts >= rel * counts.max())
    return int(idx[0]), int(idx[-1])


def detect_frame(rgb: np.ndarray, dark: int = 80) -> tuple[int, int, int, int]:
    """(x0, y0, x1, y1) of the matplotlib axes frame.

    The spines are rows / columns with many dark pixels (counts, not contiguous
    runs: red label text is drawn over the spines, hiding >10% of one at
    times). Drawing lines spanning the whole axes can score as high, but they
    are clipped to the axes, while nothing long and dark lies outside it, so the
    spines are the outermost lines scoring at least half the top count."""
    black = rgb[..., :3].max(axis=-1) < dark
    y0, y1 = _outer_lines(black.sum(axis=1))
    x0, x1 = _outer_lines(black.sum(axis=0))
    return x0, y0, x1, y1


def vis_affine(frame: tuple[int, int, int, int], img_w: int, img_h: int) -> tuple[float, float, float, float]:
    """(ox, oy, sx, sy) with vis = o + s * image_px; the frame spans the whole image.

    Raises ValueError when the frame is implausible (non-square pixels or a
    scale far from the ~1.54 px/unit of the 2000 px renders)."""
    x0, y0, x1, y1 = frame
    sx, sy = (x1 - x0) / img_w, (y1 - y0) / img_h
    if abs(sx / sy - 1) > 0.01 or not 1.4 < sx < 1.7:
        raise ValueError(f"implausible coco_vis frame {frame} for a {img_w}x{img_h} image")
    return float(x0), float(y0), sx, sy


# ----------------------------------------------------------- contact sheet ---


def _scan_svg(job: tuple[str, str]) -> tuple[str, str, dict[tuple[int, int], list[str]]]:
    """Thing instances of one SVG with the inkscape layer labels of their elements."""
    archive, path = job
    layers: dict[tuple[int, int], set[str]] = defaultdict(set)
    stack = [(ET.parse(path).getroot(), "")]
    while stack:
        el, layer = stack.pop()
        layer = el.get(_LABEL_ATTR, layer)
        try:
            key = (int(el.get("semantic-id")), int(el.get("instance-id")))
        except (TypeError, ValueError):
            key = None
        if key and key[0] in THING_SIDS and key[1] >= 0:
            layers[key].add(layer)
        stack.extend((c, layer) for c in el)
    return archive, Path(path).stem, {k: sorted(v) for k, v in layers.items()}


def _drawing_key(archive: str, stem: str) -> tuple[str, str]:
    return ("A" if archive == "train-00" else "B", stem.split("-")[0])


def _crop_padded(img: np.ndarray, region, fill: int) -> np.ndarray:
    x0, y0, x1, y1 = (int(round(v)) for v in region)
    out = np.full((y1 - y0, x1 - x0) + img.shape[2:], fill, dtype=img.dtype)
    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, img.shape[1]), min(y1, img.shape[0])
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = img[sy0:sy1, sx0:sx1]
    return out


def _cell(crop: np.ndarray, box_cell, color) -> np.ndarray:
    cell = np.ascontiguousarray(cv2.resize(crop, (_CELL, _CELL), interpolation=cv2.INTER_AREA))
    x0, y0, x1, y1 = (int(round(v)) for v in box_cell)
    cv2.rectangle(cell, (x0, y0), (x1, y1), color, 1)
    return cell


def _crop_region(box: np.ndarray) -> np.ndarray:
    """Square image-px region around a box with a margin, wide enough to the
    right of the box's left edge for the coco_vis label printed there."""
    m = max(40.0, 0.35 * max(box[2] - box[0], box[3] - box[1]))
    x0, y0 = box[0] - m, box[1] - m
    x1, y1 = max(box[2] + m, box[0] + _LABEL_W), box[3] + m
    half = max(x1 - x0, y1 - y0) / 2
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    return np.array([cx - half, cy - half, cx + half, cy + half])


def crop_pair(src: Path, archive: str, stem: str, iid: int, box: np.ndarray) -> tuple[np.ndarray, tuple]:
    """Side-by-side [ink crop | coco_vis crop] around one object, captioned with
    the tile name. Returns (image, detected coco_vis frame)."""
    ink = load_ink(src / archive / f"{stem}.png")
    img_h, img_w = ink.shape
    with Image.open(src / archive / "coco_vis" / f"{stem}.png") as im:
        vis = np.asarray(im.convert("RGB"))
    frame = detect_frame(vis)
    ox, oy, sx, sy = vis_affine(frame, img_w, img_h)

    region = _crop_region(box)
    box_cell = (box - np.tile(region[:2], 2)) * (_CELL / (region[2] - region[0]))

    ink_rgb = np.repeat(ink[..., None], 3, axis=2)
    left = _cell(_crop_padded(ink_rgb, region, 255), box_cell, _INK_BOX)
    vis_region = (ox + region[0] * sx, oy + region[1] * sy, ox + region[2] * sx, oy + region[3] * sy)
    right = _cell(_crop_padded(vis, vis_region, 255), box_cell, _VIS_BOX)

    pair = np.full((_CELL + _CAPTION_H, 2 * _CELL + 4, 3), 255, dtype=np.uint8)
    pair[:_CELL, :_CELL] = left
    pair[:_CELL, _CELL + 4 :] = right
    cv2.putText(pair, f"{archive}/{stem} iid={iid}", (4, _CELL + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    return pair, frame


def _grid(header: str, pairs: list[np.ndarray]) -> np.ndarray:
    ph, pw = pairs[0].shape[:2] if pairs else (_CELL + _CAPTION_H, 2 * _CELL + 4)
    rows = max(1, -(-len(pairs) // _PAIRS_PER_ROW))
    head = 30
    sheet = np.full((head + rows * (ph + _GAP), _PAIRS_PER_ROW * (pw + _GAP), 3), 255, dtype=np.uint8)
    cv2.putText(sheet, header, (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1, cv2.LINE_AA)
    for i, p in enumerate(pairs):
        r, c = divmod(i, _PAIRS_PER_ROW)
        y, x = head + r * (ph + _GAP), c * (pw + _GAP)
        sheet[y : y + ph, x : x + pw] = p
    return sheet


def _tile_boxes(src: Path, archive: str, stem: str):
    """({(sid, iid): box_px}, (img_w, img_h)) for one tile."""
    tile = parse_tile(src / archive / f"{stem}.svg", archive)
    with Image.open(src / archive / f"{stem}.png") as im:
        img_w, img_h = im.size
    origin, scale = _px_affine(tile, img_w, img_h)
    boxes = {k: instance_box_px(v, origin, scale) for k, v in thing_instances(tile).items()}
    return boxes, (img_w, img_h)


def _pick(src: Path, sid: int, cands: list[tuple[str, str, int]], k: int, rng) -> list[tuple[str, str, int, np.ndarray]]:
    """Up to k objects of `sid` from distinct tiles. First pass: distinct
    drawings and boxes fully inside the image; second pass relaxes both."""
    order = rng.permutation(len(cands))
    parsed: dict[tuple[str, str], tuple] = {}
    chosen, used_tiles, used_drawings = [], set(), set()
    for strict in (True, False):
        for i in order:
            if len(chosen) == k:
                return chosen
            archive, stem, iid = cands[i]
            drawing = _drawing_key(archive, stem)
            if (archive, stem) in used_tiles or (strict and drawing in used_drawings):
                continue
            if not (src / archive / "coco_vis" / f"{stem}.png").exists():
                continue
            if (archive, stem) not in parsed:
                parsed[(archive, stem)] = _tile_boxes(src, archive, stem)
            boxes, (img_w, img_h) = parsed[(archive, stem)]
            box = boxes.get((sid, iid))
            if box is None:
                continue
            inside = box[0] >= 0 and box[1] >= 0 and box[2] <= img_w and box[3] <= img_h
            if strict and not inside:
                continue
            chosen.append((archive, stem, iid, box))
            used_tiles.add((archive, stem))
            used_drawings.add(drawing)
    return chosen


def _sheet_worker(job) -> str:
    sid, cands, layer_counts, src, out_dir, k, seed = job
    picks = _pick(src, sid, cands, k, np.random.default_rng([seed, sid]))
    pairs, lines = [], []
    for archive, stem, iid, box in picks:
        try:
            pair, frame = crop_pair(src, archive, stem, iid, box)
        except ValueError as e:  # implausible coco_vis frame: keep the evidence trail honest
            lines.append(f"# skipped {archive}/{stem} iid={iid}: {e}")
            continue
        pairs.append(pair)
        lines.append(
            f"{archive}\t{stem}\t{iid}\t{' '.join(f'{v:.1f}' for v in box)}\t{' '.join(map(str, frame))}"
        )
    header = f"sid {sid}: {len(pairs)} objects (left: ink + analytic box; right: coco_vis + converted box)"
    sheet = Image.fromarray(_grid(header, pairs))  # 256-colour palette keeps the committed sheets ~3x smaller
    sheet = sheet.quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    sheet.save(out_dir / f"sid_{sid:02d}.png", optimize=True)
    top = ", ".join(f"{name or '(none)'} ({n})" for name, n in layer_counts.most_common(5))
    txt = [
        f"# sid {sid}: {len(cands)} objects in all tiles; {len(pairs)} shown (seed {seed})",
        f"# common layers (objects per inkscape layer label, all tiles): {top}",
        "# archive\tstem\tinstance_id\tbox_px x0 y0 x1 y1\tcoco_vis_frame x0 y0 x1 y1",
        *lines,
    ]
    (out_dir / f"sid_{sid:02d}.txt").write_text("\n".join(txt) + "\n", encoding="utf-8")
    return f"sid {sid}: {len(pairs)} pairs from {len(cands)} objects"


def cmd_contact_sheet(args) -> int:
    src, out = Path(args.src), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    jobs = [(a, str(p)) for a in ARCHIVES for p in sorted((src / a).glob("*.svg"))]
    cands: dict[int, list] = defaultdict(list)
    layers: dict[int, Counter] = defaultdict(Counter)
    with Pool(args.workers) as pool:
        for archive, stem, inst in pool.imap_unordered(_scan_svg, jobs, chunksize=64):
            for (sid, iid), labels in inst.items():
                cands[sid].append((archive, stem, iid))
                layers[sid].update(labels)
        sheet_jobs = [
            (sid, sorted(cands[sid]), layers[sid], src, out, args.per_class, args.seed) for sid in THING_SIDS
        ]
        for msg in pool.imap(_sheet_worker, sheet_jobs):
            print(msg, flush=True)
    return 0


# ------------------------------------------------------------- bbox-raster ---


def _raster_worker(job) -> tuple[list[dict], int]:
    src, archive, stem, min_side = job
    tile = parse_tile(src / archive / f"{stem}.svg", archive)
    with Image.open(src / archive / f"{stem}.png") as im:
        img_w, img_h = im.size
    rows = instance_ious(tile, img_w, img_h, min_side)
    for r in rows:
        r["image_id"] = f"{archive}_{stem}"
    return rows, len(thing_instances(tile))


def cmd_bbox_raster(args) -> int:
    src = Path(args.src)
    tiles = [
        (a, p.stem)
        for a in ARCHIVES
        for p in sorted((src / a).glob("*.svg"))
        if p.with_suffix(".png").exists()
    ]
    rng = np.random.default_rng(args.seed)
    picked = [tiles[i] for i in sorted(rng.choice(len(tiles), size=min(args.n, len(tiles)), replace=False))]
    with Pool(args.workers) as pool:
        results = pool.map(_raster_worker, [(src, a, s, args.min_side) for a, s in picked], chunksize=4)
    rows = [r for rs, _ in results for r in rs]
    n_things = sum(n for _, n in results)
    ious = np.array([r["iou"] for r in rows])
    failures = [r for r in rows if r["iou"] < args.min_iou]
    pass_rate = 1.0 - len(failures) / len(rows) if rows else 0.0
    report = {
        "seed": args.seed,
        "n_tiles": len(picked),
        "min_side_px": args.min_side,
        "iou_threshold": args.min_iou,
        "required_pass_rate": args.required,
        "n_thing_objects": n_things,
        "n_checked": len(rows),
        "n_below_min_side": n_things - len(rows),
        "pass_rate": pass_rate,
        "passed": bool(rows) and pass_rate >= args.required,
        "iou_quantiles": {
            q: float(np.quantile(ious, float(q))) for q in ("0.0", "0.001", "0.01", "0.05", "0.5")
        } if rows else {},
        "failures_by_sid": dict(sorted(Counter(r["sid"] for r in failures).items())),
        "failures": failures,
        "tiles": [f"{a}_{s}" for a, s in picked],
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=1, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        f"checked {len(rows)} objects on {len(picked)} tiles: pass rate {pass_rate:.4%} "
        f"(IoU >= {args.min_iou}), {len(failures)} failures -> {'PASS' if report['passed'] else 'FAIL'}"
    )
    return 0 if report["passed"] else 1


# ---------------------------------------------------------------- overlays ---


_FILL_ALPHA = 0.45  # opacity of filled segmentation polygons
_PAIR_CAPTION_H = 28


def _class_color(cls: int) -> tuple[int, int, int]:
    """Saturated RGB colour per class index (golden-angle hue steps keep neighbours apart)."""
    hsv = np.uint8([[[int(cls * 137.5) % 180, 255, 210]]])
    return tuple(int(v) for v in cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)[0, 0])


def _save_palette_png(rgb: np.ndarray, path: Path) -> None:
    """256-colour palette PNG: line drawings stay sharp and the files ~3x smaller."""
    im = Image.fromarray(rgb).quantize(256, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    im.save(path, optimize=True)


def overlay_labels(rgb: np.ndarray, label_text: str, names: dict[int, str]) -> np.ndarray:
    """Draw YOLO label lines on an RGB image: `cls cx cy w h` lines as boxes captioned with the
    class name, longer lines (`cls x1 y1 x2 y2 ...`) as translucent filled polygons with outlines."""
    h, w = rgb.shape[:2]
    fill = rgb.copy()
    boxes, polys = [], []
    for line in label_text.splitlines():
        vals = line.split()
        if not vals:
            continue
        cls, xy = int(vals[0]), np.array(vals[1:], dtype=np.float64)
        color = _class_color(cls)
        if len(xy) == 4:
            cx, cy, bw, bh = xy * (w, h, w, h)
            p0 = (int(round(cx - bw / 2)), int(round(cy - bh / 2)))
            p1 = (int(round(cx + bw / 2)), int(round(cy + bh / 2)))
            boxes.append((p0, p1, color, names.get(cls, str(cls))))
        else:
            pts = np.round(xy.reshape(-1, 2) * (w, h)).astype(np.int32)
            cv2.fillPoly(fill, [pts], color)
            polys.append((pts, color))
    out = cv2.addWeighted(fill, _FILL_ALPHA, rgb, 1 - _FILL_ALPHA, 0) if polys else rgb.copy()
    for pts, color in polys:
        cv2.polylines(out, [pts], True, color, 1, cv2.LINE_8)
    for p0, p1, color, name in boxes:
        cv2.rectangle(out, p0, p1, color, 1)
        cv2.putText(out, name, (p0[0], max(p0[1] - 3, 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)
    return out


def cmd_overlays(args) -> int:
    ds, out = Path(args.dataset), Path(args.out)
    names = yaml.safe_load((ds / "data.yaml").read_text(encoding="utf-8"))["names"]
    images = sorted((ds / "images").glob("*/*.png"))
    pick = np.random.default_rng(args.seed).choice(len(images), size=min(args.n, len(images)), replace=False)
    out.mkdir(parents=True, exist_ok=True)
    for i in sorted(pick):
        img, split = images[i], images[i].parent.name
        lbl = ds / "labels" / split / f"{img.stem}.txt"
        with Image.open(img) as im:
            rgb = np.asarray(im.convert("RGB"))
        text = lbl.read_text(encoding="utf-8") if lbl.exists() else ""
        _save_palette_png(overlay_labels(rgb, text, names), out / f"{split}_{img.stem}.png")
    print(f"{len(pick)} overlays of {len(images)} images -> {out}")
    return 0


def cmd_pair_sheet(args) -> int:
    """One image per cross-space candidate pair: the two ink tiles side by side, captioned."""
    src, out = Path(args.src), Path(args.out)
    pairs = json.loads(Path(args.stats).read_text(encoding="utf-8"))["cross_space_candidates"]
    out.mkdir(parents=True, exist_ok=True)
    side = args.size
    for k, p in enumerate(pairs):
        sheet = np.full((side + _PAIR_CAPTION_H, 2 * side + _GAP), 255, dtype=np.uint8)
        for col, key in enumerate(("a", "b")):
            archive, stem = p[key].rsplit("_", 1)
            ink = load_ink(src / archive / f"{stem}.png")
            cell = cv2.resize(ink, (side, round(side * ink.shape[0] / ink.shape[1])), interpolation=cv2.INTER_AREA)
            x = col * (side + _GAP)
            sheet[: cell.shape[0], x : x + side] = cell
        sheet[:side, side : side + _GAP] = 160  # separator
        caption = f"#{k:02d} cos={p['cosine']:.4f}  {p['a']} ({p['group_a']})  |  {p['b']} ({p['group_b']})"
        cv2.putText(sheet, caption, (6, side + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.55, 0, 1, cv2.LINE_AA)
        Image.fromarray(sheet).save(out / f"pair_{k:02d}.png", optimize=True)
    print(f"{len(pairs)} pair sheets -> {out}")
    return 0



# --------------------------------------------------------------------- CLI ---


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fpyolo.qa", description="Real-data QA tools.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    cs = sub.add_parser("contact-sheet", help="per-class [ink | coco_vis] crop pairs for naming classes")
    cs.add_argument("--src", type=Path, required=True, help="FloorPlanCAD root (holds train-00, ...)")
    cs.add_argument("--out", type=Path, required=True)
    cs.add_argument("--per-class", type=int, default=12)
    cs.add_argument("--seed", type=int, default=0)
    cs.add_argument("--workers", type=int, default=16)
    cs.set_defaults(func=cmd_contact_sheet)

    br = sub.add_parser("bbox-raster", help="analytic bbox vs. tight bbox of re-rasterized polylines")
    br.add_argument("--src", type=Path, required=True, help="FloorPlanCAD root (holds train-00, ...)")
    br.add_argument("--out", type=Path, required=True, help="JSON report path")
    br.add_argument("--n", type=int, default=500, help="number of random tiles")
    br.add_argument("--seed", type=int, default=0)
    br.add_argument("--min-side", type=float, default=4.0, help="skip objects with a box side below this (px)")
    br.add_argument("--min-iou", type=float, default=0.9)
    br.add_argument("--required", type=float, default=0.99, help="required fraction of objects with IoU >= min-iou")
    br.add_argument("--workers", type=int, default=16)
    br.set_defaults(func=cmd_bbox_raster)

    ov = sub.add_parser("overlays", help="draw the labels of n random images of a built D or W dataset")
    ov.add_argument("--dataset", type=Path, required=True, help="dataset folder holding data.yaml, images/, labels/")
    ov.add_argument("--out", type=Path, required=True)
    ov.add_argument("--n", type=int, default=50)
    ov.add_argument("--seed", type=int, default=0)
    ov.set_defaults(func=cmd_overlays)

    ps = sub.add_parser("pair-sheet", help="side-by-side ink tiles of the cross-space candidate pairs in stats.json")
    ps.add_argument("--stats", type=Path, required=True, help="manifests/stats.json of a build")
    ps.add_argument("--src", type=Path, required=True, help="FloorPlanCAD root (holds train-00, ...)")
    ps.add_argument("--out", type=Path, required=True)
    ps.add_argument("--size", type=int, default=640, help="side of each tile in the sheet, px")
    ps.set_defaults(func=cmd_pair_sheet)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
