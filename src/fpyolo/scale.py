"""Estimate mm per pixel from numeric dimension annotations in FloorPlanCAD SVGs.

A dimension text such as "3600" (mm) sits next to a dimension line drawn on the same
layer (matched from the text centre by default). The line's length in viewBox units, times pixels per unit, is its length in
pixels, so mm/px = value_mm / length_px. The SVG is read directly (not via
`svgparse`) because dimension lines and texts are normally unlabeled.
"""
import argparse
import json
import math
import random
import re
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from svgelements import Matrix

INKSCAPE = "http://www.inkscape.org/namespaces/inkscape"
MAX_ANGLE_DEG = 2.0  # largest allowed deviation between text direction and segment
MAX_MID_DIST = 3.0  # largest allowed text-anchor to segment-midpoint distance (viewBox units)
MIN_SEGMENT_LEN = 1.0  # centre mode ignores candidate segments shorter than this (ticks, arrows)
CHAR_WIDTH_EM = 0.6  # estimated glyph advance, in font-size units
MIN_ESTIMATES = 3  # a tile needs this many estimates to be used in the measurement
CONSISTENT_TOL = 0.10  # an estimate agrees with its tile median when within this relative band
CONSISTENT_SHARE = 0.5  # a tile is consistent when at least this share of estimates agree
MIN_CONSISTENT = 20  # fewer consistent tiles than this makes the conclusion "inconclusive"
MAX_SCAN = 400  # extra scanning (to reach MIN_CONSISTENT) stops after this many tiles
CONSTANT_IQR_RATIO = 0.10  # IQR / median at or below this means "constant"
MODES = ("centre", "anchor")  # "anchor" is the literal rule: text start point, no minimum length
_DIGITS = re.compile(r"[0-9]{2,}")
_PATH_TOKEN = re.compile(r"[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class DimText:
    value_mm: float
    x: float
    y: float
    angle_deg: float  # text direction in viewBox coordinates, in [0, 360)
    layer: str
    font_size: float = 2.0  # viewBox units (the real files use 2)


@dataclass(frozen=True)
class Segment:
    layer: str
    x0: float
    y0: float
    x1: float
    y1: float


@dataclass(frozen=True)
class Match:
    text: DimText
    segment: Segment
    length: float  # viewBox units
    mm_per_px: float


def _local(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _own_matrix(elem) -> np.ndarray:
    t = elem.get("transform")
    if not t:
        return np.eye(3)
    m = Matrix(t)
    return np.array([[m.a, m.c, m.e], [m.b, m.d, m.f], [0.0, 0.0, 1.0]])


def _apply(m: np.ndarray, x: float, y: float) -> tuple[float, float]:
    return float(m[0, 0] * x + m[0, 1] * y + m[0, 2]), float(m[1, 0] * x + m[1, 1] * y + m[1, 2])


def _path_segment(d: str):
    """((x0, y0), (x1, y1)) when `d` is exactly one absolute "M x,y L x,y", else None."""
    tok = _PATH_TOKEN.findall(d)
    if len(tok) != 6 or tok[0] != "M" or tok[3] != "L":
        return None
    try:
        v = [float(tok[i]) for i in (1, 2, 4, 5)]
    except ValueError:
        return None
    return (v[0], v[1]), (v[2], v[3])


def _viewbox_width(root) -> float:
    vb = root.get("viewBox")
    if not vb:
        raise ValueError("SVG has no viewBox")
    return float(re.split(r"[\s,]+", vb.strip())[2])


def read_tile_geometry(svg_path) -> tuple[list[DimText], list[Segment], float]:
    """All numeric dimension texts, all straight segments, and the viewBox width."""
    root = ET.parse(svg_path).getroot()
    texts: list[DimText] = []
    segments: list[Segment] = []

    def walk(elem, parent_m: np.ndarray, layer: str) -> None:
        for child in elem:
            m = parent_m @ _own_matrix(child)
            name = _local(child.tag)
            if name == "g":
                lyr = layer
                if child.get(f"{{{INKSCAPE}}}groupmode") == "layer":
                    lyr = child.get(f"{{{INKSCAPE}}}label", "")
                walk(child, m, lyr)
            elif name == "text":
                content = "".join(child.itertext()).strip()
                if _DIGITS.fullmatch(content):
                    x, y = _apply(m, float(child.get("x", 0)), float(child.get("y", 0)))
                    ang = math.degrees(math.atan2(m[1, 0], m[0, 0])) % 360.0
                    scale = math.sqrt(abs(m[0, 0] * m[1, 1] - m[0, 1] * m[1, 0]))
                    try:
                        fs = float(child.get("font-size", 2.0)) * scale
                    except ValueError:
                        fs = 2.0 * scale
                    texts.append(DimText(float(content), x, y, ang, layer, fs))
            elif name == "path":
                pts = _path_segment(child.get("d", ""))
                if pts:
                    (ax, ay), (bx, by) = _apply(m, *pts[0]), _apply(m, *pts[1])
                    segments.append(Segment(layer, ax, ay, bx, by))
            elif name == "line":
                try:
                    p0 = _apply(m, float(child.get("x1", 0)), float(child.get("y1", 0)))
                    p1 = _apply(m, float(child.get("x2", 0)), float(child.get("y2", 0)))
                except ValueError:
                    continue
                segments.append(Segment(layer, *p0, *p1))

    walk(root, np.eye(3), "")
    return texts, segments, _viewbox_width(root)


def numeric_dimension_texts(svg_path) -> list[DimText]:
    """`<text>` elements whose stripped content is digits only (at least 2 digits)."""
    return read_tile_geometry(svg_path)[0]


def text_reference_point(t: DimText, mode: str = "centre") -> tuple[float, float]:
    """The point matched against segment midpoints.

    "anchor": the `<text>` start point as written. "centre": the anchor moved half the
    estimated text width (n_chars * 0.6 * font-size) along the text direction.
    """
    if mode == "anchor":
        return t.x, t.y
    if mode != "centre":
        raise ValueError(f"unknown mode {mode!r}, expected one of {MODES}")
    half = len(str(int(t.value_mm))) * CHAR_WIDTH_EM * t.font_size / 2.0
    a = math.radians(t.angle_deg)
    return t.x + half * math.cos(a), t.y + half * math.sin(a)


def _best_segment(t: DimText, segments, mode: str = "centre") -> Segment | None:
    rx, ry = text_reference_point(t, mode)
    min_len = MIN_SEGMENT_LEN if mode == "centre" else 0.0
    best, best_d = None, math.inf
    for s in segments:
        if s.layer != t.layer:
            continue
        dx, dy = s.x1 - s.x0, s.y1 - s.y0
        if (dx == 0.0 and dy == 0.0) or math.hypot(dx, dy) < min_len:
            continue
        diff = abs(math.degrees(math.atan2(dy, dx)) - t.angle_deg) % 180.0
        if min(diff, 180.0 - diff) > MAX_ANGLE_DEG:
            continue
        d = math.hypot((s.x0 + s.x1) / 2 - rx, (s.y0 + s.y1) / 2 - ry)
        if d <= MAX_MID_DIST and d < best_d:
            best, best_d = s, d
    return best


def match_dimension_line(t: DimText, segments, mode: str = "centre") -> float | None:
    """Length (viewBox units) of the nearest same-layer parallel segment near the text.

    Parallel means within 2 degrees of the text direction (modulo 180); near means the
    segment midpoint is within 3 viewBox units of the text reference point (the text
    centre by default, or the start anchor with mode="anchor"; see `text_reference_point`).
    In centre mode segments shorter than 1 viewBox unit are ignored.
    """
    s = _best_segment(t, segments, mode)
    return None if s is None else math.hypot(s.x1 - s.x0, s.y1 - s.y0)


def match_tile(svg_path, img_w: int, mode: str = "centre") -> list[Match]:
    texts, segments, vb_w = read_tile_geometry(svg_path)
    px_per_unit = img_w / vb_w
    out = []
    for t in texts:
        s = _best_segment(t, segments, mode)
        if s is None:
            continue
        length = math.hypot(s.x1 - s.x0, s.y1 - s.y0)
        out.append(Match(t, s, length, t.value_mm / (length * px_per_unit)))
    return out


def measure_tile(svg_path, img_w: int, mode: str = "centre") -> list[float]:
    """mm/px estimates, one per dimension text that found its line."""
    return [m.mm_per_px for m in match_tile(svg_path, img_w, mode)]


def _png_width(png_path) -> int:
    with Image.open(png_path) as im:
        return im.size[0]


def _scan(job):
    """Worker (module level): (archive, stem, estimates) for one tile."""
    archive, stem, svg, png, mode = job
    return archive, stem, measure_tile(svg, _png_width(png), mode)


def list_tiles(src: Path) -> list[tuple[str, str, Path, Path]]:
    tiles = []
    for arch in sorted(p for p in Path(src).iterdir() if p.is_dir()):
        for svg in sorted(arch.glob("*.svg")):
            png = svg.with_suffix(".png")
            if png.exists():
                tiles.append((arch.name, svg.stem, svg, png))
    return tiles


def consistency(estimates) -> float:
    """Share of `estimates` within +-10% of their median."""
    med = float(np.median(estimates))
    return float(np.mean([abs(e - med) <= CONSISTENT_TOL * med for e in estimates]))


def is_consistent(estimates) -> bool:
    return consistency(estimates) >= CONSISTENT_SHARE


def select_tiles(src: Path, n: int, seed: int, workers: int, mode: str = "centre",
                 min_consistent: int = MIN_CONSISTENT, max_scan: int = MAX_SCAN):
    """Shuffle all tiles with `seed`; keep the ones having >= 3 estimates.

    Stops once `n` tiles are kept and either `min_consistent` of them are consistent or
    `max_scan` tiles have been scanned (so extra tiles are added only when needed).
    """
    tiles = [(a, s, svg, png, mode) for a, s, svg, png in list_tiles(src)]
    random.Random(seed).shuffle(tiles)
    chosen, scanned, n_cons = [], 0, 0
    if workers <= 1:
        results = map(_scan, tiles)
        pool = None
    else:
        pool = Pool(workers)
        results = pool.imap(_scan, tiles, chunksize=8)
    try:
        for archive, stem, est in results:
            scanned += 1
            if len(est) >= MIN_ESTIMATES:
                chosen.append((archive, stem, est))
                n_cons += is_consistent(est)
            if len(chosen) >= n and (n_cons >= min_consistent or scanned >= max_scan):
                break
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    return chosen, scanned, len(tiles)


def summarize(chosen, min_consistent: int = MIN_CONSISTENT) -> dict:
    """Per-tile medians and consistency; the overall statistics use consistent tiles only."""
    tiles = []
    for a, s, est in chosen:
        tiles.append({
            "image_id": f"{a}_{s}",
            "archive": a,
            "stem": s,
            "n_estimates": len(est),
            "median": float(np.median(est)),
            "consistency": consistency(est),
            "consistent": is_consistent(est),
            "estimates": [float(e) for e in est],
        })
    medians = [t["median"] for t in tiles if t["consistent"]]
    res = {"tiles": tiles, "n_tiles": len(tiles), "n_consistent": len(medians),
           "min_consistent": min_consistent}
    if medians:
        q1, med, q3 = (float(v) for v in np.percentile(medians, [25, 50, 75]))
        res.update(overall_median=med, q1=q1, q3=q3, iqr=q3 - q1, iqr_over_median=(q3 - q1) / med)
    else:
        res.update(overall_median=None, q1=None, q3=None, iqr=None, iqr_over_median=None)
    if len(medians) < min_consistent:
        res["conclusion"] = "inconclusive"
    else:
        res["conclusion"] = "constant" if res["iqr_over_median"] <= CONSTANT_IQR_RATIO else "varies"
    return res


def draw_overlay(svg_path, png_path, out_path, mode: str = "centre") -> None:
    """Ink image with each matched text reference point (red) and its segment (blue)."""
    from fpyolo.ink import load_ink

    ink = load_ink(png_path)
    h, w = ink.shape
    img = Image.fromarray(np.repeat(ink[..., None], 3, axis=2), "RGB")
    _, _, vb_w = read_tile_geometry(svg_path)
    k = w / vb_w
    d = ImageDraw.Draw(img)
    for m in match_tile(svg_path, w, mode):
        s, t = m.segment, m.text
        rx, ry = text_reference_point(t, mode)
        d.line([(s.x0 * k, s.y0 * k), (s.x1 * k, s.y1 * k)], fill=(0, 80, 255), width=3)
        d.ellipse([rx * k - 6, ry * k - 6, rx * k + 6, ry * k + 6], outline=(255, 0, 0), width=3)
        d.text((rx * k + 8, ry * k - 12), f"{t.value_mm:g}", fill=(255, 0, 0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.quantize(colors=16).save(out_path, optimize=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fpyolo.scale", description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, type=Path, help="dir holding the archive dirs (train-00, ...)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--mode", choices=MODES, default="centre", help="text reference point")
    ap.add_argument("--min-consistent", type=int, default=MIN_CONSISTENT)
    ap.add_argument("--max-scan", type=int, default=MAX_SCAN)
    ap.add_argument("--qa-dir", type=Path, default=None, help="write overlay PNGs here")
    ap.add_argument("--qa-n", type=int, default=10)
    args = ap.parse_args(argv)

    chosen, scanned, total = select_tiles(args.src, args.n, args.seed, args.workers, args.mode,
                                          args.min_consistent, args.max_scan)
    res = summarize(chosen, args.min_consistent)
    res.update(mode=args.mode, n_requested=args.n, seed=args.seed, n_scanned=scanned, n_total_tiles=total,
               iqr_ratio_threshold=CONSTANT_IQR_RATIO)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.qa_dir is not None:
        for a, s, _ in chosen[: args.qa_n]:
            draw_overlay(args.src / a / f"{s}.svg", args.src / a / f"{s}.png", args.qa_dir / f"{a}_{s}.png", args.mode)
    print(f"{res['n_tiles']} tiles with >= {MIN_ESTIMATES} estimates, {res['n_consistent']} consistent "
          f"(scanned {scanned}/{total}); conclusion: {res['conclusion']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    # Run the importable module so classes pickled to workers are fpyolo.scale.*.
    from fpyolo.scale import main as _main

    sys.exit(_main())
