"""Estimate mm per pixel from numeric dimension annotations in FloorPlanCAD SVGs.

A dimension text such as "3600" (mm) sits next to a dimension line drawn on the same
layer. The line's length in viewBox units, times pixels per unit, is its length in
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
MIN_ESTIMATES = 3  # a tile needs this many estimates to be used in the measurement
CONSTANT_IQR_RATIO = 0.10  # IQR / median at or below this means "constant"
_DIGITS = re.compile(r"[0-9]{2,}")
_PATH_TOKEN = re.compile(r"[A-Za-z]|[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


@dataclass(frozen=True)
class DimText:
    value_mm: float
    x: float
    y: float
    angle_deg: float  # text direction in viewBox coordinates, modulo 180, in [0, 180)
    layer: str


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
                    ang = math.degrees(math.atan2(m[1, 0], m[0, 0])) % 180.0
                    texts.append(DimText(float(content), x, y, ang, layer))
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


def _best_segment(t: DimText, segments) -> Segment | None:
    best, best_d = None, math.inf
    for s in segments:
        if s.layer != t.layer:
            continue
        dx, dy = s.x1 - s.x0, s.y1 - s.y0
        if dx == 0.0 and dy == 0.0:
            continue
        diff = abs(math.degrees(math.atan2(dy, dx)) - t.angle_deg) % 180.0
        if min(diff, 180.0 - diff) > MAX_ANGLE_DEG:
            continue
        d = math.hypot((s.x0 + s.x1) / 2 - t.x, (s.y0 + s.y1) / 2 - t.y)
        if d <= MAX_MID_DIST and d < best_d:
            best, best_d = s, d
    return best


def match_dimension_line(t: DimText, segments) -> float | None:
    """Length (viewBox units) of the nearest same-layer parallel segment near the text.

    Parallel means within 2 degrees of the text direction (modulo 180); near means the
    segment midpoint is within 3 viewBox units of the text anchor.
    """
    s = _best_segment(t, segments)
    return None if s is None else math.hypot(s.x1 - s.x0, s.y1 - s.y0)


def match_tile(svg_path, img_w: int) -> list[Match]:
    texts, segments, vb_w = read_tile_geometry(svg_path)
    px_per_unit = img_w / vb_w
    out = []
    for t in texts:
        s = _best_segment(t, segments)
        if s is None:
            continue
        length = math.hypot(s.x1 - s.x0, s.y1 - s.y0)
        out.append(Match(t, s, length, t.value_mm / (length * px_per_unit)))
    return out


def measure_tile(svg_path, img_w: int) -> list[float]:
    """mm/px estimates, one per dimension text that found its line."""
    return [m.mm_per_px for m in match_tile(svg_path, img_w)]


def _png_width(png_path) -> int:
    with Image.open(png_path) as im:
        return im.size[0]


def _scan(job):
    """Worker (module level): (archive, stem, estimates) for one tile."""
    archive, stem, svg, png = job
    return archive, stem, measure_tile(svg, _png_width(png))


def list_tiles(src: Path) -> list[tuple[str, str, Path, Path]]:
    tiles = []
    for arch in sorted(p for p in Path(src).iterdir() if p.is_dir()):
        for svg in sorted(arch.glob("*.svg")):
            png = svg.with_suffix(".png")
            if png.exists():
                tiles.append((arch.name, svg.stem, svg, png))
    return tiles


def select_tiles(src: Path, n: int, seed: int, workers: int):
    """Shuffle all tiles with `seed`; keep the first `n` having >= 3 estimates."""
    tiles = list_tiles(src)
    random.Random(seed).shuffle(tiles)
    chosen, scanned = [], 0
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
                if len(chosen) >= n:
                    break
    finally:
        if pool is not None:
            pool.terminate()
            pool.join()
    return chosen, scanned, len(tiles)


def summarize(chosen) -> dict:
    medians = [float(np.median(est)) for _, _, est in chosen]
    res = {
        "tiles": [
            {
                "image_id": f"{a}_{s}",
                "archive": a,
                "stem": s,
                "n_estimates": len(est),
                "median": float(np.median(est)),
                "estimates": [float(e) for e in est],
            }
            for a, s, est in chosen
        ],
        "n_tiles": len(chosen),
    }
    if not medians:
        res.update(overall_median=None, q1=None, q3=None, iqr=None, iqr_over_median=None, conclusion="undetermined")
        return res
    q1, med, q3 = (float(v) for v in np.percentile(medians, [25, 50, 75]))
    ratio = (q3 - q1) / med
    res.update(
        overall_median=med,
        q1=q1,
        q3=q3,
        iqr=q3 - q1,
        iqr_over_median=ratio,
        conclusion="constant" if ratio <= CONSTANT_IQR_RATIO else "varies",
    )
    return res


def draw_overlay(svg_path, png_path, out_path) -> None:
    """Ink image with each matched text anchor (red) and its segment (blue)."""
    from fpyolo.ink import load_ink

    ink = load_ink(png_path)
    h, w = ink.shape
    img = Image.fromarray(np.repeat(ink[..., None], 3, axis=2), "RGB")
    _, _, vb_w = read_tile_geometry(svg_path)
    k = w / vb_w
    d = ImageDraw.Draw(img)
    for m in match_tile(svg_path, w):
        s, t = m.segment, m.text
        d.line([(s.x0 * k, s.y0 * k), (s.x1 * k, s.y1 * k)], fill=(0, 80, 255), width=3)
        d.ellipse([t.x * k - 6, t.y * k - 6, t.x * k + 6, t.y * k + 6], outline=(255, 0, 0), width=3)
        d.text((t.x * k + 8, t.y * k - 12), f"{t.value_mm:g}", fill=(255, 0, 0))
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    img.quantize(colors=16).save(out_path, optimize=True)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fpyolo.scale", description=__doc__.split("\n")[0])
    ap.add_argument("--src", required=True, type=Path, help="dir holding the archive dirs (train-00, ...)")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--qa-dir", type=Path, default=None, help="write overlay PNGs here")
    ap.add_argument("--qa-n", type=int, default=10)
    args = ap.parse_args(argv)

    chosen, scanned, total = select_tiles(args.src, args.n, args.seed, args.workers)
    res = summarize(chosen)
    res.update(n_requested=args.n, seed=args.seed, n_scanned=scanned, n_total_tiles=total,
               iqr_ratio_threshold=CONSTANT_IQR_RATIO)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if args.qa_dir is not None:
        for a, s, _ in chosen[: args.qa_n]:
            draw_overlay(args.src / a / f"{s}.svg", args.src / a / f"{s}.png", args.qa_dir / f"{a}_{s}.png")
    print(f"{res['n_tiles']}/{args.n} tiles with >= {MIN_ESTIMATES} estimates "
          f"(scanned {scanned}/{total}); conclusion: {res['conclusion']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    # Run the importable module so classes pickled to workers are fpyolo.scale.*.
    from fpyolo.scale import main as _main

    sys.exit(_main())
