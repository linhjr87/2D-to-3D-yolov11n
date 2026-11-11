"""Parse a FloorPlanCAD SVG into labeled primitives with exact extents.

Only elements carrying integer `semantic-id` and `instance-id` attributes are
returned. Geometry is in viewBox units with every `transform` (own and
ancestors') already applied; `svgelements` supplies exact arc/ellipse extents.
"""
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from svgelements import SVG, Arc, Circle, Close, Ellipse, Path as SvgPath, Polygon, Rect, Shape

MAX_STEP = 0.05  # maximum distance between consecutive polyline samples (viewBox units)
_SAMPLE_STEP = 0.04  # sampling pitch; below MAX_STEP so chord error never matters
_CLOSED_SHAPES = (Circle, Ellipse, Rect, Polygon)


@dataclass(frozen=True)
class Primitive:
    semantic_id: int
    instance_id: int
    bbox: tuple[float, float, float, float]  # (x0, y0, x1, y1), viewBox units
    polyline: np.ndarray  # (N, 2) float64, viewBox units, step <= MAX_STEP
    closed: bool


@dataclass(frozen=True)
class Tile:
    archive: str
    stem: str
    viewbox: tuple[float, float, float, float]
    primitives: list[Primitive]
    skipped: int  # labeled elements with malformed ids or unusable geometry


def _arc_length_bound(arc: Arc) -> float:
    """Cheap upper bound on arc length (Arc.length() is adaptive and ~1000x slower)."""
    r = max(arc.center.distance_to(arc.prx), arc.center.distance_to(arc.pry))
    return abs(arc.sweep) * r


def _sample(path: SvgPath) -> np.ndarray:
    pts = []
    for seg in path:
        if seg.end is None or seg.start is None:
            continue
        length = _arc_length_bound(seg) if isinstance(seg, Arc) else seg.length()
        if not math.isfinite(length) or length == 0:
            continue
        n = int(math.ceil(length / _SAMPLE_STEP)) + 1
        t = np.linspace(0.0, 1.0, n)
        p = np.asarray(seg.npoint(t), dtype=np.float64)
        pts.append(p if not pts else p[1:])
    if not pts:
        return np.empty((0, 2), dtype=np.float64)
    return np.concatenate(pts)


def _parse_ids(values: dict):
    """Return (sid, iid), None if unlabeled, or 'bad' for malformed ids."""
    sid, iid = values.get("semantic-id"), values.get("instance-id")
    if sid is None or iid is None:
        # a lone id that is not an integer is still a data-quality signal
        for v in (sid, iid):
            if v is not None:
                try:
                    int(v)
                except ValueError:
                    return "bad"
        return None
    try:
        return int(sid), int(iid)
    except ValueError:
        return "bad"


def parse_tile(svg_path: Path, archive: str) -> Tile:
    svg_path = Path(svg_path)
    svg = SVG.parse(str(svg_path), reify=True)
    vb = svg.viewbox
    viewbox = (float(vb.x), float(vb.y), float(vb.x + vb.width), float(vb.y + vb.height))
    prims: list[Primitive] = []
    skipped = 0
    for el in svg.elements():
        if not isinstance(el, Shape):
            continue
        ids = _parse_ids(el.values)
        if ids is None:
            continue
        if ids == "bad":
            skipped += 1
            continue
        path = SvgPath(el)
        poly = _sample(path)
        bbox = el.bbox()
        if bbox is None or len(poly) == 0 or not all(math.isfinite(float(v)) for v in bbox):
            skipped += 1
            continue
        if isinstance(el, SvgPath):
            closed = len(path) > 0 and isinstance(path[-1], Close)
        else:
            closed = isinstance(el, _CLOSED_SHAPES)
        prims.append(
            Primitive(ids[0], ids[1], tuple(float(v) for v in bbox), poly, closed)
        )
    return Tile(archive, svg_path.stem, viewbox, prims, skipped)
