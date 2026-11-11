"""YOLO segmentation labels (model W): wall-body polygons from wall strokes.

Walls are drawn as closed outlines (two parallel strokes closed at the ends),
sometimes hatched inside, and cut at the image border. All geometry is built in
pixels of the true image size and converted to normalized [0, 1] coordinates
(SVG coordinate / viewBox size) at the end.
"""
from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry.polygon import orient

from fpyolo.svgparse import Tile

SIMPLIFY_PX = 0.05  # drop (near-)collinear samples before noding; far below a pixel
GRID_PX = 1e-3  # snap-rounding grid for noding: merges endpoints that differ by float noise
COVERAGE_TOL_PX = 1.5  # a stroke this close to the region boundary counts as covered
_FRAME_EPS_PX = 1e-6


@dataclass(frozen=True)
class WallResult:
    polygons: list[shapely.Polygon]  # normalized [0, 1], possibly with holes
    coverage: float | None  # None when no selected stroke lies inside the image
    stroke_len_px: float  # length of the selected strokes inside the image


def _stroke_lines(tile: Tile, semantic_ids: set[int], img_w: int, img_h: int) -> np.ndarray:
    """Selected primitives as simplified LineStrings in pixels (object array)."""
    x0, y0, x1, y1 = tile.viewbox  # (min_x, min_y, max_x, max_y)
    scale = np.array([img_w / (x1 - x0), img_h / (y1 - y0)])
    lines = [
        shapely.LineString((p.polyline - (x0, y0)) * scale)
        for p in tile.primitives
        if p.semantic_id in semantic_ids and len(p.polyline) >= 2
    ]
    arr = np.empty(len(lines), dtype=object)
    arr[:] = lines
    return shapely.simplify(arr, SIMPLIFY_PX, preserve_topology=False)


def _gap_segments(lines: np.ndarray, gap_px: float) -> np.ndarray:
    """Join each free stroke end to its nearest other stroke when 0 < distance < gap_px."""
    starts, ends = shapely.get_point(lines, 0), shapely.get_point(lines, -1)
    is_open = shapely.distance(starts, ends) > 0
    owner = np.concatenate([np.flatnonzero(is_open)] * 2)
    pts = np.concatenate([starts[is_open], ends[is_open]])
    if len(pts) == 0:
        return np.empty(0, dtype=object)
    e, l = shapely.STRtree(lines).query(pts, predicate="dwithin", distance=gap_px)
    other = l != owner[e]
    e, l = e[other], l[other]
    d = shapely.distance(pts[e], lines[l])
    order = np.lexsort((d, e))  # per endpoint, nearest other stroke first
    e, l, d = e[order], l[order], d[order]
    first = np.unique(e, return_index=True)[1]
    e, l, d = e[first], l[first], d[first]
    gap = (d > 0) & (d < gap_px)
    return shapely.shortest_line(pts[e[gap]], lines[l[gap]])


def _clip_to_frame(lines: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Line parts inside the image; lines already inside are returned unchanged."""
    b = shapely.bounds(lines).reshape(-1, 4)
    outside = (b[:, 0] < 0) | (b[:, 1] < 0) | (b[:, 2] > img_w) | (b[:, 3] > img_h)
    out = lines.copy()
    out[outside] = shapely.intersection(lines[outside], shapely.box(0, 0, img_w, img_h))
    return out


def _faces(lines: np.ndarray, img_w: int, img_h: int) -> np.ndarray:
    """Polygonize the lines plus the image frame after noding all intersections."""
    frame = shapely.box(0, 0, img_w, img_h).exterior
    noded = shapely.unary_union(np.append(lines, frame), grid_size=GRID_PX)
    parts = shapely.get_parts(noded)
    parts = parts[shapely.get_type_id(parts) == 1]  # LineStrings only (drop touch points)
    return shapely.get_parts(shapely.polygonize(parts))


def _frame_length(face: shapely.Polygon, img_w: int, img_h: int) -> float:
    """Length of the face's exterior lying on the image frame."""
    c = shapely.get_coordinates(face.exterior)
    a, b = c[:-1], c[1:]
    on = np.zeros(len(a), dtype=bool)
    for axis, lim in ((0, 0.0), (0, img_w), (1, 0.0), (1, img_h)):
        on |= (np.abs(a[:, axis] - lim) <= _FRAME_EPS_PX) & (np.abs(b[:, axis] - lim) <= _FRAME_EPS_PX)
    return float(np.linalg.norm(b - a, axis=1)[on].sum())


def _wall_face_mask(
    faces: np.ndarray, img_w: int, img_h: int, max_thickness_px: float, max_border_frac: float
) -> np.ndarray:
    """Thin faces (inscribed radius <= max_thickness/2) not lying mostly on the frame."""
    radius = shapely.length(shapely.maximum_inscribed_circle(faces))  # polylabel distance
    border = np.array([_frame_length(f, img_w, img_h) for f in faces])
    return (radius <= max_thickness_px / 2) & (border / shapely.length(faces) < max_border_frac)


def _coverage(strokes: np.ndarray, region: shapely.Geometry, total: float) -> float:
    """Fraction of stroke length within COVERAGE_TOL_PX of the region boundary."""
    if region.is_empty:
        return 0.0
    band = shapely.buffer(shapely.boundary(region), COVERAGE_TOL_PX)
    return float(shapely.length(shapely.intersection(strokes, band)).sum() / total)


def wall_regions(
    tile: Tile,
    semantic_ids: set[int],
    img_w: int,
    img_h: int,
    gap_px: float = 3.0,
    max_thickness_px: float = 60.0,
    max_border_frac: float = 0.4,
) -> WallResult:
    """Wall bodies of one tile: one polygon per connected wall region.

    Strokes of `semantic_ids` are scaled to pixels, small gaps between a free
    end and another stroke are bridged, everything is clipped to the image,
    noded together with the image frame and polygonized. Faces that are thin
    (wall bodies, not rooms) and not mostly on the frame (not border slivers)
    are merged; each connected component is one polygon. Coverage is measured
    on the strokes inside the image, before gap closing.
    """
    lines = _stroke_lines(tile, semantic_ids, img_w, img_h)
    strokes = _clip_to_frame(lines, img_w, img_h)
    total = float(shapely.length(strokes).sum())
    if total == 0:
        return WallResult([], None, 0.0)
    gaps = _clip_to_frame(_gap_segments(lines, gap_px), img_w, img_h)
    faces = _faces(np.concatenate([strokes, gaps]), img_w, img_h)
    keep = _wall_face_mask(faces, img_w, img_h, max_thickness_px, max_border_frac)
    region = shapely.unary_union(faces[keep])
    scale = np.array([img_w, img_h], dtype=np.float64)
    polygons = [shapely.transform(p, lambda c: c / scale) for p in shapely.get_parts(region)]
    return WallResult(polygons, _coverage(strokes, region, total), total)


def _segment_index(ring: np.ndarray, p: np.ndarray) -> int:
    """Index i of the closed ring's segment ring[i] -> ring[i+1] nearest to point p."""
    a, ab = ring, np.roll(ring, -1, axis=0) - ring
    t = np.clip(((p - a) * ab).sum(1) / np.maximum((ab**2).sum(1), 1e-300), 0.0, 1.0)
    return int(np.argmin(np.linalg.norm(a + t[:, None] * ab - p, axis=1)))


def _bridge(ring: np.ndarray, hole: np.ndarray) -> np.ndarray:
    """Splice `hole` into `ring` along their shortest connecting segment."""
    a, b = shapely.get_coordinates(
        shapely.shortest_line(shapely.LinearRing(hole), shapely.LinearRing(ring))
    )  # a on the hole, b on the ring
    i, j = _segment_index(ring, b), _segment_index(hole, a)
    loop = np.vstack([a, hole[j + 1 :], hole[: j + 1], a])  # full hole, from a back to a
    return np.vstack([ring[: i + 1], b, loop, b, ring[i + 1 :]])


def keyhole_ring(poly: shapely.Polygon) -> np.ndarray:
    """One (N, 2) ring encoding `poly` with its holes joined by zero-width bridges.

    The exterior is counter-clockwise and holes clockwise, so the ring fills
    correctly under both even-odd and non-zero rules. Holes are spliced in
    nearest-first, each by its shortest segment to the ring built so far; a
    bridge therefore never crosses a hole that is not yet spliced in. The ring
    is open (the first point is not repeated).
    """
    poly = orient(poly, sign=1.0)
    ring = np.asarray(poly.exterior.coords)[:-1]
    holes = [np.asarray(h.coords)[:-1] for h in poly.interiors]
    while holes:
        current = shapely.LinearRing(ring)
        k = int(np.argmin([shapely.distance(shapely.LinearRing(h), current) for h in holes]))
        ring = _bridge(ring, holes.pop(k))
    return ring


def format_yolo_seg(yolo_index: int, ring: np.ndarray) -> str:
    """'<index> x1 y1 x2 y2 ...' with 6 decimals, values clipped to [0, 1]."""
    vals = np.clip(np.asarray(ring, dtype=np.float64).ravel(), 0.0, 1.0)
    return f"{yolo_index} " + " ".join(f"{v:.6f}" for v in vals)
