"""YOLO detection labels (model D) from parsed tiles.

An object is every (semantic_id in the class map, instance_id >= 0) pair; its
box is the union of its primitives' bboxes. Boxes are normalized by the viewBox
(never by a hard-coded 1000), while padding and size statistics are computed in
pixels of the true image size.
"""
from dataclasses import dataclass

from fpyolo.svgparse import Tile

SMALL_AREA_PX2 = 32 * 32  # COCO "small": area < 32^2 px^2


@dataclass(frozen=True)
class YoloBox:
    yolo_index: int
    cx: float  # normalized to [0, 1]
    cy: float
    w: float
    h: float


def _pad(lo: float, hi: float, min_side: float) -> tuple[float, float, bool]:
    """Pad [lo, hi] evenly to at least min_side; returns (lo, hi, was_padded)."""
    if hi - lo >= min_side:
        return lo, hi, False
    mid = (lo + hi) / 2
    return mid - min_side / 2, mid + min_side / 2, True


def thing_boxes(
    tile: Tile,
    sem_to_yolo: dict[int, int],
    img_w: int,
    img_h: int,
    min_side_px: float = 2.0,
    min_visible: float = 0.30,
) -> tuple[list[YoloBox], dict[str, int]]:
    """Return (boxes, stats) for one tile.

    `tile.viewbox` is (min_x, min_y, max_x, max_y) as stored by `parse_tile`.
    Per object: union of primitive bboxes -> convert to pixels -> pad each side
    shorter than `min_side_px` evenly -> visible fraction = clipped area /
    padded area (so a zero-area line never divides by zero) -> drop if below
    `min_visible`, otherwise clip to the image.

    Stats: `n_instances` objects considered (before dropping); `dropped_cut`
    dropped by the visibility rule; `padded` objects that needed padding;
    `small_lt_32px` kept objects whose final box area is < 32*32 px^2 (COCO's
    "small" definition).
    """
    vx0, vy0, vx1, vy1 = tile.viewbox
    sx, sy = img_w / (vx1 - vx0), img_h / (vy1 - vy0)

    objects: dict[tuple[int, int], list[float]] = {}
    for p in tile.primitives:
        if p.semantic_id not in sem_to_yolo or p.instance_id < 0:
            continue
        x0, y0, x1, y1 = p.bbox
        u = objects.get((p.semantic_id, p.instance_id))
        if u is None:
            objects[(p.semantic_id, p.instance_id)] = [x0, y0, x1, y1]
        else:
            u[0], u[1] = min(u[0], x0), min(u[1], y0)
            u[2], u[3] = max(u[2], x1), max(u[3], y1)

    stats = {"n_instances": len(objects), "dropped_cut": 0, "padded": 0, "small_lt_32px": 0}
    boxes: list[YoloBox] = []
    for (sid, _iid), (x0, y0, x1, y1) in sorted(objects.items()):
        px0, px1 = (x0 - vx0) * sx, (x1 - vx0) * sx
        py0, py1 = (y0 - vy0) * sy, (y1 - vy0) * sy
        px0, px1, padx = _pad(px0, px1, min_side_px)
        py0, py1, pady = _pad(py0, py1, min_side_px)
        if padx or pady:
            stats["padded"] += 1
        cx0, cx1 = max(px0, 0.0), min(px1, float(img_w))
        cy0, cy1 = max(py0, 0.0), min(py1, float(img_h))
        visible = max(cx1 - cx0, 0.0) * max(cy1 - cy0, 0.0)
        if visible / ((px1 - px0) * (py1 - py0)) < min_visible:
            stats["dropped_cut"] += 1
            continue
        if (cx1 - cx0) * (cy1 - cy0) < SMALL_AREA_PX2:
            stats["small_lt_32px"] += 1
        boxes.append(
            YoloBox(
                sem_to_yolo[sid],
                (cx0 + cx1) / 2 / img_w,
                (cy0 + cy1) / 2 / img_h,
                (cx1 - cx0) / img_w,
                (cy1 - cy0) / img_h,
            )
        )
    return boxes, stats


def format_yolo_det(boxes: list[YoloBox]) -> str:
    """One line `idx cx cy w h` per box (6 decimals), joined by newlines."""
    return "\n".join(
        f"{b.yolo_index} {b.cx:.6f} {b.cy:.6f} {b.w:.6f} {b.h:.6f}" for b in boxes
    )
