"""Model W evaluation: pooled pixel IoU at native resolution (threshold chosen on val) and mask mAP.

The headline `test_pixel_iou` is class-agnostic: the label region is the union of all W
classes, and a pixel is predicted when the highest confidence among the predicted
instances covering it (any class) is >= t. t* is the first threshold of
THRESHOLDS that maximizes the pooled IoU on val; test is measured at t* only.
`per_class_pixel_iou` repeats this per class at the same t* (null when the class
has no label pixels in test).

`wall_thickness_mm` (test pass only): per test image whose class-agnostic label mask and
prediction mask (score >= t*) are both non-empty, thickness = 2 * area / perimeter of the
mask; the error is (prediction - label) thickness in mm (px * mm per px).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

from fpyolo.evaluate_d import (
    CONF, IMGSZ, IOU, MAX_DET, iter_predict, label_path, read_label_rows, split_images,
    ultralytics_val, val_summary, write_json,
)
from fpyolo.train import REPO, apply_device_env

THRESHOLDS = np.round(np.arange(1, 20) * 0.05, 2)  # 0.05 ... 0.95
_SUBPIXEL = 4  # cv2.fillPoly fractional bits: vertices kept to 1/16 px
_CHUNK = 16  # instances per vectorized max in max_score_map
SCALE_JSON = REPO / "reports" / "scale.json"
THICKNESS_METHOD = ("2*area/perimeter of the class-agnostic mask; area = pixel count, perimeter = total "
                    "cv2.arcLength of all external and hole contours (RETR_CCOMP, pixel-centre contours); "
                    "exact for an infinitely long straight bar, a few percent low for a finite one")


def gt_union_mask(label_path: Path, w: int, h: int, cls: int | None = None) -> np.ndarray:
    """(h, w) bool union of the label polygons (all classes, or only `cls`).

    Each line is one keyhole ring (holes joined by zero-width bridges), filled on
    its own with cv2.fillPoly, so holes stay empty. Normalized coordinates are
    continuous image coordinates; pixel (i, j) has its centre at (j + 0.5, i + 0.5).
    fillPoly also fills the pixels its boundary passes through, so the mask is the
    region of pixel centres inside a polygon grown by up to 1 px (Ultralytics
    rasterizes its training masks with fillPoly too).
    """
    canvas = np.zeros((h, w), np.uint8)
    for row in read_label_rows(Path(label_path)):
        if cls is not None and int(row[0]) != cls:
            continue
        pts = np.array(row[1:], dtype=float).reshape(-1, 2) * (w, h) - 0.5  # to pixel-centre coords
        cv2.fillPoly(canvas, [np.round(pts * (1 << _SUBPIXEL)).astype(np.int32)], 1, shift=_SUBPIXEL)
    return canvas.astype(bool)


def max_score_map(result, h: int, w: int, cls: int | None = None) -> np.ndarray:
    """(h, w) float32 per-pixel max confidence over the predicted instances (of `cls` if
    given); 0 where no instance covers the pixel. Masks must be at the image's native
    resolution (predict with retina_masks=True)."""
    if result.masks is None:
        return np.zeros((h, w), np.float32)
    data, conf = result.masks.data, result.boxes.conf
    if tuple(data.shape[1:]) != (h, w):
        raise ValueError(f"masks are {tuple(data.shape[1:])} but the image is {(h, w)}: "
                         "predict with retina_masks=True")
    if cls is not None:
        keep = result.boxes.cls == cls
        data, conf = data[keep], conf[keep]
    score = torch.zeros((h, w), dtype=torch.float32, device=data.device)
    for i in range(0, len(conf), _CHUNK):
        part = (data[i:i + _CHUNK] > 0) * conf[i:i + _CHUNK, None, None].float()
        score = torch.maximum(score, part.amax(dim=0))
    return score.cpu().numpy()


def mask_thickness_px(mask: np.ndarray) -> float | None:
    """2 * area / perimeter of a binary mask in px (perimeter: every external and hole contour);
    None for an empty mask or one without a measurable perimeter (e.g. a single pixel)."""
    m = (np.asarray(mask) > 0).astype(np.uint8)
    area = int(m.sum())
    if not area:
        return None
    contours, _ = cv2.findContours(m, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
    perimeter = sum(cv2.arcLength(c, True) for c in contours)
    return 2.0 * area / perimeter if perimeter > 0 else None


class ThicknessErrors:
    """Per-image wall-thickness errors (prediction - label, px), accumulated one image at a time."""

    def __init__(self):
        self.diffs: list[float] = []

    def add(self, gt: np.ndarray, score: np.ndarray, t_star: float) -> None:
        gt_t = mask_thickness_px(gt)
        pred_t = mask_thickness_px(np.asarray(score) >= t_star) if gt_t is not None else None
        if gt_t is not None and pred_t is not None:
            self.diffs.append(pred_t - gt_t)

    def summary(self, mm_per_px: float | None) -> dict:
        d = np.asarray(self.diffs, dtype=float)
        nan = math.nan
        if mm_per_px is None or not len(d):
            mae = med = bias = nan
        else:
            e = d * mm_per_px
            mae, med, bias = float(np.abs(e).mean()), float(np.median(np.abs(e))), float(e.mean())
        return {"mae": mae, "median_abs": med, "bias": bias, "n_images": len(d),
                "mm_per_px": mm_per_px, "method": THICKNESS_METHOD}


def default_mm_per_px(path: Path = SCALE_JSON) -> float | None:
    """`overall_median` of reports/scale.json (None if the file is missing). The scale is only a
    constant mm/px when its conclusion says so; otherwise a warning is printed."""
    path = Path(path)
    if not path.is_file():
        return None
    doc = json.loads(path.read_text(encoding="utf-8"))
    if doc.get("conclusion") != "constant":
        print(f"warning: {path} concludes {doc.get('conclusion')!r}, not a constant mm/px", file=sys.stderr)
    return float(doc["overall_median"])


class PixelCounts:
    """Pooled label/prediction pixel counts for every threshold, from one pass per image."""

    def __init__(self, thresholds: Sequence[float]):
        self.t = np.asarray(thresholds, dtype=np.float64)
        if self.t.ndim != 1 or not len(self.t) or np.any(np.diff(self.t) <= 0) or self.t[0] <= 0:
            raise ValueError("thresholds must be positive and strictly increasing")
        self.n_gt = 0
        self.pred = np.zeros(len(self.t) + 1, np.int64)  # pixels by number of thresholds <= score
        self.inter = np.zeros(len(self.t) + 1, np.int64)  # the same, label pixels only

    def add(self, gt: np.ndarray, score: np.ndarray) -> None:
        gt = np.asarray(gt, dtype=bool)
        if gt.shape != score.shape:
            raise ValueError(f"label mask {gt.shape} and score map {score.shape} differ")
        hit = score >= self.t[0]  # other pixels are predicted at no threshold
        k = np.searchsorted(self.t, score[hit], side="right")
        self.pred += np.bincount(k, minlength=len(self.pred))
        self.inter += np.bincount(k[gt[hit]], minlength=len(self.inter))
        self.n_gt += int(np.count_nonzero(gt))

    def iou(self) -> np.ndarray:
        """IoU at each threshold (a pixel is predicted if score >= t); NaN if the union is empty."""
        pred = np.cumsum(self.pred[::-1])[::-1][1:]  # pixels with score >= t[j]
        inter = np.cumsum(self.inter[::-1])[::-1][1:]
        union = self.n_gt + pred - inter
        return np.divide(inter, union, out=np.full(len(self.t), np.nan), where=union > 0)


def iou_curve(pairs: Iterable[tuple[np.ndarray, np.ndarray]], thresholds: Sequence[float]) -> np.ndarray:
    """Pooled (micro) IoU over all (label mask, score map) pairs, for every threshold."""
    counts = PixelCounts(thresholds)
    for gt, score in pairs:
        counts.add(gt, score)
    return counts.iou()


def select_threshold(curve: np.ndarray, thresholds: Sequence[float]) -> tuple[float, float]:
    """(t*, IoU at t*): the first threshold with the highest IoU."""
    curve = np.asarray(curve, dtype=float)
    if np.isnan(curve).all():
        raise ValueError("the IoU curve is undefined: no label and no predicted pixel")
    i = int(np.nanargmax(curve))
    return float(thresholds[i]), float(curve[i])


def predict_stream(weights: Path, images: Sequence[Path], *, device: str) -> Iterator[tuple[Path, object]]:
    """(image, Ultralytics Result) pairs one image at a time, with native-resolution masks."""
    if not images:
        return
    from ultralytics import YOLO

    yield from iter_predict(YOLO(str(weights)), images, imgsz=IMGSZ, conf=CONF, iou=IOU,
                            max_det=MAX_DET, retina_masks=True, device=device)


def evaluate(weights: Path, dataset: Path, out: Path, device: str, limit: int | None = None,
             mm_per_px: float | None = None) -> dict:
    weights, dataset, out = (Path(p).resolve() for p in (weights, dataset, out))
    names = {int(k): v for k, v in
             yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))["names"].items()}
    out.mkdir(parents=True, exist_ok=True)

    val_imgs = split_images(dataset, "val", limit)
    val = PixelCounts(THRESHOLDS)
    for img, r in predict_stream(weights, val_imgs, device=device):
        h, w = r.orig_shape
        val.add(gt_union_mask(label_path(img), w, h), max_score_map(r, h, w))
    val_curve = val.iou()
    t_star, val_iou = select_threshold(val_curve, THRESHOLDS)

    test_imgs = split_images(dataset, "test", limit)
    test = {c: PixelCounts([t_star]) for c in [None, *sorted(names)]}  # None: all classes
    thickness = ThicknessErrors()
    for img, r in predict_stream(weights, test_imgs, device=device):
        h, w = r.orig_shape
        for c, counts in test.items():
            gt, score = gt_union_mask(label_path(img), w, h, cls=c), max_score_map(r, h, w, cls=c)
            counts.add(gt, score)
            if c is None:
                thickness.add(gt, score, t_star)

    m = ultralytics_val(weights, dataset, "test", out, device, test_imgs if limit is not None else None)
    ultra = val_summary(m, device)
    metrics = {
        "t_star": t_star,
        "val_iou_at_t_star": val_iou,
        "test_pixel_iou": float(test[None].iou()[0]),
        "mask_map50_95": m.seg.map,
        "mask_map50": m.seg.map50,
        "mask_map_note": ultra["note"],
        "per_class_pixel_iou": {names[c]: float(test[c].iou()[0]) if test[c].n_gt else None
                                for c in sorted(names)},
        "wall_thickness_mm": thickness.summary(mm_per_px),
        "thresholds": THRESHOLDS,
        "val_iou_curve": val_curve,
        "weights": str(weights), "dataset": str(dataset), "limit": limit, "device": device,
        "n_val_images": len(val_imgs), "n_test_images": len(test_imgs),
        "ultralytics_val": ultra,
    }
    write_json(out / "metrics.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fpyolo.evaluate_w", description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path, help="<out>/W (data.yaml, images/, labels/)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", required=True, help="'cpu' or one GPU index; there is no default")
    ap.add_argument("--limit", type=int, help="CPU smoke runs only: first N images (by id) of val and test")
    ap.add_argument("--mm-per-px", type=float,
                    help=f"scale for the wall-thickness error (default: overall_median of {SCALE_JSON.relative_to(REPO)}; "
                         "null in metrics.json if that file is missing)")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be >= 1")
    if args.mm_per_px is not None and not args.mm_per_px > 0:
        ap.error("--mm-per-px must be > 0")
    mm_per_px = args.mm_per_px if args.mm_per_px is not None else default_mm_per_px()
    apply_device_env(args.device)
    m = evaluate(args.weights, args.dataset, args.out, args.device, args.limit, mm_per_px)
    print(f"t* {m['t_star']:.2f}: val IoU {m['val_iou_at_t_star']:.4f}, test IoU {m['test_pixel_iou']:.4f}, "
          f"mask mAP50-95 {m['mask_map50_95']:.4f} -> {Path(args.out) / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
