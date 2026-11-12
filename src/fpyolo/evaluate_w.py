"""Model W evaluation: pooled pixel IoU at native resolution (threshold chosen on val) and mask mAP.

The headline `test_pixel_iou` is class-agnostic: the label region is the union of all W
classes, and a pixel is predicted when the highest confidence among the predicted
instances covering it (any class) is >= t. t* is the first threshold of
THRESHOLDS that maximizes the pooled IoU on val; test is measured at t* only.
`per_class_pixel_iou` repeats this per class at the same t* (null when the class
has no label pixels in test).
"""
from __future__ import annotations

import argparse
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
from fpyolo.train import apply_device_env

THRESHOLDS = np.round(np.arange(1, 20) * 0.05, 2)  # 0.05 ... 0.95
_SUBPIXEL = 4  # cv2.fillPoly fractional bits: vertices kept to 1/16 px
_CHUNK = 16  # instances per vectorized max in max_score_map


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


def evaluate(weights: Path, dataset: Path, out: Path, device: str, limit: int | None = None) -> dict:
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
    for img, r in predict_stream(weights, test_imgs, device=device):
        h, w = r.orig_shape
        for c, counts in test.items():
            counts.add(gt_union_mask(label_path(img), w, h, cls=c), max_score_map(r, h, w, cls=c))

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
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be >= 1")
    apply_device_env(args.device)
    m = evaluate(args.weights, args.dataset, args.out, args.device, args.limit)
    print(f"t* {m['t_star']:.2f}: val IoU {m['val_iou_at_t_star']:.4f}, test IoU {m['test_pixel_iou']:.4f}, "
          f"mask mAP50-95 {m['mask_map50_95']:.4f} -> {Path(args.out) / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
