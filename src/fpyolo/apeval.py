"""COCO-style box AP over fixed class sets, with per-image weights and a cluster bootstrap.

Pure numpy. Matching and the 101-point interpolated AP replicate pycocotools
COCOeval for boxes with iscrowd=0 and area = box area (tests compare against it):
10 IoU thresholds 0.50:0.05:0.95, detections sorted by score per image and class
(stable sort) and truncated to `max_dets` per image and class, greedy matching
that prefers labels inside the area range, then all images' detections of a class
concatenated in image-id order and stably sorted by score.

How weights avoid re-matching: `build_cache` matches once and keeps, per area range
and IoU threshold, only the true positives and a few correction entries (see
`_AreaTable`). At each true positive, weighted TP and TP+FP counts are cumulative
sums of the image weights, so `ap_table(..., weights=...)` is a few vectorized passes
over the cached arrays.

Weight convention: weight k behaves as k copies of the image whose copies sit
immediately after the original in image-id order, with each detection's k copies
adjacent in the score-sorted order. This equals literal duplication through
`build_cache` except where detections tie on score within one image and class
(literal copies then form image blocks), and at cross-image score ties when literal
duplicate keys would sort elsewhere.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np

GT = tuple[np.ndarray, np.ndarray]  # boxes (N, 4) xyxy px, labels (N,) int
Pred = tuple[np.ndarray, np.ndarray, np.ndarray]  # boxes (M, 4) xyxy px, labels (M,) int, scores (M,)

# Same expressions as pycocotools Params.setDetParams, so the floats are identical.
IOU_THRS = np.linspace(0.5, 0.95, int(np.round((0.95 - 0.5) / 0.05)) + 1, endpoint=True)
REC_THRS = np.linspace(0.0, 1.00, int(np.round((1.00 - 0.0) / 0.01)) + 1, endpoint=True)
AREA_RANGES = {"all": (0, 1e5**2), "small": (0, 32**2), "medium": (32**2, 96**2),
               "large": (96**2, 1e5**2)}
_T, _R = len(IOU_THRS), len(REC_THRS)
_I50, _I75 = 0, 5  # positions of IoU 0.50 and 0.75 in IOU_THRS
_MATCH_THRS = np.minimum(IOU_THRS, 1 - 1e-10)  # COCOeval's `iou = min([t, 1-1e-10])`


@dataclass(frozen=True)
class _AreaTable:
    """Matching outcome for one area range, over all IoU thresholds.

    A detection "counts" (is not ignored, COCOeval's not dtIgnore) when it is a true
    positive or an unmatched detection inside the area range. TP+FP at a position is
    the weighted count of in-range detections so far plus the corrections: +1 for a
    true positive outside the range, -1 for an in-range detection matched to an
    ignored label. Segments are (threshold, class) pairs: segment = t * K + k.
    """

    in_area: np.ndarray | None  # (D,) bool, box area inside the range; None when all are
    n_gt: np.ndarray  # (I, K) int64, non-ignored labels per image and class
    bounds: np.ndarray  # (U,) increasing detection positions where weight sums are split
    tp_det: np.ndarray  # (P,) flat detection index of each TP, sorted by (segment, rank)
    tp_cls: np.ndarray  # (P,) class index of each TP
    tp_key: np.ndarray  # (P,) segment * (R + 1), the base of the recall-bin key
    tp_seg_lo: np.ndarray  # (P,) position in tp_* of the first TP of the same segment
    tp_hi: np.ndarray  # (P,) bounds index (U = end) just after the TP
    tp_lo: np.ndarray  # (P,) bounds index of the first detection of the TP's class
    corr_det: np.ndarray  # (Q,) flat detection index of each correction, sorted by (t, index)
    corr_sign: np.ndarray  # (Q,) int64, +1 or -1
    corr_upto: np.ndarray  # (P,) corrections up to and including each TP (exclusive end)
    corr_lo: np.ndarray  # (P,) first correction of each TP's segment


@dataclass(frozen=True)
class EvalCache:
    """Per-image matching results of one model on one image set.

    Detections of all images sit in one flat array, grouped by class and, within a
    class, in COCOeval's global score order. Image i is image_ids[i].
    """

    image_ids: tuple[str, ...]  # sorted
    classes: np.ndarray  # (K,) sorted labels that occur in the labels or the predictions
    det_img: np.ndarray  # (D,) image index of each detection
    areas: dict[str, _AreaTable]  # keyed like AREA_RANGES


def _gt_arrays(v) -> tuple[np.ndarray, np.ndarray]:
    if v is None:
        return np.zeros((0, 4)), np.zeros(0, np.int64)
    boxes = np.asarray(v[0], float).reshape(-1, 4)
    labels = np.asarray(v[1]).astype(np.int64).reshape(-1)
    if len(boxes) != len(labels):
        raise ValueError(f"{len(boxes)} label boxes but {len(labels)} classes")
    return boxes, labels


def _pred_arrays(v) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if v is None:
        return np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0)
    boxes = np.asarray(v[0], float).reshape(-1, 4)
    labels = np.asarray(v[1]).astype(np.int64).reshape(-1)
    scores = np.asarray(v[2], float).reshape(-1)
    if not len(boxes) == len(labels) == len(scores):
        raise ValueError(f"{len(boxes)} boxes, {len(labels)} classes, {len(scores)} scores")
    return boxes, labels, scores


def _xywh(b: np.ndarray) -> np.ndarray:
    return np.c_[b[:, :2], b[:, 2:] - b[:, :2]]


def _iou(d: np.ndarray, g: np.ndarray) -> np.ndarray:
    """(D, G) IoU of xywh boxes, computed like pycocotools' bbIou (no +1, iscrowd=0)."""
    iw = (np.minimum(d[:, None, 0] + d[:, None, 2], g[None, :, 0] + g[None, :, 2])
          - np.maximum(d[:, None, 0], g[None, :, 0]))
    ih = (np.minimum(d[:, None, 1] + d[:, None, 3], g[None, :, 1] + g[None, :, 3])
          - np.maximum(d[:, None, 1], g[None, :, 1]))
    inter = iw * ih
    union = (d[:, None, 2] * d[:, None, 3] + g[None, :, 2] * g[None, :, 3]) - inter
    return np.divide(inter, union, out=np.zeros_like(inter), where=(iw > 0) & (ih > 0))


def _match(iou: np.ndarray, gt_ignore: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """COCOeval.evaluateImg greedy matching at every IoU threshold.

    iou is (D, G) with detections in score order. Returns (matched, matched_ignored),
    both (T, D) bool. A detection takes the unmatched label with the best IoU
    (>= threshold), preferring labels that are not ignored; on IoU ties the later
    label wins, as COCOeval's `<` comparison does.
    """
    n_det, n_gt = iou.shape
    matched = np.zeros((_T, n_det), bool)
    matched_ign = np.zeros((_T, n_det), bool)
    if n_det == 0 or n_gt == 0:
        return matched, matched_ign
    order = np.argsort(gt_ignore, kind="mergesort")  # COCOeval puts ignored labels last
    iou, ign = iou[:, order], gt_ignore[order]
    taken = np.zeros((_T, n_gt), bool)
    rows = np.arange(_T)
    for d in np.flatnonzero(iou.max(axis=1) >= _MATCH_THRS[0]):  # the others never match
        ok = (iou[d] >= _MATCH_THRS[:, None]) & ~taken
        regular = ok & ~ign
        cand = np.where(regular.any(axis=1)[:, None], regular, ok)
        g = n_gt - 1 - np.argmax(np.where(cand, iou[d], -1.0)[:, ::-1], axis=1)
        hit = cand[rows, g]
        taken[rows[hit], g[hit]] = True
        matched[hit, d] = True
        matched_ign[hit, d] = ign[g[hit]]
    return matched, matched_ign


def _area_table(tp, corr, in_area, det_cls, class_lo, n_cls, n_gt) -> _AreaTable:
    n_det = tp.shape[1]
    tp_t, tp_det = np.nonzero(tp)  # row-major: sorted by threshold, then flat index
    tp_cls = det_cls[tp_det]
    tp_seg = tp_t * n_cls + tp_cls
    cls_lo = class_lo[tp_cls]
    bounds = np.unique(np.r_[cls_lo, tp_det + 1])
    bounds = bounds[bounds < n_det]
    corr_t, corr_det = np.nonzero(corr)
    corr_key = corr_t * n_det + corr_det
    # Index bookkeeping; every searched array is sorted, keys t * D + index order
    # (threshold, position) like the TP and correction lists:
    # - tp_seg_lo, side "left" on tp_seg: first TP of the TP's own segment.
    # - tp_hi: where tp_det + 1 sits in bounds, so cum[tp_hi] includes the TP itself;
    #   tp_det + 1 == D is not in bounds and maps to U, the total.
    # - tp_lo: where the class start sits in bounds (always present).
    # - corr_upto, side "right": a correction on the TP itself is included.
    # - corr_lo, side "left" on the segment start: earlier segments are excluded.
    return _AreaTable(
        in_area=None if in_area.all() else in_area,
        n_gt=n_gt,
        bounds=bounds,
        tp_det=tp_det,
        tp_cls=tp_cls,
        tp_key=tp_seg * (_R + 1),
        tp_seg_lo=np.searchsorted(tp_seg, tp_seg, side="left"),
        tp_hi=np.searchsorted(bounds, tp_det + 1),
        tp_lo=np.searchsorted(bounds, cls_lo),
        corr_det=corr_det,
        corr_sign=corr[corr_t, corr_det].astype(np.int64),
        corr_upto=np.searchsorted(corr_key, tp_t * n_det + tp_det, side="right"),
        corr_lo=np.searchsorted(corr_key, tp_t * n_det + cls_lo, side="left"),
    )


def build_cache(gts: Mapping[str, GT], preds: Mapping[str, Pred], max_dets: int = 300) -> EvalCache:
    """Match predictions to labels per image and class at all IoU thresholds and area
    ranges (small/medium/large by box area, limits 32**2 and 96**2, as COCO).

    Images are the union of both key sets: an image missing from `preds` has no
    predictions, one missing from `gts` has no labels.
    """
    if max_dets < 1:
        raise ValueError(f"max_dets must be positive, got {max_dets}")
    image_ids = tuple(sorted(set(gts) | set(preds)))
    g = [(np.zeros(0, np.int64), np.zeros((0, 4)), np.zeros(0, np.int64))]
    d = [(np.zeros(0, np.int64), np.zeros((0, 4)), np.zeros(0, np.int64), np.zeros(0))]
    for i, key in enumerate(image_ids):
        boxes, labels = _gt_arrays(gts.get(key))
        g.append((np.full(len(labels), i), boxes, labels))
        boxes, labels, scores = _pred_arrays(preds.get(key))
        d.append((np.full(len(labels), i), boxes, labels, scores))
    g_img, g_box, g_cls = (np.concatenate(x) for x in zip(*g))
    d_img, d_box, d_cls, d_score = (np.concatenate(x) for x in zip(*d))
    classes = np.unique(np.r_[g_cls, d_cls])
    n_img, n_cls, n_area = len(image_ids), len(classes), len(AREA_RANGES)
    g_k, d_k = np.searchsorted(classes, g_cls), np.searchsorted(classes, d_cls)

    # Per image and class, as COCOeval: labels in input order, detections by score
    # (stable), at most max_dets of them.
    go = np.lexsort((g_k, g_img))
    g_box, g_grp = _xywh(g_box[go]), (g_img * n_cls + g_k)[go]
    do = np.lexsort((-d_score, d_k, d_img))
    d_grp = (d_img * n_cls + d_k)[do]
    keep = do[np.arange(len(do)) - np.searchsorted(d_grp, d_grp) < max_dets]
    d_img, d_k, d_score, d_box = d_img[keep], d_k[keep], d_score[keep], _xywh(d_box[keep])
    d_grp = d_img * n_cls + d_k

    lo = np.array([r[0] for r in AREA_RANGES.values()], float)[:, None]
    hi = np.array([r[1] for r in AREA_RANGES.values()], float)[:, None]
    g_area, d_area = g_box[:, 2] * g_box[:, 3], d_box[:, 2] * d_box[:, 3]
    g_ign = (g_area < lo) | (g_area > hi)  # (A, G)
    d_in = ~((d_area < lo) | (d_area > hi))  # (A, D)
    n_gt = np.stack([np.bincount(g_grp[~ign], minlength=n_img * n_cls).reshape(n_img, n_cls)
                     for ign in g_ign])

    matched = np.zeros((n_area, _T, len(d_grp)), bool)
    matched_ign = np.zeros_like(matched)
    pairs = np.intersect1d(g_grp, d_grp)  # only these can have matches
    for g0, g1, d0, d1 in zip(*(np.searchsorted(x, pairs, side=s).tolist()
                                for x in (g_grp, d_grp) for s in ("left", "right"))):
        iou, memo = _iou(d_box[d0:d1], g_box[g0:g1]), {}
        for a in range(n_area):
            ign = g_ign[a, g0:g1]
            if ign.tobytes() not in memo:  # area ranges often share an ignore pattern
                memo[ign.tobytes()] = _match(iou, ign)
            matched[a, :, d0:d1], matched_ign[a, :, d0:d1] = memo[ign.tobytes()]
    tp = matched & ~matched_ign
    counted = tp | (~matched & d_in[:, None, :])  # COCOeval: not dtIgnore
    corr = counted.astype(np.int8) - d_in[:, None, :]

    # All images of a class, as COCOeval accumulates them: stable sort by score of the
    # per-image lists concatenated in image order.
    order = np.lexsort((-d_score, d_k))
    det_cls = d_k[order]
    class_lo = np.r_[0, np.cumsum(np.bincount(det_cls, minlength=n_cls))].astype(np.int64)
    areas = {name: _area_table(tp[a][:, order], corr[a][:, order], d_in[a][order], det_cls,
                               class_lo, n_cls, n_gt[a])
             for a, name in enumerate(AREA_RANGES)}
    return EvalCache(image_ids, classes, d_img[order], areas)


def _cumsum0(x: np.ndarray) -> np.ndarray:
    out = np.zeros(len(x) + 1, np.int64)
    np.cumsum(x, out=out[1:])
    return out


def _ap_matrix(cache: EvalCache, area: str, w_img: np.ndarray) -> np.ndarray:
    """(K, T) AP of every cache class with integer image weights w_img (I,).

    NaN where the weighted class has no non-ignored label. The interpolated precision
    at recall threshold r is the best precision among true positives with recall >= r
    (other points never exceed the preceding true positive), so per (threshold, class)
    segment the TPs are binned by how many recall thresholds they reach.
    """
    tab, n_cls = cache.areas[area], len(cache.classes)
    w_det = w_img[cache.det_img]
    sums = np.add.reduceat(w_det if tab.in_area is None else w_det * tab.in_area, tab.bounds) \
        if len(tab.bounds) else np.zeros(0, np.int64)
    cum = _cumsum0(sums)  # cum[u]: weight of detections before bounds[u] (u = U: all)
    counted = cum[tab.tp_hi] - cum[tab.tp_lo]
    if len(tab.corr_det):
        cc = _cumsum0(tab.corr_sign * w_det[tab.corr_det])
        counted += cc[tab.corr_upto] - cc[tab.corr_lo]
    tc = _cumsum0(w_det[tab.tp_det])
    tp = tc[1:] - tc[tab.tp_seg_lo]
    npig = w_img @ tab.n_gt
    rc = tp / np.maximum(npig, 1)[tab.tp_cls]
    pr = tp / (counted + np.spacing(1))  # COCOeval: tp / (fp + tp + eps)
    key = tab.tp_key + np.searchsorted(REC_THRS, rc, side="right")
    best = np.zeros(_T * n_cls * (_R + 1))
    if len(key):
        starts = np.flatnonzero(np.r_[True, key[1:] != key[:-1]])
        best[key[starts]] = np.maximum.reduceat(pr, starts)
    # q[s, j] = max(best[s, R - j:]) = precision at REC_THRS[R - 1 - j] (reversed; fine for the mean)
    q = np.maximum.accumulate(best.reshape(_T * n_cls, _R + 1)[:, :0:-1], axis=1)
    ap = q.mean(axis=1).reshape(_T, n_cls).T.copy()
    ap[npig == 0] = np.nan
    return ap


def _rows(cache: EvalCache, classes: Sequence[int]) -> np.ndarray:
    k_of = {c: k for k, c in enumerate(cache.classes.tolist())}
    return np.array([k_of.get(int(c), -1) for c in classes], dtype=np.int64)


def _take(ap: np.ndarray, rows: np.ndarray) -> np.ndarray:
    out = np.full((len(rows), _T), np.nan)
    out[rows >= 0] = ap[rows[rows >= 0]]
    return out


def _mean_ap(ap: np.ndarray) -> float:
    """Mean over the classes (rows) that have labels; NaN if none has."""
    ok = ~np.isnan(ap).any(axis=1)
    return float(ap[ok].mean()) if ok.any() else float("nan")


def _weight_vector(cache: EvalCache, weights: Mapping[str, int] | None) -> np.ndarray:
    if weights is None:
        return np.ones(len(cache.image_ids), np.int64)
    w = np.zeros(len(cache.image_ids), np.int64)
    index = {k: i for i, k in enumerate(cache.image_ids)}
    for key, v in weights.items():
        if v < 0 or v != int(v):
            raise ValueError(f"weight of {key!r} must be a non-negative integer, got {v!r}")
        if key in index:  # other images have neither labels nor predictions here
            w[index[key]] = int(v)
    return w


def ap_table(cache: EvalCache, classes: Sequence[int], area: str = "all",
             weights: Mapping[str, int] | None = None) -> np.ndarray:
    """(C, 10) AP per class and IoU threshold; NaN for a class without labels.

    `weights` are per-image multiplicities: weight k counts the image as k copies
    (exact convention at score ties: module docstring), and images absent from
    `weights` count 0 times.
    """
    if area not in AREA_RANGES:
        raise ValueError(f"area must be one of {list(AREA_RANGES)}, got {area!r}")
    return _take(_ap_matrix(cache, area, _weight_vector(cache, weights)), _rows(cache, classes))


def summarize(cache: EvalCache, core: Sequence[int], all_classes: Sequence[int]) -> dict:
    """Headline mAP50-95 over `core` plus the same numbers over `all_classes`.

    The "all" means and `by_size` average over the classes of `all_classes` that have
    labels (in that size range); classes without labels are listed in
    `missing_classes`. Raises ValueError if a core class has no labels.
    """
    core, all_classes = [int(c) for c in core], [int(c) for c in all_classes]
    if not core or len(set(all_classes)) != len(all_classes) or not set(core) <= set(all_classes):
        raise ValueError("core must be a non-empty subset of all_classes, without duplicates")
    rows = _rows(cache, all_classes)
    ones = np.ones(len(cache.image_ids), np.int64)
    ap = _take(_ap_matrix(cache, "all", ones), rows)
    n_gt = np.r_[cache.areas["all"].n_gt.sum(axis=0), 0][rows]  # row -1 (class not in cache) -> 0
    missing = [c for c, n in zip(all_classes, n_gt) if n == 0]
    if missing_core := [c for c in core if c in missing]:
        raise ValueError(f"core classes without labels: {missing_core}")
    pos = {c: j for j, c in enumerate(all_classes)}
    core_ap = ap[[pos[c] for c in core]]
    headline = _mean_ap(core_ap)
    return {
        "headline_map50_95": headline,
        "map50_95_core": headline,
        "map50_core": _mean_ap(core_ap[:, [_I50]]),
        "map75_core": _mean_ap(core_ap[:, [_I75]]),
        "map50_95_all": _mean_ap(ap),
        "map50_all": _mean_ap(ap[:, [_I50]]),
        "map75_all": _mean_ap(ap[:, [_I75]]),
        "per_class": {c: {"ap50_95": float(ap[j].mean()), "ap50": float(ap[j, _I50]),
                          "n_gt": int(n_gt[j])} for j, c in enumerate(all_classes)},
        "by_size": {s: _mean_ap(_take(_ap_matrix(cache, s, ones), rows))
                    for s in ("small", "medium", "large")},
        "missing_classes": missing,
    }


def bootstrap_delta(caches_a: Sequence[EvalCache], caches_b: Sequence[EvalCache],
                    core: Sequence[int], cluster_of: Mapping[str, int],
                    n: int = 1000, seed: int = 0) -> dict:
    """Cluster bootstrap of delta = mean core headline of A's caches - that of B's.

    Each of the n iterations draws the distinct clusters of the cache images with
    replacement (as many draws as clusters); an image's weight is how often its
    cluster was drawn, and the same draw weights every cache of A and B. A resample
    that leaves a core class without labels averages over the core classes it still
    has; such iterations are counted in `n_degenerate`. One that loses every core
    class has no delta: it is counted too and left out of the percentiles (ValueError
    if no iteration has a delta). `delta` is the estimate on the original data;
    `ci_low`/`ci_high` are the 2.5th/97.5th percentiles.
    """
    caches_a, caches_b = list(caches_a), list(caches_b)
    if not caches_a or not caches_b or n < 1:
        raise ValueError("need at least one cache for each model and n >= 1")
    caches, core = caches_a + caches_b, [int(c) for c in core]
    missing = sorted({k for c in caches for k in c.image_ids if k not in cluster_of})
    if missing:
        raise KeyError(f"{len(missing)} images have no cluster, e.g. {missing[0]!r}")
    clusters = sorted({cluster_of[k] for c in caches for k in c.image_ids})
    pos = {c: j for j, c in enumerate(clusters)}
    img_cluster = [np.array([pos[cluster_of[k]] for k in c.image_ids], np.int64) for c in caches]
    rows = [_rows(c, core) for c in caches]

    def delta(counts: np.ndarray) -> tuple[float, set[int]]:
        heads, absent = [], set()
        for cache, ic, r in zip(caches, img_cluster, rows):
            ap = _take(_ap_matrix(cache, "all", counts[ic]), r)
            absent.update(c for c, a in zip(core, ap[:, 0]) if np.isnan(a))
            heads.append(_mean_ap(ap))
        return float(np.mean(heads[:len(caches_a)]) - np.mean(heads[len(caches_a):])), absent

    point, absent = delta(np.ones(len(clusters), np.int64))
    if absent:
        raise ValueError(f"core classes without labels: {sorted(absent)}")
    rng = np.random.default_rng(seed)
    deltas, n_degenerate = np.empty(n), 0
    for j in range(n):
        counts = np.bincount(rng.integers(0, len(clusters), len(clusters)), minlength=len(clusters))
        deltas[j], absent = delta(counts)
        n_degenerate += bool(absent)
    finite = deltas[np.isfinite(deltas)]
    if not len(finite):
        raise ValueError("every resample lost all core classes: no finite delta")
    ci_low, ci_high = np.percentile(finite, [2.5, 97.5])
    return {"delta": point, "ci_low": float(ci_low), "ci_high": float(ci_high),
            "n_degenerate": n_degenerate}
