"""Model D evaluation: COCO-style AP over the fixed class sets (apeval), plus Ultralytics' own val.

Writes into --out: metrics.json (summarize output + run paths + Ultralytics val numbers),
per_class.csv, preds.npz (the raw predictions, re-read by fpyolo.compare) and
Ultralytics' val plots (confusion matrix, PR curves).

preds.npz layout (np.savez_compressed, no pickles): `ids` (I,) str, sorted image ids;
`offsets` (I + 1,) int64; `boxes` (M, 4) xyxy px, `cls` (M,), `scores` (M,), with the
predictions of image ids[i] in rows offsets[i]:offsets[i + 1] (possibly none).
"""
from __future__ import annotations

import argparse
import csv
import ctypes
import json
import math
import shutil
import sys
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from fpyolo.apeval import GT, Pred, build_cache, summarize
from fpyolo.schema import Schema, load_schema
from fpyolo.train import SCHEMA_PATH, apply_device_env

IMG_SUFFIXES = {".png", ".jpg", ".jpeg"}
# Prediction settings shared by every evaluation (global constraints).
IMGSZ, CONF, IOU, MAX_DET = 1024, 0.001, 0.7, 300
# Ultralytics val loader, explicit to bound host RAM: on a GPU Ultralytics would otherwise start
# up to 8 persistent worker processes, each prefetching 2 batches (on CPU it forces 0 workers).
VAL_BATCH, VAL_WORKERS = 16, 2
ULTRA_NOTE = ("Ultralytics' own averaging: mean over the classes that have labels in this split "
              "(absent classes are dropped); not the fixed class sets used above")


def split_images(dataset_dir: Path, split: str, limit: int | None = None) -> list[Path]:
    """Images of `<dataset>/images/<split>` sorted by id (file stem); the first `limit` if given."""
    d = Path(dataset_dir) / "images" / split
    if not d.is_dir():
        raise FileNotFoundError(f"no image directory {d}")
    images = sorted((p for p in d.iterdir() if p.suffix.lower() in IMG_SUFFIXES), key=lambda p: p.stem)
    return images if limit is None else images[:limit]


def label_path(image: Path) -> Path:
    """<dataset>/images/<split>/<id>.png -> <dataset>/labels/<split>/<id>.txt"""
    return image.parents[2] / "labels" / image.parent.name / f"{image.stem}.txt"


def read_label_rows(path: Path) -> list[list[str]]:
    """Whitespace-split non-empty lines; a missing file has no objects (Ultralytics convention)."""
    if not path.is_file():
        return []
    return [line.split() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_split_gt(dataset_dir: Path, split: str, limit: int | None = None) -> dict[str, GT]:
    """YOLO box labels of a split as pixel xyxy boxes, scaled by each image's true size."""
    gts = {}
    for img in split_images(dataset_dir, split, limit):
        with Image.open(img) as im:
            w, h = im.size
        rows = np.array(read_label_rows(label_path(img)), dtype=float).reshape(-1, 5)
        xc, yc, bw, bh = (rows[:, 1:] * [w, h, w, h]).T
        boxes = np.stack([xc - bw / 2, yc - bh / 2, xc + bw / 2, yc + bh / 2], axis=1)
        gts[img.stem] = (boxes, rows[:, 0].astype(np.int64))
    return gts


def iter_predict(model, images: Sequence[Path], **kwargs) -> Iterator[tuple[Path, object]]:
    """(image, Ultralytics Result) pairs, one predict call per image.

    Never pass Ultralytics a list of images: 8.4 turns a list source into in-memory images
    run as ONE batch of len(list) (check_source -> autocast_list -> LoadPilAndNumpy), even
    with stream=True, so a whole split was decoded and inferred at once (~48 GB on val).
    """
    for img in images:
        [r] = model.predict(source=str(img), stream=False, verbose=False, **kwargs)
        if Path(r.path).stem != img.stem:
            raise RuntimeError(f"prediction for {r.path} arrived in place of {img}")
        yield img, r


def predict_split(weights: Path, images: Sequence[Path], imgsz: int = IMGSZ, conf: float = CONF,
                  iou: float = IOU, max_det: int = MAX_DET, *, device: str) -> dict[str, Pred]:
    """Box predictions keyed by image id (stem); images without detections map to empty arrays."""
    if not images:
        return {}
    from ultralytics import YOLO

    preds = {}
    for img, r in iter_predict(YOLO(str(weights)), images, imgsz=imgsz, conf=conf, iou=iou,
                               max_det=max_det, device=device):
        b = r.boxes  # owned copies: nothing of the Result outlives this iteration
        preds[img.stem] = (b.xyxy.cpu().numpy().copy(), b.cls.cpu().numpy().astype(np.int64),
                           b.conf.cpu().numpy().copy())
    return preds


def write_preds(path: Path, preds: Mapping[str, Pred]) -> None:
    """Save predictions so that `read_preds` returns the same arrays (values and dtypes)."""
    ids = sorted(preds)
    parts = [tuple(np.asarray(x) for x in preds[k]) for k in ids]
    boxes = [b.reshape(-1, 4) for b, _, _ in parts]
    cls = [c.reshape(-1) for _, c, _ in parts]
    scores = [s.reshape(-1) for _, _, s in parts]
    for k, b, c, s in zip(ids, boxes, cls, scores):
        if not len(b) == len(c) == len(s):
            raise ValueError(f"{k}: {len(b)} boxes, {len(c)} classes, {len(s)} scores")
    counts = [len(c) for c in cls]
    np.savez_compressed(
        path,
        ids=np.array(ids, dtype=str),
        offsets=np.r_[0, np.cumsum(counts, dtype=np.int64)].astype(np.int64),
        boxes=np.concatenate(boxes) if ids else np.zeros((0, 4), np.float32),
        cls=np.concatenate(cls) if ids else np.zeros(0, np.int64),
        scores=np.concatenate(scores) if ids else np.zeros(0, np.float32),
    )


def read_preds(path: Path) -> dict[str, Pred]:
    with np.load(path, allow_pickle=False) as z:
        ids, off, boxes, cls, scores = (z[k] for k in ("ids", "offsets", "boxes", "cls", "scores"))
    return {str(k): (boxes[off[i]:off[i + 1]], cls[off[i]:off[i + 1]], scores[off[i]:off[i + 1]])
            for i, k in enumerate(ids)}


def _strict(o):
    if isinstance(o, Mapping):
        return {str(k): _strict(v) for k, v in o.items()}
    if isinstance(o, (list, tuple, np.ndarray)):
        return [_strict(v) for v in o]
    if isinstance(o, np.generic):
        o = o.item()
    if isinstance(o, float) and not math.isfinite(o):
        return None
    return o


def write_json(path: Path, obj) -> None:
    """Strict JSON: NaN/inf become null, numpy scalars and arrays become plain values."""
    text = json.dumps(_strict(obj), indent=2, allow_nan=False, ensure_ascii=False)
    Path(path).write_text(text + "\n", encoding="utf-8")


def core_indices(manifest_dir: Path, schema: Schema) -> list[int]:
    """core_classes.json semantic ids as D YOLO indices."""
    doc = json.loads((Path(manifest_dir) / "core_classes.json").read_text(encoding="utf-8"))
    yolo = schema.yolo_index("D")
    return sorted(yolo[int(s)] for s in doc["semantic_ids"])


def class_key(index: int, names: Mapping[int, str]) -> str:
    return f"{index}_{names[index]}"


def _subset_data_yaml(dataset: Path, split: str, images: Sequence[Path], root: Path) -> Path:
    """A dataset of symlinks to `images` and their labels, so that Ultralytics val sees only
    them and writes its label cache under `root`, not into the real dataset."""
    if root.exists():
        shutil.rmtree(root)
    for img in images:
        for src, kind in ((img, "images"), (label_path(img), "labels")):
            if src.is_file():
                dst = root / kind / split / src.name
                dst.parent.mkdir(parents=True, exist_ok=True)
                dst.symlink_to(src.resolve())
    names = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))["names"]
    data = {"path": str(root), "train": f"images/{split}", "val": f"images/{split}",
            "test": f"images/{split}", "names": names}
    (root / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    return root / "data.yaml"


def ultralytics_val(weights: Path, dataset: Path, split: str, out: Path, device: str,
                    images: Sequence[Path] | None = None):
    """Ultralytics' model.val with its plots (confusion matrix, curves) saved into `out`.

    `images` restricts it to a subset (smoke runs with --limit)."""
    from ultralytics import YOLO

    data = dataset / "data.yaml"
    if images is not None:
        data = _subset_data_yaml(dataset, split, images, out / "val_subset")
    model = YOLO(str(weights))
    model.add_callback("on_val_batch_end", trim_heap)
    return model.val(data=str(data), split=split, imgsz=IMGSZ, conf=CONF, iou=IOU, max_det=MAX_DET,
                     batch=VAL_BATCH, workers=VAL_WORKERS, device=device, plots=True,
                     project=str(out.parent), name=out.name, exist_ok=True)


def _libc_malloc_trim():
    try:
        return ctypes.CDLL("libc.so.6").malloc_trim
    except (OSError, AttributeError):  # not glibc
        return None


_MALLOC_TRIM = _libc_malloc_trim()


def trim_heap(*_) -> None:
    """Give freed heap memory back to the OS (an Ultralytics callback, hence the ignored argument).

    glibc keeps freed pages of the variable-size per-batch buffers, so without this the RSS of
    W val crept up ~6 MB per image (measured flat with a trim after every batch).
    """
    if _MALLOC_TRIM is not None:
        _MALLOC_TRIM(0)


def val_summary(m, device: str) -> dict:
    """Box numbers and speed (ms/image) of an Ultralytics val result, reported as-is."""
    if device == "cpu":
        device_name = "cpu"
    else:
        import torch

        device_name = torch.cuda.get_device_name(int(device))
    return {"note": ULTRA_NOTE, "box_map50_95": m.box.map, "box_map50": m.box.map50,
            "precision": m.box.mp, "recall": m.box.mr,
            "speed_ms": {k: m.speed[k] for k in ("preprocess", "inference", "postprocess")},
            "batch": VAL_BATCH, "device_name": device_name}


def _write_per_class(path: Path, per_class: Mapping[str, dict]) -> None:
    cols = ["yolo_index", "name", "schema_id", "core", "n_gt", "ap50_95", "ap50"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for key, row in per_class.items():
            vals = [int(key.split("_", 1)[0])] + [row[c] for c in cols[1:]]
            wr.writerow(["" if isinstance(v, float) and math.isnan(v) else v for v in vals])


def evaluate(weights: Path, dataset: Path, split: str, manifest: Path, out: Path, device: str,
             schema_path: Path = SCHEMA_PATH, limit: int | None = None) -> dict:
    weights, dataset, manifest, out = (Path(p).resolve() for p in (weights, dataset, manifest, out))
    schema_path = Path(schema_path).resolve()
    schema = load_schema(schema_path)
    names = schema.yolo_names("D")
    data_names = yaml.safe_load((dataset / "data.yaml").read_text(encoding="utf-8"))["names"]
    if {int(k): v for k, v in data_names.items()} != names:
        raise ValueError(f"{dataset / 'data.yaml'} class names differ from the D classes of {schema_path}")
    all_classes = sorted(names)
    core = core_indices(manifest, schema)
    out.mkdir(parents=True, exist_ok=True)

    images = split_images(dataset, split, limit)
    gts = load_split_gt(dataset, split, limit)
    preds = predict_split(weights, images, device=device)
    write_preds(out / "preds.npz", preds)

    core_used = core
    if limit is not None:  # a small smoke subset may miss core classes
        n_gt = np.bincount(np.concatenate([c for _, c in gts.values()] + [np.zeros(0, np.int64)]),
                           minlength=len(all_classes))
        core_used = [c for c in core if n_gt[c] > 0]
    summary = summarize(build_cache(gts, preds), core_used, all_classes)
    schema_ids = {i: c.schema_id for i, c in enumerate(schema.for_model("D"))}
    summary["per_class"] = {class_key(c, names): {"name": names[c], "schema_id": schema_ids[c],
                                                  "core": c in core, **v}
                            for c, v in summary["per_class"].items()}
    summary["missing_classes"] = [class_key(c, names) for c in summary["missing_classes"]]

    val = ultralytics_val(weights, dataset, split, out, device, images if limit is not None else None)
    metrics = {
        **summary,
        "core": [class_key(c, names) for c in core_used],
        "core_reduced": core_used != core,
        "weights": str(weights), "dataset": str(dataset), "split": split, "manifest": str(manifest),
        "schema": str(schema_path), "limit": limit, "n_images": len(gts), "device": device,
        "ultralytics_val": val_summary(val, device),
    }
    _write_per_class(out / "per_class.csv", summary["per_class"])
    write_json(out / "metrics.json", metrics)
    return metrics


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fpyolo.evaluate_d", description=__doc__.split("\n")[0])
    ap.add_argument("--weights", required=True, type=Path)
    ap.add_argument("--dataset", required=True, type=Path, help="<out>/D (data.yaml, images/, labels/)")
    ap.add_argument("--split", default="test", choices=["val", "test"])
    ap.add_argument("--manifest", required=True, type=Path, help="<out>/manifests (core_classes.json)")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", required=True, help="'cpu' or one GPU index; there is no default")
    ap.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    ap.add_argument("--limit", type=int, help="CPU smoke runs only: evaluate the first N images (by id)")
    args = ap.parse_args(argv)
    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be >= 1")
    apply_device_env(args.device)
    m = evaluate(args.weights, args.dataset, args.split, args.manifest, args.out, args.device,
                 args.schema, args.limit)
    print(f"headline mAP50-95 (core) {m['headline_map50_95']:.4f}, all {m['map50_95_all']:.4f} "
          f"-> {Path(args.out) / 'metrics.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
