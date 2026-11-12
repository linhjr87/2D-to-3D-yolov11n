"""Run both models on new images and write one JSON per image keyed by schema_id."""
from __future__ import annotations

import argparse
import sys
from collections.abc import Iterator, Sequence
from pathlib import Path

import cv2
import numpy as np

from fpyolo.evaluate_d import iter_predict, write_json
from fpyolo.ink import load_ink
from fpyolo.schema import Schema, load_schema
from fpyolo.train import SCHEMA_PATH, _sha256, apply_device_env

IMGSZ = 1024
IOU = 0.7
MAX_DET = 300
D_CONF = 0.25
IMAGE_SUFFIXES = (".png", ".jpg", ".jpeg")

Ring = list[list[float]]
Polygon = tuple[Ring, list[Ring]]


def mask_to_polygons(mask: np.ndarray) -> list[Polygon]:
    """(exterior, holes) per connected component of a binary mask, in pixel (x, y)."""
    contours, hierarchy = cv2.findContours(
        (np.asarray(mask) > 0).astype(np.uint8), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    if hierarchy is None:
        return []
    hierarchy = hierarchy[0]  # rows: next, previous, first child, parent

    def ring(i: int) -> Ring:
        return contours[i].reshape(-1, 2).astype(float).tolist()

    out: list[Polygon] = []
    for i, (_, _, _, parent) in enumerate(hierarchy):
        if parent != -1:
            continue
        holes = [ring(j) for j, h in enumerate(hierarchy) if h[3] == i]
        out.append((ring(i), holes))
    return out


def _round_ring(ring: Ring) -> Ring:
    return [[round(float(x), 2), round(float(y), 2)] for x, y in ring]


def to_json(image_name: str, w: int, h: int, d_out: Sequence, w_out: Sequence, schema: Schema,
            model_ids: dict[str, str], thresholds: dict[str, float] | None = None) -> dict:
    """JSON-ready dict. Class ids are schema_ids; YOLO indices never leave this function.

    d_out: (yolo_class, score, (x1, y1, x2, y2)) per box.
    w_out: (yolo_class, score, binary mask at native resolution) per instance.
    """
    d_defs, w_defs = schema.for_model("D"), schema.for_model("W")
    objects = []
    for cls, score, box in d_out:
        c = d_defs[int(cls)]
        objects.append({"schema_id": c.schema_id, "name": c.name, "score": round(float(score), 2),
                        "bbox_xyxy": [round(float(v), 2) for v in box]})
    walls = []
    for cls, score, mask in w_out:
        c = w_defs[int(cls)]
        for exterior, holes in mask_to_polygons(mask):
            if len(exterior) < 3:
                continue
            walls.append({"schema_id": c.schema_id, "name": c.name, "score": round(float(score), 2),
                          "exterior": _round_ring(exterior),
                          "holes": [_round_ring(r) for r in holes if len(r) >= 3]})
    doc = {"image": image_name, "width": int(w), "height": int(h), "schema_version": schema.version,
           "models": dict(model_ids)}
    if thresholds is not None:
        doc["thresholds"] = dict(thresholds)
    doc["objects"] = objects
    doc["walls"] = walls
    return doc


def _predict_one(model, path: Path, array: np.ndarray | None, **kwargs):
    """One Result for one image. Never hands Ultralytics a list (it would become one batch)."""
    if array is None:
        [(_, r)] = iter_predict(model, [path], **kwargs)
        return r
    [r] = model.predict(source=array, stream=False, verbose=False, **kwargs)
    return r


def extract_d(result) -> list[tuple[int, float, list[float]]]:
    b = result.boxes
    if b is None or len(b) == 0:
        return []
    xyxy, cls, conf = (t.cpu().numpy() for t in (b.xyxy, b.cls, b.conf))
    return [(int(c), float(s), box.tolist()) for c, s, box in zip(cls, conf, xyxy)]


def extract_w(result, h: int, w: int) -> Iterator[tuple[int, float, np.ndarray]]:
    """Per-instance native-resolution binary masks, converted one at a time."""
    if result.masks is None or len(result.boxes) == 0:
        return
    data = result.masks.data
    if tuple(data.shape[1:]) != (h, w):
        raise ValueError(f"masks are {tuple(data.shape[1:])} but the image is {(h, w)}: "
                         "predict with retina_masks=True")
    cls, conf = result.boxes.cls.cpu().numpy(), result.boxes.conf.cpu().numpy()
    for i in range(len(conf)):
        yield int(cls[i]), float(conf[i]), (data[i] > 0).cpu().numpy()


def predict_image(d_model, w_model, path: Path, schema: Schema, model_ids: dict[str, str], *,
                  d_conf: float, w_conf: float, device: str, from_floorplancad_png: bool) -> dict:
    array = None
    if from_floorplancad_png:
        array = np.repeat(load_ink(path)[..., None], 3, axis=2)
    common = dict(imgsz=IMGSZ, iou=IOU, max_det=MAX_DET, device=device)
    rd = _predict_one(d_model, path, array, conf=d_conf, **common)
    h, w = rd.orig_shape
    d_out = extract_d(rd)
    del rd
    rw = _predict_one(w_model, path, array, conf=w_conf, retina_masks=True, **common)
    if tuple(rw.orig_shape) != (h, w):
        raise RuntimeError(f"D and W disagree on the size of {path}: {(h, w)} vs {tuple(rw.orig_shape)}")
    w_out = list(extract_w(rw, h, w))
    del rw
    return to_json(path.name, w, h, d_out, w_out, schema, model_ids,
                   thresholds={"D": d_conf, "W": w_conf})


def collect_inputs(inp: Path) -> list[Path]:
    """The file itself, or the sorted images of a directory (non-recursive)."""
    inp = Path(inp)
    if inp.is_dir():
        files = sorted(p for p in inp.iterdir() if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES)
    elif inp.is_file():
        files = [inp]
    else:
        raise FileNotFoundError(f"input {inp} does not exist")
    if not files:
        raise ValueError(f"no {'/'.join(IMAGE_SUFFIXES)} images in {inp}")
    stems = [p.stem for p in files]
    dup = sorted({s for s in stems if stems.count(s) > 1})
    if dup:
        raise ValueError(f"inputs share a stem, their JSON files would collide: {dup}")
    return files


def check_model_names(model, schema: Schema, key: str, weights: Path) -> None:
    """Fail unless the model's class names are exactly the schema's classes of `key`, in order."""
    got = {int(k): v for k, v in model.names.items()}
    want = schema.yolo_names(key)
    if got == want:
        return
    if len(got) != len(want):
        detail = f"{len(got)} classes, the schema has {len(want)}"
    else:
        i = next(i for i in sorted(want) if got.get(i) != want[i])
        detail = f"class {i} is {got.get(i)!r}, the schema has {want[i]!r}"
    raise ValueError(f"{weights}: class names differ from the {key} classes of schema "
                     f"{schema.version} ({detail})")


def model_id(weights: Path) -> str:
    return f"{Path(weights).name}@{_sha256(Path(weights))[:12]}"


def run(d_weights: Path, w_weights: Path, w_conf: float, schema_path: Path, inp: Path, out: Path,
        device: str, from_floorplancad_png: bool = False, d_conf: float = D_CONF,
        overwrite: bool = False) -> list[Path]:
    schema = load_schema(schema_path)
    files = collect_inputs(inp)
    out = Path(out)
    targets = [out / f"{p.stem}.json" for p in files]
    existing = [t for t in targets if t.exists()]
    if existing and not overwrite:
        raise FileExistsError(f"{len(existing)} output file(s) exist, e.g. {existing[0]}; "
                              "pass --overwrite to replace them")
    from ultralytics import YOLO

    d_model, w_model = YOLO(str(d_weights)), YOLO(str(w_weights))
    check_model_names(d_model, schema, "D", d_weights)
    check_model_names(w_model, schema, "W", w_weights)
    model_ids = {"D": model_id(d_weights), "W": model_id(w_weights)}
    out.mkdir(parents=True, exist_ok=True)
    for path, target in zip(files, targets):
        doc = predict_image(d_model, w_model, path, schema, model_ids, d_conf=d_conf, w_conf=w_conf,
                            device=device, from_floorplancad_png=from_floorplancad_png)
        write_json(target, doc)
        print(f"{path.name}: {len(doc['objects'])} objects, {len(doc['walls'])} wall polygons -> {target}")
    return targets


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fpyolo.predict", description=__doc__.split("\n")[0])
    ap.add_argument("--d-weights", required=True, type=Path)
    ap.add_argument("--w-weights", required=True, type=Path)
    ap.add_argument("--w-conf", required=True, type=float, help="wall threshold t* chosen on val")
    ap.add_argument("--d-conf", type=float, default=D_CONF, help="object score threshold")
    ap.add_argument("--schema", type=Path, default=SCHEMA_PATH)
    ap.add_argument("--input", required=True, type=Path, help="an image or a directory of images")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--device", required=True, help="'cpu' or one GPU index; there is no default")
    ap.add_argument("--from-floorplancad-png", action="store_true",
                    help="inputs are FloorPlanCAD RGBA PNGs: convert to ink images first")
    ap.add_argument("--overwrite", action="store_true", help="replace existing JSON files")
    args = ap.parse_args(argv)
    for name in ("d_conf", "w_conf"):
        if not 0.0 <= getattr(args, name) <= 1.0:
            ap.error(f"--{name.replace('_', '-')} must be in [0, 1]")
    apply_device_env(args.device)
    run(args.d_weights, args.w_weights, args.w_conf, args.schema, args.input, args.out, args.device,
        args.from_floorplancad_png, args.d_conf, args.overwrite)
    return 0


if __name__ == "__main__":
    sys.exit(main())
