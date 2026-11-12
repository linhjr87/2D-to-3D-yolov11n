"""Ultralytics training wrapper: pinned configs, explicit device, env.json record."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import torch
import yaml

from fpyolo.schema import load_schema

REPO = Path(__file__).resolve().parents[2]
SCHEMA_PATH = REPO / "configs" / "classes_v0.yaml"
SMOKE = {"epochs": 1, "imgsz": 256, "fraction": 0.01, "batch": 4}


def train_kwargs(cfg: dict, seed: int, data: str, device: str, smoke: bool = False) -> dict:
    """Keyword arguments for ``YOLO(cfg["model"]).train(...)``.

    ``optimizer: auto`` (or a missing optimizer, which Ultralytics turns into auto)
    is rejected: auto silently picks the optimizer and learning rate from the
    iteration count, which would make runs with different batch sizes incomparable.
    """
    optimizer = cfg.get("optimizer")
    if optimizer is None or str(optimizer).lower() == "auto":
        raise ValueError(f"optimizer must be set explicitly, 'auto' is forbidden (got {optimizer!r})")
    kw = {k: v for k, v in cfg.items() if k != "model"}
    kw.update(data=str(data), seed=int(seed), device=str(device))
    if smoke:
        kw.update(SMOKE)
    return kw


def _git_commit() -> str | None:
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return out.stdout.strip() or None


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _gpu_info(device: str) -> dict:
    """Name and free/total memory of one CUDA device (initializes a context on it)."""
    try:
        idx = int(device)
    except ValueError:
        raise ValueError(f"device must be 'cpu' or a single GPU index, got {device!r}") from None
    free, total = torch.cuda.mem_get_info(idx)
    props = torch.cuda.get_device_properties(idx)
    return {
        "index": idx,
        "name": torch.cuda.get_device_name(idx),
        # same format as `nvidia-smi --query-gpu=pci.bus_id`, e.g. 00000000:3B:00.0
        "pci_bus_id": f"{props.pci_domain_id:08X}:{props.pci_bus_id:02X}:{props.pci_device_id:02X}.0",
        "free_mib": free // 2**20,
        "total_mib": total // 2**20,
    }


def apply_device_env(device: str) -> None:
    """Hide every GPU when the run is on the CPU (Ultralytics' select_device does not).

    Must run before the first CUDA call.
    """
    if device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""


def env_record(device: str, data_yaml: Path | None = None) -> dict:
    """Reproducibility record written next to each run (all keys always present).

    ``split_manifest_sha256`` hashes ``<data_yaml dir>/../manifests/tiles.csv``.
    """
    import ultralytics

    manifest = None
    if data_yaml is not None:
        tiles = Path(data_yaml).resolve().parent.parent / "manifests" / "tiles.csv"
        if tiles.is_file():
            manifest = _sha256(tiles)
    return {
        "ultralytics": ultralytics.__version__,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": None if device == "cpu" else _gpu_info(device),
        "device": device,
        "git_commit": _git_commit(),
        "schema_version": load_schema(SCHEMA_PATH).version,
        "split_manifest_sha256": manifest,
    }


def _positive_int(text: str) -> int:
    """argparse type: Ultralytics treats batch < 1 as auto-batch, so reject it."""
    try:
        value = int(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected an integer, got {text!r}") from None
    if value < 1:
        raise argparse.ArgumentTypeError(
            f"must be >= 1 (got {value}); Ultralytics would turn batch < 1 into auto-batch, "
            "which breaks identical D batches"
        )
    return value


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fpyolo.train", description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path, help="Ultralytics data.yaml")
    ap.add_argument("--seed", required=True, type=int)
    ap.add_argument("--project", required=True, type=Path)
    ap.add_argument("--name", required=True)
    ap.add_argument("--device", required=True, help="'cpu' or one GPU index; there is no default")
    ap.add_argument("--batch", type=_positive_int, help="override the config batch (e.g. after an OOM trial)")
    ap.add_argument("--smoke", action="store_true", help="epochs=1 imgsz=256 fraction=0.01 batch=4")
    args = ap.parse_args(argv)

    cfg = yaml.safe_load(args.config.read_text())
    if args.batch is not None:
        cfg["batch"] = args.batch
    kw = train_kwargs(cfg, args.seed, str(args.data.resolve()), args.device, smoke=args.smoke)

    project = args.project.resolve()
    run_dir = project / args.name
    if run_dir.exists():
        # Ultralytics would silently write to <name>2; env.json must sit next to the weights.
        ap.error(f"{run_dir} already exists; pick a new --name (to continue a run use YOLO(last.pt).train(resume=True))")
    apply_device_env(args.device)
    record = env_record(args.device, args.data)
    run_dir.mkdir(parents=True)
    (run_dir / "env.json").write_text(json.dumps(record, indent=2) + "\n")

    from ultralytics import YOLO

    # exist_ok: the directory already exists because env.json was written above.
    YOLO(cfg["model"]).train(project=str(project), name=args.name, exist_ok=True, **kw)
    return 0


if __name__ == "__main__":
    # Run the importable module so classes pickled to workers are fpyolo.train.*.
    from fpyolo.train import main as _main

    sys.exit(_main())
