"""Run state, resume and post-run checks for the seed sweep (``scripts/run_all_seeds.sh``).

Commands (all take a run directory, i.e. ``<project>/<name>``)::

    python -m fpyolo.runstate state  <run_dir>
    python -m fpyolo.runstate resume <run_dir> --device <idx>
    python -m fpyolo.runstate verify <run_dir> --config <train yaml> [--batch N]

How "done" is decided (ultralytics 8.4.210, ``engine/trainer.py`` and ``utils/torch_utils.py``)
-----------------------------------------------------------------------------------------------
* ``save_model`` (trainer.py lines 782-848) writes ``weights/last.pt`` after every epoch with the
  optimizer state and ``epoch >= 0`` (line 846 ``last.pt``, lines 847-848 ``best.pt``).
* When the loop ends, for ANY reason (all epochs done, early stopping, ``time``), line 709 calls
  ``final_eval`` (lines 1021-1030), whose first act is ``strip_optimizer(self.last)``.
* ``strip_optimizer`` (torch_utils.py lines 895-953) sets ``optimizer``/``ema``/``scaler``... to
  ``None`` and ``epoch`` to ``-1`` (lines 949-951) and rewrites the file.

So a finished run is exactly: ``weights/best.pt`` exists and ``weights/last.pt`` is a stripped
checkpoint (``epoch == -1`` and no optimizer). This covers early stopping without having to
re-implement ``EarlyStopping`` from ``results.csv`` (whose rule depends on the fitness
definition of each task). ``results.csv`` is deliberately not used: its last row is written
before the strip, so it cannot tell "killed inside ``final_eval``" from "finished".

A run killed between its last ``save_model`` and the strip has ``last.pt`` with
``epoch == epochs - 1`` and an optimizer. It is ``partial`` here, but Ultralytics cannot resume it:
``resume_training`` asserts ``0 < start_epoch < epochs`` (trainer.py line 1149). ``resume``
refuses it with a message; recovery is to strip the checkpoints by hand (see ``_FINISHED_MSG``).

States
------
``absent``   the run directory does not exist
``running``  the run directory holds a locked ``.fpyolo.lock``: a process is training it right now
``done``     ``best.pt`` exists and ``last.pt`` is stripped (see above); cannot be resumed
``partial``  ``last.pt`` is a loadable checkpoint with optimizer state: resume with ``resume``
``broken``   the directory exists but there is nothing to resume from (no ``last.pt``, an
             unreadable one, or a stripped ``last.pt`` without ``best.pt``). ``fpyolo.train``
             refuses an existing directory, so a human must move or delete it.

Locking: ``fpyolo.train`` and ``resume`` hold ``flock`` on ``<run>/.fpyolo.lock`` for their whole
life (``fpyolo.train.acquire_run_lock``); the kernel releases it when the process dies. ``state``
only probes an existing lock file, so it never creates files in a run directory.

Resume itself is ``YOLO(last.pt).train(resume=True, device=<idx>)`` (``check_resume``, trainer.py
lines 1040-1087): every other argument is restored from the checkpoint.

Batch check (``verify``)
------------------------
Ultralytics writes ``args.yaml`` once, in ``BaseTrainer.__init__`` (trainer.py line 169). The
first-epoch OOM handler halves only ``self.args.batch`` (line 589), so ``args.yaml`` keeps the
requested batch. The effective batch is in the checkpoints: ``save_model`` stores
``train_args = vars(self.args)`` (line 830) and ``strip_optimizer`` keeps it (torch_utils.py line 952).
``verify`` therefore reads ``train_args["batch"]`` from ``weights/last.pt`` (``best.pt`` as a fallback)
and also fails if the run's log contains Ultralytics' "Reducing to batch" warning.
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import yaml

import fpyolo  # noqa: F401  (forces CUDA_DEVICE_ORDER=PCI_BUS_ID before torch/CUDA is used)
from fpyolo.train import LOCK_NAME, RunLocked, acquire_run_lock, apply_device_env, env_record, release_run_lock

STATES = ("absent", "running", "partial", "done", "broken")
BATCH_WARNING = "Reducing to batch"  # Ultralytics' first-epoch OOM message (trainer.py lines 593-596)

_FINISHED_MSG = (
    "training of {run_dir} reached its last epoch, but the final strip/validation did not run, so Ultralytics "
    "cannot resume it (resume_training needs start_epoch < epochs). Do NOT delete it. Finish it by stripping "
    "the checkpoints, after which its state is 'done':\n"
    "  python -c \"from ultralytics.utils.torch_utils import strip_optimizer as s; "
    "[s('{run_dir}/weights/%s.pt' % n) for n in ('last', 'best')]\""
)


def _load_ckpt(path: Path) -> dict | None:
    """Load a checkpoint on the CPU; None if the file is missing or not a checkpoint dict."""
    import torch

    if not path.is_file():
        return None
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
    except Exception:  # truncated write, wrong format, ...
        return None
    return ckpt if isinstance(ckpt, dict) and "epoch" in ckpt else None


def _is_locked(run_dir: Path) -> bool:
    """True if another process holds ``<run_dir>/.fpyolo.lock``. Only probes an existing file."""
    import fcntl

    lock = run_dir / LOCK_NAME
    if not lock.is_file():
        return False
    with open(lock, "a") as f:
        try:
            fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False  # closing the file released the probe lock


def _ckpt_state(run_dir: Path) -> str:
    """``absent`` / ``partial`` / ``done`` / ``broken`` from the files alone (ignores locks)."""
    if not run_dir.exists():
        return "absent"
    last = _load_ckpt(run_dir / "weights" / "last.pt")
    if last is None:
        return "broken"
    if last.get("epoch") == -1 and last.get("optimizer") is None:
        return "done" if (run_dir / "weights" / "best.pt").is_file() else "broken"
    return "partial"


def run_state(run_dir: str | Path) -> str:
    """One of ``absent``, ``running``, ``partial``, ``done``, ``broken`` (see the module docstring)."""
    run_dir = Path(run_dir)
    if run_dir.exists() and _is_locked(run_dir):
        return "running"
    return _ckpt_state(run_dir)


def next_env_resume_path(run_dir: str | Path) -> Path:
    """``env_resume_<n>.json`` with n one above the highest existing number (starting at 1)."""
    run_dir = Path(run_dir)
    nums = [int(m.group(1)) for p in run_dir.glob("env_resume_*.json")
            if (m := re.fullmatch(r"env_resume_(\d+)\.json", p.name))]
    return run_dir / f"env_resume_{max(nums, default=0) + 1}.json"


def resume_run(run_dir: str | Path, device: str, yolo_cls=None) -> None:
    """Continue a ``partial`` run on ``device`` (a single GPU index, nvidia-smi numbering)."""
    run_dir = Path(run_dir)
    lock = None
    if run_dir.is_dir():
        try:
            lock = acquire_run_lock(run_dir)  # first thing: nobody else may be training this run
        except RunLocked as e:
            raise SystemExit(f"cannot resume {run_dir}: {e}") from None
    try:
        _resume_locked(run_dir, device, yolo_cls)
    finally:
        if lock is not None:
            release_run_lock(lock)


def _resume_locked(run_dir: Path, device: str, yolo_cls) -> None:
    state = _ckpt_state(run_dir)
    if state != "partial":
        raise SystemExit(f"cannot resume {run_dir}: state is {state!r}, only 'partial' runs can be resumed")
    last = run_dir / "weights" / "last.pt"
    ckpt = _load_ckpt(last) or {}
    train_args = ckpt.get("train_args") or {}
    epochs = train_args.get("epochs")
    if isinstance(epochs, int) and ckpt.get("epoch", -1) >= epochs - 1:
        raise SystemExit("cannot resume: " + _FINISHED_MSG.format(run_dir=run_dir))
    data = train_args.get("data")
    apply_device_env(device)
    record = env_record(device, Path(data) if data else None)
    next_env_resume_path(run_dir).write_text(json.dumps(record, indent=2) + "\n")
    if yolo_cls is None:
        from ultralytics import YOLO as yolo_cls
    yolo_cls(str(last)).train(resume=True, device=device)


def _norm_bus(bus: str) -> str:
    return str(bus).strip().upper()


def pci_problems(env: dict, smi_csv: str) -> list[str]:
    """Compare an env record's GPU with ``nvidia-smi --query-gpu=index,pci.bus_id --format=csv`` output."""
    gpu = env.get("gpu")
    if env.get("device") == "cpu" or not gpu:
        return [f"ran on device {env.get('device')!r}, not on a GPU"]
    table = {}
    for line in smi_csv.strip().splitlines()[1:]:
        idx, _, bus = (part.strip() for part in line.partition(","))
        table[idx] = _norm_bus(bus)
    idx = str(gpu.get("index"))
    if idx not in table:
        return [f"GPU index {idx} not in the nvidia-smi table"]
    if _norm_bus(gpu.get("pci_bus_id", "")) != table[idx]:
        return [f"GPU index {idx}: recorded pci_bus_id {gpu.get('pci_bus_id')} != nvidia-smi {table[idx]}"]
    return []


def _effective_batch(run_dir: Path):
    """``train_args["batch"]`` of ``weights/last.pt`` (``best.pt`` as a fallback), None if unreadable."""
    for name in ("last.pt", "best.pt"):
        ckpt = _load_ckpt(run_dir / "weights" / name)
        args = (ckpt or {}).get("train_args")
        if isinstance(args, dict) and "batch" in args:
            return args["batch"]
    return None


def verify_run(run_dir: str | Path, expected_batch: int, smi_csv: str | None,
               log_path: str | Path | None = None) -> list[str]:
    """Problems found in a run directory (empty list = fine).

    * ``args.yaml``: optimizer SGD.
    * Batch: the effective batch stored in the checkpoints (``train_args``) equals the requested one,
      and ``log_path`` (if given) has no "Reducing to batch" line. ``args.yaml`` cannot show a
      first-epoch OOM halving, see the module docstring.
    * ``env.json`` / ``env_resume_*.json``: a GPU device; with ``smi_csv`` also that its PCI bus id
      matches the nvidia-smi row of that index.
    """
    run_dir = Path(run_dir)
    problems: list[str] = []
    args_path = run_dir / "args.yaml"
    if not args_path.is_file():
        problems.append("args.yaml missing")
    else:
        args = yaml.safe_load(args_path.read_text()) or {}
        if str(args.get("optimizer")) != "SGD":
            problems.append(f"optimizer is {args.get('optimizer')!r}, expected 'SGD'")
    batch = _effective_batch(run_dir)
    if batch is None:
        problems.append("no readable checkpoint (weights/last.pt, best.pt) to read the effective batch from")
    elif batch != expected_batch:
        problems.append(f"effective batch is {batch!r} (checkpoint train_args), requested {expected_batch}: "
                        "Ultralytics halved it after a first-epoch OOM")
    if log_path is not None and Path(log_path).is_file():
        with open(log_path, errors="replace") as f:
            if any(BATCH_WARNING in line for line in f):
                problems.append(f"{Path(log_path).name} contains '{BATCH_WARNING}': the batch was reduced after an OOM")
    records = [run_dir / "env.json", *sorted(run_dir.glob("env_resume_*.json"))]
    if not records[0].is_file():
        problems.append("env.json missing")
    for rec in records:
        if not rec.is_file():
            continue
        env = json.loads(rec.read_text())
        if smi_csv is not None:
            problems += [f"{rec.name}: {p}" for p in pci_problems(env, smi_csv)]
        elif env.get("device") == "cpu":
            problems.append(f"{rec.name}: ran on the cpu")
    return problems


def _query_nvidia_smi() -> str | None:
    if shutil.which("nvidia-smi") is None:
        return None
    out = subprocess.run(["nvidia-smi", "--query-gpu=index,pci.bus_id", "--format=csv"],
                         capture_output=True, text=True)
    return out.stdout if out.returncode == 0 else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m fpyolo.runstate", description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("state", help="print absent | running | partial | done | broken")
    p.add_argument("run_dir", type=Path)
    p = sub.add_parser("resume", help="continue a partial run from weights/last.pt")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--device", required=True, help="one GPU index (nvidia-smi numbering) or 'cpu'; no default")
    p = sub.add_parser("verify", help="check optimizer, batch and GPU record of a run (exit 1 on problems)")
    p.add_argument("run_dir", type=Path)
    p.add_argument("--config", type=Path, required=True, help="the train yaml the run was launched with")
    p.add_argument("--batch", type=int, help="requested batch if it was overridden with --batch")
    p.add_argument("--log", type=Path, help="training log to scan for Ultralytics' 'Reducing to batch' warning")
    p.add_argument("--no-nvidia-smi", action="store_true", help="skip the PCI bus id cross-check")
    args = ap.parse_args(argv)

    if args.cmd == "state":
        print(run_state(args.run_dir))
        return 0
    if args.cmd == "resume":
        resume_run(args.run_dir, args.device)
        return 0
    expected = args.batch if args.batch is not None else yaml.safe_load(args.config.read_text())["batch"]
    smi = None if args.no_nvidia_smi else _query_nvidia_smi()
    if smi is None and not args.no_nvidia_smi:
        print("note: nvidia-smi not available, PCI bus id cross-check skipped")
    problems = verify_run(args.run_dir, expected, smi, args.log)
    for line in problems:
        print(f"PROBLEM {args.run_dir.name}: {line}")
    if not problems:
        print(f"ok {args.run_dir.name}")
    return 1 if problems else 0


if __name__ == "__main__":
    sys.exit(main())
