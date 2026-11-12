"""Run-state detection, resume entry point and run verification (fake run dirs, no training)."""
import json
import subprocess
import sys

import pytest
import torch
import yaml

from conftest import CONFIGS
from fpyolo import runstate
from fpyolo.runstate import (
    next_env_resume_path, pci_problems, resume_run, run_state, verify_run,
)


def _ckpt(path, *, finished, data="/x/D/data.yaml"):
    """Tiny stand-in for what Ultralytics writes: save_model (epoch >= 0 + optimizer) or strip_optimizer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": -1 if finished else 7,
            "optimizer": None if finished else {"state": {}, "param_groups": []},
            "train_args": {"data": data, "batch": 32, "optimizer": "SGD"},
        },
        path,
    )


def _run(tmp_path, name, *, last=None, best=False):
    run = tmp_path / name
    run.mkdir()
    if last is not None:
        _ckpt(run / "weights" / "last.pt", finished=last == "finished")
    if best:
        _ckpt(run / "weights" / "best.pt", finished=True)
    return run


def test_absent(tmp_path):
    assert run_state(tmp_path / "nope") == "absent"


def test_done_when_last_is_stripped_and_best_exists(tmp_path):
    assert run_state(_run(tmp_path, "r", last="finished", best=True)) == "done"


def test_partial_when_last_still_has_optimizer(tmp_path):
    assert run_state(_run(tmp_path, "r", last="running", best=True)) == "partial"


def test_partial_even_if_results_csv_looks_complete(tmp_path):
    # killed during final_eval: last.pt is not stripped yet, so the run is resumable, not done
    run = _run(tmp_path, "r", last="running", best=True)
    (run / "results.csv").write_text("epoch\n" + "\n".join(str(i) for i in range(1, 301)) + "\n")
    assert run_state(run) == "partial"


@pytest.mark.parametrize("last,best", [
    (None, False),          # directory exists but training never saved a checkpoint (e.g. first-epoch crash)
    ("finished", False),    # stripped last.pt but no best.pt
])
def test_broken_when_nothing_to_resume(tmp_path, last, best):
    assert run_state(_run(tmp_path, "r", last=last, best=best)) == "broken"


def test_broken_when_last_is_corrupt(tmp_path):
    run = _run(tmp_path, "r", best=True)
    (run / "weights").mkdir(exist_ok=True)
    (run / "weights" / "last.pt").write_bytes(b"truncated")
    assert run_state(run) == "broken"


def test_broken_when_last_is_not_a_checkpoint(tmp_path):
    run = _run(tmp_path, "r", best=True)
    torch.save([1, 2, 3], run / "weights" / "last.pt")
    assert run_state(run) == "broken"


def test_state_with_spaces_and_non_ascii(spaced_tmp):
    assert run_state(_run(spaced_tmp, "run é", last="running", best=True)) == "partial"


def test_state_cli(tmp_path):
    run = _run(tmp_path, "r", last="finished", best=True)
    out = subprocess.run([sys.executable, "-m", "fpyolo.runstate", "state", str(run)],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "done"


def test_next_env_resume_path(tmp_path):
    assert next_env_resume_path(tmp_path).name == "env_resume_1.json"
    (tmp_path / "env_resume_1.json").write_text("{}")
    (tmp_path / "env_resume_3.json").write_text("{}")
    assert next_env_resume_path(tmp_path).name == "env_resume_4.json"


class _FakeYOLO:
    calls: list = []

    def __init__(self, weights):
        self.weights = weights

    def train(self, **kw):
        _FakeYOLO.calls.append((self.weights, kw))


def test_resume_run_writes_env_and_calls_train(tmp_path, monkeypatch):
    run = _run(tmp_path, "r", last="running", best=True)
    _FakeYOLO.calls = []
    seen = {}
    monkeypatch.setattr(runstate, "apply_device_env", lambda d: seen.setdefault("device", d))
    monkeypatch.setattr(runstate, "env_record", lambda d, y=None: {"device": d, "data_yaml": str(y)})
    resume_run(run, "cpu", yolo_cls=_FakeYOLO)
    assert seen == {"device": "cpu"}
    assert _FakeYOLO.calls == [(str(run / "weights" / "last.pt"), {"resume": True, "device": "cpu"})]
    rec = json.loads((run / "env_resume_1.json").read_text())
    assert rec == {"device": "cpu", "data_yaml": "/x/D/data.yaml"}  # data path comes from the checkpoint
    resume_run(run, "cpu", yolo_cls=_FakeYOLO)
    assert (run / "env_resume_2.json").exists()


@pytest.mark.parametrize("state", ["done", "broken", "absent"])
def test_resume_refuses_non_partial(tmp_path, state):
    run = {
        "done": lambda: _run(tmp_path, "r", last="finished", best=True),
        "broken": lambda: _run(tmp_path, "r"),
        "absent": lambda: tmp_path / "r",
    }[state]()
    _FakeYOLO.calls = []
    with pytest.raises(SystemExit, match=state):
        resume_run(run, "cpu", yolo_cls=_FakeYOLO)
    assert _FakeYOLO.calls == []
    assert not list(tmp_path.glob("r/env_resume_*.json"))


def test_resume_cli_requires_device(tmp_path):
    out = subprocess.run([sys.executable, "-m", "fpyolo.runstate", "resume", str(tmp_path)],
                         capture_output=True, text=True)
    assert out.returncode == 2 and "--device" in out.stderr


# ---- verify ----

def _good_run(tmp_path, *, batch=32, optimizer="SGD", device="1", bus="00000000:3B:00.0"):
    run = tmp_path / "r"
    run.mkdir()
    (run / "args.yaml").write_text(yaml.safe_dump({"optimizer": optimizer, "batch": batch}))
    (run / "env.json").write_text(json.dumps(
        {"device": device, "gpu": {"index": int(device), "pci_bus_id": bus}}))
    return run


SMI = "index, pci.bus_id\n0, 00000000:1A:00.0\n1, 00000000:3B:00.0\n"


def test_pci_problems():
    env = {"device": "1", "gpu": {"index": 1, "pci_bus_id": "00000000:3b:00.0"}}
    assert pci_problems(env, SMI) == []  # case-insensitive
    bad = {"device": "0", "gpu": {"index": 0, "pci_bus_id": "00000000:3B:00.0"}}
    assert "pci_bus_id" in pci_problems(bad, SMI)[0]
    assert "index 9" in pci_problems({"device": "9", "gpu": {"index": 9, "pci_bus_id": "x"}}, SMI)[0]
    assert pci_problems({"device": "cpu", "gpu": None}, SMI) != []  # a real run must not be on the cpu


def test_verify_ok(tmp_path):
    assert verify_run(_good_run(tmp_path), expected_batch=32, smi_csv=SMI) == []


def test_verify_batch_halved(tmp_path):
    problems = verify_run(_good_run(tmp_path, batch=16), expected_batch=32, smi_csv=SMI)
    assert len(problems) == 1 and "batch" in problems[0] and "16" in problems[0]


def test_verify_optimizer_not_sgd(tmp_path):
    problems = verify_run(_good_run(tmp_path, optimizer="AdamW"), expected_batch=32, smi_csv=SMI)
    assert any("optimizer" in p for p in problems)


def test_verify_missing_files_and_pci_mismatch(tmp_path):
    run = tmp_path / "empty"
    run.mkdir()
    problems = verify_run(run, expected_batch=32, smi_csv=None)
    assert any("args.yaml" in p for p in problems) and any("env.json" in p for p in problems)
    other = tmp_path / "other"
    other.mkdir()
    wrong = _good_run(other, bus="00000000:99:00.0")
    assert any("pci_bus_id" in p for p in verify_run(wrong, expected_batch=32, smi_csv=SMI))


def test_verify_checks_resume_records_too(tmp_path):
    run = _good_run(tmp_path)
    (run / "env_resume_1.json").write_text(json.dumps(
        {"device": "0", "gpu": {"index": 0, "pci_bus_id": "00000000:3B:00.0"}}))
    problems = verify_run(run, expected_batch=32, smi_csv=SMI)
    assert any("env_resume_1.json" in p for p in problems)


def test_verify_cli_expected_batch_from_config(tmp_path):
    run = _good_run(tmp_path, batch=32)
    cfg = CONFIGS / "train_d_yolo11n.yaml"
    ok = subprocess.run([sys.executable, "-m", "fpyolo.runstate", "verify", str(run), "--config", str(cfg),
                         "--no-nvidia-smi"], capture_output=True, text=True)
    assert ok.returncode == 0, ok.stdout + ok.stderr
    bad = subprocess.run([sys.executable, "-m", "fpyolo.runstate", "verify", str(run), "--config", str(cfg),
                          "--batch", "16", "--no-nvidia-smi"], capture_output=True, text=True)
    assert bad.returncode == 1 and "batch" in bad.stdout
