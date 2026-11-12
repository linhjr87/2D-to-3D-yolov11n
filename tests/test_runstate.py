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
from fpyolo.train import LOCK_NAME, RunLocked, acquire_run_lock


def _ckpt(path, *, finished, data="/x/D/data.yaml", batch=32, epoch=None, epochs=None):
    """Tiny stand-in for what Ultralytics writes: save_model (epoch >= 0 + optimizer) or strip_optimizer."""
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": -1 if finished else (7 if epoch is None else epoch),
            "optimizer": None if finished else {"state": {}, "param_groups": []},
            "train_args": {"data": data, "batch": batch, "optimizer": "SGD",
                           **({"epochs": epochs} if epochs is not None else {})},
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


HOLDER = """
import fcntl, sys, time
f = open(sys.argv[1], "a")
fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
print("locked", flush=True)
time.sleep(60)
"""


@pytest.fixture
def lock_holder():
    """Start a subprocess that holds the flock on a given lock file; killed at teardown."""
    procs = []

    def start(lock_path):
        proc = subprocess.Popen([sys.executable, "-c", HOLDER, str(lock_path)], stdout=subprocess.PIPE, text=True)
        procs.append(proc)
        assert proc.stdout.readline().strip() == "locked"
        return proc

    yield start
    for proc in procs:
        proc.kill()
        proc.wait()


def test_state_running_while_locked_then_partial(tmp_path, lock_holder):
    run = _run(tmp_path, "r", last="running", best=True)
    assert run_state(run) == "partial"
    holder = lock_holder(run / LOCK_NAME)
    assert run_state(run) == "running"
    out = subprocess.run([sys.executable, "-m", "fpyolo.runstate", "state", str(run)],
                         capture_output=True, text=True, check=True)
    assert out.stdout.strip().splitlines()[-1] == "running"
    holder.kill()
    holder.wait()
    assert run_state(run) == "partial"  # the kernel drops the lock with the process


def test_state_does_not_create_a_lock_file(tmp_path):
    run = _run(tmp_path, "r", last="finished", best=True)
    run_state(run)
    assert not (run / LOCK_NAME).exists()


def test_resume_refuses_locked_run(tmp_path, lock_holder):
    run = _run(tmp_path, "r", last="running", best=True)
    lock_holder(run / LOCK_NAME)
    _FakeYOLO.calls = []
    with pytest.raises(SystemExit, match="locked"):
        resume_run(run, "cpu", yolo_cls=_FakeYOLO)
    assert _FakeYOLO.calls == [] and not list(run.glob("env_resume_*.json"))


def test_resume_cli_refuses_locked_run(tmp_path, lock_holder):
    run = _run(tmp_path, "r", last="running", best=True)
    lock_holder(run / LOCK_NAME)
    out = subprocess.run([sys.executable, "-m", "fpyolo.runstate", "resume", str(run), "--device", "cpu"],
                         capture_output=True, text=True)
    assert out.returncode != 0 and "locked" in out.stderr


def test_resume_holds_the_lock_while_training(tmp_path, monkeypatch):
    run = _run(tmp_path, "r", last="running", best=True)
    monkeypatch.setattr(runstate, "env_record", lambda d, y=None: {"device": d})

    class Probe:
        def __init__(self, weights):
            pass

        def train(self, **kw):
            with pytest.raises(RunLocked):  # a second locker is refused while train() runs
                acquire_run_lock(run)

    resume_run(run, "cpu", yolo_cls=Probe)


def test_resume_refuses_run_that_reached_its_last_epoch(tmp_path):
    run = tmp_path / "r"
    _ckpt(run / "weights" / "last.pt", finished=False, epoch=299, epochs=300)
    _ckpt(run / "weights" / "best.pt", finished=True)
    assert run_state(run) == "partial"
    _FakeYOLO.calls = []
    with pytest.raises(SystemExit) as e:
        resume_run(run, "cpu", yolo_cls=_FakeYOLO)
    msg = str(e.value)
    assert "Do NOT delete" in msg and "strip_optimizer" in msg and str(run) in msg
    assert _FakeYOLO.calls == []


def test_resume_allows_run_before_its_last_epoch(tmp_path, monkeypatch):
    run = tmp_path / "r"
    _ckpt(run / "weights" / "last.pt", finished=False, epoch=298, epochs=300)
    _ckpt(run / "weights" / "best.pt", finished=True)
    monkeypatch.setattr(runstate, "env_record", lambda d, y=None: {"device": d})
    _FakeYOLO.calls = []
    resume_run(run, "cpu", yolo_cls=_FakeYOLO)
    assert len(_FakeYOLO.calls) == 1


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


# ---- verify: only the last attempt in the log counts ----

OOM_LINE = "WARNING CUDA out of memory with batch=32. Reducing to batch=16 and retrying (1/3).\n"


def _attempt(*lines):
    return f"{runstate.ATTEMPT_MARKER} 2026-11-12T00:00:00Z GPU 0: start x\n" + "".join(lines)


def test_old_attempt_with_oom_warning_then_clean_new_attempt_passes(tmp_path):
    run = _good_run(tmp_path)
    log = tmp_path / "r.log"
    log.write_text(_attempt("epoch 1\n", OOM_LINE) + _attempt("epoch 1\n", "epoch 2\n"))
    assert verify_run(run, expected_batch=32, smi_csv=SMI, log_path=log) == []


def test_warning_in_the_last_attempt_fails(tmp_path):
    run = _good_run(tmp_path)
    log = tmp_path / "r.log"
    log.write_text(_attempt("epoch 1\n") + _attempt(OOM_LINE))
    assert len(verify_run(run, expected_batch=32, smi_csv=SMI, log_path=log)) == 1


def test_verify_output_appended_to_the_log_does_not_retrigger(tmp_path):
    run = _good_run(tmp_path)
    log = tmp_path / "r.log"
    log.write_text(_attempt(OOM_LINE))
    first = verify_run(run, expected_batch=32, smi_csv=SMI, log_path=log)
    assert first
    with open(log, "a") as f:  # what the sweep's `tee -a` does
        f.writelines(f"PROBLEM r: {p}\n" for p in first)
    log.write_text(log.read_text() + _attempt("clean\n"))  # then a clean rerun starts a new attempt
    with open(log, "a") as f:
        f.write("ok r\n")
    assert verify_run(run, expected_batch=32, smi_csv=SMI, log_path=log) == []
    # and without a new attempt the appended PROBLEM lines alone do not add a second problem
    log.write_text(_attempt("clean\n") + "PROBLEM r: " + first[0] + "\n")
    assert verify_run(run, expected_batch=32, smi_csv=SMI, log_path=log) == []


# ---- verify ----

def _good_run(tmp_path, *, batch=32, optimizer="SGD", device="1", bus="00000000:3B:00.0", args_batch=32):
    """A run dir like Ultralytics leaves it: args.yaml is written once with the REQUESTED batch,
    the checkpoints carry the effective batch in train_args (they differ after an OOM halving)."""
    run = tmp_path / "r"
    run.mkdir()
    (run / "args.yaml").write_text(yaml.safe_dump({"optimizer": optimizer, "batch": args_batch}))
    (run / "env.json").write_text(json.dumps(
        {"device": device, "gpu": {"index": int(device), "pci_bus_id": bus}}))
    _ckpt(run / "weights" / "last.pt", finished=True, batch=batch)
    _ckpt(run / "weights" / "best.pt", finished=True, batch=batch)
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


def test_verify_batch_halved_in_checkpoint_although_args_yaml_says_32(tmp_path):
    run = _good_run(tmp_path, batch=16, args_batch=32)
    problems = verify_run(run, expected_batch=32, smi_csv=SMI)
    assert len(problems) == 1 and "effective batch" in problems[0] and "16" in problems[0]


def test_verify_falls_back_to_best_pt(tmp_path):
    run = _good_run(tmp_path, batch=16)
    (run / "weights" / "last.pt").unlink()
    assert any("16" in p for p in verify_run(run, expected_batch=32, smi_csv=SMI))
    (run / "weights" / "best.pt").unlink()
    assert any("no readable checkpoint" in p for p in verify_run(run, expected_batch=32, smi_csv=SMI))


def test_verify_log_with_reducing_to_batch(tmp_path):
    run = _good_run(tmp_path)
    log = tmp_path / "r.log"
    log.write_text("epoch 1\nWARNING CUDA out of memory with batch=32. Reducing to batch=16 and retrying (1/3).\n")
    problems = verify_run(run, expected_batch=32, smi_csv=SMI, log_path=log)
    assert len(problems) == 1 and "lowered the batch" in problems[0]
    assert "Reducing to batch" not in problems[0]  # verify's own output must not re-trigger the grep
    log.write_text("epoch 1\n")
    assert verify_run(run, expected_batch=32, smi_csv=SMI, log_path=log) == []
    assert verify_run(run, expected_batch=32, smi_csv=SMI, log_path=tmp_path / "missing.log") == []


def test_verify_optimizer_not_sgd(tmp_path):
    problems = verify_run(_good_run(tmp_path, optimizer="AdamW"), expected_batch=32, smi_csv=SMI)
    assert any("optimizer" in p for p in problems)


def test_verify_missing_files_and_pci_mismatch(tmp_path):
    run = tmp_path / "empty"
    run.mkdir()
    problems = verify_run(run, expected_batch=32, smi_csv=None)
    assert any("args.yaml" in p for p in problems) and any("env.json" in p for p in problems)
    assert any("no readable checkpoint" in p for p in problems)
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
    log = tmp_path / "r.log"
    log.write_text("Reducing to batch=16\n")
    logged = subprocess.run([sys.executable, "-m", "fpyolo.runstate", "verify", str(run), "--config", str(cfg),
                             "--log", str(log), "--no-nvidia-smi"], capture_output=True, text=True)
    assert logged.returncode == 1 and "lowered the batch" in logged.stdout


def test_attempt_marker_glued_to_an_unterminated_line(tmp_path):
    """A run killed mid progress bar leaves no trailing newline; the next header must still count."""
    from fpyolo.runstate import ATTEMPT_MARKER, _last_attempt_has_batch_warning

    log = tmp_path / "x.log"
    log.write_text("Reducing to batch 16\n 4/30 progress" + ATTEMPT_MARKER + " resume\nclean epoch\n")
    assert not _last_attempt_has_batch_warning(log)
