"""scripts/run_all_seeds.sh with fake run dirs and a stub interpreter (no GPU, no training)."""
import os
import signal
import subprocess
import sys
import time

import pytest
import torch

from conftest import REPO

SCRIPT = REPO / "scripts" / "run_all_seeds.sh"

STUB = """#!/usr/bin/env bash
# stands in for `python`: only the `-m fpyolo.train|runstate` calls the script makes
trace="$STUB_TRACE"
case "$2" in
fpyolo.runstate)
    case "$3" in
    state) [ -d "$4" ] && echo done || echo absent ;;
    verify) echo "ok $(basename "$4")" ;;
    *) echo "unexpected runstate call: $*" >&2; exit 9 ;;
    esac ;;
fpyolo.train)
    shift 2
    while [ $# -gt 0 ]; do
        case "$1" in --name) name=$2 ;; --device) dev=$2 ;; --project) proj=$2 ;; --batch) batch=$2 ;; esac
        shift
    done
    echo "start $name $dev ${batch:-config} $(date +%s%N)" >> "$trace"
    echo $$ > "$proj/$name.pid"
    sleep "${STUB_SLEEP:-0.3}"
    [ "$name" = "${STUB_FAIL:-}" ] && exit 3
    mkdir -p "$proj/$name"
    echo "end $name $dev x $(date +%s%N)" >> "$trace" ;;
*) echo "unexpected call: $*" >&2; exit 9 ;;
esac
"""


def _env(tmp_path, **extra):
    stub = tmp_path / "fakepython"
    stub.write_text(STUB)
    stub.chmod(0o755)
    env = dict(os.environ, PYTHON=str(stub), RUNS_DIR=str(tmp_path / "runs"),
               DATA_ROOT=str(tmp_path / "data"), STUB_TRACE=str(tmp_path / "trace"))
    for kind in "DW":
        (tmp_path / "data" / kind).mkdir(parents=True, exist_ok=True)
        (tmp_path / "data" / kind / "data.yaml").write_text("path: x\n")
    env.update(extra)
    env.pop("GPUS", None) if "GPUS" not in extra else None
    return env


def _run(env, **kw):
    return subprocess.run(["bash", str(SCRIPT)], env=env, capture_output=True, text=True, **kw)


def _events(tmp_path):
    rows = [line.split() for line in (tmp_path / "trace").read_text().splitlines()]
    return [(kind, name, dev, batch, int(t)) for kind, name, dev, batch, t in rows]


def test_gpus_required(tmp_path):
    env = _env(tmp_path)
    out = _run(env)
    assert out.returncode == 1 and "GPUS is required" in out.stderr
    assert not (tmp_path / "trace").exists()


@pytest.mark.parametrize("gpus", ["0,1", "a", "1 1"])
def test_bad_gpus_rejected(tmp_path, gpus):
    out = _run(_env(tmp_path, GPUS=gpus))
    assert out.returncode == 1 and "GPU" in out.stderr
    assert not (tmp_path / "trace").exists()


def test_bad_batch_rejected(tmp_path):
    out = _run(_env(tmp_path, GPUS="0", BATCH_D="0"))
    assert out.returncode == 1 and "BATCH_D" in out.stderr


def test_queue_runs_each_job_once_one_per_gpu(tmp_path):
    out = _run(_env(tmp_path, GPUS="3 5", BATCH_D="24"))
    assert out.returncode == 0, out.stderr
    ev = _events(tmp_path)
    starts = [e for e in ev if e[0] == "start"]
    assert sorted(e[1] for e in starts) == sorted(
        f"{k}_s{s}" for k in ("d11n", "d8n", "w11n") for s in (0, 1, 2))
    assert {e[2] for e in starts} == {"3", "5"}
    # batch override applies to D only
    assert all((e[3] == "24") == e[1].startswith("d") for e in starts)
    # no two jobs overlap on one GPU, and never more than 2 run at once
    for dev in ("3", "5"):
        own = sorted((e for e in ev if e[2] == dev), key=lambda e: e[4])
        assert [e[0] for e in own] == ["start", "end"] * (len(own) // 2)
    live = peak = 0
    for e in sorted(ev, key=lambda e: e[4]):
        live += 1 if e[0] == "start" else -1
        peak = max(peak, live)
    assert peak == 2
    # second invocation: everything is done, nothing starts
    (tmp_path / "trace").unlink()
    again = _run(_env(tmp_path, GPUS="3 5"))
    assert again.returncode == 0 and "skipping" in again.stdout
    assert not (tmp_path / "trace").exists()


def test_failed_run_is_reported_but_queue_continues(tmp_path):
    out = _run(_env(tmp_path, GPUS="0", STUB_FAIL="d8n_s1"))
    assert out.returncode == 1
    assert "FAILED runs: d8n_s1" in out.stderr
    assert len([e for e in _events(tmp_path) if e[0] == "start"]) == 9


def test_sigterm_reaches_the_training_process(tmp_path):
    env = _env(tmp_path, GPUS="0", STUB_SLEEP="30")
    proc = subprocess.Popen(["bash", str(SCRIPT)], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    pidfile = tmp_path / "runs" / "d11n_s0.pid"
    for _ in range(100):
        if pidfile.exists() and pidfile.read_text().strip():
            break
        time.sleep(0.1)
    child = int(pidfile.read_text())
    proc.send_signal(signal.SIGTERM)
    proc.communicate(timeout=20)
    time.sleep(0.2)
    with pytest.raises(ProcessLookupError):
        os.kill(child, 0)


def _fake_run(runs, name, last, best=True):
    w = runs / name / "weights"
    w.mkdir(parents=True)
    fin = last == "finished"
    torch.save({"epoch": -1 if fin else 5, "optimizer": None if fin else {}, "train_args": {"data": "/x"}},
               w / "last.pt")
    if best:
        torch.save({"epoch": -1, "optimizer": None}, w / "best.pt")


def test_dry_run_reports_skip_resume_fail_start(tmp_path):
    runs = tmp_path / "runs with space"
    _fake_run(runs, "d11n_s0", "finished")
    _fake_run(runs, "d8n_s0", "running")
    (runs / "w11n_s0").mkdir()
    env = _env(tmp_path, GPUS="0 1", DRY_RUN="1", BATCH_D="24", RUNS_DIR=str(runs), PYTHON=sys.executable)
    out = _run(env)
    lines = out.stdout.splitlines()
    assert "[dry-run] GPU 0: d11n_s0 state=done -> skip" in lines
    assert "[dry-run] GPU 1: d8n_s0 state=partial -> resume" in lines
    assert "[dry-run] GPU 0: w11n_s0 state=broken -> FAIL" in lines
    assert "[dry-run] GPU 1: d11n_s1 state=absent -> start" in lines
    assert out.returncode == 1  # the broken run is a failure
    resume = lines[lines.index("[dry-run] GPU 1: d8n_s0 state=partial -> resume") + 1]
    assert "fpyolo.runstate resume" in resume and "--device 1" in resume
    assert "runs\\ with\\ space/d8n_s0" in resume  # paths with spaces are quoted
    start = lines[lines.index("[dry-run] GPU 1: d11n_s1 state=absent -> start") + 1]
    assert "--batch 24" in start and "--seed 1" in start and "--device 1" in start
    # a dry run touches nothing
    assert not (runs / "d11n_s1").exists() and not (tmp_path / "trace").exists()
    w_start = [l for l in lines if "w11n_s1" in l and "fpyolo.train" in l][0]
    assert "--batch" not in w_start and "/W/data.yaml" in w_start
