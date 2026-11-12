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
# stands in for `python`: only the `-m fpyolo.train|runstate` calls the script makes.
# A run dir holding a marker file ".running" / ".partial" reports that state, any other dir is done.
trace="$STUB_TRACE"
case "$2" in
fpyolo.runstate)
    case "$3" in
    state)
        echo "import noise on stdout"
        if [ -f "$4/.running" ]; then echo running
        elif [ -f "$4/.partial" ]; then echo partial
        elif [ -d "$4" ]; then echo done
        else echo absent; fi ;;
    resume)
        echo "resume $(basename "$4") $6 x $(date +%s%N)" >> "$trace"
        rm -f "$4/.partial" ;;
    verify)
        echo "verify $(basename "$4") - $* $(date +%s%N)" >> "$trace"
        case " $* " in *" --log "*) ;; *) echo "verify without --log" >&2; exit 9 ;; esac
        if [ "$(basename "$4")" = "${STUB_VERIFY_FAIL:-}" ]; then echo "PROBLEM $(basename "$4"): x"; exit 1; fi
        echo "ok $(basename "$4")" ;;
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
    """(kind, name, device, batch, nanoseconds) rows of the stub trace; kind is start/end/resume/verify."""
    rows = [line.split() for line in (tmp_path / "trace").read_text().splitlines()]
    return [(r[0], r[1], r[2], r[3], int(r[-1])) for r in rows]


def test_gpus_required(tmp_path):
    env = _env(tmp_path)
    out = _run(env)
    assert out.returncode == 1 and "GPUS is required" in out.stderr
    assert not (tmp_path / "trace").exists()


@pytest.mark.parametrize("gpus", ["0,1", "a", "1 1", "   "])
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
    ev = [e for e in ev if e[0] in ("start", "end")]
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
    # every finished run was verified once
    assert sorted(e[1] for e in _events(tmp_path) if e[0] == "verify") == sorted(e[1] for e in starts)
    # second invocation: everything is done, nothing starts, but every run is verified again
    (tmp_path / "trace").unlink()
    again = _run(_env(tmp_path, GPUS="3 5"))
    assert again.returncode == 0 and "skipping" in again.stdout
    kinds = [e[0] for e in _events(tmp_path)]
    assert kinds == ["verify"] * 9


def test_verify_failure_of_a_done_run_fails_the_sweep(tmp_path):
    (tmp_path / "runs" / "d11n_s0").mkdir(parents=True)  # done already
    out = _run(_env(tmp_path, GPUS="0", STUB_VERIFY_FAIL="d11n_s0"))
    assert out.returncode == 1 and "FAILED runs: d11n_s0" in out.stderr
    assert "PROBLEM d11n_s0" in (tmp_path / "runs" / "d11n_s0.log").read_text()  # also kept in the log


def test_partial_run_is_resumed_then_verified(tmp_path):
    run = tmp_path / "runs" / "d8n_s0"
    run.mkdir(parents=True)
    (run / ".partial").write_text("")
    out = _run(_env(tmp_path, GPUS="2"))
    assert out.returncode == 0, out.stderr
    mine = [e for e in _events(tmp_path) if e[1] == "d8n_s0"]
    assert [e[0] for e in mine] == ["resume", "verify"]  # resumed (no fresh start), then checked
    assert mine[0][2] == "2" and mine[0][4] < mine[1][4]
    assert sum(1 for e in _events(tmp_path) if e[0] == "start") == 8


def test_running_run_is_skipped_with_a_warning(tmp_path):
    run = tmp_path / "runs" / "w11n_s1"
    run.mkdir(parents=True)
    (run / ".running").write_text("")
    out = _run(_env(tmp_path, GPUS="0"))
    assert out.returncode == 3  # nothing failed, but a run is missing: `sweep && evaluate` must stop
    assert "WARNING" in out.stderr and "w11n_s1" in out.stderr
    names = [e[1] for e in _events(tmp_path)]
    assert "w11n_s1" not in names and len([e for e in _events(tmp_path) if e[0] == "start"]) == 8


def test_failed_and_running_lists_are_both_printed_and_failure_wins(tmp_path):
    run = tmp_path / "runs" / "w11n_s1"
    run.mkdir(parents=True)
    (run / ".running").write_text("")
    out = _run(_env(tmp_path, GPUS="0", STUB_FAIL="d8n_s1"))
    assert out.returncode == 1
    assert "FAILED runs: d8n_s1" in out.stderr and "another process is training: w11n_s1" in out.stderr


def test_start_rotates_an_existing_log_and_writes_an_attempt_header(tmp_path):
    from fpyolo.runstate import ATTEMPT_MARKER

    runs = tmp_path / "runs"
    runs.mkdir()
    (runs / "d11n_s0.log").write_text("old attempt: Reducing to batch=16\n")  # run dir was discarded, log remains
    out = _run(_env(tmp_path, GPUS="0", STUB_SLEEP="0"))
    assert out.returncode == 0, out.stderr
    (old,) = runs.glob("d11n_s0.log.*.old")
    assert old.read_text() == "old attempt: Reducing to batch=16\n"
    log = (runs / "d11n_s0.log").read_text()
    assert log.startswith("\n" + ATTEMPT_MARKER) and "Reducing to batch" not in log
    assert log.count(ATTEMPT_MARKER) == 1


def test_resume_appends_to_the_log_without_rotating(tmp_path):
    runs = tmp_path / "runs"
    (runs / "d8n_s0").mkdir(parents=True)
    (runs / "d8n_s0" / ".partial").write_text("")
    (runs / "d8n_s0.log").write_text("first attempt\n")
    assert _run(_env(tmp_path, GPUS="0", STUB_SLEEP="0")).returncode == 0
    assert not list(runs.glob("d8n_s0.log.*.old"))
    assert (runs / "d8n_s0.log").read_text().startswith("first attempt\n\n=== fpyolo-attempt")


def test_script_and_runstate_share_the_attempt_marker():
    from fpyolo.runstate import ATTEMPT_MARKER

    assert f'echo "{ATTEMPT_MARKER} ' in SCRIPT.read_text()


HOLD_SWEEP_LOCK = """
import fcntl, sys, time
f = open(sys.argv[1], "a")
fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
print("locked", flush=True)
time.sleep(60)
"""


def test_second_sweep_on_the_same_runs_dir_refuses(tmp_path):
    env = _env(tmp_path, GPUS="0")
    (tmp_path / "runs").mkdir()
    holder = subprocess.Popen([sys.executable, "-c", HOLD_SWEEP_LOCK, str(tmp_path / "runs" / ".sweep.lock")],
                              stdout=subprocess.PIPE, text=True)
    try:
        assert holder.stdout.readline().strip() == "locked"
        out = _run(env)
    finally:
        holder.kill()
        holder.wait()
    assert out.returncode == 1 and "another sweep is running" in out.stderr
    assert not (tmp_path / "trace").exists()
    # once the holder is gone the lock is free again
    assert _run(_env(tmp_path, GPUS="0", STUB_SLEEP="0")).returncode == 0


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
