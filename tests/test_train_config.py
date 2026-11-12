import json
import os
import subprocess
import sys
import types

import pytest
import yaml

from conftest import CONFIGS, REPO
from fpyolo import train
from fpyolo.train import apply_device_env, env_record, main, train_kwargs


def _load(name):
    return yaml.safe_load((CONFIGS / name).read_text())


def test_d_configs_differ_only_in_model():
    a = _load("train_d_yolo11n.yaml")
    b = _load("train_d_yolov8n.yaml")
    assert a["model"] == "yolo11n.pt" and b["model"] == "yolov8n.pt"
    assert _load("train_w_yolo11n_seg.yaml")["model"] == "yolo11n-seg.pt"
    a.pop("model")
    b.pop("model")
    assert a == b


def test_train_kwargs_values():
    cfg = _load("train_d_yolo11n.yaml")
    kw = train_kwargs(cfg, seed=1, data="x/data.yaml", device="3")
    assert kw["imgsz"] == 1024 and kw["epochs"] == 300 and kw["patience"] == 50
    assert kw["optimizer"] == "SGD" and kw["lr0"] == 0.01
    assert kw["flipud"] == 0.5 and kw["fliplr"] == 0.5 and kw["degrees"] == 0
    assert kw["hsv_h"] == kw["hsv_s"] == kw["hsv_v"] == 0
    assert kw["deterministic"] is True and kw["workers"] == 8 and kw["batch"] == 32
    assert kw["seed"] == 1 and kw["device"] == "3" and kw["data"] == "x/data.yaml"
    assert "model" not in kw  # the model is a constructor argument, not a train argument


def test_auto_optimizer_rejected():
    cfg = dict(_load("train_d_yolo11n.yaml"), optimizer="auto")
    with pytest.raises(ValueError, match="auto"):
        train_kwargs(cfg, seed=0, data="d.yaml", device="cpu")
    cfg.pop("optimizer")  # a missing optimizer would fall back to Ultralytics' "auto"
    with pytest.raises(ValueError, match="optimizer"):
        train_kwargs(cfg, seed=0, data="d.yaml", device="cpu")


def test_device_required(capsys):
    argv = ["--config", "c.yaml", "--data", "d.yaml", "--seed", "0", "--project", "p", "--name", "n"]
    with pytest.raises(SystemExit) as e:
        main(argv)
    assert e.value.code == 2
    assert "--device" in capsys.readouterr().err


def test_env_record_keys():
    rec = env_record("cpu")
    assert set(rec) == {
        "ultralytics", "torch", "cuda", "gpu", "device",
        "git_commit", "schema_version", "split_manifest_sha256",
    }
    assert rec["device"] == "cpu" and rec["gpu"] is None
    assert rec["schema_version"] == "v0"
    assert rec["split_manifest_sha256"] is None
    json.dumps(rec)


def test_env_record_manifest_hash(tmp_path):
    (tmp_path / "manifests").mkdir()
    (tmp_path / "manifests" / "tiles.csv").write_bytes(b"abc")
    (tmp_path / "D").mkdir()
    rec = env_record("cpu", data_yaml=tmp_path / "D" / "data.yaml")
    assert rec["split_manifest_sha256"] == (
        "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
    )
    # no manifest next to the dataset: the key stays present with None
    other = tmp_path / "other" / "D"
    other.mkdir(parents=True)
    assert env_record("cpu", data_yaml=other / "data.yaml")["split_manifest_sha256"] is None


def test_env_record_gpu_branch_without_touching_a_gpu(monkeypatch):
    fake = types.SimpleNamespace(
        get_device_name=lambda i: f"Fake GPU {i}",
        get_device_properties=lambda i: types.SimpleNamespace(
            pci_domain_id=0, pci_bus_id=0x3B, pci_device_id=0
        ),
        mem_get_info=lambda i: (10 * 2**30, 24 * 2**30),
    )
    monkeypatch.setattr(train.torch, "cuda", fake)
    rec = env_record("3")
    assert rec["device"] == "3"
    assert rec["gpu"] == {"index": 3, "name": "Fake GPU 3", "pci_bus_id": "00000000:3B:00.0", "free_mib": 10240, "total_mib": 24576}
    with pytest.raises(ValueError):
        env_record("0,1")


def test_smoke_overrides():
    cfg = _load("train_d_yolo11n.yaml")
    kw = train_kwargs(cfg, seed=0, data="d.yaml", device="cpu", smoke=True)
    assert (kw["epochs"], kw["imgsz"], kw["fraction"], kw["batch"]) == (1, 256, 0.01, 4)


def test_main_refuses_existing_run_dir(tmp_path):
    (tmp_path / "run").mkdir()
    cfg = CONFIGS / "train_d_yolo11n.yaml"
    with pytest.raises(SystemExit):
        main(["--config", str(cfg), "--data", "d.yaml", "--seed", "0",
              "--project", str(tmp_path), "--name", "run", "--device", "cpu"])


def test_import_forces_pci_bus_id_order():
    env = dict(os.environ, CUDA_DEVICE_ORDER="FASTEST_FIRST")
    out = subprocess.run(
        [sys.executable, "-c", "import fpyolo, os; print(os.environ['CUDA_DEVICE_ORDER'])"],
        env=env, capture_output=True, text=True, check=True,
    )
    assert out.stdout.strip() == "PCI_BUS_ID"


def test_cpu_device_hides_gpus(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    apply_device_env("cpu")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == ""


def test_gpu_device_leaves_visibility_alone(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "0,1")
    apply_device_env("1")
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "0,1"


@pytest.mark.parametrize("bad", ["0", "-1", "-16"])
def test_batch_below_one_rejected(bad, capsys, tmp_path):
    # Ultralytics treats batch < 1 as auto-batch, which would break identical D batches.
    cfg = CONFIGS / "train_d_yolo11n.yaml"
    with pytest.raises(SystemExit) as e:
        main(["--config", str(cfg), "--data", "d.yaml", "--seed", "0", "--project", str(tmp_path),
              "--name", "run", "--device", "cpu", "--batch", bad])
    assert e.value.code == 2
    assert "--batch" in capsys.readouterr().err
    assert not (tmp_path / "run").exists()


def test_train_main_holds_the_run_lock_while_training(tmp_path, monkeypatch):
    import ultralytics
    from fpyolo.train import LOCK_NAME, RunLocked, acquire_run_lock

    seen = {}

    class FakeYOLO:
        def __init__(self, model):
            pass

        def train(self, **kw):
            run = tmp_path / "run"
            seen["lock_file"] = (run / LOCK_NAME).exists()
            with pytest.raises(RunLocked):  # a second process would be refused now
                acquire_run_lock(run)
            seen["trained"] = True

    monkeypatch.setattr(ultralytics, "YOLO", FakeYOLO)
    cfg = CONFIGS / "train_d_yolo11n.yaml"
    assert main(["--config", str(cfg), "--data", "d.yaml", "--seed", "0", "--project", str(tmp_path),
                 "--name", "run", "--device", "cpu"]) == 0
    assert seen == {"lock_file": True, "trained": True}


PROBE = """
import fcntl, sys, time
f = open(sys.argv[1], "a")
fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
print("locked", flush=True)
time.sleep(float(sys.argv[2]))
"""


def _probe(path, seconds):
    proc = subprocess.Popen([sys.executable, "-c", PROBE, str(path), str(seconds)], stdout=subprocess.PIPE, text=True)
    assert proc.stdout.readline().strip() == "locked"
    return proc


def test_acquire_run_lock_waits_out_a_short_state_probe(tmp_path):
    from fpyolo.train import LOCK_NAME, acquire_run_lock, release_run_lock

    proc = _probe(tmp_path / LOCK_NAME, 0.5)  # like `runstate state` holding the lock for a moment
    try:
        lock = acquire_run_lock(tmp_path)  # retries for ~2 s instead of failing at once
        release_run_lock(lock)
    finally:
        proc.wait()


def test_acquire_run_lock_gives_up_when_the_run_is_really_busy(tmp_path, monkeypatch):
    from fpyolo import train as train_mod
    from fpyolo.train import LOCK_NAME, RunLocked, acquire_run_lock

    monkeypatch.setattr(train_mod, "LOCK_RETRIES", 3)
    proc = _probe(tmp_path / LOCK_NAME, 30)
    try:
        with pytest.raises(RunLocked):
            acquire_run_lock(tmp_path)
    finally:
        proc.kill()
        proc.wait()
