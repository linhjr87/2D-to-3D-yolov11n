import os
import subprocess
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import CONFIGS, REPO
from fpyolo import qa
from fpyolo.schema import load_schema

SCHEMA = load_schema(CONFIGS / "classes_v0.yaml")
D_NAMES, W_NAMES = SCHEMA.yolo_names("D"), SCHEMA.yolo_names("W")
SMOKE_D = REPO / "runs" / "smoke" / "d" / "weights" / "last.pt"
SMOKE_W = REPO / "runs" / "smoke" / "w" / "weights" / "last.pt"
H, W = 120, 160


def _dataset_root(root, stems_d, stems_w, split="test"):
    for model, stems, label in (("D", stems_d, "0 0.5 0.5 0.25 0.25\n"),
                                ("W", stems_w, "0 0.1 0.1 0.9 0.1 0.9 0.2 0.1 0.2\n")):
        for kind in ("images", "labels"):
            (root / model / kind / split).mkdir(parents=True, exist_ok=True)
        for s in stems:
            Image.new("RGB", (W, H), "white").save(root / model / "images" / split / f"{s}.png")
            (root / model / "labels" / split / f"{s}.txt").write_text(label, encoding="utf-8")
    return root


class _Boxes(SimpleNamespace):
    def __len__(self):
        return len(self.conf)


class _FakeModel:
    def __init__(self, names, kind, log):
        self.names, self.kind, self.log = names, kind, log

    def predict(self, source, **kw):
        self.log.append((self.kind, source, kw))
        if self.kind == "D":
            data = torch.tensor([[20.0, 20.0, 80.0, 60.0, 0.9, 3.0], [1.0, 1.0, 9.0, 9.0, 0.3, 0.0]])
            return [SimpleNamespace(path=str(source), orig_shape=(H, W), masks=None,
                                    boxes=_Boxes(xyxy=data[:, :4], conf=data[:, 4], cls=data[:, 5]))]
        m = torch.zeros((1, H, W))
        m[0, 90:100, 10:150] = 1
        return [SimpleNamespace(path=str(source), orig_shape=(H, W), masks=SimpleNamespace(data=m),
                                boxes=_Boxes(conf=torch.tensor([0.8]), cls=torch.tensor([0.0])))]


def _args(root, out, **kw):
    base = dict(d_weights="d.pt", w_weights="w.pt", w_conf=0.35, d_conf=0.25, dataset_root=root, split="test",
                n=2, seed=0, out=out, device="cpu", schema=CONFIGS / "classes_v0.yaml")
    return SimpleNamespace(**{**base, **kw})


def test_pred_overlays_with_mocked_models(spaced_tmp, monkeypatch):
    import ultralytics

    root = _dataset_root(spaced_tmp / "ds", ["a", "b", "c", "d_only"], ["a", "b", "c", "w_only"])
    log, events = [], []
    monkeypatch.setattr(ultralytics, "YOLO",
                        lambda p: events.append("load") or _FakeModel(D_NAMES if "d.pt" in str(p) else W_NAMES,
                                                                      "D" if "d.pt" in str(p) else "W", log))
    monkeypatch.setattr(qa, "apply_device_env", lambda dev: events.append(f"env:{dev}"))
    out = spaced_tmp / "out"
    assert qa.cmd_pred_overlays(_args(root, out)) == 0
    assert events[0] == "env:cpu" and "load" in events          # device first, then models
    files = sorted(p.name for p in out.glob("*.png"))
    assert len(files) == 2 and all(f.startswith("test_") for f in files)
    assert not any("only" in f for f in files)                   # only images present in both D and W
    with Image.open(out / files[0]) as im:
        assert im.width >= 2 * W and im.height >= H
        rgb = np.asarray(im.convert("RGB"))
    left, right = rgb[:, :W + 20], rgb[:, -(W + 20):]
    assert (left != right).any()
    # one image per predict call, a path string (never a list), with the requested thresholds
    assert all(isinstance(src, str) for _, src, _ in log)
    assert len(log) == 4
    d_calls = [kw for k, _, kw in log if k == "D"]
    w_calls = [kw for k, _, kw in log if k == "W"]
    assert all(kw["conf"] == 0.25 and kw["device"] == "cpu" for kw in d_calls)
    assert all(kw["conf"] == 0.35 and kw["retina_masks"] is True and kw["device"] == "cpu" for kw in w_calls)


def test_pred_overlays_is_seeded_and_n_is_capped(spaced_tmp, monkeypatch):
    import ultralytics

    root = _dataset_root(spaced_tmp / "ds", list("abcdef"), list("abcdef"))
    monkeypatch.setattr(ultralytics, "YOLO",
                        lambda p: _FakeModel(D_NAMES if "d.pt" in str(p) else W_NAMES,
                                             "D" if "d.pt" in str(p) else "W", []))
    monkeypatch.setattr(qa, "apply_device_env", lambda dev: None)
    names = []
    for k in range(2):
        qa.cmd_pred_overlays(_args(root, spaced_tmp / f"o{k}", n=3, seed=5))
        names.append(sorted(p.name for p in (spaced_tmp / f"o{k}").glob("*.png")))
    assert names[0] == names[1] and len(names[0]) == 3
    qa.cmd_pred_overlays(_args(root, spaced_tmp / "all", n=99))
    assert len(list((spaced_tmp / "all").glob("*.png"))) == 6


def test_pred_overlays_rejects_wrong_class_names(spaced_tmp, monkeypatch):
    import ultralytics

    root = _dataset_root(spaced_tmp / "ds", ["a"], ["a"])
    monkeypatch.setattr(ultralytics, "YOLO", lambda p: _FakeModel({0: "x"}, "D", []))
    monkeypatch.setattr(qa, "apply_device_env", lambda dev: None)
    with pytest.raises(ValueError, match="class names"):
        qa.cmd_pred_overlays(_args(root, spaced_tmp / "o"))


def test_pred_overlays_without_common_images_fails(spaced_tmp, monkeypatch):
    root = _dataset_root(spaced_tmp / "ds", ["a"], ["b"])
    monkeypatch.setattr(qa, "apply_device_env", lambda dev: None)
    with pytest.raises(ValueError, match="both"):
        qa.cmd_pred_overlays(_args(root, spaced_tmp / "o"))


def test_pred_overlays_cli_parser():
    ns = qa.build_parser().parse_args(
        ["pred-overlays", "--d-weights", "d.pt", "--w-weights", "w.pt", "--w-conf", "0.4",
         "--dataset-root", "data/x", "--out", "o", "--device", "cpu"])
    assert ns.split == "test" and ns.n == 20 and ns.seed == 0 and ns.d_conf == 0.25 and ns.w_conf == 0.4


@pytest.mark.skipif(not (SMOKE_D.is_file() and SMOKE_W.is_file()),
                    reason="needs the Task 11 CPU smoke weights in runs/smoke")
def test_pred_overlays_smoke_weights_on_cpu(spaced_tmp):
    root = _dataset_root(spaced_tmp / "ds", ["a"], ["a"])
    out = spaced_tmp / "out"
    proc = subprocess.run(
        [sys.executable, "-m", "fpyolo.qa", "pred-overlays", "--d-weights", str(SMOKE_D),
         "--w-weights", str(SMOKE_W), "--w-conf", "0.05", "--dataset-root", str(root), "--n", "1",
         "--out", str(out), "--device", "cpu"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, capture_output=True, text=True)
    assert proc.returncode == 0, proc.stderr[-2000:]
    [png] = list(out.glob("*.png"))
    with Image.open(png) as im:
        assert im.width >= 2 * W
