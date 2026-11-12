"""scripts/fetch_dataset.sh against a local fake archive (HF_LOCAL_DIR skips the download)."""
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

import pytest
import yaml

from conftest import REPO


def _fake_tar(root: Path) -> Path:
    src = root / "src"
    (src / "fpcad_v0" / "D" / "images").mkdir(parents=True)
    (src / "fpcad_v0" / "W" / "images").mkdir(parents=True)
    (src / "fpcad_v0" / "manifests").mkdir(parents=True)
    (src / "fpcad_v0" / "D" / "images" / "a.png").write_bytes(b"img")
    os.link(src / "fpcad_v0" / "D" / "images" / "a.png", src / "fpcad_v0" / "W" / "images" / "a.png")
    (src / "fpcad_v0" / "D" / "data.yaml").write_text("path: /old/D\ntrain: images\n")
    (src / "fpcad_v0" / "W" / "data.yaml").write_text("path: /old/W\ntrain: images\n")
    (src / "fpcad_v0" / "manifests" / "tiles.csv").write_text("id,split\n1,train\n")
    hf = root / "hf"
    hf.mkdir()
    subprocess.run(["tar", "-cf", str(hf / "fpcad_v0.tar"), "-C", str(src), "fpcad_v0"], check=True)
    digest = hashlib.sha256((hf / "fpcad_v0.tar").read_bytes()).hexdigest()
    (hf / "fpcad_v0.tar.sha256").write_text(f"{digest}  fpcad_v0.tar\n")
    return hf


@pytest.fixture
def checkout(spaced_tmp):
    """A fake checkout (path with spaces) holding the script and the 'committed' manifests."""
    co = spaced_tmp / "checkout"
    (co / "scripts").mkdir(parents=True)
    shutil.copy(REPO / "scripts" / "fetch_dataset.sh", co / "scripts")
    (co / "data" / "fpcad_v0" / "manifests").mkdir(parents=True)
    (co / "data" / "fpcad_v0" / "manifests" / "tiles.csv").write_text("id,split\n1,train\n")
    return co


def _fetch(co, hf, **env):
    return subprocess.run(["bash", str(co / "scripts" / "fetch_dataset.sh")], capture_output=True, text=True,
                          env=dict(os.environ, HF_LOCAL_DIR=str(hf), **env))


def test_fetch_installs_and_rewrites_paths(checkout, spaced_tmp):
    hf = _fake_tar(spaced_tmp)
    out = _fetch(checkout, hf)
    assert out.returncode == 0, out.stderr
    for kind in "DW":
        cfg = yaml.safe_load((checkout / "data" / "fpcad_v0" / kind / "data.yaml").read_text())
        assert cfg["path"] == str(checkout / "data" / "fpcad_v0" / kind)
        assert cfg["train"] == "images"
    assert (checkout / "data" / "fpcad_v0" / "D" / "images" / "a.png").stat().st_nlink == 2  # hardlink kept
    assert not list((checkout / "data").glob(".fetch_tmp.*"))
    # second run refuses, FORCE=1 redoes it
    again = _fetch(checkout, hf)
    assert again.returncode == 1 and "FORCE=1" in again.stderr
    assert _fetch(checkout, hf, FORCE="1").returncode == 0


def test_fetch_rejects_bad_checksum(checkout, spaced_tmp):
    hf = _fake_tar(spaced_tmp)
    (hf / "fpcad_v0.tar.sha256").write_text("0" * 64 + "  fpcad_v0.tar\n")
    out = _fetch(checkout, hf)
    assert out.returncode != 0
    assert not (checkout / "data" / "fpcad_v0" / "D").exists()


def test_fetch_rejects_manifest_differing_from_committed(checkout, spaced_tmp):
    hf = _fake_tar(spaced_tmp)
    (checkout / "data" / "fpcad_v0" / "manifests" / "tiles.csv").write_text("other\n")
    out = _fetch(checkout, hf)
    assert out.returncode == 1 and "tiles.csv" in out.stderr
    assert not (checkout / "data" / "fpcad_v0" / "D").exists()
