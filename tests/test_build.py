import csv
import json
import os
import pickle
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pytest
import yaml
from PIL import Image

from fpyolo.build import TileResult, main
from fpyolo.schema import load_schema
from minidata import SCHEMA, make_minidata, write_cfg

COLUMNS = [
    "archive", "stem", "md5", "keep", "drop_reason", "group", "cluster", "split", "n_things",
    "wall_strokes", "wall_coverage", "in_W", "align_frac", "width", "height",
]


def build(src, out, cfg, *extra, workers="1") -> int:
    return main(["--src", str(src), "--out", str(out), "--config", str(cfg), "--workers", workers, *extra])


def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def image_id(r):
    return f'{r["archive"]}_{r["stem"]}'


def row(rows, iid):
    (r,) = [r for r in rows if image_id(r) == iid]
    return r


def dup_pair(rows):
    by_md5 = defaultdict(list)
    for r in rows:
        by_md5[r["md5"]].append(r)
    (pair,) = [rs for rs in by_md5.values() if len(rs) > 1]
    return pair


def kept_ids_of_dup_pair(rows):
    return [image_id(r) for r in dup_pair(rows) if r["keep"] == "True"]


def yolo_names(path):
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))["names"]


def test_build_end_to_end(spaced_tmp):
    md = make_minidata(spaced_tmp)
    out = spaced_tmp / "out"
    cfg = write_cfg(spaced_tmp / "build test.yaml", manual_merges=[[["A", "0004"], ["B", "0012"]]])
    assert build(md.src, out, cfg, workers="2") == 0

    rows = read_csv(out / "manifests/tiles.csv")
    assert list(rows[0]) == COLUMNS and len(rows) == 15
    # exact dedup keeps the copy with more things; both copies end in one cluster (cosine 1)
    assert kept_ids_of_dup_pair(rows) == [md.expected_keeper]
    dropped = [r for r in dup_pair(rows) if r["keep"] == "False"]
    assert md.expected_keeper in dropped[0]["drop_reason"] and dropped[0]["split"] == ""
    assert len({r["cluster"] for r in dup_pair(rows)}) == 1
    # one drawing number across train-01 / test-00 -> one group -> one split
    kept = [r for r in rows if r["keep"] == "True"]
    shared = [r for r in kept if r["group"].endswith(md.shared_prefix)]
    assert len(shared) == 2 and len({r["split"] for r in shared}) == 1
    assert {r["split"] for r in kept} == {"train", "val", "test"}
    assert row(rows, "train-00_0004-0002")["cluster"] == row(rows, "test-00_0012-0002")["cluster"]  # manual merge

    d, w = out / "D", out / "W"
    # low-coverage wall tile: kept for D, excluded from W
    assert row(rows, md.low_cov_id)["in_W"] == "False"
    assert any((d / "labels").rglob(f"{md.low_cov_id}.txt"))
    assert not any((w / "labels").rglob(f"{md.low_cov_id}.txt"))
    for r in kept:
        lbl = d / "labels" / r["split"] / f"{image_id(r)}.txt"
        assert (d / "images" / r["split"] / f"{image_id(r)}.png").exists() and lbl.exists()
        assert len(lbl.read_text().splitlines()) == int(r["n_things"])
    assert row(rows, "train-00_0003-0001")["n_things"] == "0"  # the empty tile is a negative

    keeper = row(rows, md.expected_keeper)
    assert keeper["n_things"] == "2" and keeper["wall_strokes"] == "1" and keeper["align_frac"] == "1.0"
    keeper_lbl = (d / "labels" / keeper["split"] / f"{md.expected_keeper}.txt").read_text()
    assert keeper_lbl.endswith("\n") and sorted(l.split()[0] for l in keeper_lbl.splitlines()) == ["0", "6"]
    w_lbl = (w / "labels" / keeper["split"] / f"{md.expected_keeper}.txt").read_text().splitlines()
    assert len(w_lbl) == 1 and w_lbl[0].startswith("0 ")  # one wall body (YOLO index 0)
    cw = row(rows, "train-00_0004-0002")  # room walls + one curtain wall outline
    cw_lbl = (w / "labels" / cw["split"] / "train-00_0004-0002.txt").read_text().splitlines()
    assert sorted(l.split()[0] for l in cw_lbl) == ["0", "1"]

    r999 = row(rows, md.id_999)
    assert (r999["width"], r999["height"]) == ("1000", "999")
    with Image.open(d / "images" / r999["split"] / f"{md.id_999}.png") as im:
        assert im.mode == "RGB" and im.size == (1000, 999)
        rgb = np.asarray(im)
    assert (rgb[..., 0] == rgb[..., 2]).all() and (rgb < 255).any()
    w_img = w / "images" / r999["split"] / f"{md.id_999}.png"
    assert os.stat(w_img).st_ino == os.stat(d / "images" / r999["split"] / f"{md.id_999}.png").st_ino

    schema = load_schema(SCHEMA)
    data_d = yaml.safe_load((d / "data.yaml").read_text(encoding="utf-8"))
    assert Path(data_d["path"]).is_absolute() and Path(data_d["path"]) == d.resolve()
    assert yolo_names(d / "data.yaml") == schema.yolo_names("D")
    assert yolo_names(w / "data.yaml") == schema.yolo_names("W")
    core = json.load(open(out / "manifests/core_classes.json"))
    assert core["min_train"] == 1 and core["min_test"] == 1
    schema_id_of = {c.semantic_id: c.schema_id for c in schema.for_model("D")}
    assert core["semantic_ids"] and core["schema_ids"] == [schema_id_of[s] for s in core["semantic_ids"]]

    stats = json.load(open(out / "manifests/stats.json"))
    assert {"dedup", "clusters", "cross_space_candidates", "split_report", "box_stats",
            "w_candidates", "w_coverage", "misaligned"} <= set(stats)
    assert stats["dedup"]["kept"] == 14 and stats["dedup"]["agree_things_frac"] == 0.0
    assert stats["dedup"]["missing_instance_frac"] == 0.5  # window missing in one of two copies
    assert stats["w_candidates"]["2"]["would_enable"] is True and stats["w_candidates"]["2"]["enabled"] is True
    assert stats["w_candidates"]["33"]["n_tiles"] == 0
    assert stats["misaligned"]["n_misaligned"] == 0


def test_stop_when_too_many_wall_tiles_excluded(spaced_tmp):
    md = make_minidata(spaced_tmp)
    out = spaced_tmp / "out"
    cfg = write_cfg(spaced_tmp / "cfg.yaml", walls={"stop_if_excluded_frac": 0.0})
    assert build(md.src, out, cfg) == 3
    assert (out / "manifests/tiles.csv").exists()  # evidence is written ...
    assert json.load(open(out / "manifests/stats.json"))["w_coverage"]["excluded"]["all"]["excluded"] == 1
    assert not (out / "W").exists()  # ... but no dataset


def test_misaligned_tiles_exit_4(spaced_tmp):
    md = make_minidata(spaced_tmp)
    png = md.src / "test-00" / "0011-0001.png"
    with Image.open(png) as im:
        arr = np.asarray(im)
    Image.fromarray(np.roll(arr, (30, 30), axis=(0, 1))).save(png)
    out = spaced_tmp / "out"
    assert build(md.src, out, write_cfg(spaced_tmp / "cfg.yaml")) == 4
    assert float(row(read_csv(out / "manifests/tiles.csv"), "test-00_0011-0001")["align_frac"]) < 0.9
    assert json.load(open(out / "manifests/stats.json"))["misaligned"]["n_misaligned"] == 1
    assert not (out / "D").exists()


def test_reuse_manifest_reproduces_split(spaced_tmp):
    md = make_minidata(spaced_tmp)
    first = spaced_tmp / "first"
    assert build(md.src, first, write_cfg(spaced_tmp / "a.yaml")) == 0
    other = write_cfg(spaced_tmp / "b.yaml", seed=7, ratios=[0.4, 0.3, 0.3])
    assert build(md.src, spaced_tmp / "fresh", other) == 0
    reused = spaced_tmp / "reused"
    assert build(md.src, reused, other, "--reuse-manifest", str(first / "manifests/tiles.csv")) == 0

    def column(out, name):
        return [r[name] for r in read_csv(out / "manifests/tiles.csv")]

    assert column(spaced_tmp / "fresh", "split") != column(first, "split")  # the other config does differ
    for name in ("keep", "cluster", "split"):
        assert column(reused, name) == column(first, name)
    for r in read_csv(reused / "manifests/tiles.csv"):
        if r["keep"] == "True":
            assert (reused / "D" / "labels" / r["split"] / f"{image_id(r)}.txt").exists()


def test_official_split_mode(spaced_tmp):
    md = make_minidata(spaced_tmp)
    out = spaced_tmp / "out"
    assert build(md.src, out, write_cfg(spaced_tmp / "cfg.yaml", split_mode="official")) == 0
    kept = [r for r in read_csv(out / "manifests/tiles.csv") if r["keep"] == "True"]
    assert {image_id(r) for r in kept if r["split"] == "test"} == {image_id(r) for r in kept if r["archive"] == "test-00"}
    val = [r for r in kept if r["split"] == "val"]
    assert val and {r["cluster"] for r in val}.isdisjoint(r["cluster"] for r in kept if r["split"] == "train")


def test_cli_cache_loads_in_process(spaced_tmp):
    # `python -m fpyolo.build` runs the module as __main__; its cache must still name fpyolo.build classes
    md = make_minidata(spaced_tmp)
    out = spaced_tmp / "out"
    cmd = [sys.executable, "-m", "fpyolo.build", "--src", str(md.src), "--out", str(out),
           "--config", str(write_cfg(spaced_tmp / "cfg.yaml")), "--workers", "1"]
    assert subprocess.run(cmd, cwd=spaced_tmp, capture_output=True).returncode == 0
    files = list((out / "cache").rglob("*.pkl"))
    assert len(files) == 15
    assert all(isinstance(pickle.loads(f.read_bytes()), TileResult) for f in files)


def test_manual_merges_are_validated(spaced_tmp):
    md = make_minidata(spaced_tmp)
    unquoted = write_cfg(spaced_tmp / "a.yaml", manual_merges=[[["A", 4], ["B", "0012"]]])
    with pytest.raises(ValueError, match="quoted strings"):
        build(md.src, spaced_tmp / "out", unquoted)
    unknown = write_cfg(spaced_tmp / "b.yaml", manual_merges=[[["A", "0004"], ["B", "9999"]]])
    with pytest.raises(ValueError, match="unknown drawing groups"):
        build(md.src, spaced_tmp / "out", unknown)
