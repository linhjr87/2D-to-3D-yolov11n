"""A tiny FloorPlanCAD-like source tree (3 archives, 15 tiles) for the build tests.

Every PNG is rendered from its own SVG (labeled and unlabeled strokes), so the
SVG/PNG alignment check passes unless a test moves the image on purpose.
"""
import copy
import shutil
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

from fpyolo.qa import render_polylines
from fpyolo.svgparse import parse_tile
from svgfixtures import write_svg

SCHEMA = Path(__file__).resolve().parents[1] / "configs" / "classes_v0.yaml"
TEST_CFG = {
    "schema": str(SCHEMA),
    "archives": ["train-00", "train-01", "test-00"],
    "split_mode": "cluster",
    "ratios": [0.6, 0.2, 0.2],
    "seed": 0,
    "near_dup": {"threshold": 0.95, "min_ink": 0.01, "thumb": 32},
    "boxes": {"min_side_px": 2, "min_visible": 0.30},
    "walls": {
        "gap_px": 3, "max_thickness_px": 60, "cap_px": 32, "coverage_min": 0.80,
        "stop_if_excluded_frac": 0.30, "candidates": [2, 33], "candidate_enable_median_cov": 0.80,
    },
    "core": {"min_train": 1, "min_test": 1},
    "manual_merges": [],
}


def write_cfg(path: Path, **changes) -> Path:
    cfg = copy.deepcopy(TEST_CFG)
    for key, value in changes.items():
        if isinstance(value, dict):
            cfg[key].update(value)
        else:
            cfg[key] = value
    path.write_text(yaml.safe_dump(cfg, allow_unicode=True), encoding="utf-8")
    return path


@dataclass(frozen=True)
class MiniData:
    src: Path
    expected_keeper: str  # copy with more things in the byte-identical pair
    low_cov_id: str  # wall tile whose strokes close no region (coverage 0)
    shared_prefix: str  # drawing number shared by a train-01 and a test-00 tile
    id_999: str  # tile whose PNG is 1000 px wide and 999 px high


def _path(d: str, sid: int | None = None, iid: int = -1) -> str:
    label = "" if sid is None else f' semantic-id="{sid}" instance-id="{iid}"'
    return f'<path d="{d}" fill="none" stroke="rgb(255,0,0)" stroke-width="0.1"{label}/>'


def _rect(x, y, w, h, sid=None, iid=-1) -> str:
    return _path(f"M {x} {y} L {x + w} {y} L {x + w} {y + h} L {x} {y + h} Z", sid, iid)


def wall(x, y, w=50, h=2, sid=1) -> str:
    """A closed wall outline (2 units = 20 px thick by default)."""
    return _rect(x, y, w, h, sid)


def door(x, y, iid) -> str:
    """Single door (sid 3): leaf from the hinge plus a quarter-circle swing."""
    return _path(f"M {x} {y} L {x + 8} {y} A 8 8 0 0 1 {x} {y + 8}", 3, iid)


def window(x, y, iid, sid=9) -> str:
    return _rect(x, y, 10, 2, sid, iid)


def hatch(x, y) -> list[str]:
    """Unlabeled hatching: 25 lines of 60 units -> 15,000 ink px (1.5% of a tile)."""
    return [_path(f"M {x} {y + 0.8 * k} L {x + 60} {y + 0.8 * k}") for k in range(25)]


def _render_png(svg_body: str, png: Path, size: tuple[int, int], tmp: Path) -> None:
    """RGBA PNG (PIL size = (w, h)) whose alpha is the 1-px raster of every stroke."""
    # parse_tile only returns labeled elements, so render a copy where all are labeled
    everything = svg_body.replace('stroke-width="0.1"/>', 'stroke-width="0.1" semantic-id="0" instance-id="-1"/>')
    tile = parse_tile(write_svg(tmp, everything), "render")
    w, h = size
    scale = np.array([w / 100.0, h / 100.0])
    mask = render_polylines([p.polyline * scale for p in tile.primitives], h, w)
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    rgba[mask] = (255, 0, 0, 255)
    Image.fromarray(rgba, "RGBA").save(png)


def _tile(root: Path, archive: str, stem: str, elements: list[str], size=(1000, 1000), image_from=None) -> None:
    d = root / archive
    d.mkdir(parents=True, exist_ok=True)
    body = "\n".join(elements)
    write_svg(d / f"{stem}.svg", body)
    if image_from is not None:
        shutil.copyfile(image_from, d / f"{stem}.png")
    else:
        _render_png(body, d / f"{stem}.png", size, root / "render.svg")


def make_minidata(root: Path) -> MiniData:
    src = Path(root) / "Floor Plan CAD é"
    src.mkdir(parents=True)
    room = [wall(20, 20, 34, 2), wall(20, 52, 34, 2), wall(20, 22, 2, 30), wall(52, 22, 2, 30)]

    # byte-identical pair: 0002-0001 labels the window, 0001-0001 left it unlabeled
    full = [wall(10, 20), door(70, 40, 0), window(20, 60, 1), *hatch(20, 70)]
    _tile(src, "train-00", "0002-0001", full)
    partial = [wall(10, 20), door(70, 40, 0), _rect(20, 60, 10, 2), *hatch(20, 70)]
    _tile(src, "train-00", "0001-0001", partial, image_from=src / "train-00" / "0002-0001.png")

    _tile(src, "train-00", "0003-0001", [])  # empty tile
    _tile(src, "train-00", "0004-0002", [*room, door(60, 60, 0), wall(5, 80, 40, 2, sid=2)])  # + curtain wall
    _tile(src, "train-00", "0005-0001", [wall(10, 10), window(30, 50, 0), door(70, 70, 1)], size=(1000, 999))
    _tile(src, "train-00", "0014-0001", [door(10, 10, 0), window(50, 50, 1)])
    _tile(src, "train-00", "0015-0001", [wall(30, 30), door(40, 60, 0)])

    _tile(src, "train-01", "0007-0001", [wall(10, 40), door(30, 70, 0)])
    _tile(src, "train-01", "0009-0001", [_path("M 10 50 L 90 50", 1), door(30, 70, 0)])  # open wall stroke
    _tile(src, "train-01", "0010-0003", [door(10, 10, 0), door(40, 10, 1), window(60, 60, 2)])

    _tile(src, "test-00", "0007-0005", [wall(15, 15), window(40, 70, 0)])
    _tile(src, "test-00", "0011-0001", [wall(10, 85), door(20, 20, 0), window(60, 30, 1)])
    _tile(src, "test-00", "0012-0002", [door(50, 50, 0)])
    _tile(src, "test-00", "0013-0001", [*room, window(70, 70, 0)])
    _tile(src, "test-00", "0016-0001", [door(15, 60, 0), window(70, 15, 1)])
    (src / "render.svg").unlink()
    return MiniData(
        src=src,
        expected_keeper="train-00_0002-0001",
        low_cov_id="train-01_0009-0001",
        shared_prefix="0007",
        id_999="train-00_0005-0001",
    )
