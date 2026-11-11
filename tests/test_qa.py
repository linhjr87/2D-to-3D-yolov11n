import json
from dataclasses import replace

import numpy as np
import pytest
from PIL import Image

from fpyolo.qa import box_iou, detect_frame, instance_ious, main, render_polylines, tight_bbox
from svgfixtures import tile_box, tile_with, write_rgba_png, write_svg


def test_render_segment_tight_bbox():
    mask = render_polylines([np.array([[10.0, 10.0], [20.0, 30.0]])], 50, 40)
    assert mask.dtype == bool and mask.shape == (50, 40)
    assert tight_bbox(mask) == (10, 10, 20, 30)


def test_render_is_one_pixel_and_unaliased():
    mask = render_polylines([np.array([[5.0, 7.0], [25.0, 7.0]])], 20, 30)
    assert mask.sum() == 21 and mask[7, 5:26].all()


def test_tight_bbox_empty_is_none():
    assert tight_bbox(np.zeros((8, 8), dtype=bool)) is None


def test_box_iou():
    assert box_iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert box_iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0
    assert box_iou((0, 0, 10, 10), (5, 0, 15, 10)) == pytest.approx(1 / 3)


def test_detect_frame_ignores_shorter_and_thick_lines():
    rgb = np.full((300, 400, 3), 255, dtype=np.uint8)
    rgb[40, 50:351] = rgb[260, 50:351] = 0  # top / bottom spines
    rgb[40:261, 50] = rgb[40:261, 350] = 0  # left / right spines
    rgb[100:103, 60:300] = 0  # thick but shorter black stroke (e.g. text)
    rgb[41, 52:349] = 30  # a dark row hugging the top spine must not count as the bottom
    assert detect_frame(rgb) == (50, 40, 350, 260)
    # Real data (train-00/0376-0015): a drawing line spanning the whole axes is as
    # long as a spine; the spines are the outermost long lines.
    rgb[150, 50:351] = 0
    rgb[40:261, 200] = 0
    assert detect_frame(rgb) == (50, 40, 350, 260)
    # Real data (train-00/0000-0002): red label text is drawn over a spine.
    rgb[100:110, 350] = (255, 0, 0)
    assert detect_frame(rgb) == (50, 40, 350, 260)


def test_instance_ious_arc_and_rotated_ellipse(tmp_path):
    t = tile_with(
        tmp_path,
        '<path d="M 40,50 A 10,10 0 0 0 60,50" semantic-id="3" instance-id="0"/>'
        '<ellipse cx="30" cy="30" rx="8" ry="2" transform="rotate(30 30 30)" semantic-id="9" instance-id="1"/>'
        '<path d="M 70,70 L 70.2,70.2" semantic-id="4" instance-id="0"/>'  # 2 px: below the 4 px floor
        '<path d="M 0,0 L 90,90" semantic-id="1" instance-id="-1"/>',  # wall: not a thing
    )
    rows = instance_ious(t, 1000, 1000, min_side_px=4.0)
    assert [(r["sid"], r["iid"]) for r in rows] == [(3, 0), (9, 1)]
    assert all(r["iou"] >= 0.99 for r in rows)


def test_instance_ious_half_pixel_float_noise_is_not_an_error():
    # Real data (train-01/0897-0007): CAD edges land exactly on x.5 px and the
    # polyline carries ~1e-12 float noise; independent rounding flipped a whole pixel.
    t = tile_box(4.5, 14.75, 23.5, 15.75)  # y edges at 147.5 / 157.5 px
    p = t.primitives[0]
    noisy = replace(t, primitives=[replace(p, polyline=p.polyline - 1e-12)])
    (row,) = instance_ious(noisy, 1000, 1000, min_side_px=4.0)
    assert row["iou"] == 1.0


def test_instance_ious_flags_wrong_bbox(tmp_path):
    t = tile_with(tmp_path, '<path d="M 40,50 A 10,10 0 0 0 60,50" semantic-id="3" instance-id="0"/>')
    p = t.primitives[0]
    chord_only = replace(t, primitives=[replace(p, bbox=(40.0, 50.0, 60.0, 51.0))])
    (row,) = instance_ious(chord_only, 1000, 1000, min_side_px=4.0)
    assert row["iou"] < 0.2


def _mini_floorplancad(root):
    """One train-00 tile with a single door (sid 3) and its coco_vis render
    (2000 px, axes frame at (255, 240)-(1795, 1780) as in the real data)."""
    tiles = root / "train-00"
    (tiles / "coco_vis").mkdir(parents=True)
    write_svg(tiles / "0000-0000.svg", '<path d="M 40,50 A 10,10 0 0 0 60,50" semantic-id="3" instance-id="0"/>')
    write_rgba_png(tiles / "0000-0000.png", strokes=[(400, 500, 600, 600, 255)])
    vis = np.full((2000, 2000, 3), 255, dtype=np.uint8)
    vis[240, 255:1796] = vis[1780, 255:1796] = 0
    vis[240:1781, 255] = vis[240:1781, 1795] = 0
    Image.fromarray(vis).save(tiles / "coco_vis" / "0000-0000.png")
    for archive in ("train-01", "test-00"):
        (root / archive).mkdir()
    return root


def test_cli_bbox_raster_on_spaced_path(spaced_tmp):
    src = _mini_floorplancad(spaced_tmp / "Floor Plan CAD")
    out = spaced_tmp / "qa out" / "bbox_raster.json"
    assert main(["bbox-raster", "--src", str(src), "--n", "5", "--out", str(out), "--workers", "1"]) == 0
    report = json.loads(out.read_text(encoding="utf-8"))
    assert report["n_tiles"] == 1 and report["n_checked"] == 1 and report["pass_rate"] == 1.0
    assert report["tiles"] == ["train-00_0000-0000"]


def test_cli_contact_sheet_on_spaced_path(spaced_tmp):
    src = _mini_floorplancad(spaced_tmp / "Floor Plan CAD")
    out = spaced_tmp / "sheets é"
    assert main(["contact-sheet", "--src", str(src), "--out", str(out), "--per-class", "3", "--workers", "1"]) == 0
    assert len(list(out.glob("sid_*.png"))) == 30 and len(list(out.glob("sid_*.txt"))) == 30
    rows = [l for l in (out / "sid_03.txt").read_text(encoding="utf-8").splitlines() if not l.startswith("#")]
    assert rows == ["train-00\t0000-0000\t0\t400.0 500.0 600.0 600.0\t255 240 1795 1780"]


def _built_minidata(root):
    from fpyolo.build import main as build_main
    from minidata import make_minidata, write_cfg

    md = make_minidata(root)
    out = root / "built é"
    assert build_main(["--src", str(md.src), "--out", str(out), "--config", str(write_cfg(root / "cfg.yaml")),
                       "--workers", "1"]) == 0
    return md, out


def test_overlays_one_image_per_sampled_tile_for_D_and_W(spaced_tmp):
    _, out = _built_minidata(spaced_tmp)
    for kind in ("D", "W"):
        dst = spaced_tmp / f"overlays {kind}"
        assert main(["overlays", "--dataset", str(out / kind), "--n", "4", "--seed", "0", "--out", str(dst)]) == 0
        pngs = sorted(dst.glob("*.png"))
        assert len(pngs) == 4
        assert all(p.stem.split("_", 1)[0] in ("train", "val", "test") for p in pngs)
    # a W overlay of a wall tile is not just the ink image: the wall body is painted
    wall_tile = next((out / "W" / "labels").rglob("train-00_0015-0001.txt"))
    split = wall_tile.parent.name
    dst = spaced_tmp / "all W"
    assert main(["overlays", "--dataset", str(out / "W"), "--n", "100", "--seed", "0", "--out", str(dst)]) == 0
    with Image.open(dst / f"{split}_train-00_0015-0001.png") as im:
        rgb = np.asarray(im.convert("RGB")).astype(int)
    with Image.open(out / "W" / "images" / split / "train-00_0015-0001.png") as im:
        ink = np.asarray(im.convert("RGB")).astype(int)
    assert np.abs(rgb - ink)[305:315, 305:795].mean() > 20  # inside the 30..80 x 30..32 wall body


def test_pair_sheet_one_image_per_candidate(spaced_tmp):
    md, out = _built_minidata(spaced_tmp)
    stats = spaced_tmp / "stats.json"
    pairs = [{"a": "train-00_0004-0002", "b": "test-00_0013-0001", "group_a": "A/0004", "group_b": "B/0013",
              "cosine": 0.91},
             {"a": "train-00_0005-0001", "b": "train-01_0007-0001", "group_a": "A/0005", "group_b": "B/0007",
              "cosine": 0.88}]
    stats.write_text(json.dumps({"cross_space_candidates": pairs}), encoding="utf-8")
    dst = spaced_tmp / "cross space"
    assert main(["pair-sheet", "--stats", str(stats), "--src", str(md.src), "--out", str(dst)]) == 0
    assert sorted(p.name for p in dst.glob("*.png")) == ["pair_00.png", "pair_01.png"]
