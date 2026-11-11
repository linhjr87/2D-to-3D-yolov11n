import json

import pytest
from PIL import Image

from fpyolo.scale import (
    DimText,
    Segment,
    consistency,
    is_consistent,
    main,
    match_dimension_line,
    measure_tile,
    numeric_dimension_texts,
    summarize,
    text_reference_point,
)
from svgfixtures import write_rgba_png, write_svg


def layer(name, body, transform=""):
    t = f' transform="{transform}"' if transform else ""
    return (
        f'<g id="layer{name}" inkscape:groupmode="layer" inkscape:label="{name}"{t}>\n{body}\n</g>'
    )


def text(content, x, y, rot=0.0):
    return (
        f'<text font-size="2" transform="rotate({rot},{x},{y})" x="{x}" y="{y}">\n'
        f"   {content}\n  </text>"
    )


def seg(x0, y0, x1, y1):
    return f'<path d="M {x0},{y0} L {x1},{y1}" fill="none" stroke="black"/>'


def write(tmp, body, name="0000-0000.svg"):
    # write_svg's root has no inkscape namespace, so declare it on the root here
    p = write_svg(tmp / name, body)
    p.write_text(
        p.read_text(encoding="utf-8").replace(
            "<svg ", '<svg xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" ', 1
        ),
        encoding="utf-8",
    )
    return p


def test_horizontal_text_matches_segment(spaced_tmp):
    p = write(spaced_tmp, layer("DIM", text("3600", 50, 45) + seg(32, 46, 68, 46)))
    assert measure_tile(p, 1000) == pytest.approx([10.0])


def test_vertical_text_matches_vertical_segment(spaced_tmp):
    p = write(spaced_tmp, layer("DIM", text("3600", 45, 50, rot=90) + seg(46, 32, 46, 68)))
    assert measure_tile(p, 1000) == pytest.approx([10.0])


def test_rotation_outside_180_is_normalised(spaced_tmp):
    p = write(spaced_tmp, layer("DIM", text("3600", 45, 50, rot=-450.0) + seg(46, 32, 46, 68)))
    (t,) = numeric_dimension_texts(p)
    assert t.angle_deg % 180 == pytest.approx(90.0)
    assert measure_tile(p, 1000) == pytest.approx([10.0])


def test_text_fields(spaced_tmp):
    p = write(spaced_tmp, layer("DIM", text("3600", 50, 45)))
    (t,) = numeric_dimension_texts(p)
    assert t == DimText(3600.0, 50.0, 45.0, 0.0, "DIM")


def test_non_digit_and_short_texts_skipped(spaced_tmp):
    body = text("C1818", 50, 45) + text("5", 50, 44) + text("12x3", 50, 43) + text("烟道", 50, 42)
    body += text("1 2", 50, 41) + seg(32, 46, 68, 46)
    p = write(spaced_tmp, layer("DIM", body))
    assert numeric_dimension_texts(p) == []
    assert measure_tile(p, 1000) == []


def test_no_matching_segment_gives_empty(spaced_tmp):
    p = write(spaced_tmp, layer("DIM", text("3600", 50, 45)))
    assert measure_tile(p, 1000) == []


def test_other_layer_segment_not_matched(spaced_tmp):
    p = write(spaced_tmp, layer("DIM", text("3600", 50, 45)) + layer("WALL", seg(32, 46, 68, 46)))
    assert measure_tile(p, 1000) == []


def test_deviation_limit_two_degrees():
    t = DimText(3600.0, 50.0, 45.0, 0.0, "DIM")
    # tan(1.9 deg) * 36 ~= 1.194 rise over 36 units -> accepted; tan(2.1 deg) -> rejected
    import math

    def tilted(deg):
        dy = 36 * math.tan(math.radians(deg))
        return Segment("DIM", 32.0, 45.0 - dy / 2, 68.0, 45.0 + dy / 2)

    assert match_dimension_line(t, [tilted(1.9)]) == pytest.approx(36 / math.cos(math.radians(1.9)))
    assert match_dimension_line(t, [tilted(2.1)]) is None
    assert match_dimension_line(t, [tilted(-1.9)]) is not None


def test_perpendicular_segment_not_matched():
    t = DimText(3600.0, 50.0, 45.0, 0.0, "DIM")
    assert match_dimension_line(t, [Segment("DIM", 50.0, 30.0, 50.0, 60.0)]) is None


def test_midpoint_distance_limit_anchor_mode():
    t = DimText(3600.0, 50.0, 45.0, 0.0, "DIM")
    near, far = Segment("DIM", 32.0, 47.9, 68.0, 47.9), Segment("DIM", 32.0, 48.1, 68.0, 48.1)
    assert match_dimension_line(t, [near], mode="anchor") == pytest.approx(36.0)
    assert match_dimension_line(t, [far], mode="anchor") is None


def test_midpoint_distance_limit_centre_mode():
    # "3600" is 4 chars * 0.6 * 2 = 4.8 wide: the centre is 2.4 right of the start anchor
    t = DimText(3600.0, 47.6, 45.0, 0.0, "DIM")
    assert match_dimension_line(t, [Segment("DIM", 32.0, 47.9, 68.0, 47.9)]) == pytest.approx(36.0)
    assert match_dimension_line(t, [Segment("DIM", 32.0, 48.1, 68.0, 48.1)]) is None


def test_reference_point_follows_text_direction():
    assert text_reference_point(DimText(3600.0, 10.0, 20.0, 0.0, "L")) == pytest.approx((12.4, 20.0))
    assert text_reference_point(DimText(3600.0, 10.0, 20.0, 90.0, "L")) == pytest.approx((10.0, 22.4))
    assert text_reference_point(DimText(3600.0, 10.0, 20.0, 180.0, "L")) == pytest.approx((7.6, 20.0))
    assert text_reference_point(DimText(36.0, 10.0, 20.0, 0.0, "L", 3.0)) == pytest.approx((11.8, 20.0))
    assert text_reference_point(DimText(3600.0, 10.0, 20.0, 90.0, "L"), "anchor") == (10.0, 20.0)


def test_text_start_far_left_of_line_still_matches_in_centre_mode():
    # start anchor is 2.4 left and 2 below the midpoint: 3.1 away, too far for the literal rule
    t = DimText(3600.0, 47.6, 46.0, 0.0, "DIM")
    line = Segment("DIM", 32.0, 44.0, 68.0, 44.0)
    assert match_dimension_line(t, [line]) == pytest.approx(36.0)
    assert match_dimension_line(t, [line], mode="anchor") is None


def test_tick_near_start_anchor_not_chosen_in_centre_mode():
    t = DimText(3600.0, 47.6, 45.0, 0.0, "DIM")
    tick = Segment("DIM", 47.0, 45.2, 48.5, 45.2)  # 1.5 long, 0.6 from the anchor
    line = Segment("DIM", 32.0, 46.0, 68.0, 46.0)  # midpoint 1 from the centre
    assert match_dimension_line(t, [tick, line]) == pytest.approx(36.0)
    assert match_dimension_line(t, [tick, line], mode="anchor") == pytest.approx(1.5)


def test_segments_shorter_than_one_unit_ignored_in_centre_mode():
    t = DimText(3600.0, 47.6, 45.0, 0.0, "DIM")
    short = Segment("DIM", 49.6, 45.0, 50.4, 45.0)  # 0.8 long, right at the centre
    assert match_dimension_line(t, [short]) is None
    assert match_dimension_line(t, [short], mode="anchor") == pytest.approx(0.8)  # literal rule keeps it
    assert match_dimension_line(t, [Segment("DIM", 49.5, 45.0, 50.5, 45.0)]) == pytest.approx(1.0)


def test_font_size_read_from_text(spaced_tmp):
    body = '<text font-size="3" x="50" y="45">\n  3600\n </text>'
    p = write(spaced_tmp, layer("DIM", body))
    (t,) = numeric_dimension_texts(p)
    assert t.font_size == 3.0
    assert text_reference_point(t) == pytest.approx((53.6, 45.0))


def test_consistency_share_within_ten_percent():
    assert consistency([10.0, 10.5, 9.5, 20.0]) == pytest.approx(0.75)
    assert consistency([10.0, 11.0, 9.0]) == pytest.approx(1.0)  # median 10: both within +-10%
    assert consistency([10.0, 11.5, 20.0, 40.0]) == pytest.approx(0.0)  # median 15.75
    assert is_consistent([5.0, 10.0, 10.0, 40.0])  # median 10: exactly half agree
    assert not is_consistent([10.0, 20.0, 40.0])


def test_summarize_uses_consistent_tiles_only_and_inconclusive_below_minimum():
    good = [("train-00", f"0000-{i:04d}", [10.0, 10.0, 10.0, 90.0]) for i in range(3)]
    noisy = [("train-00", "0001-0000", [1.0, 10.0, 100.0])]
    res = summarize(good + noisy, min_consistent=4)
    assert res["n_tiles"] == 4 and res["n_consistent"] == 3
    assert res["overall_median"] == pytest.approx(10.0) and res["iqr"] == pytest.approx(0.0)
    assert res["conclusion"] == "inconclusive"
    assert [t["consistent"] for t in res["tiles"]] == [True, True, True, False]
    res = summarize(good + noisy, min_consistent=3)
    assert res["conclusion"] == "constant"


def test_summarize_varies_when_consistent_medians_spread():
    tiles = [("train-00", f"0000-{i:04d}", [m, m, m]) for i, m in enumerate([5.0, 10.0, 20.0, 25.0])]
    res = summarize(tiles, min_consistent=4)
    assert res["conclusion"] == "varies" and res["iqr_over_median"] > 0.10


def test_summarize_without_any_consistent_tile():
    res = summarize([("train-00", "0000-0000", [1.0, 10.0, 100.0])], min_consistent=1)
    assert res["overall_median"] is None and res["conclusion"] == "inconclusive"


def test_reversed_segment_and_wraparound_angle():
    # text at 179.5 deg is nearly horizontal; the segment runs right to left
    t = DimText(3600.0, 50.0, 45.0, 179.5, "DIM")
    assert match_dimension_line(t, [Segment("DIM", 68.0, 46.0, 32.0, 46.0)]) == pytest.approx(36.0)


def test_closest_midpoint_wins():
    t = DimText(3600.0, 50.0, 45.0, 0.0, "DIM")
    segs = [Segment("DIM", 40.0, 47.0, 60.0, 47.0), Segment("DIM", 32.0, 46.0, 68.0, 46.0)]
    assert match_dimension_line(t, segs) == pytest.approx(36.0)


def test_layer_and_group_transform_applied(spaced_tmp):
    # the layer is shifted by (10, 0): both text and segment move together
    body = layer("DIM", text("3600", 40, 45) + seg(22, 46, 58, 46), transform="translate(10,0)")
    p = write(spaced_tmp, body)
    (t,) = numeric_dimension_texts(p)
    assert (t.x, t.y) == pytest.approx((50.0, 45.0))
    assert measure_tile(p, 1000) == pytest.approx([10.0])


def test_own_transform_on_segment_and_line_element(spaced_tmp):
    body = text("3600", 50, 45)
    body += '<path d="M 22,46 L 58,46" transform="translate(10,0)"/>'
    p = write(spaced_tmp, layer("DIM", body))
    assert measure_tile(p, 1000) == pytest.approx([10.0])
    body = text("3600", 50, 45) + '<line x1="32" y1="46" x2="68" y2="46"/>'
    p = write(spaced_tmp, layer("DIM", body), name="0000-0001.svg")
    assert measure_tile(p, 1000) == pytest.approx([10.0])


def test_curved_and_multi_point_paths_are_not_segments(spaced_tmp):
    body = text("3600", 50, 45)
    body += '<path d="M 32,46 A 4,4 0 0,0 68,46"/>'
    body += '<path d="M 32,46 L 50,46 L 68,46"/>'
    p = write(spaced_tmp, layer("DIM", body))
    assert measure_tile(p, 1000) == []


def test_img_width_scales_estimate(spaced_tmp):
    p = write(spaced_tmp, layer("DIM", text("3600", 50, 45) + seg(32, 46, 68, 46)))
    assert measure_tile(p, 999) == pytest.approx([3600 / (36 * 9.99)])


def test_viewbox_width_used(spaced_tmp):
    p = write_svg(spaced_tmp / "a.svg", layer("DIM", text("3600", 50, 45) + seg(32, 46, 68, 46)), viewbox="0 0 200 200")
    p.write_text(
        p.read_text(encoding="utf-8").replace(
            "<svg ", '<svg xmlns:inkscape="http://www.inkscape.org/namespaces/inkscape" ', 1
        ),
        encoding="utf-8",
    )
    # 36 units of 200 across 1000 px = 180 px -> 20 mm/px
    assert measure_tile(p, 1000) == pytest.approx([20.0])


def make_src(root, tiles_per_archive=3, n_dims=3):
    for a in ("train-00", "train-01", "test-00"):
        d = root / a
        d.mkdir(parents=True)
        for i in range(tiles_per_archive):
            body = ""
            for k in range(n_dims):
                y = 20 + 10 * k
                body += text("3600", 47.6, y) + seg(32, y + 1, 68, y + 1)
            write(d, layer("DIM", body), name=f"0000-000{i}.svg")
            write_rgba_png(d / f"0000-000{i}.png", size=(1000, 1000), strokes=[(10, 10, 20, 20, 255)])


def test_cli_writes_json_and_overlays(spaced_tmp):
    src, out, qa = spaced_tmp / "src", spaced_tmp / "out" / "scale.json", spaced_tmp / "qa"
    make_src(src)
    rc = main(["--src", str(src), "--n", "4", "--seed", "0", "--out", str(out),
               "--qa-dir", str(qa), "--qa-n", "2", "--workers", "1", "--min-consistent", "4"])
    assert rc == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["n_tiles"] == 4
    assert len(res["tiles"]) == 4
    assert all(t["n_estimates"] == 3 and t["median"] == pytest.approx(10.0) for t in res["tiles"])
    assert res["overall_median"] == pytest.approx(10.0)
    assert res["iqr"] == pytest.approx(0.0)
    assert res["n_consistent"] == 4 and res["conclusion"] == "constant"
    assert res["mode"] == "centre"
    pngs = sorted(qa.glob("*.png"))
    assert len(pngs) == 2
    assert Image.open(pngs[0]).size == (1000, 1000)
    # same seed, same selection
    out2 = spaced_tmp / "out2.json"
    main(["--src", str(src), "--n", "4", "--seed", "0", "--out", str(out2), "--workers", "1",
          "--min-consistent", "4"])
    assert [t["image_id"] for t in json.loads(out2.read_text())["tiles"]] == [t["image_id"] for t in res["tiles"]]


def test_cli_conclusion_varies(spaced_tmp):
    src = spaced_tmp / "src"
    make_src(src, tiles_per_archive=2)
    # make one archive 2x coarser by changing the labelled value
    for f in (src / "train-00").glob("*.svg"):
        f.write_text(f.read_text(encoding="utf-8").replace("3600", "7200"), encoding="utf-8")
    out = spaced_tmp / "s.json"
    assert main(["--src", str(src), "--n", "6", "--seed", "0", "--out", str(out), "--workers", "1",
                 "--min-consistent", "6"]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["conclusion"] == "varies"


def test_cli_fewer_tiles_than_requested_still_writes(spaced_tmp):
    src = spaced_tmp / "src"
    make_src(src, tiles_per_archive=1)
    out = spaced_tmp / "s.json"
    assert main(["--src", str(src), "--n", "40", "--seed", "0", "--out", str(out), "--workers", "1"]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["n_tiles"] == 3 and res["n_requested"] == 40


def test_tiles_with_fewer_than_three_estimates_not_selected(spaced_tmp):
    src = spaced_tmp / "src"
    make_src(src, tiles_per_archive=1, n_dims=2)
    out = spaced_tmp / "s.json"
    assert main(["--src", str(src), "--n", "5", "--seed", "0", "--out", str(out), "--workers", "1"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["n_tiles"] == 0


def test_cli_scans_further_until_enough_consistent_tiles(spaced_tmp):
    src = spaced_tmp / "src"
    make_src(src, tiles_per_archive=3)  # 9 tiles, all consistent
    out = spaced_tmp / "s.json"
    assert main(["--src", str(src), "--n", "2", "--seed", "0", "--out", str(out), "--workers", "1",
                 "--min-consistent", "5"]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["n_tiles"] == 5 and res["n_consistent"] == 5 and res["conclusion"] == "constant"
    # max_scan caps the extra scanning
    assert main(["--src", str(src), "--n", "2", "--seed", "0", "--out", str(out), "--workers", "1",
                 "--min-consistent", "9", "--max-scan", "4"]) == 0
    res = json.loads(out.read_text(encoding="utf-8"))
    assert res["n_scanned"] == 4 and res["n_tiles"] == 4 and res["conclusion"] == "inconclusive"


def test_cli_inconclusive_by_default_with_few_tiles(spaced_tmp):
    src = spaced_tmp / "src"
    make_src(src, tiles_per_archive=2)
    out = spaced_tmp / "s.json"
    assert main(["--src", str(src), "--n", "4", "--seed", "0", "--out", str(out), "--workers", "1"]) == 0
    assert json.loads(out.read_text(encoding="utf-8"))["conclusion"] == "inconclusive"
