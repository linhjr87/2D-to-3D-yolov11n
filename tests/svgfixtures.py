"""Helpers to build tiny SVG / RGBA PNG fixtures and in-memory Tiles for tests."""
from pathlib import Path

import numpy as np
from PIL import Image

from fpyolo.svgparse import Primitive, Tile, parse_tile


def write_svg(path, body: str, viewbox: str = "0 0 100 100") -> Path:
    path = Path(path)
    path.write_text(
        '<?xml version="1.0" encoding="utf-8"?>\n'
        f'<svg version="1.1" viewBox="{viewbox}" xmlns="http://www.w3.org/2000/svg">\n'
        f"{body}\n</svg>\n",
        encoding="utf-8",
    )
    return path


def write_rgba_png(path, size=(1000, 1000), strokes=()) -> Path:
    """Write a transparent RGBA PNG; `size` is PIL (width, height) order.

    Each stroke is (x0, y0, x1, y1, alpha): a filled yellow rectangle (inclusive
    pixel bounds) with the given alpha.
    """
    path = Path(path)
    w, h = size
    arr = np.zeros((h, w, 4), dtype=np.uint8)
    for x0, y0, x1, y1, alpha in strokes:
        arr[y0 : y1 + 1, x0 : x1 + 1] = (255, 255, 0, alpha)
    Image.fromarray(arr, "RGBA").save(path)
    return path


def tile_with(tmp_path, body: str) -> Tile:
    return parse_tile(write_svg(Path(tmp_path) / "0000-0000.svg", body), "train-00")


def tile_box(x0, y0, x1, y1, sid=3, iid=0) -> Tile:
    """A Tile with one rectangular primitive, built without any SVG."""
    poly = np.array(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]], dtype=np.float64
    )
    prim = Primitive(sid, iid, (x0, y0, x1, y1), poly, True)
    return Tile("train-00", "0000-0000", (0.0, 0.0, 100.0, 100.0), [prim], 0)
