import cv2
import numpy as np

from fpyolo.ink import ink_fraction, load_ink, write_ink_rgb
from svgfixtures import write_rgba_png


def test_load_ink_inverts_alpha(tmp_path):
    p = write_rgba_png(tmp_path / "a.png", size=(10, 10), strokes=[(0, 0, 0, 0, 255), (2, 2, 2, 2, 128)])
    ink = load_ink(p)
    assert ink.dtype == np.uint8 and ink.shape == (10, 10)
    assert ink[0, 0] == 0 and ink[5, 5] == 255 and ink[2, 2] == 127


def test_write_ink_rgb_three_equal_channels(tmp_path):
    ink = load_ink(write_rgba_png(tmp_path / "a.png", size=(12, 8), strokes=[(1, 1, 3, 3, 200)]))
    out = tmp_path / "out.png"
    write_ink_rgb(ink, out)
    img = cv2.imread(str(out), cv2.IMREAD_UNCHANGED)
    assert img.shape == (8, 12, 3)
    assert (img[..., 0] == img[..., 1]).all() and (img[..., 1] == img[..., 2]).all()
    assert (img[..., 0] == ink).all()


def test_999px_shape_kept(tmp_path):
    p = write_rgba_png(tmp_path / "a.png", size=(1000, 999))
    assert load_ink(p).shape == (999, 1000)


def test_ink_fraction(tmp_path):
    ink = load_ink(write_rgba_png(tmp_path / "a.png", size=(10, 10), strokes=[(0, 0, 9, 0, 255)]))
    assert ink_fraction(ink) == 0.1
    assert ink_fraction(np.full((4, 4), 255, np.uint8)) == 0.0


def test_spaced_paths(spaced_tmp):
    ink = load_ink(write_rgba_png(spaced_tmp / "é.png", size=(6, 6), strokes=[(0, 0, 1, 1, 255)]))
    write_ink_rgb(ink, spaced_tmp / "o.png")
    assert (spaced_tmp / "o.png").exists()
