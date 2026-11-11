"""RGBA PNG tiles to black-on-white "ink" images (ink = 255 - alpha)."""
from pathlib import Path

import numpy as np
from PIL import Image


def load_ink(png_path: Path) -> np.ndarray:
    """H x W uint8 array, 0 = full ink, 255 = blank; true image size kept."""
    with Image.open(png_path) as im:
        alpha = np.asarray(im.convert("RGBA"))[..., 3]
    return (255 - alpha).astype(np.uint8)


def write_ink_rgb(ink: np.ndarray, out_path: Path) -> None:
    """Save as a PNG with three identical channels."""
    rgb = np.repeat(ink[..., None], 3, axis=2)
    Image.fromarray(rgb, "RGB").save(out_path)


def ink_fraction(ink: np.ndarray) -> float:
    """Fraction of pixels carrying any ink (value < 255)."""
    return float((ink < 255).mean())
