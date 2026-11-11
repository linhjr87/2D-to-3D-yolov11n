"""Exact dedup, ink thumbnails, near-duplicate pairs and building clusters."""
from collections import defaultdict

import cv2
import numpy as np

TileId = tuple[str, str]  # (archive, stem)


def exact_duplicate_sets(md5_by_tile: dict[TileId, str]) -> list[list[TileId]]:
    """Sets of tiles sharing one md5 (>= 2 tiles); members and sets sorted."""
    by_md5: dict[str, list[TileId]] = defaultdict(list)
    for tile, digest in md5_by_tile.items():
        by_md5[digest].append(tile)
    sets = [sorted(ts) for ts in by_md5.values() if len(ts) >= 2]
    return sorted(sets)


def choose_keeper(dup_set: list[TileId], thing_count: dict[TileId, int],
                  wall_strokes: dict[TileId, int]) -> TileId:
    """Most thing instances, then most wall strokes, then smallest (archive, stem)."""
    return min(dup_set, key=lambda t: (-thing_count[t], -wall_strokes[t], t))


def drawing_group(t: TileId) -> tuple[str, str]:
    """Source drawing: train-00 has its own numbering, train-01 and test-00 share one."""
    archive, stem = t
    return ("A" if archive == "train-00" else "B", stem.split("-")[0])


def thumbnail(ink: np.ndarray, size: int = 32) -> np.ndarray:
    """Unit-norm float32 vector (size*size) of the area-averaged ink density.

    A tile with no ink gives the zero vector (never valid for linking anyway).
    """
    density = (255.0 - ink.astype(np.float32)) / 255.0
    small = cv2.resize(density, (size, size), interpolation=cv2.INTER_AREA)
    vec = small.reshape(-1).astype(np.float32)
    norm = float(np.linalg.norm(vec))
    return vec / norm if norm > 0 else np.zeros_like(vec)


def near_duplicate_pairs(thumbs: np.ndarray, valid: np.ndarray, threshold: float = 0.95,
                         block: int = 2048) -> list[tuple[int, int]]:
    """Pairs (i, j), i < j, of valid rows with cosine >= threshold (rows are unit norm).

    The similarity matrix is computed block by block, so memory stays O(block^2).
    """
    idx = np.flatnonzero(valid)
    x = np.ascontiguousarray(thumbs[idx], dtype=np.float32)
    pairs: list[tuple[int, int]] = []
    for a in range(0, len(idx), block):
        for b in range(a, len(idx), block):
            sims = x[a:a + block] @ x[b:b + block].T
            rows, cols = np.nonzero(sims >= threshold)
            rows, cols = rows + a, cols + b
            keep = rows < cols
            pairs.extend(zip(idx[rows[keep]].tolist(), idx[cols[keep]].tolist()))
    return sorted(pairs)


class UnionFind:
    def __init__(self):
        self.parent: dict = {}

    def find(self, x):
        self.parent.setdefault(x, x)
        root = x
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[x] != root:  # path compression
            self.parent[x], x = root, self.parent[x]
        return root

    def union(self, a, b) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[max(ra, rb)] = min(ra, rb)  # smaller key stays the root


def build_clusters(tiles: list[TileId], pairs: list[tuple[int, int]]) -> dict[TileId, int]:
    """Cluster id per tile: drawing groups merged through near-duplicate pairs.

    All tiles of a drawing group share a cluster. Ids start at 0, ordered by
    the smallest drawing group of each cluster (independent of input order).
    """
    uf = UnionFind()
    for t in tiles:
        uf.find(drawing_group(t))
    for i, j in pairs:
        uf.union(drawing_group(tiles[i]), drawing_group(tiles[j]))
    roots = sorted({uf.find(drawing_group(t)) for t in tiles})
    number = {root: n for n, root in enumerate(roots)}
    return {t: number[uf.find(drawing_group(t))] for t in tiles}
