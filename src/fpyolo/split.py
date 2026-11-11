"""Stratified train/val/test split that assigns whole drawing clusters."""

import random
from collections import Counter, defaultdict
from typing import Literal

from fpyolo.dedup import TileId

SETS = ("train", "val", "test")
SetName = Literal["train", "val", "test"]
# Weight of the tile-count stratum relative to one class stratum.
TILE_WEIGHT = 10.0
# An overshooting set pays this factor more than an under-filled one, so a huge cluster is not
# dumped into the small sets just because their quota is reached in one step.
OVERSHOOT_WEIGHT = 10.0


def _aggregate(cluster_of, class_counts):
    """Per-cluster tile and class counts, plus per-class instance totals and cluster counts."""
    n_tiles: dict[int, int] = Counter()
    cnt: dict[int, Counter] = defaultdict(Counter)
    for tile, k in cluster_of.items():
        n_tiles[k] += 1
        for c, n in class_counts.get(tile, {}).items():
            if n > 0:
                cnt[k][c] += n
    total: Counter = Counter()
    n_clusters: Counter = Counter()
    for counts in cnt.values():
        total.update(counts)
        n_clusters.update(counts.keys())
    return n_tiles, cnt, total, n_clusters


def _fill_loss(x: float) -> float:
    """Squared distance of a set's fill fraction x (assigned / quota) from 1; overshoot costs more."""
    return (x - 1) ** 2 if x <= 1 else OVERSHOOT_WEIGHT * (x - 1) ** 2


def _assignment_cost(have: float, add: float, target: float) -> float:
    """Change in the fill loss when `add` is put into a set holding `have` of its `target` quota."""
    return _fill_loss((have + add) / target) - _fill_loss(have / target)


def stratified_cluster_split(
    cluster_of: dict[TileId, int],
    class_counts: dict[TileId, Counter],
    ratios: tuple[float, float, float] = (0.7, 0.1, 0.2),
    seed: int = 0,
) -> dict[TileId, SetName]:
    """Greedy cluster-level split: rare-class clusters first, each to the cheapest set.

    `class_counts[tile][c]` is the number of instances of class c in the tile (wall pseudo-class 1
    is 1 when the tile has wall strokes). Clusters are ordered by the rarest class they contain
    (min total over present classes), then larger first, ties shuffled with `seed`; clusters
    without any class come last. A cluster goes to the set s minimising the sum over its strata
    (each present class, plus the tile count weighted by TILE_WEIGHT) of the change in
    fill loss of assigned[s][stratum] / quota (see _fill_loss); the quota of a set is
    ratio_s * total, at least one average cluster's worth for classes.
    """
    if len(ratios) != len(SETS) or min(ratios) <= 0 or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError(f"ratios must be three positive numbers summing to 1, got {ratios}")
    n_tiles, cnt, total, n_clusters = _aggregate(cluster_of, class_counts)
    n_total = sum(n_tiles.values())
    assigned_tiles = [0] * len(SETS)
    assigned = [Counter() for _ in SETS]

    rng = random.Random(seed)
    clusters = sorted(n_tiles)
    tie_break = {k: rng.random() for k in clusters}
    rarity = {k: min((total[c] for c in cnt[k]), default=float("inf")) for k in clusters}
    clusters.sort(key=lambda k: (rarity[k], -n_tiles[k], tie_break[k]))

    cluster_set: dict[int, int] = {}
    for k in clusters:
        costs = []
        for s, ratio in enumerate(ratios):
            cost = TILE_WEIGHT * _assignment_cost(assigned_tiles[s], n_tiles[k], ratio * n_total)
            for c, n in cnt[k].items():
                # a quota is never smaller than one average cluster's share, so that a rare class
                # can reach a small set without counting as an overshoot
                quota = max(ratio * total[c], total[c] / n_clusters[c])
                cost += _assignment_cost(assigned[s][c], n, quota)
            costs.append(cost)
        s = costs.index(min(costs))
        cluster_set[k] = s
        assigned_tiles[s] += n_tiles[k]
        assigned[s].update(cnt[k])
    return {tile: SETS[cluster_set[k]] for tile, k in cluster_of.items()}


def split_report(assign: dict[TileId, SetName], class_counts: dict[TileId, Counter]) -> dict:
    """Actual tile ratios and per-class instance counts per set (plain JSON types)."""
    tiles = {s: 0 for s in SETS}
    per_class: dict[int, dict[str, int]] = defaultdict(lambda: {s: 0 for s in SETS})
    for tile, s in assign.items():
        tiles[s] += 1
        for c, n in class_counts.get(tile, {}).items():
            if n > 0:
                per_class[c][s] += n
    n_total = sum(tiles.values())
    return {
        "tiles": tiles,
        "tile_ratio": {s: (tiles[s] / n_total if n_total else 0.0) for s in SETS},
        "class_instances": {str(c): per_class[c] for c in sorted(per_class)},
        "classes_absent_from_some_set": [
            c for c in sorted(per_class) if any(v == 0 for v in per_class[c].values())
        ],
    }
