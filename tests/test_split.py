import json
import random
from collections import Counter, defaultdict

from fpyolo.split import split_report, stratified_cluster_split

SETS = ("train", "val", "test")
RATIOS = (0.7, 0.1, 0.2)


def _tile(k, i):
    return ("train-00", f"{k:04d}-{i:04d}")


def _random_problem(n_clusters=200, seed=123, n_classes=6):
    """Clusters of 1-10 tiles; each tile holds a few instances of random classes."""
    rng = random.Random(seed)
    cluster_of, class_counts = {}, {}
    for k in range(n_clusters):
        for i in range(rng.randint(1, 10)):
            t = _tile(k, i)
            cluster_of[t] = k
            class_counts[t] = Counter(
                {3 + rng.randrange(n_classes): rng.randint(1, 5) for _ in range(rng.randint(0, 3))}
            )
    return cluster_of, class_counts


def _tile_ratios(assign):
    n = len(assign)
    return [sum(1 for s in assign.values() if s == name) / n for name in SETS]


def test_cluster_integrity():
    cluster_of, class_counts = _random_problem()
    assign = stratified_cluster_split(cluster_of, class_counts)
    assert set(assign) == set(cluster_of)
    sets_of = defaultdict(set)
    for t, k in cluster_of.items():
        sets_of[k].add(assign[t])
    assert all(len(v) == 1 for v in sets_of.values())


def test_tile_ratios_within_2pts():
    cluster_of, class_counts = _random_problem()
    assign = stratified_cluster_split(cluster_of, class_counts)
    for got, want in zip(_tile_ratios(assign), RATIOS):
        assert abs(got - want) <= 0.02


def test_rare_class_in_every_split():
    cluster_of, class_counts = _random_problem()
    rare_clusters = random.Random(1).sample(sorted(set(cluster_of.values())), 6)
    for k in rare_clusters:  # class 40 lives in exactly 6 clusters, in one tile each
        class_counts[next(t for t, c in cluster_of.items() if c == k)][40] += 3
    assign = stratified_cluster_split(cluster_of, class_counts)
    where = {assign[t] for t, c in class_counts.items() if c.get(40)}
    assert where == set(SETS)


def test_clusters_without_classes_still_get_a_set():
    cluster_of = {_tile(k, 0): k for k in range(10)}
    class_counts = {t: Counter() for t in cluster_of}
    assign = stratified_cluster_split(cluster_of, class_counts)
    assert sorted(Counter(assign.values()).items()) == [("test", 2), ("train", 7), ("val", 1)]


def test_deterministic():
    # identical single-tile clusters: every cluster is a tie, so the seed decides who goes where
    cluster_of = {_tile(k, 0): k for k in range(40)}
    class_counts = {t: Counter({3: 1}) for t in cluster_of}
    a = stratified_cluster_split(cluster_of, class_counts, seed=0)
    assert a == stratified_cluster_split(cluster_of, class_counts, seed=0)
    assert a != stratified_cluster_split(cluster_of, class_counts, seed=1)


def test_huge_cluster_reported():
    cluster_of, class_counts = {}, {}
    for i in range(30):  # one cluster holding 30% of all tiles
        t = _tile(0, i)
        cluster_of[t], class_counts[t] = 0, Counter({3: 1})
    for k in range(1, 71):
        t = _tile(k, 0)
        cluster_of[t], class_counts[t] = k, Counter({3: 1})
    assign = stratified_cluster_split(cluster_of, class_counts)
    assert {assign[_tile(0, i)] for i in range(30)} == {"train"}
    report = split_report(assign, class_counts)
    assert report["tile_ratio"]["train"] >= 0.7
    assert report["tile_ratio"]["train"] == sum(1 for s in assign.values() if s == "train") / 100


def test_split_report_contents_and_json():
    cluster_of = {_tile(k, 0): k for k in range(4)}
    class_counts = {
        _tile(0, 0): Counter({3: 2, 1: 1}),
        _tile(1, 0): Counter({3: 1}),
        _tile(2, 0): Counter({3: 4}),
        _tile(3, 0): Counter({5: 1}),
    }
    assign = {_tile(0, 0): "train", _tile(1, 0): "train", _tile(2, 0): "test", _tile(3, 0): "train"}
    report = split_report(assign, class_counts)
    assert report["tiles"] == {"train": 3, "val": 0, "test": 1}
    assert report["tile_ratio"] == {"train": 0.75, "val": 0.0, "test": 0.25}
    assert report["class_instances"]["3"] == {"train": 3, "val": 0, "test": 4}
    assert report["class_instances"]["1"] == {"train": 1, "val": 0, "test": 0}
    assert report["classes_absent_from_some_set"] == [1, 3, 5]
    json.dumps(report)  # must be serializable as-is for stats.json
