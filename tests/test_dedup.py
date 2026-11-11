import numpy as np

from fpyolo.dedup import (
    build_clusters,
    choose_keeper,
    drawing_group,
    exact_duplicate_sets,
    near_duplicate_pairs,
    thumbnail,
)
from fpyolo.ink import ink_fraction


def _t(i):
    return ("train-00", f"0001-{i:04d}")


def test_exact_sets():
    tiles = [_t(i) for i in range(5)]
    md5 = dict(zip(tiles, ["a", "a", "b", "c", "c"]))
    assert exact_duplicate_sets(md5) == [[tiles[0], tiles[1]], [tiles[3], tiles[4]]]


def test_exact_sets_stable_regardless_of_insertion_order():
    tiles = [_t(i) for i in range(4)]
    md5 = {tiles[3]: "x", tiles[1]: "y", tiles[2]: "x", tiles[0]: "y"}
    assert exact_duplicate_sets(md5) == [[tiles[0], tiles[1]], [tiles[2], tiles[3]]]


def test_keeper_rule():
    a, b = ("train-00", "0001-0001"), ("train-00", "0002-0001")
    assert choose_keeper([a, b], {a: 5, b: 7}, {a: 99, b: 0}) == b
    assert choose_keeper([a, b], {a: 7, b: 7}, {a: 3, b: 4}) == b
    assert choose_keeper([b, a], {a: 7, b: 7}, {a: 4, b: 4}) == a


def test_drawing_group():
    assert drawing_group(("train-00", "0012-0010")) == ("A", "0012")
    assert drawing_group(("test-00", "0001-0023")) == ("B", "0001")
    assert drawing_group(("train-01", "0001-0087")) == ("B", "0001")


def test_thumbnail_unit_norm():
    rng = np.random.default_rng(0)
    ink = rng.integers(0, 256, size=(1000, 1000), dtype=np.uint8)
    v = thumbnail(ink)
    assert v.shape == (1024,) and v.dtype == np.float32
    assert abs(np.linalg.norm(v) - 1.0) < 1e-6


def test_thumbnail_accepts_all_tile_shapes_and_size_param():
    for shape in [(1000, 1000), (999, 1000), (1000, 999), (999, 999)]:
        ink = np.full(shape, 255, np.uint8)
        ink[100:200, 100:200] = 0
        assert thumbnail(ink).shape == (1024,)
    assert thumbnail(ink, size=8).shape == (64,)


def test_thumbnail_blank_is_zero_vector():
    v = thumbnail(np.full((1000, 1000), 255, np.uint8))
    assert v.shape == (1024,) and v.dtype == np.float32
    assert not v.any() and np.isfinite(v).all()


def test_thumbnail_same_drawing_similar_different_drawing_not():
    a = np.full((1000, 1000), 255, np.uint8)
    a[100:110, 50:900] = 0
    b = a.copy()
    b[500:502, 500:502] = 0  # tiny change
    c = np.full((1000, 1000), 255, np.uint8)
    c[600:900, 100:110] = 0
    ta, tb, tc = thumbnail(a), thumbnail(b), thumbnail(c)
    assert float(ta @ tb) > 0.95
    assert float(ta @ tc) < 0.5


def _unit(*xs):
    v = np.zeros(8, np.float32)
    v[: len(xs)] = xs
    return v / np.linalg.norm(v)


def test_pairs_threshold_and_blocking():
    thumbs = np.stack([
        _unit(1, 0),            # 0
        _unit(0, 1),            # 1
        _unit(1, 0),            # 2: identical to 0
        _unit(0.9, np.sqrt(1 - 0.81)),  # 3: cosine 0.9 with 0 -> below threshold
        _unit(0, 0, 1),         # 4
    ])
    valid = np.ones(5, bool)
    for block in (2, 3, 2048):
        assert near_duplicate_pairs(thumbs, valid, block=block) == [(0, 2)]
    # lowering the threshold exposes the 0.9 pair, still ordered i<j
    assert near_duplicate_pairs(thumbs, valid, threshold=0.85, block=2) == [(0, 2), (0, 3), (2, 3)]


def test_pairs_cross_block_pair_found_once():
    rng = np.random.default_rng(1)
    thumbs = rng.normal(size=(7, 16)).astype(np.float32)
    thumbs /= np.linalg.norm(thumbs, axis=1, keepdims=True)
    thumbs[6] = thumbs[0]
    assert near_duplicate_pairs(thumbs, np.ones(7, bool), block=3) == [(0, 6)]


def test_near_empty_tiles_never_linked():
    ink = np.full((1000, 1000), 255, np.uint8)
    ink[10, 10:110] = 0  # 100 px of ink -> fraction 1e-4
    assert ink_fraction(ink) < 0.01
    thumbs = np.stack([thumbnail(ink), thumbnail(ink)])
    valid = np.array([ink_fraction(ink) >= 0.01] * 2)
    assert not valid.any()
    assert near_duplicate_pairs(thumbs, valid) == []
    # sanity: the same vectors do link when marked valid
    assert near_duplicate_pairs(thumbs, np.ones(2, bool)) == [(0, 1)]


def test_pairs_only_valid_rows_participate():
    thumbs = np.stack([_unit(1), _unit(1), _unit(1)])
    assert near_duplicate_pairs(thumbs, np.array([True, False, True])) == [(0, 2)]


def test_clusters_union_groups():
    tiles = [
        ("train-00", "0001-0001"), ("train-00", "0001-0002"),  # g1 = A0001
        ("train-00", "0002-0001"),                             # g2 = A0002
        ("train-00", "0003-0001"),                             # g3 = A0003
        ("test-00", "0004-0001"), ("train-01", "0004-0002"),   # B0004 (two archives)
    ]
    cid = build_clusters(tiles, [(0, 2)])
    assert cid[tiles[0]] == cid[tiles[1]] == cid[tiles[2]]
    assert cid[tiles[3]] != cid[tiles[0]]
    assert cid[tiles[4]] == cid[tiles[5]] != cid[tiles[0]]
    assert sorted(set(cid.values())) == [0, 1, 2]
    assert set(cid) == set(tiles)


def test_clusters_numbering_stable_and_transitive():
    tiles = [("train-00", "0003-0001"), ("train-00", "0001-0001"),
             ("train-00", "0002-0001"), ("train-00", "0004-0001")]
    pairs = [(0, 2), (2, 3)]  # chain 3-2-4
    cid = build_clusters(tiles, pairs)
    assert cid[tiles[0]] == cid[tiles[2]] == cid[tiles[3]]
    assert cid[tiles[1]] == 0 and cid[tiles[0]] == 1
    assert build_clusters(tiles[::-1], [(3, 1), (1, 0)]) == cid
