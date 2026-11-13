# YOLO11n on FloorPlanCAD: detection (D) and wall segmentation (W), evaluation report

Date: 2026-11-12 · Branch `feat/fpyolo-v0` · Spec: `docs/specs/2026-11-10-floorplancad-yolo11n-design.md` (cited as "spec §x")

> **Status: final.** All 9 training runs (D-yolo11n, D-yolov8n, W-yolo11n-seg × seeds 0, 1, 2) and their test
> evaluations are finished; §6 holds the results (filled 2026-11-13). Every result number is copied from
> `reports/tables.md`, `results/*/metrics.json` or `results/compare.json`; none is estimated or invented.

## Contents

1. Summary
2. Method
3. Data issues found in FloorPlanCAD
4. Dedup, clustering and re-splitting
5. Final class table
6. Results
7. Measured mm/px scale
8. Label-noise ceiling
9. Literature context (not a fair comparison)
10. Decisions made during implementation
11. Known limitations
12. How to reproduce

---

## 1. Summary

- **Goal (spec §1–2).** A foundation model for the architectural part of a 2D-plan-to-3D pipeline: **D** detects
  the 30 FloorPlanCAD "thing" classes (doors, windows, furniture, sanitary/kitchen fixtures, airconditioner,
  stairs/elevators/escalators) with axis-aligned boxes; **W** segments walls as polygons. Both are YOLO11n
  fine-tunes; D is compared with a YOLOv8n baseline under the same protocol.
- **The official FloorPlanCAD split is not usable for honest numbers.** 5,323 of the 5,502 official test tiles
  come from drawings that also have tiles in the official train set, and 1,186 official test tiles have a
  byte-identical copy in train under another drawing number (§3). All data was therefore deduplicated and
  re-split by building cluster: **13,352 unique tiles, 1,108 clusters, 9,341 / 1,341 / 2,670 tiles in
  train / val / test** (§4).
- **The labels are noisy.** Byte-identical copies of the same image disagree: only 59.9% of copy sets have the
  same per-class thing counts, and 21.6% of thing instances are absent from at least one copy (§8).
- **Scale.** FloorPlanCAD tiles are drawn at a constant **10.0 mm/px** (33 of 40 sampled tiles internally
  consistent, IQR/median 0.0004) (§7). Wall-thickness error is therefore reported in mm.
- **D results (§6.1–6.4).** YOLO11n reaches **core mAP50-95 0.725 ± 0.003** (mAP50 0.790; all-30 0.659) over
  3 seeds on the test split. YOLOv8n reaches **0.725 ± 0.003**. Δ = +0.0004, 95% cluster-bootstrap CI
  [−0.0036, +0.0046]: **no real difference** under the pre-registered rule, so accuracy gives no reason to
  prefer either model. The 3 seeds measure training noise only, not split noise. Airconditioner:
  **AP50-95 0.695 ± 0.012, AP50 0.742**. The weakest core classes are bay_window (0.19, confused with window),
  escalator, opening_symbol, bath and bed.
- **W results (§6.5).** **Pixel IoU 0.699 ± 0.006** (class-agnostic; wall alone 0.71), with t* = 0.20 / 0.25 / 0.35
  per seed. Mask mAP50-95 is 0.225: long, connected wall instances are hard to match one-to-one, and pixel IoU is
  the headline by design. Wall-thickness median |error| is 13–14 mm, with a bias of −6 to −8 mm (thinner
  predictions), plausibly caused mostly by the ~1 px wider fillPoly labels.
- **Deliverables.** `runs/d11n_s1/weights/best.pt` (D) and `runs/w11n_s1/weights/best.pt` with t* = 0.25 (W).
  Inference takes about 3.2 ms/image (D) on an RTX A5000 (§6.6).
- **Errors (§6.7).** In 12 of 20 inspected test images, some "false positives" are real but unlabelled objects.
  W's main failure is missing wall pieces (9 of 20).

## 2. Method

### 2.1 Data source

`<FloorPlanCAD>` = `dataset/FloorPlanCAD/{train-00,train-01,test-00}` (read-only): 3,760 + 6,401 + 5,502 = **15,663
tiles**, each `NNNN-MMMM.svg` + `.png` (`NNNN` = source drawing, `MMMM` = crop index), plus the dataset authors'
`coco_vis/` renders. 15,151 PNGs are 1000×1000; 512 have a 999 px side (322 at 1000×999, 184 at 999×1000,
6 at 999×999) (counted from `width`/`height` in `data/fpcad_v0/manifests/tiles.csv`).

### 2.2 Image preprocessing (spec §6.3)

- Image = **ink** = `255 − alpha` of the RGBA PNG, saved as 3 identical channels. RGB is discarded so the
  model cannot learn CAD layer colours (which will not exist on project drawings). Ultralytics' `cv2.imread`
  would otherwise see coloured strokes on black.
- True image size kept (999 px images stay 999 px). Labels are normalised by the SVG `viewBox` size
  (`0 0 100 100`), never by 1000.
- SVG/PNG alignment was verified on the whole dataset: `align_frac` = 1.0 on all 15,228 tiles with labelled
  primitives, 0 misaligned (`stats.json` → `misaligned`).

### 2.3 Labels for D (spec §6.5)

- One object = one (`semantic-id` ∈ 3–32, `instance-id` ≥ 0) pair. Box = union of the exact geometric extents of
  all its primitives (lines, circular/elliptical arcs including their extrema, circles, rotated ellipses with
  their `transform` applied), in SVG coordinates, normalised by the viewBox.
- Box rules: a side < 2 px is padded evenly to 2 px (59 boxes); border-cut objects with < 30% of their unclipped
  box visible are dropped (never fired, see §11); no box is dropped for being small.
- Tiles without things are kept as negatives (3,875 tiles).
- 99,755 boxes in total: 69,692 / 10,141 / 19,922 in train / val / test; 8,646 (8.7%) smaller than 32 px
  (`stats.json` → `box_stats`).
- **Check against re-rasterisation** (spec §7.1): each instance rasterised separately at native resolution, its
  tight ink box compared with the analytic box on 500 random tiles: 4,128 of 4,128 eligible instances
  (both sides ≥ 4 px) reach IoU ≥ 0.9; minimum IoU 0.980 (`reports/qa/bbox_raster.json`).

### 2.4 Labels for W (spec §6.6)

- Classes: wall (`semantic-id` 1) always; curtain wall (2) and railing (33) were enabled after the closure check
  (both close into thin bodies with median coverage 1.00 on 1,843 and 2,376 kept tiles; `stats.json` →
  `w_candidates`, `reports/qa/README.md`). W YOLO indices: wall 0, curtain_wall 1, railing 2.
- Region construction (`src/fpyolo/walls.py`, thresholds in `configs/build.yaml` → `walls`): strokes of the
  class with exact arcs → close gaps ≤ 3 px (`gap_px`) → **end caps**: connect an open wall end to the facing
  stroke end ≤ 32 px away (`cap_px`, ruling R10, §10) → add the four image-border edges → polygonize →
  keep "thin" faces (largest inscribed circle radius ≤ 30 px, `max_thickness_px: 60`) and drop narrow
  border slivers → union.
- One instance = one connected wall body. Holes are encoded as one polygon with zero-width bridges (keyhole);
  re-rasterisation reaches IoU ≥ 0.99 (unit-tested).
- **Coverage** = share of the class's stroke length within 1.5 px of the generated region (boundary or
  interior; ruling R11). Tiles with W-class strokes and coverage < 0.80 are excluded from W in all three sets
  (still used for D). Exclusion: **1,266 of 12,220 = 10.4%** (train 10.4%, val 10.8%, test 9.9%), well under the
  30% stop line. Coverage quantiles: 10% → 0.79, 25% → 0.95, median 1.00 (`stats.json` → `w_coverage`).
- W dataset: 8,442 / 1,213 / 2,431 images (train / val / test, including wall-free negatives), 56,317 polygons.
  Because low-coverage tiles are excluded, **W's test set is biased toward drawings with clean, closed walls**
  (spec §6.6).

### 2.5 Training (spec §6.7; `configs/train_*.yaml`)

| Setting | Value |
|---|---|
| Models | D: `yolo11n.pt` and baseline `yolov8n.pt` (COCO-pretrained); W: `yolo11n-seg.pt` |
| Image size | `imgsz=1024` |
| Schedule | up to 300 epochs, `patience=50` |
| Optimizer | `SGD`, `lr0=0.01` (Ultralytics' SGD: momentum 0.937, Nesterov); `optimizer=auto` is refused by the wrapper (`docs/ultralytics_notes.md`) |
| Batch | 32 for all runs (identical for both D models). 1-epoch trial on an RTX A5000 24 GB: peak 12.2 GB (yolo11n), 11.6 GB (yolov8n), 13.6 GB (W), no OOM batch halving (`README.md`, GPU trial table) |
| Augmentation | Ultralytics defaults except `hsv_h=hsv_s=hsv_v=0`, `fliplr=0.5`, `flipud=0.5`, `degrees=0` |
| Seeds | 0, 1, 2 for each of the 3 configurations (9 runs), `deterministic: true` |
| Checkpoint | `best.pt` = Ultralytics' fitness on val (see §11 for what that means for W) |
| Software | Python 3.12, ultralytics 8.4.210, torch 2.15.0 (CUDA 13.2 build); each run records versions, GPU name and PCI bus id, schema version and the SHA-256 of `tiles.csv` in `env.json` |

`fpyolo.runstate verify` checks after every run that `optimizer: SGD` and that the **effective** batch stored in
`weights/last.pt` equals the requested one (Ultralytics silently halves the batch after a first-epoch OOM).

### 2.6 Evaluation (spec §6.8)

- **Test is used only once, for the final numbers.** Checkpoints, W's confidence threshold and every other choice
  come from val.
- **D**: COCO-style AP computed by the project's own evaluator (`src/fpyolo/apeval.py`, matches pycocotools to
  ~1e-16 on a fixture): IoU 0.50:0.05:0.95, 101-point interpolation, `max_det=300`, predictions at `conf=0.001`,
  `iou=0.7`. **Headline = mAP50-95 over the 27 core classes** fixed in `data/fpcad_v0/manifests/core_classes.json`
  at build time, before any training. The all-30-class number is always reported alongside, with absent classes
  listed in `missing_classes` (the denominator never changes silently). Ultralytics' own "all" mAP is not used:
  it averages only classes with labels and fills absent classes with the mean (`docs/ultralytics_notes.md`).
- Also: mAP50, mAP75, per-class AP with test instance counts (airconditioner reported separately), AP by size
  (COCO small/medium/large), confusion matrix and speed (ms/image on an RTX A5000) from Ultralytics' val.
- **"Real difference"** (YOLO11n − YOLOv8n): 95% CI from a bootstrap over test **clusters** (1,000 resamples, the
  same resample for both models) excludes 0, **and** |Δ| > 2 × pooled across-seed standard deviation
  (pooled = √((s_a² + s_b²)/2), sample std, ddof=1) (`src/fpyolo/compare.py`).
- **W**: main metric = pooled pixel IoU of the class-agnostic wall region (union of wall, curtain wall and railing)
  at native resolution (`retina_masks=True`), threshold `t*` chosen on val from 0.05…0.95 (step 0.05) and frozen
  for test. Also per-class pixel IoU at `t*`, mask mAP50-95 and the wall-thickness error in mm (§6.5).

## 3. Data issues found in FloorPlanCAD

All counts below were re-computed from `data/fpcad_v0/manifests/tiles.csv` (one row per original tile: archive,
stem, PNG md5, drawing group, cluster, new split) and agree with spec §5.3, which verified them independently on
2026-11-11.

### 3.1 The official split leaks by drawing

- `train-01` and `test-00` share one drawing numbering: **1,045 drawing prefixes appear in both**, with 0 shared
  file names, i.e. crops of the same drawing were divided between the two archives.
- **5,323 of the 5,502 official test tiles (96.7%)** belong to a drawing that also has tiles in `train-01`.
- Visually confirmed example (spec §5.3): `train-01/0001-0087` and `test-00/0001-0023` are the same apartment plan.
- After clustering (§4), **5,409 of 5,502 official test tiles (98.3%)** sit in a building cluster that also holds
  official-train tiles.
- `train-00` has its own numbering: 175 names collide with `test-00` and 203 with `train-01`, but with different
  content (different md5). Output files are therefore named `<archive>_<stem>`.

### 3.2 Byte-identical images under different drawing numbers

- **1,163 sets of byte-identical PNGs cover 3,474 tiles (22%)** (`stats.json` → `dedup`).
- Every set spans more than one drawing number, and in every set all copies have the same crop index `MMMM`
  (e.g. `train-00/0012-0010` and `train-00/0021-0010`). These are most likely repeated floors of one building
  exported as separate drawings.
- Set sizes: 723 pairs, 202 triples, 96 of 4, …, up to 15 copies.
- Archives spanned: 711 sets span `test-00` and `train-01`; 233 are inside `train-01`, 174 inside `test-00`,
  45 inside `train-00`. **No set spans `train-00` and the `train-01`/`test-00` group.**
- **1,186 official test tiles have an exact copy in official train** under another drawing number — a second,
  independent leak.
- Near-duplicates go further: at cosine ≥ 0.95 on ink thumbnails, 234 clusters merge more than one drawing
  group, covering 4,297 of the 13,352 kept tiles (§4.2).

### 3.3 Identical images, different labels

Within the 1,163 identical sets (`stats.json` → `dedup`, computed in `src/fpyolo/build.py`):

| Measure | Value |
|---|---|
| Sets whose copies have the same per-class thing counts | 59.9% |
| Sets whose copies have the same number of wall strokes | 45.6% |
| Thing instances absent from at least one copy (Σ(max − min copy count) / Σ max, per set and class) | 21.6% |

The pixels are identical, so the difference is annotation: one annotator labelled objects another missed.
This caps achievable scores (§8).

### 3.4 Other data facts

- **Class numbering differs from SymPoint's `SVG_CATEGORIES`.** In the release, IDs 3–32 are things and
  {1, 2, 33, 34, 35} are area classes (1 = wall, not 33). Names were fixed from the authors' own `coco_vis`
  labels (§5, `reports/class_names.md`). The Voxel51/HF name list does not match `coco_vis` and was not used.
- **SVGs are pre-clipped to the tile.** 0 of 76,753 primitives in a 300-tile sample extend outside the
  viewBox; about 14% of objects touch the border. The unclipped size of a border-cut object is unknown, so the
  30%-visible rule cannot fire (`reports/qa/README.md`, `box_stats.dropped_cut = 0`).
- **Columns** are mostly unlabelled; some carry the wall ID (spec §5.6). Not corrected.
- **Seat rows (ID 34) and parking spots (ID 35)** are not trained (spec §3). Their symbols remain in the images
  (about 319 and 1,323 tiles, spec §3) and are background for D.
- **Possible missed labels seen in QA** (`reports/qa/README.md`): long glazing strips in
  `test-00_1117-0091` without D boxes (probably curtain walls, a W class); one small window-like symbol
  unlabelled in `test-00_0257-0622`.

## 4. Dedup, clustering and re-splitting

### 4.1 Step 1: exact duplicates

- Key: md5 of the PNG bytes. Each set keeps **one** tile: the one with the most thing instances, then the most
  wall strokes, then the smallest `(archive, stem)`. Rationale: missing labels are far more common than extra
  ones (§3.3), and keeping several copies would teach conflicting labels and count one image several times in test.
- Result: 15,663 − (3,474 − 1,163) = **13,352 kept, 2,311 dropped**; each dropped row in `tiles.csv` has
  `drop_reason = exact_duplicate_of:<kept image id>`.

### 4.2 Step 2: building clusters

- Initial groups: `("A", NNNN)` for `train-00`, `("B", NNNN)` for `train-01` ∪ `test-00` (shared numbering):
  1,800 drawing groups.
- Near-duplicate measure: cosine similarity of **32×32 ink-density thumbnails**; pairs with cosine ≥ **0.95**
  are linked and groups merged by union-find. Tiles with ink fraction < 0.01 are never used for linking
  (near-empty tiles all look alike and would chain unrelated drawings): 15,610 of 15,663 tiles are valid for linking.
- 9,166 linking pairs, 9,070 of them between different groups, 10 across the two numbering spaces
  (`stats.json` → `clusters`).
- **Result: 1,108 clusters; largest 187 tiles (1.4% of kept tiles); the most-merged cluster joins 40 drawing
  groups.** Top-10 sizes 187, 173, 172, 122, 122, 115, 111, 110, 110, 104. (The design review had estimated
  ~1,153 clusters with a largest of 450.)
- Example linked pairs (cosine): `test-00_0228-0017` ↔ `train-01_0223-0017` (0.996),
  `test-00_1121-0029` ↔ `train-01_0975-0029` (1.000), `train-00_0477-0005` ↔ `train-00_0478-0005` (0.995).
- **`train-00` vs the rest** (spec §6.4), checked visually (`reports/qa/README.md`, `reports/qa/cross_space*/`):
  - the 10 automatic cross-space links all join `A/0121` with `B/0543`, the same floor plan with a few-pixel
    shift and a constant crop-index offset: correct merges;
  - the 30 strongest sub-threshold candidates (cosine 0.873–0.935) were all **different buildings**: four
    `train-00` tiles covered almost entirely by uniform floor hatching match unrelated hatch-dominated tiles.
    0 manual merges (`manual_merges: []`).
- Limitation: hatch-dominated tiles can also link unrelated groups *within* a numbering space. That only makes
  clusters larger (no leakage), and the largest stays at 1.4%. Shifted crops of a repeated floor with cosine
  < 0.95 are not caught (spec §9).

### 4.3 Step 3: stratified 70/10/20 split by cluster

- All tiles of a cluster go to the same set; seed 0. Cluster-level greedy that minimises a fill loss over
  per-class instance quotas and the tile quota (ruling R12, §10).
- Result (`stats.json` → `split_report`):

| Split | Tiles | Share | Clusters | Boxes (D) | W images |
|---|---|---|---|---|---|
| train | 9,341 | 70.0% | 952 | 69,692 | 8,442 |
| val | 1,341 | 10.0% | 40 | 10,141 | 1,213 |
| test | 2,670 | 20.0% | 116 | 19,922 | 2,431 |

- **Every class appears in all three sets** (`classes_absent_from_some_set: []`); IDs 7 and 8 have exactly one
  instance each in val and in test.
- Where the official test tiles went: of the 4,528 kept `test-00` tiles, 3,058 are now in train, 554 in val and
  916 in test. The new test set mixes all three archives (916 `test-00`, 1,159 `train-01`, 595 `train-00`).
- The split files ship with the code: `data/fpcad_v0/manifests/tiles.csv` (committed with `git add -f`).
- **Val has only 40 clusters.** Checkpoint and threshold selection on val is therefore noisier than the tile
  count suggests.

## 5. Final class table

Source of truth: `configs/classes_v0.yaml` (version `v0`). `schema_id` is stable and never reused; future HVAC
classes get new IDs. Names are the dataset authors' `coco_vis` labels (all 30 confirmed on 5–12 viewed instances
each, none uncertain; evidence and CAD layer names in `reports/class_names.md`, contact sheets in
`reports/qa/classes/sid_XX.png`). Instance counts are after dedup (`stats.json` → `split_report.class_instances`).
Core = ≥ 50 train and ≥ 15 test instances (`core_classes.json`, written before training).

| schema_id | semantic-id | name | group | model / YOLO idx | train | val | test | core |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | single_door | door | D / 0 | 13,414 | 1,663 | 3,791 | yes |
| 2 | 4 | double_door | door | D / 1 | 6,943 | 1,285 | 2,155 | yes |
| 3 | 5 | sliding_door | door | D / 2 | 2,975 | 519 | 897 | yes |
| 4 | 6 | folding_door | door | D / 3 | 49 | 9 | 19 | **no** (49 train) |
| 5 | 7 | revolving_door | door | D / 4 | 5 | 1 | 1 | **no** |
| 6 | 8 | rolling_door | door | D / 5 | 6 | 1 | 1 | **no** |
| 7 | 9 | window | window | D / 6 | 8,737 | 1,421 | 2,610 | yes |
| 8 | 10 | bay_window | window | D / 7 | 282 | 47 | 102 | yes |
| 9 | 11 | blind_window | window | D / 8 | 690 | 84 | 193 | yes |
| 10 | 12 | opening_symbol | window | D / 9 | 677 | 91 | 195 | yes |
| 11 | 13 | sofa | furniture | D / 10 | 780 | 111 | 246 | yes |
| 12 | 14 | bed | furniture | D / 11 | 1,772 | 286 | 496 | yes |
| 13 | 15 | chair | furniture | D / 12 | 5,592 | 652 | 1,455 | yes |
| 14 | 16 | table | furniture | D / 13 | 2,824 | 379 | 732 | yes |
| 15 | 17 | tv_cabinet | furniture | D / 14 | 617 | 71 | 193 | yes |
| 16 | 18 | wardrobe | furniture | D / 15 | 847 | 98 | 218 | yes |
| 17 | 19 | cabinet | furniture | D / 16 | 3,569 | 454 | 1,026 | yes |
| 18 | 20 | refrigerator | sanitary-kitchen | D / 17 | 666 | 82 | 192 | yes |
| **19** | **21** | **airconditioner** | **appliance-hvac** | **D / 18** | **1,520** | **252** | **448** | **yes** |
| 20 | 22 | gas_stove | sanitary-kitchen | D / 19 | 820 | 120 | 234 | yes |
| 21 | 23 | sink | sanitary-kitchen | D / 20 | 4,590 | 780 | 1,210 | yes |
| 22 | 24 | bath | sanitary-kitchen | D / 21 | 987 | 140 | 277 | yes |
| 23 | 25 | bath_tub | sanitary-kitchen | D / 22 | 340 | 59 | 105 | yes |
| 24 | 26 | washing_machine | sanitary-kitchen | D / 23 | 691 | 116 | 206 | yes |
| 25 | 27 | squat_toilet | sanitary-kitchen | D / 24 | 1,536 | 166 | 434 | yes |
| 26 | 28 | urinal | sanitary-kitchen | D / 25 | 796 | 141 | 237 | yes |
| 27 | 29 | toilet | sanitary-kitchen | D / 26 | 2,695 | 395 | 723 | yes |
| 28 | 30 | stairs | vertical-circulation | D / 27 | 2,985 | 348 | 850 | yes |
| 29 | 31 | elevator | vertical-circulation | D / 28 | 2,070 | 333 | 589 | yes |
| 30 | 32 | escalator | vertical-circulation | D / 29 | 217 | 37 | 87 | yes |
| 31 | 1 | wall | wall | W / 0 | 8,170 tiles¹ | 1,145 tiles¹ | 2,333 tiles¹ | – |
| 32 | 2 | curtain_wall | wall | W / 1 | 1,843 kept tiles (all sets) | | | – |
| 33 | 33 | railing | wall | W / 2 | 2,376 kept tiles (all sets) | | | – |

¹ Kept tiles with wall (ID 1) strokes, counted **before** the W coverage exclusion. They are not the W dataset
sizes: W train has 8,442 images = the train tiles that pass the coverage rule (or have no W-class strokes at
all), so it includes wall-free negatives and leaves out the 899 low-coverage train tiles (§2.4).

Notes:
- The release order is SymPoint's thing order shifted by 2, **except** IDs 20–23 (refrigerator, airconditioner,
  gas stove, sink). Every name was read from `coco_vis`, not inferred from the order.
- "bath" (ID 24) is drawn as shower heads / trays; "bath_tub" (ID 25) is the tub. The authors' names are kept.
- **Airconditioner (schema_id 19) is the only HVAC-related class** and is reported separately (spec §5.4).
- All 30 thing classes are trained, including the three non-core ones, so the table stays stable for extension.
- Seat row (34) and parking spot (35) have no schema_id in v0.

## 6. Results

All tables in §6.1–6.5 are copied verbatim from `reports/tables.md`. That file is generated by
`python -m fpyolo.report --results results --manifests data/fpcad_v0/manifests --out reports/tables.md`
from `results/` after the evaluation commands in §12. Every number is on the **test** split, which was used once.
Per-seed values come from `results/<run>/metrics.json` and `results/compare.json`.

**Training length** (`runs/*/results.csv`, `patience=50`, at most 300 epochs):

| Run | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| D YOLO11n: epochs trained (best val epoch) | 188 (138) | 204 (154) | 146 (96) |
| D YOLOv8n: epochs trained (best val epoch) | 159 (109) | 281 (231) | 152 (102) |
| W YOLO11n-seg: epochs trained (best val epoch) | 300 (298) | 300 (293) | 300 (295) |

Every D run stopped early, 50 epochs after its best val fitness. No W run stopped early, and every W `best.pt`
comes from epochs 293–298 of 300. W was probably still improving when the schedule ended. Whether longer training
helps W was not tested.

**Deliverable checkpoints** (README convention: the seed whose test headline is closest to its 3-seed mean):
- **D: `runs/d11n_s1/weights/best.pt`**: core mAP50-95 0.7260 vs mean 0.7253;
- **W: `runs/w11n_s1/weights/best.pt`**, `t* = 0.25`: pixel IoU 0.7009 vs mean 0.6985.

### 6.1 D headline (test split, mean ± sample std over 3 seeds)

| Model | Seeds | mAP50-95 (core, headline) | mAP50 (core) | mAP75 (core) | mAP50-95 (all 30) | mAP50 (all 30) | mAP75 (all 30) | Precision (Ultralytics averaging) | Recall (Ultralytics averaging) | Val speed ms/img (pre+inf+post) |
|---|---|---|---|---|---|---|---|---|---|---|
| YOLO11n | 3/3 | 0.725 ± 0.003 | 0.790 ± 0.002 | 0.750 ± 0.002 | 0.659 ± 0.003 | 0.719 ± 0.002 | 0.682 ± 0.002 | 0.819 ± 0.009 | 0.721 ± 0.005 | 7.5 ± 0.4 |
| YOLOv8n | 3/3 | 0.725 ± 0.003 | 0.788 ± 0.002 | 0.748 ± 0.003 | 0.660 ± 0.004 | 0.719 ± 0.001 | 0.682 ± 0.005 | 0.810 ± 0.009 | 0.730 ± 0.018 | 7.2 ± 0.2 |

Mean ± sample standard deviation (ddof=1) over seeds. "core" = classes with enough train and test instances (core_classes.json); "all 30" averages over the classes that have test labels. Precision and recall are Ultralytics' val numbers (mean over the classes with test labels, at Ultralytics' own operating point), not the fixed class sets of the AP columns.

Per-seed headline (core mAP50-95): YOLO11n 0.7273 / 0.7260 / 0.7225; YOLOv8n 0.7258 / 0.7276 / 0.7213
(seeds 0 / 1 / 2). `missing_classes` is empty for all six runs, so the all-30 number really averages
30 classes. That number sits about 6.5 points below the core headline, mainly because of folding_door and the
two one-instance classes (§6.3).

### 6.2 YOLO11n vs YOLOv8n

| Δ mAP50-95 (core), A − B | 95% CI (cluster bootstrap) | Pooled seed std | CI excludes 0 | Δ exceeds 2 × pooled std (absolute) | Real difference | Degenerate resamples |
|---|---|---|---|---|---|---|
| +0.000 | [-0.004, +0.005] | 0.003 | no | no | no | 0 of 1000 |

A = d11n_s0, d11n_s1, d11n_s2; B = d8n_s0, d8n_s1, d8n_s2. Resampling unit: drawing cluster of the test tiles (1000 resamples, seed 0). Real = the CI excludes 0 and |Δ| exceeds twice the pooled across-seed standard deviation.

At full precision (`results/compare.json`): Δ = **+0.0004**, 95% CI **[−0.0036, +0.0046]**, pooled std 0.0029.
YOLO11n scores 0.725 ± 0.003 and YOLOv8n 0.725 ± 0.003.

- **There is no real difference under the pre-registered rule.** The CI includes 0, and |Δ| = 0.0004 is far below
  2 × pooled std = 0.0058. The seed-to-seed spread inside each model (0.0048 and 0.0063 between the best and
  worst seeds) is larger than the gap between the two models.
- **What this means for choosing a model.** This protocol gives no accuracy reason to prefer YOLO11n over YOLOv8n,
  or the reverse. The CI half-width is about 0.004, so on this test set any true gap is probably smaller than
  about half a point of mAP50-95. The choice can rest on other criteria, such as deployment tooling or
  speed on the target hardware. Speed was not benchmarked rigorously here (§6.6). YOLO11n stays the
  deliverable as planned, but the result does not support a claim that it is the more accurate detector. This
  supports the background survey's conclusion that data, rendering and split matter more than the YOLO version
  (§9), here on clean, leak-free data.
- **Limitation.** The 3 seeds measure **training noise only**. Both models were trained and tested on one cluster
  split, so the across-seed std says nothing about how much a different split would move the numbers. The
  cluster bootstrap covers sampling of the test set, not of the training set (§11).

### 6.3 Per-class analysis

| # | Class | Test n_gt | AP50-95 YOLO11n | AP50 YOLO11n | AP50-95 YOLOv8n | AP50 YOLOv8n | Flags |
|---|---|---|---|---|---|---|---|
| 1 | double_door | 2155 | 0.928 | 0.965 | 0.927 | 0.963 |  |
| 2 | tv_cabinet | 193 | 0.909 | 0.955 | 0.892 | 0.935 |  |
| 3 | elevator | 589 | 0.909 | 0.944 | 0.899 | 0.938 |  |
| 4 | toilet | 723 | 0.900 | 0.940 | 0.898 | 0.938 |  |
| 5 | wardrobe | 218 | 0.897 | 0.920 | 0.889 | 0.913 |  |
| 6 | bath_tub | 105 | 0.891 | 0.959 | 0.876 | 0.951 |  |
| 7 | single_door | 3791 | 0.888 | 0.935 | 0.889 | 0.936 |  |
| 8 | gas_stove | 234 | 0.884 | 0.936 | 0.895 | 0.943 |  |
| 9 | urinal | 237 | 0.861 | 0.954 | 0.875 | 0.957 |  |
| 10 | squat_toilet | 434 | 0.854 | 0.942 | 0.873 | 0.957 |  |
| 11 | sofa | 246 | 0.818 | 0.870 | 0.822 | 0.875 |  |
| 12 | sink | 1210 | 0.782 | 0.867 | 0.793 | 0.881 |  |
| 13 | refrigerator | 192 | 0.767 | 0.826 | 0.780 | 0.825 |  |
| 14 | table | 732 | 0.760 | 0.807 | 0.760 | 0.805 |  |
| 15 | blind_window | 193 | 0.756 | 0.835 | 0.762 | 0.837 |  |
| 16 | cabinet | 1026 | 0.738 | 0.780 | 0.736 | 0.779 |  |
| 17 | window | 2610 | 0.703 | 0.771 | 0.712 | 0.777 |  |
| **18** | **airconditioner** | **448** | **0.695** | **0.742** | **0.691** | **0.739** |  |
| 19 | washing_machine | 206 | 0.692 | 0.711 | 0.708 | 0.729 |  |
| 20 | sliding_door | 897 | 0.669 | 0.827 | 0.701 | 0.845 |  |
| 21 | chair | 1455 | 0.659 | 0.737 | 0.658 | 0.724 |  |
| 22 | stairs | 850 | 0.635 | 0.729 | 0.618 | 0.716 |  |
| 23 | bed | 496 | 0.558 | 0.562 | 0.525 | 0.544 |  |
| 24 | bath | 277 | 0.517 | 0.592 | 0.514 | 0.587 |  |
| 25 | opening_symbol | 195 | 0.364 | 0.485 | 0.369 | 0.500 |  |
| 26 | escalator | 87 | 0.353 | 0.524 | 0.317 | 0.480 |  |
| 27 | folding_door | 19 | 0.195 | 0.245 | 0.230 | 0.301 | not a core class |
| 28 | bay_window | 102 | 0.194 | 0.213 | 0.193 | 0.210 |  |
| 29 | revolving_door | 1 | 0.000 | 0.000 | 0.000 | 0.000 | not a core class |
| 30 | rolling_door | 1 | 0.000 | 0.000 | 0.000 | 0.000 | not a core class |

Mean over seeds (YOLO11n: 3, YOLOv8n: 3 run(s)). Sorted by YOLO11n AP50-95, descending; classes without test labels have no AP (n/a) and come last. The airconditioner row is in bold.

(`#` is the rank, not the schema_id. Train and test counts below come from the class table in §5. Confusions were
read from Ultralytics' normalised confusion matrix of the deliverable run,
`results/d11n_s1/confusion_matrix_normalized.png`. That matrix uses Ultralytics' own confidence and IoU
settings, so it is only qualitative.)

- **Best classes.** double_door (0.928), tv_cabinet (0.909), elevator (0.909), toilet (0.900), wardrobe (0.897).
  These symbols are distinctive and drawn in a standard way. Being rare does not by itself mean scoring low:
  tv_cabinet (617 train / 193 test) and bath_tub (340 / 105) are among the best classes.
- **Airconditioner** (schema_id 19, 1,520 train / 448 test, the only HVAC class). YOLO11n reaches AP50-95
  **0.695 ± 0.012** and AP50 **0.742 ± 0.013**; YOLOv8n reaches 0.691 ± 0.014 and 0.739 ± 0.001 (mean ± std over
  seeds, from `per_class` in `metrics.json`). It ranks 18th of 30, just below window. AP50 is only about
  5 points above AP50-95, so the losses are mostly missed or unmatched instances, not loose boxes. In the
  confusion matrix most true airconditioners are predicted correctly. The remaining mass goes mainly to
  background, with small leaks to cabinet. A qualitative example: an X-marked equipment box labelled
  airconditioner was predicted as cabinet 0.65 (`test_train-00_0459-0004.png`, §6.7).
- **Weakest core classes**, with their train / test counts (§5):
  - **bay_window** (282 / 102): 0.194. This is the weakest core class. In the confusion matrix, true bay windows
    are predicted as `window` about as often as `bay_window`: the class is both rare and drawn much like the
    frequent window class (8,737 train).
  - **escalator** (217 / 87): 0.353, with a large gap to AP50 (0.524). The class is rare, and its large, often
    border-cut boxes have no clear extent; the stairs show the same problem (0.635).
  - **opening_symbol** (677 / 195): 0.364. This is a small symbol; in the confusion matrix, part of the true opening
    symbols are predicted as single_door or double_door.
  - **bath** (987 / 277): 0.517. This class is not rare. "bath" is the shower area (§5), visually weak, and a
    sizeable share goes to background, with some confusion with bath_tub.
  - **bed** (1,772 / 496): 0.558. AP50 (0.562) is almost equal to AP50-95, so located beds are located tightly
    and the loss is beds not found at all (bed → background in the confusion matrix). This is a frequent class
    with a high miss rate. Plausible causes are heavy hatching and pillows/linen drawn inside the bed, and
    missing bed labels in training (§8); the cause was not tested.
  - **sliding_door** (2,975 / 897) has a large gap between AP50 0.827 and AP50-95 0.669. Its very thin
    boxes lose IoU fast when they are off by a pixel.
  So, of the weak classes, bay_window and escalator are explained by low counts plus look-alike classes. Bath, bed
  and opening_symbol are frequent enough, so their weakness is about appearance, class definition or
  labelling, not data quantity.
- **Non-core classes 6, 7 and 8** (semantic ids; folding_door 49 train / 19 test, revolving_door 5 / 1,
  rolling_door 6 / 1) are reported but not interpreted. revolving_door and rolling_door each have
  **one** test instance, and both get AP 0.000 in every run (in the confusion matrix the revolving door is
  predicted as single_door). With one instance, each class moves the all-30 mean by up to 1/30 = 3.3 points.
  folding_door (0.195 vs 0.230) rests on 19 instances. These three classes explain the gap between the core and the all-30 mean:
  (27 × 0.725 + 0.195 + 0 + 0) / 30 = 0.659.

### 6.4 AP by object size

| Model | small (< 32² px) | medium (32²–96² px) | large (> 96² px) |
|---|---|---|---|
| YOLO11n | 0.302 ± 0.012 | 0.669 ± 0.008 | 0.486 ± 0.005 |
| YOLOv8n | 0.305 ± 0.006 | 0.675 ± 0.008 | 0.491 ± 0.008 |

Box area in image pixels, COCO limits; mean ± std over seeds.

- 8.7% of all boxes are < 32 px (`box_stats.small_lt_32px` = 8,646 of 99,755). As expected, mAP50-95 on them is
  low (0.30): a box that is 1 px loose loses IoU fast at this size.
- **Large objects score lower than medium ones** (0.49 vs 0.67). Large boxes are dominated by stairs, escalators,
  beds and elevators, which are often cut by the tile border and have an ambiguous extent (see the stairs in
  `test_train-00_0390-0100.png`). This explanation is plausible but was not tested per class.
- The two models do not differ in any size bucket.

### 6.5 W: wall segmentation (YOLO11n-seg)

| Metric | Mean ± std over seeds (3/3 runs) |
|---|---|
| Pixel IoU (pooled, class-agnostic, at t*) | 0.699 ± 0.006 |
| t* (chosen on val) | 0.27 ± 0.08 |
| Val pixel IoU at t* | 0.698 ± 0.003 |
| Mask mAP50-95 | 0.225 ± 0.003 |
| Mask mAP50 | 0.370 ± 0.004 |
| Pixel IoU: wall | 0.714 ± 0.005 |
| Pixel IoU: curtain_wall | 0.280 ± 0.022 |
| Pixel IoU: railing | 0.176 ± 0.028 |
| Wall-thickness error: MAE (mm) | 26.97 ± 0.40 |
| Wall-thickness error: median |error| (mm) | 13.54 ± 0.69 |
| Wall-thickness error: bias, prediction − label (mm) | -6.83 ± 1.38 |
| Images in the thickness comparison | 2084 ± 30 |

Wall thickness: 2*area/perimeter of the class-agnostic mask; area = pixel count, perimeter = total cv2.arcLength of all external and hole contours (RETR_CCOMP, pixel-centre contours); exact for an infinitely long straight bar, a few percent low for a finite one; a wall cut by the tile border counts as a bar end (the border edge adds to the perimeter), which biases the thickness low for both the label and the prediction. Scale: 10 mm/px (constant). Only test images with a non-empty label mask and a non-empty prediction (score ≥ t*) count.

Per-class pixel IoU is n/a for a class without label pixels in test. Mask mAP is Ultralytics' own averaging (classes with labels only).

Per seed (`results/w11n_s{0,1,2}/metrics.json`; 2,431 test images each):

| Seed | t* (val) | Val IoU at t* | Test pixel IoU | Mask mAP50-95 | IoU wall / curtain_wall / railing | Thickness median abs error / bias (mm) | n images (thickness) |
|---|---|---|---|---|---|---|---|
| 0 | 0.20 | 0.701 | 0.703 | 0.222 | 0.718 / 0.298 / 0.198 | 13.7 / −8.4 | 2,109 |
| 1 | 0.25 | 0.696 | 0.701 | 0.225 | 0.716 / 0.286 / 0.186 | 14.2 / −6.1 | 2,092 |
| 2 | 0.35 | 0.697 | 0.691 | 0.228 | 0.708 / 0.256 / 0.144 | 12.8 / −6.0 | 2,051 |

- **Pixel IoU is about 0.70 on every seed** (0.691–0.703). The val IoU curve is flat near its maximum: within
  ±0.05 of t* it changes by less than 0.01. So the different t* values (0.20, 0.25, 0.35) matter little, and val
  and test IoU agree within 0.006.
- **Wall alone reaches 0.71. Curtain wall (0.28) and railing (0.18) are much weaker.** Both are thin, often
  single- or double-line strokes that are easy to confuse with windows and plain lines. They are also rarer
  (1,843 and 2,376 kept tiles). The headline is class-agnostic, so a curtain wall predicted as wall is not
  penalised in it.
- **Why mask mAP is low (~0.22) while pixel IoU is ~0.70.** One W instance is one *connected wall body* (§2.4),
  and a single instance often spans the whole tile. Mask AP needs a one-to-one match at IoU ≥ 0.5…0.95 per
  instance. If one long, connected body is predicted in two pieces, or two bodies are predicted as one, the
  prediction is charged a false positive and a missed instance, even when almost every pixel is right. Pixel IoU
  ignores how the region is split into instances, which is what the 3D pipeline needs (wall area and thickness).
  That is why the spec makes pooled pixel IoU the W headline. Mask mAP is reported for completeness and should not
  be read as segmentation quality.
- **Wall-thickness error.** Median |error| is **13–14 mm** (12.8–14.2 per seed, about 1.3–1.4 px at 10 mm/px), and
  the MAE is ~27 mm. The **bias is −6 to −8 mm** (−6.0 to −8.4): predicted walls are slightly thinner than the
  labels. The GT masks are rasterised with `cv2.fillPoly`, which also fills the pixels the boundary passes
  through. The GT walls are therefore about **1 px = 10 mm wider** than the analytic region (§11). So most of the
  bias **plausibly** comes from the label rasterisation, not from the model. This is not proven: the predictions
  were trained on the same fillPoly masks and could have learned the extra pixel, and no analytic-width
  comparison was run. The MAE is twice the median, so a minority of images have large errors. The qualitative
  examples suggest these come from partly missed or partly over-filled walls (§6.7), not from edge placement.
- Read next to these numbers:
  - W's test set excludes the 9.9% of test tiles with wall coverage < 0.80, so it is biased toward clean walls
    (§2.4). On the excluded tiles, IoU would likely be lower.
  - The W labels contain known noise: shafts are filled as wall (§10.4), columns are mostly unlabelled
    (§3.4), and wall-stroke counts agree in only 45.6% of identical copies (§8). Part of the gap to 1.0 is label
    disagreement, not model error.
  - Training ran the full 300 epochs, and `best.pt` came from epochs 293–298 (§6, training length). W may be
    undertrained.

### 6.6 Speed

Ultralytics' val timings (`ultralytics_val.speed_ms` in `metrics.json`, `batch=16`, `imgsz=1024`, one RTX A5000),
mean over 3 seeds, ms per image:

| Model | Preprocess | Inference | Postprocess | Total |
|---|---|---|---|---|
| D YOLO11n | 2.88 | **3.20** | 1.39 | 7.5 |
| D YOLOv8n | 2.91 | **2.96** | 1.38 | 7.2 |
| W YOLO11n-seg | 2.81 | **3.97** | 1.69 | 8.5 |

- Inference takes **about 3.2 ms per image for D YOLO11n** (3.0 ms for YOLOv8n, 4.0 ms for W). The machine was
  shared with other users' jobs. One W seed ran at about half the pre/postprocess time of the other two, which
  shows how much load affects these numbers. The 0.24 ms gap between the two D models is not a reliable speed
  comparison.
- These are batched val timings, not end-to-end deployment latency.
- Full evaluation cost (`results/eval_*.out`, `results/*.log` with GNU `time`): `evaluate_d` takes 135–158 s
  of wall time for the 2,670 test images, and `evaluate_w` takes 177–292 s (val threshold sweep plus test).
  Peak RSS is **about 2.1–2.5 GiB per evaluation** (2,149,932–2,611,356 KiB maximum resident set size, logged for
  d11n_s0 and the three W runs).

### 6.7 Qualitative examples and error categories

20 random W-test images (`reports/qualitative/*.png`; produced by `python -m fpyolo.qa pred-overlays` with the
deliverable checkpoints `runs/d11n_s1/weights/best.pt` and `runs/w11n_s1/weights/best.pt`,
`--w-conf 0.25` (= t* of w11n_s1), `--split test --n 20`, default sampling seed 0). Left: ground truth (D boxes,
W polygons). Right: D boxes with score ≥ 0.25 and W masks with score ≥ t*. W colours: green = wall,
yellow = curtain_wall, pink = railing. The image notes below come from visual inspection, not from a
measurement. "Unlabelled" means no GT box or polygon is drawn for a symbol that looks like a real object.

| Image | Errors seen |
|---|---|
| `test_test-00_0101-0020.png` | D: the 4 GT objects are found; elevator 0.90 and sliding_door 0.31 at the bottom sit on unlabelled symbols (missing labels). W: top-left L-shaped wall and parts of the left wall missed (under-segmentation). |
| `test_test-00_0257-0453.png` | No D objects. W: both walls stop ~110 px before the lower border, and the right wall is ragged (under-segmentation). |
| `test_test-00_0346-1055.png` | D: single_door 0.89 on an unlabelled, border-cut door at the top (missing label). W: wall predicted on an unhatched double-line strip and a horizontal stub that GT leaves empty (over-segmentation, possibly unlabelled walls). |
| `test_test-00_0613-0038.png` | D: 3/3 doors correct. W: thin double-line partitions that GT leaves empty are predicted as curtain_wall (over-segmentation); the top GT wall is partly predicted as curtain_wall (class swap inside W, harmless for the class-agnostic IoU); the X-marked shaft at the bottom right is partly filled. |
| `test_test-00_0860-0061.png` | No GT at all. W predicts a strip on the double line annotated 女儿墙 (parapet): over-segmentation against GT, plausibly a real but unlabelled wall. |
| `test_test-00_1159-0026.png` | D: all ~20 doors and sinks found with tight boxes. W: two short wall pieces in the middle partly missed (minor under-segmentation). |
| `test_train-00_0389-0173.png` | Essentially clean: 4/4 doors; walls match except a small spill at the bottom corner. |
| `test_train-00_0390-0100.png` | D: stairs predicted as two overlapping, oversized boxes (0.50, 0.26); the GT box covers only part of the flight (loose box; the stairs extent is ambiguous). |
| `test_train-00_0459-0004.png` | D: an X-marked equipment box labelled airconditioner predicted as cabinet 0.65 (wrong class); ~20 other objects correct. W: two railings missed (under-segmentation). |
| `test_train-00_0514-0013.png` | D: GT bath (shower area) and the living-room table missed (missed, not small); a small bathroom fixture predicted as gas_stove 0.29 (wrong class, unlabelled object); sink 0.88 on an unlabelled kitchen sink and cabinet on an unlabelled built-in wardrobe (missing labels). W: good. |
| `test_train-01_0175-0028.png` | D: one GT window split into two predicted windows (0.48, 0.4x) (split/loose box); window 0.26 on an unlabelled window strip at the bottom (missing label). W: lower wall has holes, the bottom railing is missed (under-segmentation). |
| `test_train-01_0297-2351.png` | D: doors and stairs correct; overlapping single/double door boxes on an unlabelled, border-cut door (missing label). W: the whole left wall structure is missed (large under-segmentation). |
| `test_train-01_0297-2381.png` | D: stairs 0.85 on unlabelled stair lines (missing label). W: only the left half of a solid-filled wall and none of the three thin side walls predicted (under-segmentation). |
| `test_train-01_0346-0728.png` | D: two border-cut doors without labels predicted as double_door 0.75/0.74 (missing labels; the arcs look like single doors) and stairs 0.83/0.42 on unlabelled stairs (missing label). W: good. |
| `test_train-01_0478-0801.png` | D: 5/5 correct; double_door 0.61 on a border sliver (missing label, see §10.3). W: an unlabelled column square filled (over-segmentation by the column convention, §3.4). |
| `test_train-01_0650-0074.png` | D: an unusual door symbol (labelled single_door) missed. W: the GT wall strip is missed entirely (under-segmentation); GT itself labels only the lower part of a continuous double line. |
| `test_train-01_0719-0002.png` | D: window and both airconditioners correct; chair on a border sliver (missing label); window 0.26 on a wall piece (wrong class). W: small enclosed rectangles above and below the GT wall strip filled as wall. |
| `test_train-01_0875-0013.png` | D: dense plan, most of ~50 objects right; refrigerator predicted twice (refrigerator 0.84 + washing_machine 0.71, wrong class); a GT washing_machine missed; bath detected twice; bottom sliding doors merged/shifted (loose boxes); wardrobe 0.9 on an unlabelled wardrobe (missing label). W: good. |
| `test_train-01_0966-0981.png` | D: double door correct; blind_window 0.79 on an unlabelled hatched strip (missing label); window 0.30 on a curtain wall (wrong class: a W object read as a window). W: curtain wall correct. |
| `test_train-01_1174-0105.png` | D: a second escalator 0.71 next to the labelled one (unlabelled, missing label). W: lower half of the railing missed (under-segmentation); walls predicted on unlabelled double lines (over-segmentation). |

**Tally over the 20 images** (an image counts once per category):

| Category | Images | Which |
|---|---|---|
| Apparent false positives that are real but unlabelled objects (missing labels) | **12** | 0101-0020, 0346-1055, 0514-0013, 0175-0028, 0297-2351, 0297-2381, 0346-0728, 0478-0801, 0719-0002, 0875-0013, 0966-0981, 1174-0105 (5 of them border-cut objects) |
| W under-segmentation (walls or railings missed) | **9** | 0101-0020, 0257-0453, 1159-0026, 0459-0004, 0175-0028, 0297-2351, 0297-2381, 0650-0074, 1174-0105 (railings missed in 3) |
| W over-segmentation against GT (excluding shafts) | **5** | 0346-1055, 0613-0038, 0860-0061, 0478-0801, 1174-0105 (in most, the predicted region looks like a real but unlabelled wall, parapet or column) |
| D wrong class | **5** | 0459-0004, 0514-0013, 0719-0002, 0875-0013, 0966-0981 |
| D loose, split or duplicate boxes | **3** | 0390-0100, 0175-0028, 0875-0013 |
| D missed objects | **3** | 0514-0013, 0650-0074, 0875-0013; none of them small (< 32 px) |
| W shafts / small enclosed spaces filled as wall | **2** | 0613-0038, 0719-0002 |
| No visible error | 1 | 0389-0173 (0101-0020 and 1159-0026 are close) |

What the sample shows:
- The **most common D "error" is not a model error**: 12 of 20 images contain confident detections of symbols
  that have no label. They count as false positives in AP, which links the results directly to the label-noise
  ceiling (§8).
- **Missed small objects were not seen.** The few misses involve medium or large objects with an unusual drawing
  (bath area, a door symbol, a table drawn as hatching). This sample is too small to contradict the low small-box
  AP (§6.4), which is driven by localisation at IoU ≥ 0.75.
- **W's main failure is missing whole wall pieces** (9 images), especially solid-filled walls, unhatched double
  lines and railings. Over-segmentation mostly marks real-looking walls that the labels omit.
  Shaft filling (§10.4) was seen only twice.

### 6.8 Conclusions

- **D works well on most symbol classes and is limited by labels as much as by the model.** The core mAP50-95 is
  0.725 on a leak-free, cluster-level test split. 22 of 27 core classes reach AP50-95 ≥ 0.6, and the most
  distinctive symbols (doors, elevator, toilets, urinal, bath_tub, wardrobe, tv_cabinet, gas_stove) reach
  0.85–0.93. Airconditioner reaches 0.695 AP50-95 / 0.742 AP50.
- **YOLO11n and YOLOv8n are equal within training noise** (Δ = +0.0004, CI [−0.0036, +0.0046]). Choose between
  them on non-accuracy grounds. The comparison covers training noise on one split only.
- **The weak spots are specific:** bay_window (confused with window), escalator, opening_symbol, bath and bed.
  Rarity explains the first two only.
- **W gives ~0.70 class-agnostic pixel IoU** (wall alone 0.71), with a median thickness error of 13–14 mm and a
  −6 to −8 mm bias that plausibly comes from the label rasterisation. Mask mAP (~0.22) is low by construction
  (instance matching of long, connected walls) and is not the headline. Curtain walls and railings are weak
  (0.28 / 0.18 IoU).
- **Next steps suggested by the errors** (not tested): train W longer (`best.pt` was still in the last epochs);
  check for missing labels in the test set before trusting AP differences below a few points; give bay_window
  and the vertical-circulation classes more data or merge them, if the downstream pipeline allows it.

## 7. Measured mm/px scale

Full method and per-tile table: `reports/scale.md`; numbers: `reports/scale.json` (main) and
`reports/scale_anchor.json` (comparison); overlays: `reports/qa/scale/`.

- **Method.** A dimension text (digits only, ≥ 2 digits, value in mm) is matched to a straight two-point segment on
  the same CAD layer, parallel within 2°, whose midpoint is ≤ 3 viewBox units from the **text centre**
  (start anchor + half the estimated text width); segments < 1 unit (ticks, arrow heads) are ignored.
  mm/px = value / (segment length in px). Per tile: median and consistency (share of estimates within ±10% of
  the median); a tile is consistent if ≥ 0.5. 40 tiles with ≥ 3 estimates sampled with seed 0 (90 tiles scanned).
- **Result: constant 10.0 mm/px.** 33 of 40 tiles consistent; median of their medians 10.000, Q1/Q3
  10.000/10.004, IQR/median 0.0004 (threshold 0.10). 29 of 33 medians lie within 10.00 ± 0.02; the other four are
  9.05–9.38. 74% of all 407 individual estimates lie within 5% of 10.0. Consistent tiles come from all three
  archives (test-00 16, train-01 14, train-00 3). So 1 viewBox unit = 100 mm = 10 px, and a 1000 px tile covers
  10 m × 10 m (not 20 m × 20 m as the Voxel51 card states).
- **The literal anchor rule** (match from the text's start anchor, no minimum length) gives median 22.54 mm/px with
  tile medians up to 10,000 mm/px and only 6/40 consistent tiles; it is dominated by wrong matches (extension
  ticks next to the anchor) and is not evidence of a varying scale. It is kept in `scale.md` for comparison.
- **Limits.** Only tiles with ≥ 3 matched numeric dimensions are represented (train-00 is thin: 3 tiles); the 7
  inconsistent tiles were excluded, not explained; a consistency filter cannot prove every tile has the same scale.
- **Consequence for integration (spec §6.10).** Project drawings must be rendered at **10 mm/px** with thin
  (~1 px) black strokes on white for the models to see the same symbol sizes as in training.

## 8. Label-noise ceiling

Spec §6.8 asks for the label disagreement between byte-identical copies as an estimated ceiling.

| Evidence (`stats.json` → `dedup`, 1,163 sets, 3,474 tiles) | Value |
|---|---|
| Copy sets with identical per-class thing counts | 59.9% |
| Copy sets with identical wall-stroke counts | 45.6% |
| Thing instances absent from at least one copy | **21.6%** |

How to read it:
- On identical pixels, roughly one thing instance in five is present in one copy and missing in another. If the
  test labels have a similar miss rate (they come from the same annotation process; only one copy of each test
  image survives dedup), a detector that finds every real object is still charged false positives for the
  unlabelled ones, and recall against "real objects" cannot be measured. **Precision, and hence AP, has a
  ceiling below 1** that this dataset cannot quantify exactly.
- The measure is count-based (max − min copy count per class), so it is a **lower bound** on disagreement: two
  copies with equal counts but different labelled instances count as agreeing.
- Dedup reduces the noise only for the 1,163 kept tiles (the most complete copy is kept); the other ~12,200
  tiles have a single annotation and unknown misses.
- Walls disagree even more often (45.6% agreement on stroke counts), and some columns are labelled as wall while
  most are not (spec §5.6); W's pixel IoU carries the same kind of ceiling.
- No mAP number is derived from this; it is context for interpreting §6.
- **Seen in the results:** in 12 of the 20 qualitative test images (§6.7), the deliverable detector makes
  confident detections of symbols that have no label (doors cut by the tile border, stairs, sinks, wardrobes,
  an elevator). Each one costs precision in AP. Over-segmented W regions are mostly real-looking walls,
  parapets or columns that the labels omit. The measured D precision (0.82) and W pixel IoU (0.70) are
  therefore lower bounds on agreement with the real drawings, by an amount this dataset cannot quantify. The
  same noise limits the comparison in §6.2: AP differences of a few tenths of a point between models are below
  what these labels can resolve.

## 9. Literature context (not a fair comparison)

Spec §6.8 "Context, not a fair comparison". None of the numbers below can be compared with §6:

- **Community model** `mudasir13cs/floorcad-yolov8n-detect` on Hugging Face reports **mAP50-95 0.671**, measured on
  **val**, on a different conversion and a different split. Its 4,246 training images ≈ 80% of the Voxel51 mirror,
  which contains only the official test set randomly re-split, so its val set very likely shares drawings and
  exact copies with its train set (§3). **It was deliberately not run on our test set:** it may have trained on
  part of `test-00`, which is now spread across our train, val and test (§4.3).
- **Published FloorPlanCAD results** (primitive-level panoptic symbol spotting, e.g. SymPoint and the vector models
  named in spec §3) use a **different task and metric** (per-primitive PQ/F1), not box mAP.
- **All published FloorPlanCAD numbers use the official split**, which leaks by drawing (96.7% of test tiles) and
  by exact copy (1,186 test tiles) (§3). They may all be optimistic.
- The design's background survey (in `knowledge/`, outside this repository) concluded that data, rendering and
  split matter more than the YOLO version, that mAP50-95 is the right main metric, that splits must be by drawing,
  and that ≥ 3 seeds trained to convergence are needed. This project follows those conclusions; the D comparison
  in §6.2 tests the "YOLO version matters less" claim on clean data.
- Our numbers come from a cluster-level split with dedup, ink-only rendering and a fixed core class set, so they
  should be read as a **conservative, leak-free estimate** for unseen buildings of the same style, not as a
  leaderboard entry.

## 10. Decisions made during implementation

Each entry: what was decided · why · cost if it turns out wrong. Source: the implementation ledger
(`.superpowers/sdd/2026-11-10-floorplancad-yolo11n-plan/progress.md`, "Ruling", "USER DECISION" and concern
lines). Decisions already in the spec (§10 there) are not repeated.

### 10.1 Environment and infrastructure

- **New conda env `fpyolo` (Python 3.12)**, created from conda-forge with `--override-channels` (the `defaults`
  channel stopped on a Terms-of-Service error), ultralytics 8.4.210, torch 2.15.0. · The plan mandates a
  dedicated env; the user's existing env stays untouched. · ~5 GB disk.
- **`CUDA_DEVICE_ORDER=PCI_BUS_ID` forced on import of `fpyolo`; `--device cpu` hides all GPUs;** `env.json`
  records the GPU's PCI bus id. · Ultralytics maps `--device N` to torch's `cuda:N`, and torch's default order may
  differ from `nvidia-smi` numbering, so a run could land on another user's GPU. · None.
- **Effective batch is verified from `weights/last.pt` `train_args`, plus a "Reducing to batch" grep of the last
  attempt in the run log.** · `args.yaml` is written once and never shows Ultralytics' OOM halving. · None.
- **Batch overrides only via `BATCH_D` / `BATCH_W` environment variables** (same value for both D models), never by
  editing configs. · Keeps the D comparison fair and recorded. · A rerun without the variable uses 32.
- **Run locks** (sweep-level `flock`, per-run `fcntl` lock), a fifth run state "broken", log rotation on a fresh
  start, and `runstate verify` after each run. · Guard unattended GPU-days on a shared machine. · None.
- **Evaluation memory fix:** never pass a list of images to Ultralytics `predict` (8.4.210 builds one batch of the
  whole list even with `stream=True`; a CPU smoke run reached ~48 GB RAM); predict per file. Peak host RAM after
  the fix: 2.25 GB (D, full val), ~3.3 GB (W). · Shared machine. · None.
- **Where training runs.** The user first moved training to another server (code on GitHub, dataset as one tar in
  a private Hugging Face dataset repo, `scripts/fetch_dataset.sh`, `scripts/pack_results.sh`), then decided to
  train on this machine after all, each long job in its own tmux session with a log; the handoff path is kept as
  a backup. GPUs: the controller may pick free GPUs itself after checking `nvidia-smi`; the sweep uses GPUs 6 and 7.
- **Hugging Face dataset clean-up:** after the single tar was uploaded and its size and SHA-256 were verified, the
  12,751 partial loose files from an earlier upload attempt were deleted from the dataset repo (user-approved).
  · One verified artefact instead of a half-uploaded folder. · None (the tar holds everything).
- Provenance note: commit `db77d77` (in-progress Task 12 files) was made and pushed by a separate session, not by
  the implementation workflow; later commits build on it without amending.

### 10.2 Classes

- **Groups follow spec §6.2 literally:** `appliance-hvac` holds only airconditioner; refrigerator, gas stove and
  washing machine stay `sanitary-kitchen`. · Spec text; groups do not affect training. · A YAML edit of group
  fields (schema_ids unchanged).
- **W has 3 classes: wall, curtain_wall, railing** (enabled after the closure check, median coverage 1.00 for
  both). · Spec §6.6 condition met. · Downstream code must read class names from the schema (it does); the W
  headline IoU is class-agnostic, so curtain walls and railings count as "wall area".
- **Core classes exclude IDs 6, 7, 8.** folding_door misses by **one** train instance (49 < 50). The rule was
  fixed before training and is not adjusted. · Spec §6.8. · folding_door (19 test instances) is reported in the
  all-30 number only.

### 10.3 D labels

- **R1: the 30%-visible rule is computed on the box after 2 px padding.** · A zero-area line box would otherwise
  divide by zero. · A few thin border-cut lines kept/dropped differently — moot, see next.
- **The 30%-visible rule is inert** (`dropped_cut = 0` of 99,755): SVGs are pre-clipped to the tile, so the
  unclipped size of a border-cut object is unknown. Code kept, behaviour documented. · Cost: sliver boxes at tile
  borders (objects mostly outside the tile) are kept as labels; some D label noise at borders.
- **Bbox-raster check: "box ≥ 4 px" = both sides ≥ 4 px**, both boxes snapped to the pixel grid the same way.
  · Natural reading. · Slightly different eligible set; pass rate 100% either way.
- Fixed during implementation: polylines of rotated ellipses (used by the raster checks and QA renders) were
  unrotated; the analytic boxes were already right. Found while checking the class contact sheets.

### 10.4 W labels

- **R10: wall end-capping** (connect an open wall end to a facing stroke end ≤ 32 px away, roughly perpendicular)
  on top of 3 px gap closing. · Without it 43% of wall tiles fell below 0.80 coverage (open wall ends at door
  openings), above the 30% stop line; cap histogram peaks at 20–32 px. · Possible false bridges between two
  parallel walls whose aligned ends are closer than their thickness; 0 seen in 50 overlays.
- **R11: coverage counts stroke length within 1.5 px of the region, boundary *or* interior.** · Hatching inside
  walls is also ID 1; the literal "on the boundary" definition penalised hatched concrete walls. · A few hatched
  tiles admitted whose outlines are fine anyway.
- **Accepted W label noise "type (b)": enclosed spaces narrower than ~60 px (shafts, niches, pipe chases) are
  filled as wall** by the face-thickness rule (`max_thickness_px: 60`). 5 (+1 uncertain) of 50 sampled W tiles
  affected, each by a small area (`reports/qa/README.md`). Thresholds not changed: tuning was out of scope, the
  exclusion rate (10.4%) is far below 30%, and a lower threshold risks dropping thick real walls. · W learns to
  fill small shafts; slightly lower W IoU/mask AP. **Candidate fix for v1.**
- **Border slivers** between a wall and a parallel image border are excluded (`max_border_frac` default 0.4).

### 10.5 Dedup, clustering, split

- **R2: a tile with no ink gets an all-zero thumbnail** (no NaN) and never links.
- **R7: `(archive, stem)` tuples inside dedup/split/build; the string `<archive>_<stem>` everywhere outside.**
  · Both needed. · Cosmetic.
- **R8: split manifests are committed** (`git add -f data/fpcad_v0/manifests/*`) although `data/` is ignored.
  · Spec §6.4 ships split files. · ~2 MB of CSV in git.
- **R12 (with R3): loss-based cluster greedy instead of the plan's max-need formula.** · The plan's formula left
  11–12 of 31 classes absent from some set on a realistic synthetic and failed its own test; the replacement
  meets the spec goals (tile ratios 70.0/10.0/20.0, no class absent). · A different but reproducible split.
- **Val = clusters until ≥ 10% of tiles** (the brief said 10% of clusters). · Tile share is what 70/10/20 measures.
  · Val slightly larger/smaller than a literal reading; it ends at 40 clusters.
- **Cross-space candidates = best tile pair per drawing-group pair**, skipping pairs already in one cluster.
  · The brief's "top 30 pairs" would have spent all 30 on one hatch-heavy group pair. · A same-building pair ranked
  31+ is missed (all 30 inspected were different buildings).
- **`manual_merges` format** `[[["A","0012"],["B","0731"]]]`, validated loudly. Used: none.

### 10.6 Scale measurement

- **mm/px method changed before review (Task 9 fix round 1):** match a dimension text from its **centre**
  (start anchor + half the estimated text width), ignore segments shorter than 1 viewBox unit, judge each tile by
  its consistency (share of estimates within ±10% of the tile median, consistent if ≥ 0.5), base the conclusion on
  consistent tiles only and call it "inconclusive" below 20 consistent tiles; the literal-anchor numbers are kept
  in `reports/scale.md` for comparison. · The literal rule (match from the start anchor) produced matching noise:
  only 4 of 40 tiles had a majority estimate and tile medians reached 10,000 mm/px, so its "varies" verdict was not
  evidence, and spec §6.10 needs a credible value. · Minutes of compute; the literal rule becomes a secondary result.

### 10.7 Evaluation

- **R5/R6/R9 interfaces:** `env_record` records the SHA-256 of `tiles.csv` and the schema version; `metrics.json`
  stores the dataset/split/manifest paths so `compare` can rebuild GT; core-class semantic IDs are mapped to
  YOLO indices through the schema. · Pure plumbing.
- **Bootstrap details:** per-image multiplicity weights; distinct clusters resampled; the same resample for both
  models in an iteration; CI = 2.5/97.5 percentiles; the point Δ on unweighted data. · Not fixed by the plan.
  · CI width differs slightly.
- **Bootstrap resamples missing a core class** use the present core classes and are counted in `n_degenerate`
  (all-absent resamples are excluded from the percentiles). · Raising mid-bootstrap would abort `compare`. · Slight
  CI bias, visible via `n_degenerate`.
- **Tie convention in weighted AP:** copies of a detection are adjacent in the score order. · COCO's own tie order
  is an artefact of image-id order; float32 scores rarely tie. · Tie-order noise only.
- **Test of pycocotools agreement uses `coco.eval["precision"]` at maxDets 300**, because `stats[0]` is hard-coded
  to maxDets 100.
- **W headline pixel IoU is class-agnostic** (union of the 3 W classes in GT and prediction), with per-class IoU at
  the same `t*` as extra keys. · Spec measures wall *area*; W now has 3 classes. · One extra metric to recompute.
- **Across-seed std uses ddof = 1** (sample std of 3 seeds).
- **`--limit N`** on `evaluate_d`/`evaluate_w` exists only for CPU smoke runs; `report` refuses runs evaluated with it.
- **D inference export:** `fpyolo.predict` uses `--d-conf 0.25` by default and writes one W polygon per (exterior,
  holes) component with conf ≥ `--w-conf` (= `t*`). The evaluation itself uses `conf=0.001`.
- **Wall-thickness error (spec §6.8, enabled by the constant scale):** per test image whose class-agnostic GT and
  predicted masks (score ≥ `t*`) are both non-empty, thickness = 2 · area / perimeter of the mask; error =
  prediction − label, × 10.0 mm/px; reported as MAE, median |error|, bias and n_images. · The simplest
  image-level estimate that needs no skeleton. · Biases listed in §11; may need a better definition later.

## 11. Known limitations

- **W type-(b) label noise:** shafts, niches and pipe chases narrower than ~60 px are labelled (and will be
  learned) as wall; ~5–6 of 50 sampled W tiles, small areas each (§10.4).
- **The 30%-visible rule is inert** because the SVGs are pre-clipped; tiny slivers of objects cut by the tile
  border remain as boxes (§10.3).
- **Core classes exclude IDs 6, 7 and 8;** folding_door misses by one train instance. revolving_door and
  rolling_door have one test instance each, so their AP is 0 or 1-ish noise; the all-30 mean moves by up to
  3.3 points per such instance.
- **W gained curtain_wall and railing** per the closure check; the headline pixel IoU and the thickness error are
  over all three, not wall alone (per-class IoU is reported separately).
- **GT masks via `cv2.fillPoly` are ~1 px wider** than the analytic regions (boundary pixels filled).
- **Wall thickness is an approximation:** image-level 2·area/perimeter over the class-agnostic mask
  (including curtain walls and railings); perimeter = `cv2.arcLength` of all external and hole contours
  (`evaluate_w.THICKNESS_METHOD`). It is exact for an infinitely long straight bar and a few percent low for a
  finite one (a bar of width w and length L reads ≈ w·L/(L + w)); short segments and junctions read lower.
  Walls cut by the tile border count as bar ends, because the border edge is part of the contour: a 20 px wall
  running the full 1000 px tile reads ~2% low. The same bias applies to the GT and the predicted mask, so it
  largely cancels in the error. It is a consistency measure between prediction and label, not a per-wall
  thickness.
- **3 seeds measure training noise, not split noise.** A different cluster split could move all numbers;
  the bootstrap over test clusters covers test-set sampling only.
- **Speed measured on an RTX A5000** on a shared machine via Ultralytics' val timings; not an end-to-end
  deployment latency.
- **W checkpoint selection:** Ultralytics' segmentation fitness is mask mAP50-95 **plus** box mAP50-95
  (`ultralytics/utils/metrics.py`, `SegmentMetrics.fitness`, 8.4.210), not mask mAP50-95 alone as spec §6.7 states.
  D's fitness is box mAP50-95 over classes with val labels (all 30 have val labels in this split). Not changed
  for v0; noted here as a deviation.
- **Val is 40 clusters**, so val-based choices (checkpoint, `t*`) are noisier than 1,341 tiles suggests.
- **Clustering limits:** near-duplicates below cosine 0.95 (shifted crops, edited floors) are not merged, so
  some small leakage may remain (spec §9). The measure over-merges hatch-dominated tiles, which only enlarges
  clusters.
- **Label noise** of ~22% missing instances between copies, columns sometimes labelled as wall, and unlabelled
  seat rows / parking spots in the images (background for D).
- **W test bias:** tiles with wall coverage < 0.80 (9.9% of test wall tiles) are excluded from W evaluation.
- **Domain:** FloorPlanCAD is Chinese drafting conventions, mostly residential; no project drawings were provided,
  so no domain-shift section (spec §7.6) is included.
- **Scale:** 10.0 mm/px is measured on tiles with dimension text; tiles without dimensions are assumed to share it.

## 12. How to reproduce

All commands from the repository root, env `fpyolo`. Setup, GPU selection and resume: `README.md`.

```bash
# dataset build (CPU; manifests committed under data/fpcad_v0/manifests/)
python -m fpyolo.build --src <FloorPlanCAD> --out data/fpcad_v0 --config configs/build.yaml --workers 32
# QA material (reports/qa/README.md lists every command)
python -m fpyolo.qa overlays --dataset data/fpcad_v0/D --n 50 --seed 0 --out reports/qa/overlays_D
python -m fpyolo.qa overlays --dataset data/fpcad_v0/W --n 50 --seed 0 --out reports/qa/overlays_W
# scale (§7)
python -m fpyolo.scale --src <FloorPlanCAD> --n 40 --seed 0 --out reports/scale.json --qa-dir reports/qa/scale
# training: 9 runs (in tmux), GPUs chosen from nvidia-smi
GPUS="6 7" scripts/run_all_seeds.sh 2>&1 | tee -a runs/sweep.log
# evaluation (one free GPU), then the comparison
GPU=6; D=data/fpcad_v0
for s in 0 1 2; do
    for m in d11n d8n; do
        python -m fpyolo.evaluate_d --weights runs/${m}_s$s/weights/best.pt \
            --dataset $D/D --split test --manifest $D/manifests --out results/${m}_s$s --device $GPU
    done
    python -m fpyolo.evaluate_w --weights runs/w11n_s$s/weights/best.pt --dataset $D/W --out results/w11n_s$s --device $GPU
done
python -m fpyolo.compare --a results/d11n_s{0,1,2} --b results/d8n_s{0,1,2} --manifest $D/manifests --out results/compare.json
# tables for §6 and the data statistics below
python -m fpyolo.report --results results --manifests data/fpcad_v0/manifests --out reports/tables.md
```

(Run the first `evaluate_d` alone; see the README "Evaluation" section for the memory cap and the
`labels/test.cache` race.)

### Appendix: data-statistics tables as rendered by `fpyolo.report`

Rendered on 2026-11-12 from the real `stats.json` with an empty results directory
(`python -m fpyolo.report --results <empty dir> --manifests data/fpcad_v0/manifests --out /tmp/…`):

**Deduplication (byte-identical PNG copies)**

| Tiles | Identical sets | Tiles in sets | Kept | Dropped |
|---|---|---|---|---|
| 15,663 | 1,163 | 3,474 | 13,352 | 2,311 |

**Label disagreement between identical copies**

| Sets whose copies agree on every thing count | Sets whose copies agree on the wall stroke count | Thing instances missing from at least one copy |
|---|---|---|
| 59.9% | 45.6% | 21.6% |

**Clusters and split**

| Clusters | Largest | Largest share | Top sizes | Clusters train / val / test |
|---|---|---|---|---|
| 1,108 | 187 | 1.4% | 187, 173, 172, 122, 122, 115, 111, 110, 110, 104 | 952 / 40 / 116 |

| Split | Tiles | Share |
|---|---|---|
| train | 9,341 | 70.0% |
| val | 1,341 | 10.0% |
| test | 2,670 | 20.0% |

Boxes: 99,755 in total, 59 padded to 2 px, 8,646 smaller than 32 px, 3,875 tiles without objects.

**W coverage and exclusion**

| Split | Tiles with W-class strokes | Excluded (coverage < 0.8) | Excluded share |
|---|---|---|---|
| train | 8,612 | 899 | 10.4% |
| val | 1,189 | 128 | 10.8% |
| test | 2,419 | 239 | 9.9% |
| all | 12,220 | 1,266 | 10.4% |

56,317 polygons in the W dataset. Excluded tiles are still used for D.

| W class | Tiles | Median coverage | Enabled |
|---|---|---|---|
| curtain_wall | 1,843 | 1.00 | yes |
| railing | 2,376 | 1.00 | yes |
