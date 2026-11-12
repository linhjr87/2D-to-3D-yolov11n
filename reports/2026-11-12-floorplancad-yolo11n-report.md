# YOLO11n on FloorPlanCAD: detection (D) and wall segmentation (W), evaluation report

Date: 2026-11-12 · Branch `feat/fpyolo-v0` · Spec: `docs/specs/2026-11-10-floorplancad-yolo11n-design.md` (cited as "spec §x")

> **Status: results pending.** The data pipeline, class table, split, scale measurement and every
> non-results section below are final. The 9 training runs (D-yolo11n, D-yolov8n, W-yolo11n-seg × seeds 0, 1, 2)
> started on 2026-11-12 at 11:29 UTC (tmux session `fpyolo-sweep`, GPUs 6 and 7, log `runs/sweep.log`).
> Every number that depends on a trained model is marked **Pending: filled after the sweep finishes**,
> with the command that produces it. No result number in this report is estimated or invented.

## Contents

1. Summary
2. Method
3. Data issues found in FloorPlanCAD
4. Dedup, clustering and re-splitting
5. Final class table
6. Results (pending)
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
- **Results.** Pending (§6).

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
  for test. Also per-class pixel IoU at `t*`, mask mAP50-95 and the wall-thickness error in mm (§6.3).

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

> **Pending: filled after the sweep finishes.** The tables in §6.1–6.4 are produced by
> `python -m fpyolo.report --results results --manifests data/fpcad_v0/manifests --out reports/tables.md`
> after the evaluation commands in §12. Copy them in verbatim; do not round differently.

### 6.1 D headline (test split, mean ± sample std over 3 seeds)

<!-- RESULTS: paste the "D: object detection (test split)" table from reports/tables.md -->

**Pending: filled after the sweep finishes.**

| Model | Seeds | mAP50-95 (27 core, headline) | mAP50 (core) | mAP75 (core) | mAP50-95 (all 30) | mAP50 (all 30) | mAP75 (all 30) | ms/img |
|---|---|---|---|---|---|---|---|---|
| YOLO11n | 3 | pending | pending | pending | pending | pending | pending | pending |
| YOLOv8n | 3 | pending | pending | pending | pending | pending | pending | pending |

Also report precision/recall from `results/<run>/metrics.json` (Ultralytics val block) and the
`missing_classes` list of the all-30 number (with test counts ≥ 1 for every class in this split, it is
expected to be empty; confirm).

### 6.2 YOLO11n vs YOLOv8n

<!-- RESULTS: from results/compare.json: delta (11n − v8n, core mAP50-95), 95% cluster-bootstrap CI, pooled std, ci_excludes_zero, exceeds_2x_pooled_std, real, n_degenerate -->

**Pending: filled after the sweep finishes.** Statement to write: "real difference" only if the CI excludes 0
**and** |Δ| > 2 × pooled std; otherwise "no real difference under this protocol". Note n_degenerate
(bootstrap resamples missing a core class) if > 0.

### 6.3 Per-class analysis

<!-- RESULTS: per-class AP table from reports/tables.md (sorted by AP; non-core and absent-from-test flagged; airconditioner row in bold) -->

**Pending: filled after the sweep finishes.** To write, from the per-class table and the confusion matrix
(`results/d11n_s0/confusion_matrix_normalized.png`):

- airconditioner (schema_id 19, 448 test instances): AP50-95 and AP50, mean ± std, and its main confusions;
- the weakest core classes and whether they are small-box classes (doors/windows), rare classes
  (bay_window 102, escalator 87, bath_tub 105 test instances) or visually confusable pairs
  (bath vs bath_tub, cabinet vs wardrobe vs tv_cabinet, single vs double door, window vs blind_window);
- the non-core classes (folding_door 19, revolving_door 1, rolling_door 1 test instances): report but do not
  interpret; one instance of IDs 7 or 8 moves the all-30 mean by up to 1/30 = 3.3 points.

### 6.4 AP by object size

<!-- RESULTS: by-size table from reports/tables.md (COCO small < 32², medium 32²–96², large > 96² px) -->

**Pending: filled after the sweep finishes.** 8.7% of all boxes are < 32 px (`box_stats.small_lt_32px` = 8,646
of 99,755); mAP50-95 is expected to be harsh on them (loose-by-1-px boxes lose IoU fast).

### 6.5 W: wall segmentation (YOLO11n-seg)

<!-- RESULTS: W table from reports/tables.md: t* (val), val IoU at t*, test pooled pixel IoU (class-agnostic), per-class pixel IoU (wall / curtain_wall / railing), mask mAP50-95, wall-thickness error (MAE, median |err|, bias in mm, n_images); mean ± std over 3 seeds -->

**Pending: filled after the sweep finishes.** Read next to these numbers:
- GT masks are rasterised with `cv2.fillPoly`, which also fills the pixels the boundary passes through, so
  GT walls are about **1 px wider** than the analytic region (≈ 10 mm at 10 mm/px). Predictions trained on
  fillPoly masks share this convention; ~1 px is about 5% of the width of a ~20 px wall.
- W's test set excludes the 9.9% of test tiles with wall coverage < 0.80 (biased toward clean walls).
- Thickness definition and its biases: see §11.

### 6.6 Speed

<!-- RESULTS: speed_ms (preprocess / inference / postprocess) from results/<run>/metrics.json, mean over seeds -->

**Pending: filled after the sweep finishes.** Measured by Ultralytics' val on one RTX A5000 (shared machine;
other users' jobs can slow it down) at `imgsz=1024`; ms per image, not an end-to-end latency benchmark.

### 6.7 Qualitative examples and error categories

<!-- RESULTS: 20 test images from reports/qualitative/ (ground truth left, D boxes >= 0.25 and W masks >= t* right); one line per image with the error seen -->

**Pending: filled after the sweep finishes.** Command (CPU or one free GPU):

```bash
python -m fpyolo.qa pred-overlays --d-weights runs/d11n_s0/weights/best.pt --w-weights runs/w11n_s0/weights/best.pt \
    --w-conf <t* from results/w11n_s0/metrics.json> --dataset-root data/fpcad_v0 --split test --n 20 --seed 0 \
    --out reports/qualitative --device <cpu|idx>
```

Categories to tally over the 20 images (spec §6.8): wrong class; missed small objects; loose / shifted boxes;
"false positives" that are real but unlabelled objects (§3.3, §8); W filling shafts/niches (§11); W misses at
door openings or curtain walls; anything else seen.

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
