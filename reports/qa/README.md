# Dataset QA — build v0 (`data/fpcad_v0`)

Build: `python -m fpyolo.build --src <FloorPlanCAD> --out data/fpcad_v0 --config configs/build.yaml --workers 32`
(manifests committed under `data/fpcad_v0/manifests/`). Earlier QA material in this folder:
`classes/` and `bbox_raster.json` (Task 4, class names and analytic-bbox check).

| Folder | Content | Command |
|---|---|---|
| `cross_space/` | the 30 `train-00` ↔ group-B candidate pairs (cosine in [0.85, 0.95)) | `python -m fpyolo.qa pair-sheet --stats data/fpcad_v0/manifests/stats.json --src <FloorPlanCAD> --out reports/qa/cross_space` |
| `cross_space_linked/` | the 10 `train-00` ↔ group-B pairs that linked automatically (cosine ≥ 0.95) | same tool, on a stats file listing those pairs (found from the build cache) |
| `overlays_D/`, `overlays_W/` | labels of 50 random images each (seed 0), file name `<split>_<image id>.png` | `python -m fpyolo.qa overlays --dataset data/fpcad_v0/{D,W} --n 50 --seed 0 --out reports/qa/overlays_{D,W}` |
| `coco_vis_check/` | 3 D overlays next to the dataset authors' `coco_vis` render | scratch script (overlay_labels + qa.detect_frame) |

W overlay colours: red = wall (index 0), violet = curtain_wall (1), cyan = railing (2).

## Cross-space pairs (plan Step 4b)

**Candidates, 30 pairs, cosine 0.873–0.935: all different buildings; 0 merged, 0 uncertain.**
`manual_merges` stays `[]`, so no rebuild was needed.

- Each candidate is the best tile pair for one (group A, group B) pair that is not already
  in one cluster.
- Only 4 `train-00` tiles take part: `0511-0000` (22 pairs), `0424-0005`/`0424-0003` (4),
  `0425-0003` (4). Each of them is a crop almost entirely covered by uniform floor hatching
  (diagonal lines, crossed lines). Their group-B partners are also hatch-dominated
  (basket-weave, herringbone, roof tiles, diagonal hatching) from unrelated drawings: car
  parks, shops, stair cores, roofs.
- The 32×32 ink-density thumbnail of such a tile is nearly flat, so any two hatch-filled
  tiles reach cosine ≈ 0.9 whatever the building.

| Pairs | Pattern in the B tile | Verdict |
|---|---|---|
| 00, 11, 25 | `B/0905`: shaded stair core, different layout | different |
| 01, 02, 05 | `B/0157`: dark basket-weave paving | different |
| 03 | `B/0297`: basket-weave paving | different |
| 04 | `B/0346`: crossed hatching with toilets | different |
| 06, 09, 13 | `B/0538`: roof tiles | different |
| 07 | `B/0915`: crossed hatching, fire doors | different |
| 08 | `B/0035`: diagonal hatching, curved wall | different |
| 10 | `B/0514`: elevator bank on hatching | different |
| 12, 15, 21 | `B/0655`: vertical-line paving | different |
| 14 | `B/0829`: crossed and diagonal hatching | different |
| 16–20, 22–24, 26, 27, 29 | `B/1146, 0103, 0094, 1121, 0113, 0122, 0067, 1259, 1180, 0101, 1276`: herringbone floor, various cores and offices | different |
| 28 | `B/0865`: diagonal hatching, doors | different |

**Auto-linked pairs (cosine ≥ 0.95, `clusters.n_pairs_cross_space` = 10): correct links.**
- All 10 pairs join `A/0121` with `B/0543` (5 to `test-00`, 5 to `train-01`).
- They have a constant crop-index offset (e.g. `0121-0017` ↔ `0543-0020`,
  `0121-0031` ↔ `0543-0037`) and nearly equal ink fractions (0.035–0.088).
- `cross_space_linked/pair_00.png` and `pair_07.png` show the same floor plan with a few-pixel
  shift. This is one drawing exported in both numbering spaces, so the merge into one cluster
  is right.
- Spec §6.4 expected train-00 and the rest to be "almost disjoint". They are, apart from this
  one drawing.

**Limitation of the measure.** Hatch-dominated tiles can also reach ≥ 0.95 *within* a numbering
space and merge unrelated groups. That only makes clusters larger (no leakage), and the
largest cluster is 187 tiles = 1.4% of the kept tiles.

## W candidate classes (plan Step 5)

`stats.json` `w_candidates` from the first build:

| ID | Class | Kept tiles with strokes | Median coverage | Would enable |
|---|---|---|---|---|
| 2 | curtain wall | 1,843 | 1.00 | yes |
| 33 | railing | 2,376 | 1.00 | yes |

- Both classes were enabled in `configs/classes_v0.yaml`. W YOLO indices: wall 0,
  curtain_wall 1, railing 2; schema_ids 31/32/33 unchanged.
- The build was then re-run with `--reuse-manifest`. keep, cluster and split were identical
  for all 15,663 rows.
- W exclusion after enabling: 1,266 of 12,220 tiles with W-class strokes = **10.4%**
  (train 10.4%, val 10.8%, test 9.9%). Before: 1,217 / 11,648 = 10.4%.
- In the overlays, curtain walls (`train-01_1039-0106`, curved glazing) and railings
  (`test-00_0273-0017`, `train-00_0253-0005`, `test-00_0462-0024`) close into thin bodies,
  as walls do.

## W overlays (50 images, plan Step 6)

Checked for the three failure modes named by the controller. Zoomed crops at native resolution
were used where a region was unclear.

| Failure mode | Count (tiles) | Tiles |
|---|---|---|
| (a) end cap bridging two separate parallel walls | **0** | none seen |
| (b) small enclosed space (shaft, niche < 60 px) filled as wall | **5** (+1 uncertain) | `test-00_0349-0669` (row of ~80×40 px vent shafts in a 90 px stair wall), `test-00_0462-0024` (风井 air shaft, ~52 px interior), `test-00_1285-0004` (two ~60 px niches beside a wall), `train-00_0159-0007` (~27 px pipe chase with 3 risers), `train-01_0271-0005` (~60 px unlabeled shaft-like box); uncertain: `train-01_0493-0187` (20×68 px box inside a wall junction, riser or column) |
| (c) door opening filled | **0** | doors left open in every tile with doors (about 20 tiles) |

- Cause of (b): the face-thickness rule (`max_thickness_px: 60`, inscribed radius ≤ 30 px)
  cannot tell a thin wall body from an equally thin enclosed void. Shafts wider than ~60 px
  are correctly left empty: `train-00_0310-0391` (flue), `test-00_1047-0694`,
  `train-00_0133-0001`, `train-00_0384-0025`, `train-01_0513-0053`, `train-01_0725-0249`,
  `train-01_0946-4051`, `train-01_0984-0010`, `train-00_0183-0003`.
- With 5–6 of 50 tiles affected, each by a small area, this is label noise for W. It was
  already listed as an open concern in Task 5. The thresholds were not changed: tuning was
  not in scope, and `coverage_min` was not touched.
- Other observations:
  - 4/50 W images are negatives because the drawing has no wall labels (car park, Japanese
    drawing, façade).
  - Thick multi-line walls (`train-01_0939-0610`, `train-00_0304-0053`) are filled as one
    body, consistent with the labels.
  - Border walls are closed by the frame as intended.

## D overlays (50 images) and the coco_vis cross-check

- Every box sits tight on its symbol, and the class names are plausible:
  - doors (single, double, sliding);
  - windows and bay windows;
  - furniture (bed, sofa, table, chair, wardrobe, cabinet, tv_cabinet);
  - sanitary and kitchen fixtures (sink, toilet, squat toilet, urinal, bath, bath_tub,
    gas_stove, refrigerator, washing_machine);
  - airconditioner, stairs, elevator.
- No offsets and no scale errors. 9/50 images are negatives (no things).
- Rotated drawings (`train-00_0388-0083`, `train-01_0547-0041`, `val test-00_1134-0026`) get
  axis-aligned boxes around the rotated symbols, as YOLO detection expects.
- Possible label noise seen (not changed):
  - long segmented glazing strips in `test-00_1117-0091` carry no D box; they are probably
    curtain walls (a W class);
  - one small window-like symbol in `test-00_0257-0622` is unlabeled.
- `coco_vis_check/`: `train-00_0011-0012` (a **1000×999** tile, 74 things), `train-00_0314-0449`
  and `test-00_0089-0102` match the authors' render object for object and name for name. There
  is no vertical drift on the 999 px tile.

## Data facts found by the build

- **SVG/PNG alignment:** `align_frac` = 1.0 on all 15,228 tiles with labeled primitives
  (0 misaligned).
- **The 30% visible rule never fires** (`box_stats.dropped_cut` = 0 of 99,755 objects).
  - The SVG tiles are already clipped to the viewBox: 0 of 76,753 primitives in a 300-tile
    sample extend outside it.
  - About 14% of objects touch the border. They are kept with their visible extent; the
    unclipped size cannot be known.
- **Dedup label disagreement** between byte-identical copies, over 1,163 sets:
  - same per-class thing counts in 59.9% of sets;
  - same number of wall strokes in 45.6%;
  - 21.6% of thing instances missing from at least one copy.

  These match spec §5.3 (60% / 46% / 22%).
