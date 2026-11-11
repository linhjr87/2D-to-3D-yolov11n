# Design: Fine-tuning YOLO11n on FloorPlanCAD for architecture and furniture detection (foundation stage)

Date: 2026-11-10 (revision 2 after review; revision 3 on 2026-11-11: dataset downloaded and extracted, data facts re-verified, hardware updated, document translated to English) · Status: awaiting approval · Project directory: `floorplan-yolo/` (inside `/home/linh.ba/code/yolov11n-tuning/`)

Convention: *(author's decision)* = a choice made by the author of this design that the user has not confirmed separately. Anything unmarked was either decided by the user or is a fact verified on the data.

---

## 1. Purpose and rationale

The larger project converts 2D floor plans into 3D models, and later serves an HVAC task (the "Learning to Place Air Terminals" proposal). This stage only builds a **foundation model that detects the architectural and furniture elements** of floor plans, trained on FloorPlanCAD and evaluated rigorously before it is integrated into the real system.

After this stage the user will prepare more data and design additional HVAC/MEP classes to extend the model. The output of this stage must therefore:

- be a good starting point for further fine-tuning (corresponds to experiment **E6 "pretrain on architecture"** in the survey in `knowledge/`);
- have a **class table that can be extended** without renumbering;
- be usable to **pre-label the architectural classes** on HVAC drawings later. The survey notes that HVAC images must be labeled for every class; otherwise the model learns doors and walls as background.

In the survey's pipeline, this stage is the architectural part of Stage 2 (Detection) and the wall part of Stage 3 (Geometry). Walls are the most important input for the 3D step (wall extrusion, door cut-outs).

Background reading before starting: the two files in `knowledge/` (the survey and the literature review of 11 Tier A papers; they live in the research root, not in this repository). Conclusions used directly here:

- changing the YOLO version matters less than the data, the rendering and the split;
- use mAP50-95 as the main metric;
- split by drawing to avoid leakage;
- run ≥ 3 seeds and train to convergence;
- prevent the model from learning layer colors.

## 2. Goals

- **G1. Model D (detect):** fine-tune `yolo11n.pt` to detect the **30 thing classes** of FloorPlanCAD with axis-aligned bboxes: door types, windows, furniture, sanitary/kitchen fixtures, air conditioners, stairs/elevators/escalators.
- **G2. Model W (wall seg):** fine-tune `yolo11n-seg.pt` to detect **walls** as polygons. Curtain walls and railings are included under the conditions in §6.6.
- **G3. Reproducible data pipeline:** from the original SVG + PNG, generate two Ultralytics-format datasets (one for D, one for W), with automated checks and a visual label-inspection tool.
- **G4. Evaluation on a new test set split by drawing cluster** (§6.4), as specified in §6.8. Includes a YOLOv8n baseline for D under the same protocol, and 3 seeds for the final results.
- **G5. Results report** (markdown) with tables, per-class analysis, error examples, the data issues found, and a section placing the results in the context of the literature.
- **G6. Deliverables:** weights, configs, a versioned class-table file, split files, and an inference script that outputs JSON (§6.9).

## 3. Non-goals for this stage

- HVAC/MEP classes and the project's own drawings. Exception: one qualitative trial run on a few project drawings, if the user provides them (§7).
- Tiled/SAHI inference on full drawings. FloorPlanCAD only has crops of about 1000×1000, so this cannot be evaluated.
- 3D reconstruction, scene graphs, glTF/IFC export, pipeline integration.
- Changes to the YOLO architecture (ECA, attention...). The literature review advises against investing in this early.
- Vector models (VecFormer, SymPoint, DPSS) and primitive-level PQ metrics.
- Numbers on the official FloorPlanCAD split (§6.4).
- The two area classes "seat row" (ID 34) and "parking spot" (ID 35) *(author's decision: not needed for the 3D building shell)*. Their symbols still appear in the images (seat rows in about 319 tiles, parking spots in about 1,323 tiles) but are unlabeled. For D they are background. The impact is small but must be stated in the report.
- Text reading/OCR, dimension values written on the drawing.

## 4. Things that ruin the results even when the code runs (anti-goals)

- **Numbers from a leaky split:** the same drawing, or the same building repeated under a different drawing number (typical floors), lands in two different sets; or the test set is used to select models or thresholds.
  - The official FloorPlanCAD split has both kinds of leakage (§5.3). The data is therefore re-split by cluster (§6.4).
  - The community model on HF (`mudasir13cs/floorcad-yolov8n-detect`) very likely has it too: 4,246 training images ≈ 80% of the Voxel51 mirror, and that mirror only contains the test set, randomly re-split.
- **Area classes merged into one giant box** per image, as the Voxel51 conversion does.
- **Reporting only val numbers, or only one seed,** for the main result. **Choosing the class set for mAP after looking at test results.**
- **The model learning CAD layer colors.** These colors will not exist on HVAC drawings rendered differently.
- **Class IDs hard-coded in the code,** so adding HVAC classes forces renumbering.
- **Training W on tiles where walls are clearly visible but unlabeled.** The model then learns walls as background.
- **Silent failures:** files with the same name overwriting each other when archives are merged; black-background images due to a dropped alpha channel; wrong bboxes from ignoring arcs or `transform`; normalizing labels by 1000 px while the image may be 999 px.

## 5. Constraints and data facts (re-verified on 2026-11-11 after extraction)

### 5.1 Location and scale

- Extracted data: `/home/linh.ba/code/yolov11n-tuning/dataset/FloorPlanCAD/` with `train-00/` (3,760 SVG+PNG pairs), `train-01/` (6,401), `test-00/` (5,502), 15,663 tiles in total (7.5 GB). Each pair is `NNNN-MMMM.svg` + `NNNN-MMMM.png` at the top level of its folder.
- Each folder also has `coco_vis/` (one image per tile: 3,760 / 6,401 / 5,502): 2000×2000 visualizations made by the dataset authors, with instance coloring and class names, plus matplotlib axes.
- The original archives are kept in `dataset/FloorPlanCAD/_archives/{train-00,train-01,test-00}.tar.xz` (5.4 GB; the download script `scripts/download_floorplancad.sh` writes there). All three archives have the same top-level names (`coco_vis/`, `NNNN-MMMM.*`), so each must be extracted into its own folder.
- There is no official val set.

### 5.2 PNG images

- Most are 1000×1000 (15,151 images). **512 images have one side of 999 px** (322 at 1000×999, 184 at 999×1000, 6 at 999×999).
- All 15,663 are RGBA with a **transparent background**: background pixels are `(0,0,0,0)`, about 90–99% of pixels. `cv2.imread` (used by Ultralytics) drops alpha, so it sees **colored strokes on a black background**.
- Strokes are colored by CAD layer. According to the review, every stroke has a width of 0.1 SVG units (1 px), so stroke thickness does not encode the layer.

### 5.3 SVG, drawing numbering and duplicates

- `viewBox="0 0 100 100"`. The review checked on 594 tiles that SVG coordinates match the PNG: path endpoints land exactly on the strokes. **YOLO labels are normalized by SVG coordinates divided by the viewBox size**, not by 1000 px.
- Labeled primitives are `path`, `circle`, `ellipse`, with the attributes `semantic-id` and `instance-id`. Thing paths may contain circular arcs (door swings). Labeled ellipses carry a `transform` attribute (rotated). Unlabeled primitives (text, dimension lines...) are background.
- File names are `NNNN-MMMM`: `NNNN` is the source drawing, `MMMM` the crop position.
- **Two numbering spaces:**
  - `train-00` has its own numbering. 175 names collide with `test-00` and 203 with `train-01`, but the content differs (different md5).
  - `train-01` and `test-00` **share one numbering**: 1,045 shared drawing prefixes but 0 shared names. That is, crops of the same drawing are split between the two archives.
  - Consequence: 5,323/5,502 official test tiles belong to drawings that also have tiles in `train-01`. Visually confirmed example: `train-01/0001-0087` and `test-00/0001-0023` are the same apartment plan.
- **Byte-identical images across different drawing numbers:**
  - There are **1,163 sets of byte-identical PNGs covering 3,474 tiles (22%)**.
  - Every set spans more than one drawing number, and every set has the same `MMMM`. Example: `train-00/0012-0010` and `train-00/0021-0010`. Most likely these differently numbered "drawings" are repeated floors of the same building.
  - No set spans `train-00` and the `train-01`/`test-00` group.
  - On the official split, 1,186 test tiles have an exact copy in train under another drawing number. This is a second kind of leakage.
  - According to the review, counting near-duplicates too (similarity ≥ 0.95), about 4,385 tiles in 894/1,782 groups have a copy in another group.
- **Identical images with different labels:**
  - Within the identical sets, only **60%** of sets have the same per-class thing instance counts, and **46%** have the same number of wall strokes.
  - **22%** of thing instances are missing in at least one copy.
  - This is label noise (annotators missed objects), and it caps the achievable mAP.

### 5.4 Class ID table

**The ID table in the release does NOT match SymPoint's `SVG_CATEGORIES` table.** In SymPoint's table, 1–30 are things (1 = single door) and 31–35 are areas (33 = wall). The actual data differs:

| semantic-id | instance-id | Common CAD layers | Interpretation |
|---|---|---|---|
| 1 | always −1 | WALL, A-WALL, 砼墙 (concrete wall), PUB_HATCH | **wall** (certain, checked visually) |
| 2 | always −1 | 幕墙 (curtain wall), A-GLAZ, WINDOW | **curtain wall** |
| 3–12 | ≥ 0 | WINDOW, DOOR_FIRE, A-DOOR, 门窗开启方向 (door/window opening direction) | doors, windows, opening symbols |
| 13–19 | ≥ 0 | A-FURN, 家具 (furniture) | furniture |
| 20–29 | ≥ 0 | LVTRY, TOILET, 洁具/厨具 (sanitary/kitchen fixtures) | sanitary fixtures, kitchen, appliances |
| 21 | ≥ 0 | 空调 / 空调机 / AC | **almost certainly "airconditioner"** |
| 30–32 | ≥ 0 | STAIR, EVTR, 楼梯/电梯 (stairs/elevator) | stairs, elevators, escalators |
| 33 | always −1 | 栏杆 (railing) | **railing** |
| 34 | always −1 | FUR, CHAIR, 影院 (cinema) | seat row |
| 35 | always −1 | 停车位 (parking spot), CAR | parking spot |

- Hence **areas = {1, 2, 33, 34, 35}** and **things = 3–32 (exactly 30 classes)**.
- **Class name set:** the `coco_vis` images use exactly the SymPoint name set ("single door", "window", "opening symbol", "airconditioner", "tv cabinet", "cabinet", "gas stove", "refrigerator"...). Only the **numbering** in the SVG differs from SymPoint.
- **ID → name mapping:** the implementer fixes it by matching SVG instances against the labels on `coco_vis` (§8).
- **Do not use the Voxel51/HF name list.** It has `bedside_cupboard`, `half_height_cabinet`, `high_cabinet` and 4 unnamed classes, which do not match `coco_vis`.
- **The "airconditioner" class (ID 21)** is the only currently available class related to HVAC. Report its results separately.

### 5.5 Class distribution (all 15,663 tiles, before dedup)

Notation: instance count / number of drawing groups containing the class.

- **Nearly empty:** ID 6 has 83/35, ID 7 has 7/5, ID 8 has 8/8. With 30 classes, a single test instance of ID 7 or 8 can swing mAP by about 3.3 points.
- **Few:** ID 32 has 360/130, ID 10 has 485/163, ID 25 has 766/263.
- **Many:** ID 3 has 23,159/1,430, ID 9 has 16,459/1,277, ID 4 has 12,372/1,247.
- **Small boxes:** according to the review, about 10% of thing boxes are smaller than 32×32 px; for ID 5 alone it is 58%. mAP50-95 will therefore be harsh on doors and windows.

### 5.6 How walls are drawn

- Walls are drawn as **closed outlines**: two parallel strokes closed at both ends and at door openings. Sometimes there is hatching inside (also ID 1).
- Walls are about 20 px thick. Walls are cut at the image border.
- Illustration: ID 1 strokes in red on `train-00/0035-0010`.
- **Columns:** mostly unlabeled, some carry ID 1. This is label noise.
- **The review's rough prototype** (arcs as chords, thickness limit 50 px, no gap closing), run on 489 tiles with walls:
  - 28% of tiles have less than 70% of wall stroke length on the generated region;
  - 18% have less than 50%;
  - median 3 wall bodies per tile; 14% of tiles have a wall body with holes.

### 5.7 mm/px scale: uncertain

- An estimate from one "3600" dimension line gives about 10 mm/px.
- The Voxel51 card says each tile is a 20 m × 20 m block, i.e. about 20 mm/px.
- It is unknown whether the scale is constant across tiles.

### 5.8 Hardware and environment

- **Everything (data build, tests, training, evaluation) runs on one Linux machine:** 8× NVIDIA RTX A5000 24 GB, 64 CPU cores, 251 GB RAM, 2.8 TB free disk.
- **The GPUs are shared with other users:** at the time of checking, 7 of the 8 GPUs already had 2.5–16 GB in use. Before each training run, pick GPUs with enough free memory (`nvidia-smi`) and pass them explicitly via `--device`.
- The base Python is miniconda Python 3.14, without PyTorch or Ultralytics. The project gets its own environment with **Python 3.12** (§8: pin the Ultralytics version).
- The earlier assumption of a separate Windows laptop for code and data conversion no longer applies. Paths are still passed as parameters, not hard-coded.
- Note on Python ≥ 3.14: the default multiprocessing start method is no longer `fork`, so worker functions must be importable at module level and entry points need an `if __name__ == "__main__":` guard.

### 5.9 License

FloorPlanCAD labels are CC BY-NC 4.0; Ultralytics is AGPL-3.0. Suitable for research. Record both licenses in the README.

## 6. Decided design

### 6.1 Two models, one data source

The user chose option A: **two separate YOLO11n models**.

- **D:** `yolo11n.pt`, detect, 30 thing classes.
- **W:** `yolo11n-seg.pt`, segmentation, walls (and curtain walls and railings if they qualify).

Rationale: the detector is directly comparable with the papers and with the HF model; wall polygons feed straight into the 3D step; each converter is simpler; the two models can train in parallel on separate GPUs.

Two options were rejected:
- **One seg model for everything:** needs masks for things (hard to do accurately with door swings); walls dominate the count in the same head; box mAP is hard to compare with detect baselines.
- **OBB for walls:** cannot represent L-shaped walls, chamfered walls or curved walls. Could be a side experiment later, since OBBs can be derived from W's polygons.

### 6.2 Versioned class table (class schema v0)

- A single config file (e.g. `configs/classes_v0.yaml`) is the source of truth for every script. Each class has:
  - **`schema_id`**: a stable global ID, never changed or reused;
  - the original FloorPlanCAD `semantic-id`;
  - a name;
  - a group: door / window / furniture / sanitary-kitchen / **appliance-hvac** (currently only airconditioner, ID 21) / vertical-circulation / wall;
  - the owning model (D or W), and the local YOLO index within that model.
- **Extension rule:** a new class only gets a new `schema_id`; old classes are never renumbered. Future HVAC classes get new `schema_id`s.
- The Ultralytics dataset YAML is **generated** from this file, never hand-written.
- All 30 thing classes are kept in training (including IDs 6, 7, 8), so the class table stays stable for future extensions.

### 6.3 Image preprocessing (shared by D and W)

- **Use the provided PNGs and take the alpha channel as "ink"**: pixel value = 255 − alpha, giving black strokes on white, discarding RGB color entirely *(author's decision)*.
  - Rationale: prevents learning layer colors and helps transfer to HVAC drawings.
  - Ordinary luminance grayscale is not used: about 31% of stroke pixels would be brighter than 200/255 (yellow and light-blue strokes, anti-aliasing), and the intensity would still encode the layer.
  - Save as 3 identical channels so Ultralytics reads it normally.
- Keep each image's true size (including 999 px images). Labels are normalized by the viewBox (§5.3).
- Output file names include the archive name, e.g. `train-00_0000-0002.png`, to avoid collisions.
- SVG/PNG coordinate alignment was confirmed by the review. The automated check in §7 re-confirms it on the whole dataset, including the 999 px images.

### 6.4 Dedup and split

The user chose to **re-split all data by drawing**, discarding the official split. After review, "by drawing" means **by building cluster**, since the same building can repeat under several drawing numbers.

**Step 1 — Remove exact duplicates** *(author's decision)*:
- Each set of byte-identical PNGs keeps **one** tile: the one with the most thing instances. Ties go to the one with more wall strokes, then by name order.
  - Rationale: missing labels are more common than extra labels (§5.3).
  - Keeping several copies would teach conflicting labels on the same image, and in test one image would be counted several times.
- Expected result: 15,663 − (3,474 − 1,163) = **13,352 tiles** (confirmed by md5 on the extracted data on 2026-11-11).
- Write the list of dropped tiles with reasons.

**Step 2 — Clustering:**
- Initial groups are source drawings: `("A", NNNN)` for `train-00`; `("B", NNNN)` for `train-01` ∪ `test-00`, since those two archives share numbering.
- **Merge groups that have near-duplicate tiles** (union-find) into building clusters.
  - The near-duplicate measure is chosen by the implementer, e.g. similarity on downscaled ink images.
  - Default threshold 0.95.
  - It must catch at least the pairs with the same `MMMM`. A global image hash is recommended in addition, to catch shifted pairs.
  - At 0.95 the review estimated about 1,153 clusters, the largest with 450 tiles.
- Check `train-00` against the rest separately. The review found them almost disjoint: 0 exact duplicates, about 30 weak candidates to inspect visually.
- Report the number of clusters, the largest cluster size and a few example pairs.

**Step 3 — Split by cluster:**
- Ratio about **70/10/20 for train/val/test**, counted in tiles *(author's decision, similar to ArchCAD-400K's 7:1:2 by drawing)*. All tiles of a cluster go to the same set.
- **Stratified:** choose clusters so that each thing class (if it has enough samples) and the wall class appear in all three sets, with instance ratios close to 70/10/20. Classes too rare for this are recorded.
- Fixed seed. **The list of clusters and tiles per set is written to a file** and shipped with the deliverables.

**Using the sets:**
- **Test is used only for the final evaluation.** All choices of epoch, hyperparameters and confidence thresholds are based on val only.
- D and W use **the same split**. W additionally drops tiles with low wall coverage (§6.6).
- The conversion code must support choosing the split via config. If a comparison with the papers is needed later, simply retrain on the official split (still with the dedup step).

### 6.5 Labels for model D

- Each pair (`semantic-id` ∈ 3–32, `instance-id` ≥ 0) is one object.
- **bbox = union of the exact geometric extents** of all primitives of the object:
  - line segments, circular and elliptical arcs (including arc extrema, not just the two endpoints), Bézier curves, circles;
  - **apply the `transform` attribute**;
  - compute in SVG coordinates, then normalize by the viewBox.
- **Box rules** *(author's decision; applied identically to train, val and test)*:
  - A box with a side below 2 px (thin objects such as lines) is padded evenly to 2 px, not dropped.
  - Objects cut by the image border: clip the box to the image. **Drop the object if the visible part is under 30% of the unclipped box area.**
  - Never drop a box just for being small. Small doors and windows are real, and AP by size will show that difficulty.
- Images without any things are kept as negatives.

### 6.6 Labels for model W

**Classes:**
- Wall (ID 1) always.
- Curtain wall (ID 2) and railing (ID 33) are only included if a check shows that most of their outlines close into regions the way walls do. Otherwise they are excluded and the reason recorded.

**Building wall regions:**
- Take all strokes of the class (arcs computed exactly, not as chords).
- **Close small gaps** between nearby stroke endpoints, below a threshold.
- Add the four image-border edges to close walls cut by the border, then build closed faces (polygonize on the noded lines).
- **Keep only "thin" faces** that are actual wall bodies. Filter by thickness, e.g. the largest inscribed-circle radius ≤ a threshold based on the maximum wall thickness. This removes faces that are rooms enclosed by walls.
- Union the kept faces, so pieces split by hatching merge into a wall body.

**Instances and holes:**
- One instance = **one connected wall body** (a connected component of the wall region).
- A wall body with holes is encoded as **a single polygon with zero-width "bridges"**. Each hole is connected to the outer boundary by the shortest segment, like Ultralytics' `merge_multi_segment`.
- Check: re-rasterizing the polygon must match the original region with IoU ≥ 0.99.

**Coverage and the low-coverage tile rule** *(author's decision)*:
- Tile coverage = fraction of wall stroke length lying on the boundary of the generated region.
- **Tiles with walls whose coverage is below 80% are excluded from the W dataset**, in train, val and test. They are still used for D.
- Tiles without walls are kept as W negatives.
- The threshold may only be tuned based on visual inspection on the train set, before training W.
- Report the coverage distribution and the exclusion rate per set. Note in the report: W's test set is therefore biased toward drawings with "clean" walls.

**Known label noise:** some columns are labeled as walls, most columns are not. Record this in the report; do not fix labels.

### 6.7 Training

**Initial weights:**
- `yolo11n.pt` for D, `yolo11n-seg.pt` for W (COCO pretrained).
- **Baseline:** `yolov8n.pt` for D, with identical settings. No baseline for W *(author's decision)*.

**Main settings:**
- **Image size:** `imgsz=1024`, close to the native resolution of 1000 *(author's decision)*.
- **Batch:** the largest that fits in 24 GB. Because the GPUs are shared (§5.8), the batch must fit in the memory actually free on the GPUs used; it is fixed once and kept identical for every run that is compared.
- **Schedule:** up to 300 epochs with `patience=50` *(author's decision)*. Select the checkpoint by val mAP50-95 (for W, mask mAP50-95).
- **Explicit optimizer, never `optimizer=auto`.** To the author's understanding (not yet checked on the version that will be pinned), `auto` switches between AdamW and SGD depending on the number of iterations, so two models or two runs could use different optimizers without anyone knowing. Default: SGD with Ultralytics' default `lr0` *(author's decision)*. The implementer may change it based on val, but every run used for comparison must use the same optimizer.
- **Augmentation:** Ultralytics defaults, with three changes:
  - HSV augmentation off, since images are black ink only;
  - horizontal and vertical flips on;
  - no arbitrary-angle rotation, since boxes would inflate.
- **Seeds:** experiments use 1 seed. The final configurations of D (YOLO11n and YOLOv8n) and W run **3 seeds** (0, 1, 2), 9 runs in total. With 8 GPUs, runs may execute in parallel on different free GPUs.

**Recorded for each run:** `args.yaml`, `results.csv`, Ultralytics/PyTorch/CUDA versions, GPU index and model, class-table version, split file, training time.

### 6.8 Evaluation (on the new test set)

**Class set for mAP — fixed before training, written to a file:**
- Right after splitting, count instances per class. **"Core classes"** are those with ≥ 50 instances in train **and** ≥ 15 instances in test *(author's decision)*.
- **D's headline = mean mAP50-95 over the core classes.** Always report mAP50-95 over all 30 classes alongside, and flag the classes below the threshold.
- **Compute mAP with the project's own evaluator** (e.g. pycocotools-equivalent on exported predictions) on exactly the fixed class set. Do not rely on Ultralytics' default averaging: to the author's understanding (not yet checked), it skips classes absent from the evaluation set, silently changing the class set.

**Model D:**
- Headline as above. Also:
  - mAP50, mAP75, precision/recall;
  - per-class AP, with each class's test instance count; **airconditioner** (ID 21) is called out separately;
  - AP by size small/medium/large (COCO definition);
  - confusion matrix;
  - speed (ms/image on an A5000).
- Compare YOLO11n with YOLOv8n under the same protocol.

**When a difference counts as real** *(author's decision)*:
- Report mean ± standard deviation over 3 seeds.
- Compute the 95% confidence interval of the headline difference (YOLO11n − YOLOv8n) with a **bootstrap over test clusters** (resampling building clusters, 1,000 times).
- Only call it a "real difference" when the confidence interval excludes 0 **and** the difference exceeds 2× the across-seed standard deviation.
- State the limitation: 3 seeds measure training noise, not split noise.

**Model W:**
- **Main metric: pixel IoU of the wall region,** pooled over the whole test set (total intersection / total union), comparing the union of all predicted masks with the ground-truth wall region, ignoring instances. Rationale: the 3D step needs the correct wall region, not the correct instance partition.
  - The confidence threshold for building masks is **chosen on val** (maximizing IoU), then kept fixed for test.
  - Masks are evaluated at the **image's native resolution** (`retina_masks`, or upsampling masks to the image size), not at the head's 1/4 resolution.
- Secondary: mask mAP50-95; three seeds as for D.
- If a constant mm/px scale is established, also report wall-thickness error in mm.

**Ceiling from label noise:** report the label disagreement between byte-identical copies (§5.3) as an estimated ceiling for the results.

**Context, not a fair comparison:**
- The HF model reaches mAP50-95 0.671, but measured on val, with different data and a different split.
- Primitive-level instance-spotting numbers on FloorPlanCAD use a different metric.
- All published FloorPlanCAD numbers use the official split, which leaks by drawing and by copy, so they may all be optimistic.
- Do **not** run the HF model on our test set: it may have trained on part of `test-00`, and `test-00` is now spread across all three new sets.

**Qualitative:** about 20 test images with predictions next to ground truth, and typical error categories (wrong class, missed small objects, loose boxes, errors due to missing labels...).

### 6.9 Inference output

A script takes an image (or a folder of images) and writes **one JSON per image**, containing:
- the list of D's objects: **`schema_id`** (stable global ID, §6.2), class name, confidence score, `xyxy` bbox in pixels;
- the list of W's wall regions: `schema_id`, class name, confidence score, polygon in pixels (holes as a separate list of inner rings, no bridges);
- the model versions and the class-table version.

Each model's local YOLO indices are not written to the JSON; they are internal only.

Rationale: reusable for pre-labeling HVAC drawings, and as input to Stage 2/3 later.

### 6.10 Constraints for later integration

- Input images must be rendered at the same mm/px scale and in the same style (white background, thin black strokes of about 1 px) as in training.
- The implementer measures FloorPlanCAD's mm/px scale on ≥ 20 tiles, using dimension lines with numeric values. Record the measured value, or record that the scale varies between tiles and by how much.

## 7. Definition of done

1. **Automated converter checks pass:**
   - **Unit tests** for: circular arcs (e.g. a quarter-circle door swing); ellipses with a rotating `transform`; border-cut boxes and the 30% rule; 999 px images; wall polygons (a rectangular frame gives one polygon; a ring with a hole re-rasterizes with IoU ≥ 0.99).
   - **Bbox check against re-rasterization:** rasterize each instance separately from its primitives at native resolution, take the tight box of the ink pixels, and compare with the analytic box. Require IoU ≥ 0.9 for ≥ 99% of instances with boxes ≥ 4 px. List the failing cases.
   - Instance counts per class and per set; dedup, clustering and wall-coverage statistics.
2. **Visual check:** label overlays on ≥ 50 random images for each of D and W, cross-checked against `coco_vis`.
3. **D (YOLO11n and YOLOv8n) and W** are trained with all 3 seeds. Test numbers per §6.8 exist, computed on the class set fixed in advance.
4. **Report** in `floorplan-yolo/reports/`, covering:
   - the method;
   - the data issues found, with evidence: the official split leaks by drawing, identical images across drawing numbers, inconsistent labels between copies;
   - how dedup, clustering and re-splitting were done;
   - the final class table;
   - results, per-class analysis, error examples;
   - the measured mm/px scale;
   - the decisions made during implementation.
5. **Weights, configs, split files and the inference script** run on this machine following the README.
6. **Optional, if the user provides 5–10 project drawings:** run D and W on them (rendered in the same style and at the same scale), and comment qualitatively on domain shift in the report.

The user set no numeric target. "Success" means a correct process and honest numbers, not reaching a predefined figure *(author's decision)*.

## 8. Left to the implementer

- **Names of the 30 thing IDs:** fix them by matching SVG instances against the labels on `coco_vis` (drawn by the dataset authors), together with the CAD layer names. Record the evidence in the report.
- **The near-duplicate measure** for clustering, and how to handle the weak candidates between `train-00` and the rest (within §6.4).
- **Libraries:** for reading SVG geometry (e.g. svgelements or svgpathtools) and for building/filtering polygons (e.g. shapely).
- **W's geometric thresholds:** the thickness threshold for wall faces, the gap-closing threshold. The coverage threshold may only be tuned before training W, on the train set.
- **Training tuning:** batch size, learning rate, other augmentation tweaks. Tune only on val, and use identical settings for every run that is compared.
- **Curtain walls and railings:** whether they go into W, per the check in §6.6.
- **Ultralytics version:** pin a specific release with YOLO11 (8.3.x or later), with an environment file for this machine (Python 3.12, a PyTorch build matching the installed CUDA driver). Re-check on the pinned version the two behaviors mentioned in §6.7 and §6.8 (`optimizer=auto`, mAP averaging).
- **Report format,** charts and the sub-folder structure inside `floorplan-yolo/`.

## 9. Known risks

- **Missing, inconsistent labels** (§5.3): cap mAP, and may penalize the model when it is right. Partly handled: dedup keeps the most complete copy, and the disagreement level is reported as a ceiling.
- **Clustering misses some repeats** (shifted crops, slight edits between floors): some small leakage remains. Report the measure used and its limits.
- **Walls that do not close** (drawing errors, gaps): many tiles get excluded from W. Mitigation: gap closing; if the exclusion rate is too high (> 30% of tiles with walls), stop and ask the user before training W.
- **W's masks are at 1/4 of the input resolution,** so a 20 px wall becomes about 5 px. Wall boundaries may be loose. Mitigation: compute metrics at native resolution; try a larger `imgsz` if needed.
- **Very rare classes** (IDs 6, 7, 8): unstable or zero AP. Handled by the core class set fixed in advance (§6.8). Do not merge classes without asking the user.
- **Domain shift:** FloorPlanCAD follows Chinese drafting conventions, mostly residential. Symbols on the project's drawings may differ. Early detection: the qualitative trial in §7, if drawings are available.
- **Wrong class-name table:** only makes the report wrong, not the training. Must still be fixed before writing the report.
- **Shared GPUs** (§5.8): other users' jobs can take memory mid-run and cause out-of-memory errors or slowdowns. Mitigation: check free memory before each run, keep batch size with some headroom, and make every run resumable.

## 10. Author's decisions (for the user to review)

1. Use the alpha channel as ink: black strokes on white, RGB color discarded entirely. Color images only as a side experiment if time allows.
2. `imgsz=1024`; a 640/1024 comparison only as a side experiment.
3. Remove exact duplicates, keeping the copy with the most labels. Cluster by near-duplicate images (threshold 0.95), then a stratified 70/10/20 split by cluster. Discarding the official split was the user's decision.
4. Up to 300 epochs, `patience=50`; explicit optimizer (SGD by default), never `auto`.
5. YOLOv8n baseline for D only.
6. "Seat row" and "parking spot" are not included in this stage; their symbols remain in the images, unlabeled.
7. Box rules: pad sides below 2 px; drop border-cut objects with less than 30% visible; never drop a box just for being small.
8. D's headline = mAP50-95 over the "core classes" (≥ 50 train instances and ≥ 15 test instances), fixed before training, alongside the number over all 30 classes.
9. "Real difference" = the cluster-bootstrap confidence interval excludes 0 and the difference exceeds 2× the across-seed standard deviation.
10. W's main metric is pixel IoU at native resolution, threshold chosen on val. An instance is a connected wall body; holes are encoded with bridges.
11. Tiles with wall coverage below 80% are excluded from W (still used for D).
12. The output JSON uses global `schema_id`s.
13. No target mAP threshold. The HF model is not run on the test set.
