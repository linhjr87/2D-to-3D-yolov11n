# FloorPlanCAD → YOLO11n (architecture & furniture) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

Revision 2026-11-11: the dataset is downloaded and extracted at `dataset/FloorPlanCAD/`; everything (build, tests, training, evaluation) now runs on one Linux machine with 8 shared RTX A5000 GPUs; the separate-laptop workflow and the data upload step are removed; translated to English.

**Goal:** Build a FloorPlanCAD data pipeline → two Ultralytics datasets:
- **D:** 30 thing classes as bboxes;
- **W:** walls as polygons.

Plus scripts for training, evaluation, inference and reporting. Two models are fine-tuned: `yolo11n.pt` (D, with a `yolov8n.pt` baseline) and `yolo11n-seg.pt` (W).

**Architecture:**
- **Python package `fpyolo`:** one responsibility per module (SVG reading, ink images, boxes, walls, dedup/clustering, splitting, build, AP evaluation, training, inference).
- **One Linux machine** (8× RTX A5000 24 GB, shared with other users; 64 CPU cores; 251 GB RAM): write code, run unit tests, build the dataset once into `data/fpcad_v0/` with manifests, then train and evaluate on free GPUs.
- **Class table:** one versioned YAML file is the single source of truth for every script.

**Tech Stack:** Python 3.12 (dedicated conda environment; the base Python is 3.14 without PyTorch), ultralytics (pinned 8.3.x or later, < 9), PyTorch built for the installed CUDA driver, svgelements, shapely ≥ 2.0, numpy, pillow, opencv (via ultralytics), pycocotools (for cross-checking in tests only), pyyaml, pytest.

**Spec:** `floorplan-yolo/docs/specs/2026-11-10-floorplancad-yolo11n-design.md` (read alongside; "§x" references below point into the spec).

## Global Constraints

**Data and classes:**
- Source dataset: `/home/linh.ba/code/yolov11n-tuning/dataset/FloorPlanCAD/{train-00,train-01,test-00}`, each tile is `NNNN-MMMM.svg` + `.png`, with author visualizations in `<archive>/coco_vis/<stem>.png`. Original archives are in `dataset/FloorPlanCAD/_archives/`. **Read-only: never write into `dataset/`.** Below, `<FloorPlanCAD>` means this path.
- Verified counts (2026-11-11): 3,760 / 6,401 / 5,502 tiles; 15,663 total; 1,163 byte-identical sets covering 3,474 tiles; 13,352 tiles after dedup.
- D classes = `semantic-id` 3–32 (30 classes, `instance-id` ≥ 0).
- Area classes = {1, 2, 33, 34, 35}. W = ID 1 (wall) always; IDs 2 and 33 only if they pass the closure check; IDs 34 and 35 unused.
- `schema_id` is stable, never changed or reused. Output JSON uses `schema_id`, never YOLO indices.

**Images and labels:**
- Image = ink: `255 − alpha`, saved as 3 identical channels, true size kept (some images are 999 px).
- Normalized labels = SVG coordinates / viewBox size (never divide by 1000).
- **Box rules:**
  - side < 2 px → pad evenly to 2 px;
  - drop a border-cut object if its visible part is < 30% of the unclipped box area;
  - never drop a box just for being small.

**Dedup, clustering, splitting:**
- Remove exact duplicates (md5 of the PNG bytes): keep the tile with the most thing instances, then the most wall (ID 1) strokes, then the smallest `(archive, stem)`.
- Clustering: drawing groups `("A", NNNN)` for `train-00`, `("B", NNNN)` for `train-01`/`test-00`. Merge groups when a pair of tiles is a near-duplicate with cosine ≥ **0.95** on 32×32 ink thumbnails. Tiles with ink fraction < **0.01** are never used for linking.
- Split **70/10/20** (train/val/test) by cluster, stratified, seed **0**.
- "Core classes" = ≥ **50** train instances **and** ≥ **15** test instances. Write `core_classes.json` at build time, before any training.

**Model W:**
- Tiles with walls whose coverage is < **0.80** are excluded from W (still used for D).
- If > **30%** of the tiles with walls are excluded, the build stops with exit code 3 (the user must be asked).
- Holes are encoded with bridges (keyhole); re-rasterization must reach IoU ≥ **0.99**.

**Training:**
- `imgsz=1024`, `epochs=300`, `patience=50`, `optimizer="SGD"` (`auto` forbidden), `lr0=0.01`.
- Augmentation: `hsv_h=hsv_s=hsv_v=0`, `fliplr=0.5`, `flipud=0.5`, `degrees=0`.
- Seeds {0, 1, 2}. `batch` identical between D-yolo11n and D-yolov8n.
- **GPUs are shared:** before launching any GPU job, check `nvidia-smi` and pass the chosen free GPU explicitly with `--device <idx>`. Never default to all GPUs. Long runs are started in the background (`nohup` or `tmux`) and must be resumable.

**Evaluation:**
- COCO style, IoU 0.50:0.05:0.95, 101-point interpolation, `max_det=300`. Predictions use `conf=0.001`, `iou=0.7`.
- D headline = mAP50-95 over the core classes; always report the all-30-class number alongside.
- "Real difference": the 95% cluster-bootstrap confidence interval over test clusters (**1,000** resamples) excludes 0 **and** |Δ| > 2 × the pooled across-seed standard deviation.
- W: pooled (micro) pixel IoU at native resolution. Confidence threshold chosen on val, sweeping 0.05…0.95 in steps of 0.05.

**Environment and conventions:**
- Linux only. Every path comes from the CLI/config; never hard-code dataset or output paths. Paths with spaces and non-ASCII characters must still work (cheap robustness; tested via the `spaced_tmp` fixture).
- Multiprocessing: worker functions live at module level and every CLI entry point has an `if __name__ == "__main__":` guard (required on Python ≥ 3.14, harmless on 3.12).
- README records the licenses: FloorPlanCAD labels CC BY-NC 4.0; Ultralytics AGPL-3.0.
- **Commits:** each commit step runs `git add -A && git commit -m "<message>"` inside `floorplan-yolo/`. Every commit message ends with the line `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`.
- **Image ID** used everywhere (dataset file names, GT/pred keys, `cluster_of` keys): `<archive>_<stem>`, e.g. `train-00_0000-0002`.

## Review Focus

1. **Near-empty tiles** (little ink): they look alike and can easily chain unrelated drawings into one giant cluster. Expected: they are never used for linking. Tested in Task 6 `test_near_empty_tiles_never_linked`, and by the largest-cluster-size check in Task 8.
2. **Paths with spaces or non-ASCII characters.** Expected: every CLI works. Use the `spaced_tmp` fixture in Task 1 and Task 8.
3. **Images with no predictions, or predictions but no labels,** during evaluation. Expected: AP stays finite, missed labels count as FN, extra predictions as FP. Tested in Task 10.
4. **Classes absent from test** (IDs 7, 8 nearly empty). Expected: the "all 30 classes" mAP does not silently change its denominator; absent classes are listed in `missing_classes`. The core-class headline must raise an error if a core class has no labels. Tested in Task 10.
5. **Walls running parallel close to the image border:** they create a narrow "face" between the border and the wall that is easily mistaken for wall. Expected: excluded. Tested in Task 5 `test_border_sliver_excluded`.
6. **GPU selection on a shared machine.** Expected: no script silently uses GPU 0 or all GPUs; the device is always explicit and recorded in `env.json`. Checked in Task 11 and Task 14.

---

## File map

```
floorplan-yolo/
  pyproject.toml  environment.yml  requirements-lock.txt  README.md  .gitignore
  configs/  classes_v0.yaml  build.yaml  train_d_yolo11n.yaml  train_d_yolov8n.yaml  train_w_yolo11n_seg.yaml
  src/fpyolo/
    schema.py      class table (Task 1)
    svgparse.py    SVG → Tile/Primitive (Task 2)
    ink.py         RGBA PNG → ink image (Task 2)
    boxes.py       D labels (Task 3)
    qa.py          contact sheets, bbox–raster check, overlays (Tasks 4, 8)
    walls.py       W labels (Task 5)
    dedup.py       dedup, thumbnails, near-duplicates, clustering (Task 6)
    split.py       stratified 70/10/20 cluster split (Task 7)
    build.py       dataset build CLI + manifest + core classes (Task 8)
    scale.py       mm/px measurement (Task 9)
    apeval.py      COCO-style AP + cluster bootstrap (Task 10)
    train.py       Ultralytics training wrapper (Task 11)
    evaluate_d.py  evaluate_w.py  compare.py   (Task 12)
    predict.py     JSON export (Task 13)
    report.py      markdown result tables (Task 15)
  tests/  conftest.py  svgfixtures.py  minidata.py  test_*.py
  scripts/run_all_seeds.sh   (Task 14)
  reports/  class_names.md  scale.md  qa/  <final report>
  data/ runs/ results/       (gitignored; data and large weights)
```

**Where things run:** everything runs on this machine.
- Tasks 1–10, 13 and 15: CPU only (unit tests, data build with up to 32 workers).
- Tasks 11–13 smoke runs: CPU, or one free GPU.
- Task 14: the real training runs on free GPUs; the implementer prepares and launches them only after the user confirms which GPUs may be used.
- Task 15 is written after the Task 14 results exist.

---

### Task 1: Project scaffold and class table

**Files:**
- Create: `floorplan-yolo/pyproject.toml`, `environment.yml`, `.gitignore`, `README.md` (skeleton, with a License section), `src/fpyolo/__init__.py`, `src/fpyolo/schema.py`, `configs/classes_v0.yaml`, `tests/conftest.py`, `tests/test_schema.py`

**Interfaces:**
- Produces:
  - `ClassDef(schema_id:int, semantic_id:int, name:str, group:str, model:Literal["D","W"], enabled:bool=True)` (frozen dataclass).
  - `Schema(version:str, classes:tuple[ClassDef,...])` with methods:
    - `for_model(model) -> list[ClassDef]`: enabled classes only, ordered by `schema_id`; YOLO index = position in the list;
    - `yolo_index(model) -> dict[int,int]`: `semantic_id` → YOLO index;
    - `yolo_names(model) -> dict[int,str]`;
    - `by_schema_id(sid) -> ClassDef`.
  - `load_schema(path: Path) -> Schema`.
  - pytest fixture `spaced_tmp`: `tmp_path / "dir with space" / "non-ascii-é-路径"` (created).
- **Valid groups:** `door, window, furniture, sanitary-kitchen, appliance-hvac, vertical-circulation, wall`. Any other group → `ValueError`. A duplicate `schema_id`, or a duplicate (`model`, `semantic_id`), → `ValueError`.
- **Initial content of `classes_v0.yaml`:**
  - D: `schema_id = semantic_id − 2` for `semantic_id` 3..32. Placeholder names `sid_03`…`sid_32`, except `sid` 21 named `airconditioner`.
  - Groups by range: 3–8 door, 9–12 window, 13–19 furniture, 20–29 sanitary-kitchen (21 is appliance-hvac), 30–32 vertical-circulation.
  - W: `{schema_id 31, semantic_id 1, wall}`, `{32, 2, curtain_wall, enabled false}`, `{33, 33, railing, enabled false}`.
  - Task 4 fixes the names and groups.

- [ ] **Step 1: Create the repo and environment**

```bash
cd /home/linh.ba/code/yolov11n-tuning/floorplan-yolo && git init
conda env create -f environment.yml && conda activate fpyolo     # Python 3.12
pip install -e ".[dev]" && pip freeze --exclude-editable > requirements-lock.txt
python -c "import torch; print(torch.__version__, torch.version.cuda, torch.cuda.is_available())"   # expect True
```
`environment.yml`: name `fpyolo`, `python=3.12`, `pip`. PyTorch is installed by pip as a dependency of ultralytics; if `torch.cuda.is_available()` is False, install the PyTorch wheel matching the driver's CUDA version (see `nvidia-smi`) and re-freeze.
`pyproject.toml`: package `fpyolo` (src layout). Dependencies: `ultralytics>=8.3,<9`, `svgelements`, `shapely>=2.0`, `numpy`, `pillow`, `pyyaml`. Extra `dev`: `pytest`, `pycocotools`.
`.gitignore`: `data/`, `runs/`, `results/`, `*.pt`, `__pycache__/`.

- [ ] **Step 2: Write failing tests**

```python
def test_v0_counts_and_indices():
    s = load_schema(CONFIGS / "classes_v0.yaml")
    assert [c.semantic_id for c in s.for_model("D")] == list(range(3, 33))
    assert [c.semantic_id for c in s.for_model("W")] == [1]
    assert s.yolo_index("D")[3] == 0 and s.yolo_index("D")[32] == 29
    assert s.by_schema_id(19).name == "airconditioner" and s.by_schema_id(19).group == "appliance-hvac"
    assert s.by_schema_id(31).semantic_id == 1

def test_rejects_bad_group_and_duplicates(spaced_tmp): ...   # write 3 bad YAMLs into spaced_tmp → pytest.raises(ValueError) for each
def test_disabled_classes_excluded(spaced_tmp): ...          # enable curtain_wall → W = [1, 2]; wall's YOLO index stays 0
def test_load_from_path_with_spaces(spaced_tmp): ...         # copy classes_v0.yaml into spaced_tmp → loads OK
```

- [ ] **Step 3: Run tests, must FAIL** — `python -m pytest tests/test_schema.py -v` reports `ModuleNotFoundError: fpyolo.schema`
- [ ] **Step 4: Write `schema.py` and `classes_v0.yaml`**, `conftest.py` (fixture `spaced_tmp`, constant `CONFIGS`)
- [ ] **Step 5: Run tests, must PASS** — `python -m pytest tests/test_schema.py -v`
- [ ] **Step 6: Commit** — `git add -A && git commit -m "feat: project scaffold and versioned class schema"`

---

### Task 2: SVG parsing and ink images

**Files:**
- Create: `src/fpyolo/svgparse.py`, `src/fpyolo/ink.py`, `tests/svgfixtures.py`, `tests/test_svgparse.py`, `tests/test_ink.py`

**Interfaces:**
- Produces:
  - `Primitive(semantic_id:int, instance_id:int, bbox:tuple[float,float,float,float], polyline:np.ndarray, closed:bool)`:
    - `bbox` is (x0,y0,x1,y1) in viewBox units, **after applying every `transform`** (including transforms on parent `<g>`);
    - `polyline` is an (N,2) float64 array in viewBox units, sampling step ≤ 0.05.
  - `Tile(archive:str, stem:str, viewbox:tuple[float,float,float,float], primitives:list[Primitive], skipped:int)`.
  - `parse_tile(svg_path: Path, archive: str) -> Tile`: only takes elements whose `semantic-id` and `instance-id` both parse as int. Elements with the attributes but malformed values increment `skipped` and do not raise.
  - `load_ink(png_path: Path) -> np.ndarray` (H×W uint8 = 255 − alpha).
  - `write_ink_rgb(ink: np.ndarray, out_path: Path) -> None` (PNG with 3 identical channels).
  - `ink_fraction(ink) -> float` (fraction of pixels with `ink < 255`).
  - In `tests/svgfixtures.py`:
    - `write_svg(path, body:str, viewbox="0 0 100 100")`;
    - `write_rgba_png(path, size=(1000,1000), strokes=[...])`;
    - `tile_with(tmp_path, body:str) -> Tile` (writes an SVG, then `parse_tile`);
    - `tile_box(x0, y0, x1, y1, sid=3, iid=0) -> Tile` (builds a `Tile` directly with one rectangular primitive, without SVG).
- Use `svgelements` (`SVG.parse(..., reify=True)`) to get exact arc/ellipse extents and apply transforms. Get `polyline` by sampling along the path.

- [ ] **Step 1: Write failing tests**

```python
def test_semicircle_bbox_uses_arc_extremum(tmp_path):      # endpoints alone would give height 0
    t = tile_with(tmp_path, '<path d="M 40,50 A 10,10 0 0 0 60,50" semantic-id="3" instance-id="0"/>')
    assert t.primitives[0].bbox == pytest.approx((40, 50, 60, 60), abs=1e-3)
    t = tile_with(tmp_path, '<path d="M 40,50 A 10,10 0 0 1 60,50" semantic-id="3" instance-id="0"/>')
    assert t.primitives[0].bbox == pytest.approx((40, 40, 60, 50), abs=1e-3)

def test_rotated_ellipse_bbox(tmp_path):
    t = tile_with(tmp_path, '<ellipse cx="50" cy="50" rx="20" ry="10" transform="rotate(45 50 50)" semantic-id="9" instance-id="1"/>')
    r = 250 ** 0.5
    assert t.primitives[0].bbox == pytest.approx((50 - r, 50 - r, 50 + r, 50 + r), abs=1e-3)

def test_group_transform_applied(tmp_path):
    t = tile_with(tmp_path, '<g transform="translate(10,0)"><path d="M 0,0 L 5,5" semantic-id="3" instance-id="0"/></g>')
    assert t.primitives[0].bbox == pytest.approx((10, 0, 15, 5))

def test_polyline_dense_and_on_curve(tmp_path):            # semicircle: step ≤ 0.05; distance to center (50,50) = 10 ± 1e-6
def test_unlabeled_ignored_malformed_counted(tmp_path):    # no semantic-id → ignored; semantic-id="abc" → skipped == 1
def test_viewbox_parsed(tmp_path):                         # viewbox == (0, 0, 100, 100)
def test_circle_bbox(tmp_path):                            # circle cx 30 cy 40 r 5 → (25, 35, 35, 45)

def test_load_ink_inverts_alpha(tmp_path):                 # alpha 255 yellow → 0; alpha 0 → 255; alpha 128 → 127
def test_write_ink_rgb_three_equal_channels(tmp_path):     # cv2.imread → shape (H, W, 3), three equal channels
def test_999px_shape_kept(tmp_path):                       # 1000×999 PNG → load_ink returns (999, 1000)
```

- [ ] **Step 2: Run tests, must FAIL** — `python -m pytest tests/test_svgparse.py tests/test_ink.py -v` reports an import error
- [ ] **Step 3: Write `svgparse.py` and `ink.py`**
- [ ] **Step 4: Run tests, must PASS** (same command)
- [ ] **Step 5: Commit** — `feat: SVG primitive parsing with exact extents and alpha-ink images`

---

### Task 3: D labels (thing bboxes)

**Files:**
- Create: `src/fpyolo/boxes.py`, `tests/test_boxes.py`

**Interfaces:**
- Consumes: `Tile`, `Primitive` (Task 2); `Schema.yolo_index("D")` (Task 1).
- Produces:
  - `YoloBox(yolo_index:int, cx:float, cy:float, w:float, h:float)` (normalized values).
  - `thing_boxes(tile, sem_to_yolo: dict[int,int], img_w:int, img_h:int, min_side_px:float=2.0, min_visible:float=0.30) -> tuple[list[YoloBox], dict[str,int]]`:
    - stats has keys `n_instances`, `dropped_cut`, `padded`, `small_lt_32px`;
    - each (`semantic_id` ∈ `sem_to_yolo`, `instance_id` ≥ 0) is one object; box = union of the primitives' `bbox`;
    - normalized by the viewBox; padding and "small" are computed in pixels (`img_w`, `img_h`).
  - `format_yolo_det(boxes) -> str`: one line `idx cx cy w h` per box, 6 decimals.

- [ ] **Step 1: Write failing tests** (viewBox 100, image 1000×1000 unless stated; `MAP = {3: 0}`)

```python
def test_union_per_instance():           # two primitives with (3, 7): (10,10,20,20) and (15,15,30,25) → cx .20 cy .175 w .20 h .15
def test_stuff_and_negative_instance_ignored():   # sid 1 iid -1; sid 34; sid 3 iid -1 → []
def test_cut_object_dropped_below_30pct():
    boxes, st = thing_boxes(tile_box(-50, 10, 10, 20), MAP, 1000, 1000)   # 10/60 visible
    assert boxes == [] and st["dropped_cut"] == 1
    boxes, _ = thing_boxes(tile_box(-10, 10, 30, 20), MAP, 1000, 1000)    # 0.75 visible → clipped to x 0..30
    assert boxes[0].cx == pytest.approx(0.15) and boxes[0].w == pytest.approx(0.30)
def test_thin_box_padded_to_2px():       # horizontal line y=50, x 10..20 → h == 0.002, cy == 0.5, st["padded"] == 1
def test_999px_uses_viewbox_and_true_width():  # img_w=999: cx same as for a 1000 px image; when padding, w == 2/999
def test_format_yolo_det():              # "0 0.200000 0.175000 0.200000 0.150000"
```

- [ ] **Step 2: Run tests, must FAIL** — `python -m pytest tests/test_boxes.py -v`
- [ ] **Step 3: Write `boxes.py`**
- [ ] **Step 4: Run tests, must PASS**
- [ ] **Step 5: Commit** — `feat: thing bounding boxes with cut and thin-box rules`

---

### Task 4: Real-data checks I — class names and bbox vs. re-rasterization

**Files:**
- Create: `src/fpyolo/qa.py` (subcommands `contact-sheet`, `bbox-raster`), `tests/test_qa.py`, `reports/class_names.md`
- Modify: `configs/classes_v0.yaml` (real names and groups)

**Interfaces:**
- Consumes: `parse_tile`, `load_ink` (Task 2); `thing_boxes` (Task 3).
- Produces (reused by Task 8):
  - `render_polylines(polylines_px: list[np.ndarray], h:int, w:int) -> np.ndarray[bool]` (drawn with `cv2.polylines`, thickness 1, no anti-aliasing);
  - `tight_bbox(mask) -> tuple[int,int,int,int] | None`;
  - `box_iou(a, b) -> float`.

- [ ] **Step 1: Write failing tests** — `render_polylines` of a segment (10,10)→(20,30) gives `tight_bbox == (10,10,20,30)`; `box_iou` of two identical boxes = 1, disjoint = 0; an empty mask gives `tight_bbox` `None`.
- [ ] **Step 2: Run tests, must FAIL**, then **write the code**, then **run tests, must PASS** — `python -m pytest tests/test_qa.py -v`
- [ ] **Step 3: Write the `contact-sheet` command, then run it**

```bash
python -m fpyolo.qa contact-sheet --src "<FloorPlanCAD>" --out reports/qa/classes --per-class 12 --seed 0
```
Expected output: 30 files `sid_XX.png` and files `sid_XX.txt` listing the tiles used. Each `sid_XX.png` is a grid of **side-by-side crop pairs** around the same instance, captioned with the tile name:
- left: the ink image with the box drawn;
- right: the corresponding region of `<archive>/coco_vis/<stem>.png`, where the dataset authors wrote the class name.

Converting image coordinates (0..1000) to `coco_vis` pixels (2000×2000, with a matplotlib axes frame):
- detect the axes frame (the two longest black lines in each direction) on 5 images to get the origin and scale (about 1.54 px per unit);
- verify by drawing the converted box on the `coco_vis` crop.

- [ ] **Step 4: Fix the class names** (the implementer reads the images; the user only reviews the resulting table)
  - For each `sid`, read the text label in the `coco_vis` crop in `sid_XX.png`, on ≥ 2 instances from different tiles. If two instances give different names, look at 3 more.
  - Write snake_case names following the `coco_vis` vocabulary into `classes_v0.yaml`, then fix the groups to match the names.
  - Build an evidence table in `reports/class_names.md` (sid | name | coco_vis tiles viewed | common layers).
  - Expected: 30 distinct names; `sid` 21 = airconditioner confirmed or corrected. Re-run `python -m pytest tests/test_schema.py` (adjust the first test if the name of `sid` 21 changes).
- [ ] **Step 5: Write the `bbox-raster` command, then run it**

```bash
python -m fpyolo.qa bbox-raster --src "<FloorPlanCAD>" --n 500 --seed 0 --out reports/qa/bbox_raster.json
```
For each instance with a box ≥ 4 px, compute `box_iou(analytic box, tight_bbox(render of the instance's polylines))`.
- **Pass:** ≥ 99% of instances have IoU ≥ 0.9. The JSON records the pass rate and the list of failing instances.
- **Fail:** add a reproducing test to `test_svgparse.py`, fix `svgparse.py`, then re-run.

- [ ] **Step 6: Commit** — `feat: QA contact sheets, bbox raster check, confirmed class names`

---

### Task 5: W labels (wall regions)

**Files:**
- Create: `src/fpyolo/walls.py`, `tests/test_walls.py`

**Interfaces:**
- Consumes: `Tile` (Task 2).
- Produces:
  - `WallResult(polygons: list[shapely.Polygon], coverage: float | None, stroke_len_px: float)`:
    - polygons in normalized [0,1] coordinates, possibly with holes;
    - `coverage` is `None` when the tile has no strokes of the selected classes.
  - `wall_regions(tile, semantic_ids:set[int], img_w:int, img_h:int, gap_px:float=3.0, max_thickness_px:float=60.0, max_border_frac:float=0.4) -> WallResult`.
  - `keyhole_ring(poly: shapely.Polygon) -> np.ndarray` (single (N,2) ring, holes joined to the exterior by the shortest segment).
  - `format_yolo_seg(yolo_index:int, ring:np.ndarray) -> str` (6 decimals).
- **Algorithm (in pixels, converted to normalized at the end):**
  1. Take the polylines of `semantic_ids`, scaled by `img_w/vb_w`, `img_h/vb_h`.
  2. Gap closing: for each endpoint, if another stroke is closer than `gap_px`, add a connecting segment.
  3. Add the image frame; `unary_union` to node intersections, then `polygonize`.
  4. **Keep a face** when both hold:
     - inscribed radius (distance from `polylabel` to the boundary) ≤ `max_thickness_px/2`;
     - fraction of the perimeter on the image frame < `max_border_frac`.
  5. `unary_union` the kept faces, then split into individual polygons.
  6. **Coverage:** length of the original strokes (before gap closing) within 1.5 px of the region boundary, divided by the total stroke length.

- [ ] **Step 1: Write failing tests** (viewBox 100, image 1000; 1 unit = 10 px)

```python
def test_rect_outline_is_one_body():         # frame 10..60 × 20..22 → 1 polygon, area 0.01, coverage ≥ 0.99
def test_room_interior_excluded():           # 4 walls 2 units thick around a 30×30 room → 1 wall body with 1 hole; area ≈ wall area
def test_thick_wall_kept():                  # wall 60 px thick → kept (radius 30 = 60/2)
def test_border_cut_wall_closed_by_border(): # two strokes y=40 and y=42 from x=-5 to 20, open on the left → 1 polygon from x=0 to 20
def test_border_sliver_excluded():           # wall body at y 5..7 units → no polygon in the band y 0..5
def test_hatch_lines_merged():               # rectangular frame + 5 diagonal hatch strokes inside → 1 polygon
def test_gap_closing():                      # 2 px gap at one corner: gap_px=3 → 1 polygon; gap_px=1 → 0 polygons and coverage < 0.5
def test_no_wall_strokes():                  # coverage is None, polygons == []
def test_keyhole_roundtrip_iou():            # outer ring 0..100, inner 30..70 px → fillPoly(keyhole) vs. raster of the polygon: IoU ≥ 0.99
def test_format_yolo_seg():                  # starts with "0 ", has ≥ 6 numbers, every value in [0,1]
```

- [ ] **Step 2: Run tests, must FAIL** — `python -m pytest tests/test_walls.py -v`
- [ ] **Step 3: Write `walls.py`**
- [ ] **Step 4: Run tests, must PASS**
- [ ] **Step 5: Commit** — `feat: wall body polygons with gap closing, border handling and keyhole encoding`

---

### Task 6: Dedup and clustering

**Files:**
- Create: `src/fpyolo/dedup.py`, `tests/test_dedup.py`

**Interfaces:**
- Produces:
  - `TileId = tuple[str, str]` (archive, stem).
  - `exact_duplicate_sets(md5_by_tile: dict[TileId,str]) -> list[list[TileId]]`: only sets with ≥ 2 tiles, stably sorted.
  - `choose_keeper(dup_set, thing_count: dict[TileId,int], wall_strokes: dict[TileId,int]) -> TileId`.
  - `drawing_group(t: TileId) -> tuple[str,str]`: `("A",NNNN)` for `train-00`, otherwise `("B",NNNN)`.
  - `thumbnail(ink: np.ndarray, size:int=32) -> np.ndarray`: float32 vector with L2 norm = 1, length `size*size`, area average of `(255−ink)/255`.
  - `near_duplicate_pairs(thumbs: np.ndarray, valid: np.ndarray, threshold:float=0.95, block:int=2048) -> list[tuple[int,int]]`: only pairs i<j, both `valid`; cosine computed in blocks with matrix multiplication.
  - `build_clusters(tiles: list[TileId], pairs: list[tuple[int,int]]) -> dict[TileId,int]`: union-find over drawing groups, so all tiles of a group share a cluster; cluster ids numbered from 0 in a stable order.

- [ ] **Step 1: Write failing tests**

```python
def test_exact_sets():                    # 5 tiles, md5 {a,a,b,c,c} → [[t0,t1],[t3,t4]]
def test_keeper_rule():                   # thing counts 5 and 7 → pick 7; tie → more wall strokes; still tied → smallest (archive, stem)
def test_drawing_group():                 # ("train-00","0012-0010") → ("A","0012"); ("test-00","0001-0023") → ("B","0001")
def test_thumbnail_unit_norm():           # shape (1024,), norm 1 ± 1e-6
def test_pairs_threshold_and_blocking():  # 5 thumbnails, block=2: identical pair → present; cosine-0.9 pair → absent
def test_near_empty_tiles_never_linked(): # two identical tiles with ink_fraction 0.001 → valid False → no pair
def test_clusters_union_groups():         # a pair linking groups g1–g2 → same cluster; g3 separate; tiles without pairs follow their group's cluster
```

- [ ] **Step 2: Run tests, must FAIL**, then **write `dedup.py`**, then **run tests, must PASS** — `python -m pytest tests/test_dedup.py -v`
- [ ] **Step 3: Commit** — `feat: exact dedup, near-duplicate pairs and drawing clusters`

---

### Task 7: Stratified cluster split

**Files:**
- Create: `src/fpyolo/split.py`, `tests/test_split.py`

**Interfaces:**
- Produces:
  - `stratified_cluster_split(cluster_of: dict[TileId,int], class_counts: dict[TileId, Counter[int]], ratios=(0.7,0.1,0.2), seed:int=0) -> dict[TileId, Literal["train","val","test"]]`:
    - `class_counts` holds the per-class D instance counts, plus a pseudo-class `1` = 1 if the tile has wall strokes.
  - `split_report(assign, class_counts) -> dict`: tile ratio per set; per-class instance counts per set; list of classes absent from some set.
- Algorithm (fixed by the author):

```text
aggregate per cluster: n_tiles[k], cnt[k][c]
order: clusters containing rarer classes first (key = min_c total[c] over the classes present), then larger clusters first, ties shuffled with the seed
for each cluster k: choose the set s with the largest need, where
  need(s) = max_c∈k (ratio_s·total[c] − assigned[s][c]) / total[c]   (ties broken by need in tile counts)
assign k to s and update assigned
```

- [ ] **Step 1: Write failing tests**

```python
def test_cluster_integrity():           # all tiles of a cluster in the same set
def test_tile_ratios_within_2pts():     # 200 clusters, random sizes 1–10 → |ratio − (0.7, 0.1, 0.2)| ≤ 0.02
def test_rare_class_in_every_split():   # a class present in only 6 clusters → present in train, val and test
def test_deterministic():               # same seed → same result; different seed → different
def test_huge_cluster_reported():       # one cluster holding 30% → goes to train; split_report records the actual ratios
```

- [ ] **Step 2: Run tests, must FAIL**, then **write `split.py`**, then **run tests, must PASS** — `python -m pytest tests/test_split.py -v`
- [ ] **Step 3: Commit** — `feat: stratified cluster-level train/val/test split`

---

### Task 8: Dataset build CLI and the real-data build

**Files:**
- Create: `src/fpyolo/build.py`, `configs/build.yaml`, `tests/minidata.py`, `tests/test_build.py`
- Modify: `src/fpyolo/qa.py` (add the `overlays` and `pair-sheet` subcommands)

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces:
  - **Command:** `python -m fpyolo.build --src <dir> --out <dir> --config configs/build.yaml [--workers N] [--reuse-manifest <tiles.csv>]`.
  - **Output layout:**
    - `D/images|labels/{train,val,test}/<archive>_<stem>.png|txt` and `D/data.yaml` (absolute `path`; `names` from `Schema.yolo_names("D")`);
    - `W/...` likewise; images are hardlinked, or copied if hardlinking fails;
    - `manifests/tiles.csv` with columns `archive, stem, md5, keep, drop_reason, group, cluster, split, n_things, wall_strokes, wall_coverage, in_W, align_frac, width, height`;
    - tiles without things still get an image and an **empty label file** (negatives); likewise for W;
    - `manifests/stats.json` with keys `dedup, clusters, cross_space_candidates, split_report, box_stats, w_candidates, w_coverage, misaligned`:
      - `dedup` includes the label disagreement between byte-identical copies: `agree_things_frac`, `agree_walls_frac`, `missing_instance_frac` (label-noise ceiling, §6.8);
      - `cross_space_candidates` is the 30 `train-00` ↔ group-B pairs with the highest cosine in [0.85, 0.95);
    - `manifests/core_classes.json` as `{"min_train":50,"min_test":15,"semantic_ids":[...],"schema_ids":[...]}`;
    - `cache/` holding the per-tile parse results.
  - **Exit codes:**
    - 3: the number of wall tiles excluded from W exceeds `stop_if_excluded_frac`;
    - 4: more than 1% of tiles have `align_frac` < 0.9.
- `build.yaml` contains:
  - `schema`, `archives`, `split_mode: cluster | official`, `ratios`, `seed`;
  - `near_dup: {threshold: 0.95, min_ink: 0.01, thumb: 32}`, `boxes: {min_side_px: 2, min_visible: 0.30}`;
  - `walls: {gap_px: 3, max_thickness_px: 60, coverage_min: 0.80, stop_if_excluded_frac: 0.30, candidates: [2, 33], candidate_enable_median_cov: 0.80}`;
  - `core: {min_train: 50, min_test: 15}`;
  - `manual_merges: []` (a list of group pairs, e.g. `[["A","0012"],["B","0731"]]`).
- **`align_frac`:** draw the polylines of every labeled primitive at native resolution, then compute the fraction of drawn pixels that fall in the `alpha>0` region dilated by 1 px.
- **`split_mode: official`:** test = `test-00`; val = 10% of the clusters of the rest. This mode is for later use and only needs a small test.
- **`--reuse-manifest`:** skips dedup, clustering and splitting; uses the existing `keep` and `split` from the manifest.
- **`w_candidates`:** for IDs 2 and 33, record the median coverage over the tiles containing that class and a `would_enable` flag (median ≥ 0.80). The build does **not** enable classes automatically; a human edits `classes_v0.yaml`.

- [ ] **Step 1: Write failing tests**
  - `tests/minidata.py` has `make_minidata(root) -> MiniData(src: Path, expected_keeper: str, low_cov_id: str, shared_prefix: str)`, writing 3 archives with about 14 tiles:
    - a pair of byte-identical PNGs with different labels;
    - a 999 px tile;
    - a wall tile with low coverage;
    - an empty tile;
    - `train-01`/`test-00` tiles sharing a prefix.

```python
def test_build_end_to_end(spaced_tmp):     # test config: min_train=1, min_test=1, ratios so that every set gets ≥ 1 cluster
    md = make_minidata(spaced_tmp)
    rc = main(["--src", str(md.src), "--out", str(spaced_tmp / "out"), "--config", str(TEST_CFG)])
    assert rc == 0
    rows = read_csv(spaced_tmp / "out/manifests/tiles.csv")          # local helper in the test
    assert kept_ids_of_dup_pair(rows) == [md.expected_keeper]        # the tile with more things
    assert len({r["split"] for r in rows if r["group"].endswith(md.shared_prefix) and r["keep"] == "True"}) == 1
    assert row(rows, md.low_cov_id)["in_W"] == "False"
    assert any((spaced_tmp / "out/D/labels").rglob(f"{md.low_cov_id}.txt"))
    assert yolo_names(spaced_tmp / "out/D/data.yaml") == schema.yolo_names("D")
    assert json.load(open(spaced_tmp / "out/manifests/core_classes.json"))["min_train"] == 1
def test_stop_when_too_many_wall_tiles_excluded(spaced_tmp):  # stop_if_excluded_frac=0.0 → rc == 3
def test_misaligned_tiles_exit_4(spaced_tmp):                 # shift the PNG by 30 px → rc == 4
def test_reuse_manifest_reproduces_split(spaced_tmp):         # build twice with --reuse-manifest → identical split column
```

- [ ] **Step 2: Run tests, must FAIL**, then **write `build.py`**, then **run tests, must PASS** — `python -m pytest tests/test_build.py -v`
- [ ] **Step 3: Write `qa overlays`** — `python -m fpyolo.qa overlays --dataset <out>/D --n 50 --seed 0 --out reports/qa/overlays_D`, with a W mode that draws polygons. Add a smoke test on `minidata` (produces exactly n images).
- [ ] **Step 4: Run on the real data**

```bash
python -m fpyolo.build --src "<FloorPlanCAD>" --out data/fpcad_v0 --config configs/build.yaml --workers 32
```
Expected in `stats.json`:
- `dedup.kept` = 13,352 (already confirmed by md5 on 2026-11-11; any other value is a bug);
- `clusters.n` about 1,000–1,300, largest cluster < 10% of tiles;
- `misaligned` ≤ 1%;
- W exclusion rate ≤ 30%;
- `split_report`: tile ratios within ±0.02.

If not met:
- **Largest cluster ≥ 10%:** inspect the linking pairs inside that cluster (Review Focus 1), adjust `min_ink` or the measure, record the reason.
- **Exit code 3:** stop and report to the user (§9).

- [ ] **Step 4b: Inspect the candidate pairs between `train-00` and group B** (the implementer inspects them; the user only looks at pairs marked "uncertain")
  - Run `python -m fpyolo.qa pair-sheet --stats data/fpcad_v0/manifests/stats.json --src "<FloorPlanCAD>" --out reports/qa/cross_space`: one image per pair showing the two ink tiles side by side, captioned with the cosine.
  - Inspect the 30 pairs. If a pair is the same building, add that group pair to `manual_merges` in `build.yaml` (the list of forced group merges; `build` feeds them into union-find), then rebuild. Record the conclusions in `reports/qa/README.md`.
- [ ] **Step 5: Enable W candidate classes** — read `w_candidates`. If ID 2 or 33 has `would_enable`, enable that class in `classes_v0.yaml`, then re-run Step 4 with `--reuse-manifest data/fpcad_v0/manifests/tiles.csv`.
- [ ] **Step 6: Visual check** — run `qa overlays` for D and W (50 images each), look at all of them, cross-check a few against `coco_vis`. Record observations in `reports/qa/README.md`.
- [ ] **Step 7: Commit** (code, config, `reports/qa` and `manifests/*.json|csv`; never the dataset images) — `feat: dataset build CLI, real-data build v0 and QA`

---

### Task 9: mm/px measurement

**Files:**
- Create: `src/fpyolo/scale.py`, `tests/test_scale.py`, `reports/scale.md`

**Interfaces:**
- Produces:
  - `numeric_dimension_texts(svg_path) -> list[DimText]`, with `DimText(value_mm:float, x:float, y:float, angle_deg:float, layer:str)`. Only `<text>` elements whose content is digits only (≥ 2 digits).
  - `match_dimension_line(t: DimText, segments) -> float | None`: find a line segment **on the same layer**, parallel to the text direction (deviation ≤ 2°), whose midpoint is ≤ 3 viewBox units from the text anchor; return its length in viewBox units.
  - `measure_tile(svg_path, img_w) -> list[float]`: mm/px estimates (= `value_mm / (len · img_w / vb_w)`).
  - Command `python -m fpyolo.scale --src <dir> --n 40 --seed 0 --out reports/scale.json`.

- [ ] **Step 1: Write failing tests** — text "3600" at (50,45) on layer `DIM` with a horizontal segment 32→68 at y=46 gives `[10.0]`. Text rotated 90° with a matching vertical segment also matches. Text "C1818" is skipped. No matching segment gives `[]`.
- [ ] **Step 2: Run tests, must FAIL**, then **write the code**, then **run tests, must PASS** — `python -m pytest tests/test_scale.py -v`
- [ ] **Step 3: Run on the real data** — pick 40 tiles with ≥ 3 estimates. Record the median per tile, then the median and IQR across tiles. Draw the matched text and segment on the images of 10 tiles for visual inspection.
  - **Conclusion:** "constant" if IQR/median ≤ 0.10, otherwise "varies". Record in `reports/scale.md`.
- [ ] **Step 4: Commit** — `feat: mm-per-pixel measurement from dimension annotations`

---

### Task 10: AP evaluator and cluster bootstrap

**Files:**
- Create: `src/fpyolo/apeval.py`, `tests/test_apeval.py`

**Interfaces:**
- Produces:
  - Data types:
    - `GT = tuple[np.ndarray(N,4) xyxy px, np.ndarray(N,) int]`;
    - `Pred = tuple[np.ndarray(M,4), np.ndarray(M,) int, np.ndarray(M,) float]`.
  - `build_cache(gts: dict[str,GT], preds: dict[str,Pred], max_dets:int=300) -> EvalCache`: COCO-style matching per image, per class, at 10 IoU thresholds. Small/medium/large area ranges use the GT box area, thresholds 32² and 96².
  - `ap_table(cache, classes, area="all", weights: Mapping[str,int]|None=None) -> np.ndarray`: shape (C,10), NaN when a class has no labels. `weights` are per-image multiplicities (used by the bootstrap).
  - `summarize(cache, core: Sequence[int], all_classes: Sequence[int]) -> dict` with keys:
    - `headline_map50_95` (over the core classes);
    - `map50_95_core, map50_core, map75_core, map50_95_all, map50_all, map75_all`;
    - `per_class {cls: {ap50_95, ap50, n_gt}}`, `by_size {small, medium, large}`, `missing_classes`.
    - Raise `ValueError` if a core class has no labels.
    - The "all" numbers average over the classes that have labels; `missing_classes` lists the rest.
  - `bootstrap_delta(caches_a: Sequence[EvalCache], caches_b: Sequence[EvalCache], core, cluster_of: Mapping[str,int], n:int=1000, seed:int=0) -> dict` with keys `delta, ci_low, ci_high`:
    - each iteration: resample clusters with replacement, convert to `weights`;
    - the iteration's delta = mean headline across A's seeds − mean across B's seeds.

- [ ] **Step 1: Write failing tests**

```python
def test_matches_pycocotools():           # 20 images, 3 classes, random boxes (seed 0); COCOeval bbox with maxDets [1,10,300]
    s = summarize(build_cache(gts, preds), core=[0,1,2], all_classes=[0,1,2])
    assert s["map50_95_all"] == pytest.approx(coco.stats[0], abs=1e-6)
    assert s["map50_all"] == pytest.approx(coco.stats[1], abs=1e-6)
    assert s["by_size"]["small"] == pytest.approx(coco.stats[3], abs=1e-6)
def test_perfect_predictions():           # predictions = labels → every AP == 1
def test_image_without_predictions():     # add an image with 3 labels and no predictions → AP drops, stays finite
def test_predictions_without_gt():        # an image with no labels but with predictions → counted as FP
def test_missing_class_listed_not_averaged():  # class 2 has no labels → missing_classes == [2]; core=[0,2] → ValueError
def test_weights_equal_duplication():     # weights {img0: 2} == AP with img0 duplicated
def test_bootstrap_identical_models():    # A == B → delta 0, ci (0, 0)
```

- [ ] **Step 2: Run tests, must FAIL**, then **write `apeval.py`**, then **run tests, must PASS** — `python -m pytest tests/test_apeval.py -v`
- [ ] **Step 3: Commit** — `feat: COCO-style AP evaluator with fixed class sets and cluster bootstrap`

---

### Task 11: Training wrapper, configs and smoke runs

**Files:**
- Create: `src/fpyolo/train.py`, `configs/train_d_yolo11n.yaml`, `configs/train_d_yolov8n.yaml`, `configs/train_w_yolo11n_seg.yaml`, `tests/test_train_config.py`, `docs/ultralytics_notes.md`

**Interfaces:**
- Produces:
  - **Command:** `python -m fpyolo.train --config <yaml> --data <data.yaml> --seed <int> --project <dir> --name <str> --device <cpu|idx> [--batch N] [--smoke]`. `--device` is **required** (no default), so a run never lands on a GPU by accident.
  - `train_kwargs(cfg: dict, seed:int, data:str, device:str) -> dict`: raises `ValueError` if `optimizer` is `auto`.
  - `env_record(device:str) -> dict` with keys `ultralytics, torch, cuda, gpu, device, git_commit, schema_version, split_manifest_sha256`. `gpu` is the name and free/total memory of the chosen device at launch time. Written to `<project>/<name>/env.json`.
- **Config content:**
  - All three: `imgsz: 1024, epochs: 300, patience: 50, batch: 32, optimizer: SGD, lr0: 0.01, hsv_h: 0, hsv_s: 0, hsv_v: 0, fliplr: 0.5, flipud: 0.5, degrees: 0, deterministic: true, workers: 8`.
  - `model` is respectively `yolo11n.pt`, `yolov8n.pt`, `yolo11n-seg.pt`.
- **`--smoke`:** `epochs=1, imgsz=256, fraction=0.01, batch=4` (any device).

- [ ] **Step 1: Write failing tests**

```python
def test_d_configs_differ_only_in_model():   # without the "model" key the two D dicts are equal
def test_train_kwargs_values():              # imgsz 1024, epochs 300, patience 50, optimizer "SGD", flipud 0.5, degrees 0, seed 1, device "3"
def test_auto_optimizer_rejected():          # optimizer "auto" → ValueError
def test_device_required():                  # CLI without --device → argparse error (SystemExit 2)
def test_env_record_keys():                  # all 8 keys, device == the requested one
```

- [ ] **Step 2: Run tests, must FAIL**, then **write the code**, then **run tests, must PASS** — `python -m pytest tests/test_train_config.py -v`
- [ ] **Step 3: Smoke runs** (CPU; optionally one free GPU)

```bash
python -m fpyolo.train --config configs/train_d_yolo11n.yaml --data data/fpcad_v0/D/data.yaml --seed 0 --project runs/smoke --name d --device cpu --smoke
python -m fpyolo.train --config configs/train_w_yolo11n_seg.yaml --data data/fpcad_v0/W/data.yaml --seed 0 --project runs/smoke --name w --device cpu --smoke
```
Expected: `runs/smoke/{d,w}/weights/last.pt` and `env.json` exist; `args.yaml` records `optimizer: SGD`.

- [ ] **Step 4: Read the installed Ultralytics source**, then record in `docs/ultralytics_notes.md`:
  - the version;
  - the logic of `optimizer == "auto"` (file and line);
  - how `ap_class_index` drops classes without labels.
- [ ] **Step 5: Commit** — `feat: training wrapper, pinned configs, smoke runs`

---

### Task 12: D and W evaluation and comparison

**Files:**
- Create: `src/fpyolo/evaluate_d.py`, `src/fpyolo/evaluate_w.py`, `src/fpyolo/compare.py`, `tests/test_evaluate.py`

**Interfaces:**
- Consumes: `apeval` (Task 10); `walls.format_yolo_seg` (Task 5) and the manifest (Task 8).
- Produces:
  - **`evaluate_d`:**
    - Command `python -m fpyolo.evaluate_d --weights <pt> --dataset <out>/D --split test --manifest <out>/manifests --out <dir> --device <cpu|idx>`.
    - `load_split_gt(dataset_dir, split) -> dict[str,GT]`: converts labels to pixels using each image's true size.
    - `predict_split(weights, images, imgsz=1024, conf=0.001, iou=0.7, max_det=300, device) -> dict[str,Pred]`.
    - Writes `metrics.json` (= `summarize`), `per_class.csv` and `preds.npz`.
    - Also runs `model.val(split=..., imgsz=1024)` for the confusion matrix and speed (ms/image).
  - **`compare`:**
    - Command `python -m fpyolo.compare --a <dir>... --b <dir>... --manifest <out>/manifests --out <json>`.
    - Writes the mean and standard deviation of each side's headline, the `bootstrap_delta` result (with `cluster_of` from `tiles.csv`), and `real` = (CI excludes 0) and (|Δ| > 2·sqrt((sa²+sb²)/2)).
  - **`evaluate_w`:**
    - Command `python -m fpyolo.evaluate_w --weights <pt> --dataset <out>/W --out <dir> --device <cpu|idx>`.
    - `gt_union_mask(label_path, w, h) -> np.ndarray[bool]`: fills the keyhole rings with `cv2.fillPoly`.
    - `max_score_map(result, h, w) -> np.ndarray[float32]`: per-pixel max confidence, with `retina_masks=True` masks.
    - `iou_curve(pairs, thresholds) -> np.ndarray`: pooled IoU over the whole set, for all thresholds in one pass.
    - Choose `t*` on val (thresholds 0.05…0.95), then compute test IoU at `t*`.
    - Mask mAP from `model.val(split="test")`.
    - Writes `metrics.json` with keys `t_star, val_iou_at_t_star, test_pixel_iou, mask_map50_95, mask_map50`.

- [ ] **Step 1: Write failing tests**

```python
def test_load_split_gt_999px(tmp_path):     # label "0 0.5 0.5 0.1 0.2" on a 1000×999 image → px box (450, 399.6, 550, 599.4)
def test_iou_curve_perfect_and_empty():     # score 0.8 exactly on the label region → IoU 1 for t ≤ 0.8, 0 for t > 0.8; empty prediction → 0
def test_threshold_selection_picks_max():   # IoU curve [0.2, 0.6, 0.5] → picks threshold 0.10
def test_gt_mask_keeps_hole():              # keyhole ring of an annulus → center pixel is False
def test_compare_real_flag(tmp_path):       # fake metrics: large Δ, small std → real True; Δ ~ 0 → real False
```

- [ ] **Step 2: Run tests, must FAIL**, then **write the code**, then **run tests, must PASS** — `python -m pytest tests/test_evaluate.py -v`
- [ ] **Step 3: Smoke run with the smoke models** — run `evaluate_d` and `evaluate_w` on **val**, with weights `runs/smoke/*/weights/last.pt` and `--device cpu`. Expected: `metrics.json` exists (very low numbers are normal), no errors.
- [ ] **Step 4: Commit** — `feat: D/W evaluation and seed-aware model comparison`

---

### Task 13: Inference JSON export

**Files:**
- Create: `src/fpyolo/predict.py`, `tests/test_predict.py`

**Interfaces:**
- Consumes: `Schema` (Task 1), `load_ink` (Task 2).
- Produces:
  - **Command:** `python -m fpyolo.predict --d-weights <pt> --w-weights <pt> --w-conf <t*> --schema configs/classes_v0.yaml --input <img|dir> --out <dir> --device <cpu|idx> [--from-floorplancad-png]`. One `<stem>.json` per image.
  - `mask_to_polygons(mask) -> list[tuple[list[list[float]], list[list[list[float]]]]]`: uses `cv2.findContours(RETR_CCOMP)`, returns (exterior ring, holes).
  - `to_json(image_name, w, h, d_out, w_out, schema, model_ids) -> dict`. JSON shape:

```json
{"image": "x.png", "width": 1000, "height": 1000, "schema_version": "v0",
 "models": {"D": "<file>@<sha256[:12]>", "W": "<file>@<sha256[:12]>"},
 "objects": [{"schema_id": 1, "name": "single_door", "score": 0.93, "bbox_xyxy": [0, 0, 10, 10]}],
 "walls": [{"schema_id": 31, "name": "wall", "score": 0.88, "exterior": [[0, 0]], "holes": [[[0, 0]]]}]}
```

- [ ] **Step 1: Write failing tests**

```python
def test_mask_ring_gives_one_hole():          # annulus mask → 1 exterior, 1 hole
def test_schema_ids_not_yolo_indices():       # D yolo idx 0 → schema_id 1; JSON has no "yolo_index" key
def test_empty_outputs():                     # objects == [] and walls == []
def test_from_floorplancad_png(tmp_path):     # RGBA input → converted with load_ink before prediction (mocked model)
```

- [ ] **Step 2: Run tests, must FAIL**, then **write the code**, then **run tests, must PASS** — `python -m pytest tests/test_predict.py -v`
- [ ] **Step 3: Smoke run** — 3 val images with the smoke weights produce 3 valid JSON files
- [ ] **Step 4: Commit** — `feat: per-image JSON inference output keyed by schema_id`

---

### Task 14: Runbook and the real training runs (shared GPUs)

**Files:**
- Modify: `README.md` (add Setup, GPU selection, Runs, Evaluation, Resume sections)
- Create: `scripts/run_all_seeds.sh`

**Interfaces:**
- Consumes: the CLIs from Tasks 8, 11, 12, 13.

- [ ] **Step 1: Write the README and `run_all_seeds.sh`**
  - **Script:** runs the 9 runs (D-yolo11n, D-yolov8n, W × seeds 0, 1, 2), each named `{d11n,d8n,w11n}_s{seed}`. GPUs come from a required `GPUS` variable (e.g. `GPUS="6 7" scripts/run_all_seeds.sh`); with no `GPUS` the script exits with an error. It runs one job per listed GPU in parallel and queues the rest. A run that already has `weights/best.pt` and a `results.csv` with all epochs (or early-stopped) is skipped. An interrupted run is continued with `resume`. Each run's log goes to `runs/<name>.log`.
  - **README:**
    - create the environment from `environment.yml` + `requirements-lock.txt`;
    - choosing GPUs: `nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv`; use GPUs with ≥ 20 GB free; run inside `tmux` or with `nohup`;
    - evaluation and comparison commands;
    - time estimate: several GPU-days for the 9 runs in total, divided by the number of GPUs used.
- [ ] **Step 2: Syntax check** — `bash -n scripts/run_all_seeds.sh` prints nothing, then commit `docs: runbook and seed sweep script`
- [ ] **Step 3: Ask the user which GPUs may be used**, showing the current `nvidia-smi` memory table. Do not launch GPU jobs before the user answers.
- [ ] **Step 4: 1-epoch GPU trial for D and W at `batch 32`** on one approved GPU.
  - If out of memory (or if the free memory on the approved GPUs is lower): reduce `batch` **identically** in both D configs (and set W's batch separately), record it in the README, then commit.
- [ ] **Step 5: Launch `scripts/run_all_seeds.sh` in the background** with the approved `GPUS`. While it runs, check progress from the logs; if a run dies from OOM caused by another user's job, resume it once memory is free.
- [ ] **Step 6: Evaluate** with the README commands:
  - `evaluate_d` for the 6 D models;
  - `evaluate_w` for the 3 W models;
  - `compare` between yolo11n and yolov8n.
- [ ] **Step 7: Check the runs**
  - Each run has `env.json` (with the device used), and `args.yaml` records `optimizer: SGD`.
  - `results/` has 9 `metrics.json` files and `compare.json`, plus the `confusion_matrix*.png` files.

---

### Task 15: Report

**Files:**
- Create: `src/fpyolo/report.py`, `tests/test_report.py`, `reports/2026-xx-floorplancad-yolo11n-report.md` (real date at writing time)

**Interfaces:**
- Consumes: `results/**/metrics.json`, `compare.json`, `manifests/stats.json`, `reports/scale.md`, `reports/class_names.md`.
- Produces: `python -m fpyolo.report --results results --manifests data/fpcad_v0/manifests --out reports/tables.md`, generating the tables:
  - headline mean ± standard deviation;
  - per-class table, sorted by AP, flagged "not a core class" and "absent from test", with the airconditioner row in bold;
  - by-size table;
  - W (IoU, `t*`, mask mAP);
  - data statistics (dedup, clusters, coverage, label disagreement).

- [ ] **Step 1: Write failing tests** — on sample JSON: the headline shows "0.612 ± 0.004"; absent classes are flagged; the airconditioner row is bold.
- [ ] **Step 2: Run tests, must FAIL**, then **write `report.py`**, then **run tests, must PASS** — `python -m pytest tests/test_report.py -v`
- [ ] **Step 3: Write the report** covering every item of spec §7.4:
  - the method;
  - the data issues with evidence;
  - dedup, clustering, splitting;
  - the class table;
  - results;
  - about 20 qualitative examples (use `qa overlays` on test predictions);
  - the mm/px scale;
  - literature context;
  - decisions made during implementation.
  - Add a "domain shift" section if the user has provided project drawings (§7.6).
- [ ] **Step 4: Run the whole test suite** — `python -m pytest -q`, expected: all PASS
- [ ] **Step 5: Commit** — `docs: final evaluation report`
