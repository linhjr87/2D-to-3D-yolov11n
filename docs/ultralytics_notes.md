# Ultralytics source notes

Version read: **ultralytics 8.4.210** (`ultralytics.__version__`, also recorded in each run's `env.json`).
All paths are relative to the installed package, `<env>/lib/python3.12/site-packages/ultralytics/`, and all
line numbers refer to that version. Re-check them after any upgrade.

## `optimizer == "auto"` (why the wrapper forbids it)

`engine/trainer.py`, `build_optimizer` (def at line 1171):

- Lines 1193-1203: if `name == "auto"`, the requested `lr0` is ignored (the log line at 1195 says so).
  `nc = self.data.get("nc", 10)`, `lr_fit = round(0.002 * 5 / (4 + nc), 6)`, and then
  `("MuSGD", 0.01, 0.9) if iterations > 10000 else ("AdamW", lr_fit, 0.9)` (line 1200).
  It also overwrites `self.args.optimizer` / `self.args.lr0` (line 1201) and sets `warmup_bias_lr = 0.0` (line 1202).
- `iterations` is computed at line 330 as `ceil(len(train_dataset) / max(batch_size, nbs)) * epochs`
  (`nbs` = 64 by default), and passed in at lines 331-337. So the choice depends on dataset size,
  epochs and batch size: with `epochs=300` the same data can give AdamW at one setting and MuSGD at another.
  That would make runs with different batch sizes silently incomparable.
- Lines 1191-1192: names are matched case-insensitively against `{Adam, Adamax, AdamW, NAdam, RAdam, RMSprop, SGD, MuSGD, auto}`.
  An explicit `SGD` (line 1224) gives `lr = lr0`, `momentum = args.momentum` (default 0.937, `cfg/default.yaml` line 97)
  and `nesterov=True`; the `auto` branch instead hard-codes momentum 0.9.
- The default for `optimizer` in Ultralytics is `auto`, so a config that omits the key also lands here.
  `fpyolo.train.train_kwargs` therefore raises `ValueError` for `auto` and for a missing key.
  The smoke runs' `args.yaml` show `optimizer: SGD`, `lr0: 0.01`.

## `ap_class_index` and classes without labels

`utils/metrics.py`:

- `ap_per_class` (def at line 799): line 845, `unique_classes, nt = np.unique(target_cls, return_counts=True)` with
  `nc = unique_classes.shape[0]`, "number of classes with labels". Only classes that occur in the **validation
  labels** get a row in `ap`, `p`, `r`, `f1`. A class with labels but no predictions keeps a row of zeros
  (lines 857-859), so it pulls the mean down. A class with no labels has no row at all.
- The `return` at line 894 hands `unique_classes.astype(int)` back as the `ap_class_index`; `Metric.update`
  stores it at lines 1036-1041 (`self.ap_class_index`), and is fed by `DetMetrics.process` (def at line 1155, the
  `ap_per_class` call at line 1169; `nt_per_class` at line 1182). The segmentation variant calls `ap_per_class`
  for masks at line 1342.
- `Metric.map50`, `map` and `mean_results` (lines 973-1001) average `all_ap` over those rows only:
  `self.all_ap.mean()`. **The "all" mAP is therefore the mean over classes that have val labels, and its
  denominator silently changes with the split.** Classes absent from the val labels are not counted as 0, they are dropped.
- `Metric.maps` (line 1008-1013): `np.zeros(self.nc) + self.map`, then the real AP is written at
  `ap_class_index` positions. **Absent classes are filled with the overall mean mAP, not with 0 or NaN**,
  so a per-class list taken from `maps` looks complete but contains fabricated values for absent classes.
- `DetMetrics.nt_per_class` (line 1182) is `np.bincount(target_cls, minlength=len(names))`, so it is the reliable
  way to tell which classes are absent (count 0).
- Per-class log rows (`models/yolo/detect/val.py`, `print_results`, lines 372-380) loop over `ap_class_index`, so
  absent classes are not printed. In the smoke log, D shows all 30 rows because every class has val labels in
  this split; the test split is the one where classes 7 and 8 can be nearly empty.
- Consequence for this project: do not take Ultralytics' `all` row as the "all-30-class" number. Task 12 computes the
  headline (core classes) and the all-classes number itself in `fpyolo.apeval` with fixed class sets, and lists
  absent classes as `missing_classes`. Ultralytics' own `mAP50-95` (also used for `fitness`, line 1015-1017, i.e.
  best.pt selection) is only for checkpoint selection and early stopping.

## Other behaviour that matters for the real runs

- **OOM auto-reduce on a GPU** (`engine/trainer.py`, lines 575-600): an out-of-memory error in the first epoch
  of a single-GPU run halves `self.args.batch` (up to 3 times) and retries instead of failing. This silently breaks
  "batch identical between D-yolo11n and D-yolov8n". After every run compare the `batch:` line of `args.yaml`
  with the intended value, and pick `--batch` from an OOM trial so that no retry happens.
- `batch < 1` (e.g. `-1`) triggers `auto_batch` (line 455-456); the wrapper's configs always set an explicit batch.
- **Resume** (`check_resume`, lines 1040-1087): `YOLO("<run>/weights/last.pt").train(resume=True)` rebuilds all
  arguments from the checkpoint (line 1057, `get_cfg(ckpt_args)`), so config values and `project`/`name` are taken from
  the saved run. Only `imgsz, batch, device, close_mosaic, augmentations, save_period, workers, cache, patience, time,
  freeze, val, plots, channels_last, ...` (lines 1059-1075) can be overridden; anything else is ignored with a warning
  (line 1083). The wrapper has no resume mode: it refuses an existing `<project>/<name>` directory, and
  resuming goes through Ultralytics directly (pass `device=<idx>` explicitly). `env.json` of the original launch is kept.

## GPU indices

`select_device` (`utils/torch_utils.py`, def at line 244; body 325-372): `device=cpu` goes straight to
`torch.device("cpu")` and does not hide the GPUs. A numeric device is validated with `int(x) < torch.cuda.device_count()`
(line 328) and used as torch's `cuda:<n>`; the function never sets `CUDA_VISIBLE_DEVICES` (it only translates ids when
the user already set it externally, in `parse_device`, def at line 176, lines 195-240). Torch's default `CUDA_DEVICE_ORDER=FASTEST_FIRST` need not match
`nvidia-smi` numbering, but the user approves GPUs by their `nvidia-smi` index. Therefore `fpyolo/__init__.py` forces
`CUDA_DEVICE_ORDER=PCI_BUS_ID` (assignment, so an external value cannot break it), the wrapper sets
`CUDA_VISIBLE_DEVICES=""` for `--device cpu`, and `env.json` records the opened GPU's name, memory and PCI bus id
(compare with `nvidia-smi --query-gpu=index,pci.bus_id --format=csv`).
