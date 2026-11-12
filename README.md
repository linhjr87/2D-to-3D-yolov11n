# floorplan-yolo

Converts the FloorPlanCAD dataset into Ultralytics YOLO datasets (object detection for door/window/furniture classes, wall segmentation) and trains YOLO11n models on them. The package `fpyolo` holds the dataset build, evaluation and prediction tools.

Two models are trained:

- **D**: detection, 30 object classes. `yolo11n` is compared with a `yolov8n` baseline (`configs/train_d_yolo11n.yaml`, `configs/train_d_yolov8n.yaml`).
- **W**: wall segmentation, 3 classes, `yolo11n-seg` (`configs/train_w_yolo11n_seg.yaml`).

Each model is trained with seeds 0, 1, 2: 9 runs in total, named `{d11n,d8n,w11n}_s{0,1,2}`.

## Setup

```bash
conda create -n fpyolo --override-channels -c conda-forge python=3.12 pip
conda activate fpyolo
pip install -e ".[dev]"
python -m pytest -q
```

`conda env create -f environment.yml` does the same thing, but with a recent conda it can stop with a Terms-of-Service error for the `defaults` channel. The `conda create --override-channels -c conda-forge ...` line above avoids the `defaults` channel and worked.

`requirements-lock.txt` records the exact package versions used on the build machine. It contains a CUDA 13 build of torch and its `nvidia-*` libraries, which only work with a new enough driver, so on a different machine follow "Move to another server" below instead of installing the file as it is.

## Move to another server

Training and evaluation run on a GPU server. All commands below run from the repository root (Ultralytics downloads and looks for `yolo11n.pt`, `yolo11n-seg.pt`, `yolov8n.pt` in the current directory).

1. **Code.**

   ```bash
   git clone <GitHub URL of the repository> floorplan-yolo   # `git remote -v` on the build machine shows it
   cd floorplan-yolo
   git checkout feat/fpyolo-v0
   ```

2. **Environment.** Create the conda environment as in "Setup" above, then install torch for the driver of this server. `nvidia-smi` prints the highest CUDA version the driver supports (top right); pick the matching wheel index from <https://pytorch.org/get-started/locally/> (for example `cu126`, `cu128`, `cu130`). Use the versions in `requirements-lock.txt` (`torch==2.15.0`, `torchvision==0.29.2`) if that index has them, otherwise the closest ones, and write down what you installed.

   ```bash
   pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cu130   # adjust the index
   # everything else from the lock file, without the CUDA-specific lines:
   grep -viE '^(torch|torchvision|triton|nvidia-|cuda-)' requirements-lock.txt > /tmp/requirements-nogpu.txt
   pip install -r /tmp/requirements-nogpu.txt
   pip install -e ".[dev]"
   python -c "import torch, ultralytics; print(torch.__version__, torch.cuda.is_available(), ultralytics.__version__)"
   python -m pytest -q
   ```

   The last `python -c` line must print `True` and `8.4.210` (the Ultralytics behaviour documented in `docs/ultralytics_notes.md` was read from that version). The tests need no GPU.

3. **Dataset.** It lives in a private Hugging Face dataset repo (`linhjr87/2D-to-3D-yolov11n`: `fpcad_v0.tar`, `fpcad_v0.tar.sha256`, `README.md`). Log in once (`pip install -U huggingface_hub`, then `hf auth login`, or `export HF_TOKEN=...`), then:

   ```bash
   scripts/fetch_dataset.sh            # or: scripts/fetch_dataset.sh <other HF repo>
   ```

   The script downloads into `data/_hf`, checks the SHA-256, unpacks into `data/fpcad_v0/{D,W}`, rewrites the absolute `path:` line in `D/data.yaml` and `W/data.yaml` to this checkout, and checks that the manifests in the archive are byte-identical to the committed ones in `data/fpcad_v0/manifests/` (so the train/val/test split is the one in git). It refuses to overwrite an existing `data/fpcad_v0/{D,W}` unless you run it with `FORCE=1`. `HF_LOCAL_DIR=<dir>` skips the download if you already have the two files in `<dir>`.

4. **Weights of the pretrained models.** Ultralytics downloads `yolo11n.pt`, `yolo11n-seg.pt` and `yolov8n.pt` from GitHub on first use, so the server needs outbound access to GitHub. Without it, copy these three files into the repository root.

## GPU selection

The GPUs of the server may be shared with other people. Never use all GPUs and never rely on a default: always pass the GPU index explicitly. First look at who uses what:

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv
nvidia-smi --query-gpu=index,pci.bus_id --format=csv     # for the cross-check after the runs
```

Use only GPUs with at least 20 GB free (at `batch 32`, `imgsz 1024`) that you are allowed to use. `fpyolo` sets `CUDA_DEVICE_ORDER=PCI_BUS_ID` on import, so `--device N` and `GPUS="N ..."` mean the `nvidia-smi` index `N`. Every run writes `env.json` with the device and the GPU's PCI bus id.

Long jobs must survive a closed terminal: run them inside `tmux` (or `screen`), or with `nohup ... > file 2>&1 &`.

## 1-epoch GPU trial (before the sweep)

The sweep uses `batch: 32` from the configs. Check once on one approved GPU that it fits. **Ultralytics does not fail on a GPU out-of-memory error in the first epoch: it halves `batch` (up to 3 times) and carries on.** So the trial passes only if `args.yaml` still says `batch: 32` and the log has no "Reducing to batch" line. Both D configs must use the same batch, so the trial covers all three configs.

```bash
GPU=6                       # the approved GPU
mkdir -p runs/trial
for t in d11n:train_d_yolo11n:D d8n:train_d_yolov8n:D w11n:train_w_yolo11n_seg:W; do
    IFS=: read -r name cfg kind <<< "$t"
    sed 's/^epochs: .*/epochs: 1/' "configs/$cfg.yaml" > "runs/trial/$name.yaml"
    python -m fpyolo.train --config "runs/trial/$name.yaml" --data "data/fpcad_v0/$kind/data.yaml" \
        --seed 0 --project runs/trial --name "$name" --device "$GPU" 2>&1 | tee "runs/trial/$name.log"
    grep -E '^batch:' "runs/trial/$name/args.yaml"
    grep -c 'Reducing to batch' "runs/trial/$name.log" || true      # must print 0
done
```

(Watch `nvidia-smi` in another terminal; the epoch time printed in the log also gives the time estimate below.) If the trial fails or the batch was halved, pick a smaller batch that fits, **the same for the D runs of both models** (W may differ) and run the sweep with `BATCH_D=<n>` and `BATCH_W=<m>` (see Runs). Then record the numbers in the table below and commit the README:

| | batch | note |
|---|---|---|
| D (yolo11n and yolov8n) | 32 (config default) | not yet trialled on the GPU server |
| W (yolo11n-seg) | 32 (config default) | not yet trialled on the GPU server |

Afterwards remove the trial: `rm -rf runs/trial`.

## Runs

```bash
tmux new -s sweep
mkdir -p runs
GPUS="6 7" scripts/run_all_seeds.sh 2>&1 | tee -a runs/sweep.log
# with a reduced batch:  BATCH_D=16 BATCH_W=16 GPUS="6 7" scripts/run_all_seeds.sh
```

- `GPUS` is required (space-separated indices); without it the script stops with an error. One job runs per listed GPU, the other jobs wait in a queue and start when a GPU is free. Nothing else is started: it only uses the GPUs you list.
- Optional variables: `BATCH_D`, `BATCH_W` (passed to `fpyolo.train --batch`; `BATCH_D` applies to both D models), `DATA_ROOT` (default `data/fpcad_v0`), `RUNS_DIR` (default `runs`), `PYTHON`.
- `DRY_RUN=1 GPUS="6 7" scripts/run_all_seeds.sh` prints what each GPU slot would do (start, resume, skip) and runs nothing.
- Run directories are `runs/<name>/` (for example `runs/d11n_s0/`) and logs are `runs/<name>.log` (appended on resume).
- Finished runs are skipped, interrupted runs are resumed, so after a crash, a reboot, or an out-of-memory error caused by another user's job you only start the same command again once memory is free. A run directory with nothing to resume from (the run died before saving its first checkpoint) is reported as failed: delete that directory (`rm -rf runs/<name>`), then start the command again.
- After each run the script runs `python -m fpyolo.runstate verify` and prints `ok <name>` or `PROBLEM <name>: ...`. It checks that `args.yaml` has `optimizer: SGD` and the requested `batch`, and that `env.json` records a GPU whose PCI bus id matches `nvidia-smi`.

Check progress with `tail -f runs/<name>.log`, `nvidia-smi`, and `runs/<name>/results.csv` (one row per epoch).

**Time estimate.** Each run is up to 300 epochs on about 9,300 (D) or 8,400 (W) images of 1024 px, with early stopping after 50 epochs without improvement. The 9 runs together take several GPU-days; with `n` GPUs the wall-clock time is roughly that divided by `n` (for example 2 GPUs: several days). The epoch time of the trial times at most 300 times 9 gives an upper bound on the GPU time.

### Manual checks after the sweep

The script already does these; repeat them by hand if in doubt:

```bash
for r in runs/*_s[012]; do
    echo "== $r"; grep -E '^(optimizer|batch):' "$r/args.yaml"
    python -c "import json,sys; e=json.load(open('$r/env.json')); print(e['device'], e['gpu']['name'], e['gpu']['pci_bus_id'])"
done
nvidia-smi --query-gpu=index,pci.bus_id --format=csv          # bus ids must match
```

Every `args.yaml` must show `optimizer: SGD`. All 6 D runs (both models) must show the same `batch`. A `batch` below the requested one means Ultralytics halved it after an out-of-memory error: discard that run and rerun it with a smaller `BATCH_D` for all D runs.

## Resume

`scripts/run_all_seeds.sh` resumes interrupted runs by itself. To do it for one run by hand:

```bash
python -m fpyolo.runstate state  runs/d11n_s0                  # absent | partial | done | broken
python -m fpyolo.runstate resume runs/d11n_s0 --device 6       # only for 'partial'
```

`resume` records the GPU in `runs/<name>/env_resume_<n>.json` and continues with `YOLO("<run>/weights/last.pt").train(resume=True, device="<idx>")`; Ultralytics restores all other settings from the checkpoint. A finished run's `last.pt` has been stripped of its optimizer state and cannot be resumed (state `done`). A run that finished is never restarted. The same first-epoch out-of-memory halving can happen in a resumed process, so check `batch` again afterwards.

## Evaluation

Run after all 9 runs are done. Use one approved GPU (`GPU=6`). Outputs go to `results/` (git-ignored):

```
results/d11n_s0 ... d11n_s2   evaluate_d output: metrics.json, per_class.csv, preds.npz, confusion_matrix*.png, curves
results/d8n_s0  ... d8n_s2    the same for the yolov8n baseline
results/w11n_s0 ... w11n_s2   evaluate_w output: metrics.json (t_star, test_pixel_iou, ...), confusion_matrix*.png
results/compare.json          yolo11n vs yolov8n, seed-aware, cluster bootstrap
```

**Memory.** Never pass image lists to Ultralytics predict (the code does not). Each evaluation process needs roughly 3 to 6 GB of host RAM. Run the **first evaluation alone**: the first `val` on the test split writes `data/fpcad_v0/D/labels/test.cache`, and several first runs started together would race on it. Cap it and measure it:

```bash
GPU=6; D=data/fpcad_v0
systemd-run --user --scope -p MemoryMax=12G -q \
    /usr/bin/time -v python -m fpyolo.evaluate_d --weights runs/d11n_s0/weights/best.pt \
        --dataset $D/D --split test --manifest $D/manifests --out results/d11n_s0 --device $GPU
# read "Maximum resident set size" in the /usr/bin/time output; if systemd-run is unavailable run the line without it
```

Then the rest, one after the other (to run several in parallel, use one GPU per process and make sure RAM is enough for `n` times the measured peak):

```bash
GPU=6; D=data/fpcad_v0
for s in 0 1 2; do
    for m in d11n d8n; do
        [ -f results/${m}_s$s/metrics.json ] && continue       # already done (the first one above)
        python -m fpyolo.evaluate_d --weights runs/${m}_s$s/weights/best.pt \
            --dataset $D/D --split test --manifest $D/manifests --out results/${m}_s$s --device $GPU
    done
    python -m fpyolo.evaluate_w --weights runs/w11n_s$s/weights/best.pt \
        --dataset $D/W --out results/w11n_s$s --device $GPU
done
python -m fpyolo.compare --a results/d11n_s{0,1,2} --b results/d8n_s{0,1,2} \
    --manifest $D/manifests --out results/compare.json
```

`evaluate_w` chooses the confidence threshold `t*` on the val split and reports the test split (`t_star` in `metrics.json`; `fpyolo.predict --w-conf` needs it).

Final check: `ls results/*/metrics.json | wc -l` prints 9, `results/compare.json` exists, and the D and W folders contain `confusion_matrix*.png`.

## Bring the results back

```bash
scripts/pack_results.sh      # writes results_<git short sha>_<UTC date>.tar.gz in the repository root
```

The archive holds `results/` (all metrics, `compare.json`, confusion matrices, `preds.npz`) and, per run, `args.yaml`, `env.json`, `env_resume_*.json`, `results.csv` and the `*.png` plots, plus all `runs/*.log`. It contains **no weights** (`*.pt`, a few MB each; copy `runs/*/weights/best.pt` separately if you want them).

Send it back with `scp`, or upload it to the same Hugging Face dataset repo under `results/` (`hf upload linhjr87/2D-to-3D-yolov11n results_<sha>_<date>.tar.gz results/results_<sha>_<date>.tar.gz --repo-type dataset`). Also push any README change (the batch table) to the GitHub branch.

## License

- FloorPlanCAD labels: CC BY-NC 4.0.
- Ultralytics: AGPL-3.0.
