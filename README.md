# floorplan-yolo

Converts the FloorPlanCAD dataset into Ultralytics YOLO datasets (object detection for door/window/furniture classes, wall segmentation) and trains YOLO11n models on them. The package `fpyolo` holds the dataset build, evaluation and prediction tools.

## Setup

```bash
conda env create -f environment.yml
conda activate fpyolo
pip install -e ".[dev]"
python -m pytest
```

`requirements-lock.txt` records the exact package versions used.

## License

- FloorPlanCAD labels: CC BY-NC 4.0.
- Ultralytics: AGPL-3.0.
