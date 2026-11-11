"""FloorPlanCAD to Ultralytics YOLO tooling."""
import os

# Torch's default FASTEST_FIRST order can differ from nvidia-smi; force it so `--device N` is the nvidia-smi index.
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
