# YOLOv1 — PyTorch Implementation

PyTorch implementation of YOLOv1 from *You Only Look Once: Unified, Real-Time Object Detection* (Redmon et al., 2015), with a ResNet-50 backbone and several modern adaptations.

## Key Modifications

- **ResNet-50 backbone** (pretrained on ImageNet) replaces the original custom architecture
- **Fully convolutional** — no fully connected layers; the model operates on arbitrary input sizes (subject to stride-32 alignment)
- **Rectangular input** — trains on 512×320 (16×10 grid) instead of 448×448; inference dynamically preserves aspect ratio and pads to 32-aligned dimensions
- **Bounding-box activations** — sigmoid on confidence, x, y; exp on w, h; softmax on class scores (rather than using raw logits directly)
- **Single box per cell** (B=1) for simplicity
- **Two-stage training** — head-only training first, then fine-tunes the final residual block of the backbone

## Project Structure

```
├── config.py              # Hyperparameters, paths, thresholds
├── data_prep.py           # Dataset download, split, and offline preprocessing
├── dataset.py             # YOLOv1 PyTorch Dataset (YOLOv5-format labels)
├── train.py               # Training and validation loop
├── predict.py             # Inference and visualization
├── yolov1/
│   ├── model.py           # YOLOv1 model definition (ResNet-50 + conv head)
│   ├── loss.py            # Vectorized YOLO loss function
│   └── utils.py           # Decoding, NMS, IoU, plotting utilities
└── tests/                 # Unit tests
```

## Setup

```bash
pip install torch torchvision albumentations pyyaml tqdm tensorboard Pillow
```

## Data Preparation

Downloads the Udemy Self-Driving Car dataset (11 classes) from Roboflow, removes duplicates, splits chronologically (70/20/10), and preprocesses to 512×320:

```bash
python data_prep.py --api_key YOUR_ROBOFLOW_KEY
```

Or point `DATASET_ROOT` in `config.py` to your dataset with `train/`, `valid/` splits and a `data.yaml`.

## Training

```bash
python train.py                          # fresh training (ResNet-50 initialized with imagenet weights)
python train.py --weights path.pth       # initialize from weights
python train.py --resume checkpoint.pth  # resume from a saved checkpoint
```

Training logs are written to TensorBoard.

## Inference

```bash
python predict.py --image path/to/image.jpg --weights checkpoints/best.pth
```

Optional flags: `--confidence` (default 0.25), `--iou` (default 0.4), `--image_size` (default 512).

## Tests

```bash
python -m pytest tests/
```

## Known Limitations

This is a learning-oriented implementation, not a production-ready detector. YOLOv1 is a 2015 architecture. Detection quality is limited. Pretrained weights are not included.

## References

- [You Only Look Once: Unified, Real-Time Object Detection](https://arxiv.org/abs/1506.02640) (Redmon et al., 2015)
- [Udemy Self-Driving Car Dataset](https://universe.roboflow.com/roboflow-gw7yv/self-driving-car) on Roboflow
