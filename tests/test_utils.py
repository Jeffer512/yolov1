import math
import torch
from yolov1.utils import iou, iou_vectorized, non_max_suppression


def test_iou_scalar_perfect_overlap():
    """Verifies that scalar IoU is exactly 1.0 when two boxes are identical."""
    box = [0.1, 0.1, 0.5, 0.5]
    assert iou(box, box) == 1.0


def test_iou_scalar_no_overlap():
    """Verifies that scalar IoU is exactly 0.0 when two boxes do not intersect."""
    box1 = [0.0, 0.0, 0.2, 0.2]
    box2 = [0.8, 0.8, 1.0, 1.0]
    assert iou(box1, box2) == 0.0


def test_iou_scalar_partial_overlap():
    """Verifies that scalar IoU calculates correct overlap ratios."""
    box1 = [0.0, 0.0, 0.2, 0.2]
    box2 = [0.1, 0.1, 0.3, 0.3]
    
    # Use math.isclose to handle floating-point precision tolerances safely
    assert math.isclose(iou(box1, box2), 0.01 / 0.07)


def test_iou_vectorized_math():
    """Verifies that vectorized IoU computes accurate shapes and values."""
    box1 = torch.tensor([[[[0.5, 0.5, 0.2, 0.2]]]])
    box2 = torch.tensor([[[[0.5, 0.5, 0.2, 0.2]]]])
    
    res = iou_vectorized(box1, box2)
    
    # Verify the output shape matches expectations
    assert res.shape == (1, 1, 1, 1)
    
    # Verify values match within standard epsilon bounds
    assert torch.allclose(res, torch.tensor([[[[1.0]]]]), atol=1e-4)


def test_nms_suppression():
    """Verifies that NMS correctly suppresses highly overlapping boxes of the same class."""
    detections = [
        {"bbox": [0.1, 0.1, 0.5, 0.5], "confidence": 0.9, "class_id": 0},
        {"bbox": [0.12, 0.12, 0.52, 0.52], "confidence": 0.8, "class_id": 0},  # Expected to be suppressed
        {"bbox": [0.8, 0.8, 1.0, 1.0], "confidence": 0.7, "class_id": 0}        # Expected to be kept
    ]
    
    filtered = non_max_suppression(detections, iou_threshold=0.5)
    
    assert len(filtered) == 2
    assert filtered[0]["confidence"] == 0.9
    assert filtered[1]["confidence"] == 0.7