import torch
# Note: matplotlib and numpy are lazily imported inside plot_predictions to keep production light
from config import CONF_THRESHOLD, NMS_IOU_THRESH


def load_class_names(data_yaml_path):
    """Loads class name strings from the dataset YAML configuration file."""
    import yaml
    with open(data_yaml_path, "r") as f:
        data = yaml.safe_load(f)
    return data.get("names", [])


def scale_box(bbox, metadata):
    """Reverses the dynamic scaling and padding to restore coordinates to original dimensions."""
    x1, y1, x2, y2 = bbox
    
    orig_w = metadata["orig_w"]
    orig_h = metadata["orig_h"]
    unpadded_w = metadata["unpadded_w"]
    unpadded_h = metadata["unpadded_h"]
    padded_w = metadata["padded_w"]
    padded_h = metadata["padded_h"]
    pad_left = metadata["pad_left"]
    pad_top = metadata["pad_top"]
    
    # Reverse the padding shift
    x1_pixel = (x1 * padded_w) - pad_left
    y1_pixel = (y1 * padded_h) - pad_top
    x2_pixel = (x2 * padded_w) - pad_left
    y2_pixel = (y2 * padded_h) - pad_top
    
    # Normalize relative to the UNPADDED dimensions
    x1_unpadded = x1_pixel / unpadded_w
    y1_unpadded = y1_pixel / unpadded_h
    x2_unpadded = x2_pixel / unpadded_w
    y2_unpadded = y2_pixel / unpadded_h
    
    # Clamp values strictly within [0.0, 1.0] to handle precision errors
    x1_unpadded = max(0.0, min(1.0, x1_unpadded))
    y1_unpadded = max(0.0, min(1.0, y1_unpadded))
    x2_unpadded = max(0.0, min(1.0, x2_unpadded))
    y2_unpadded = max(0.0, min(1.0, y2_unpadded))
    
    # Scale to original raw image pixel dimensions
    return [
        x1_unpadded * orig_w,
        y1_unpadded * orig_h,
        x2_unpadded * orig_w,
        y2_unpadded * orig_h,
    ]


def decode_predictions(pred_tensor, conf_threshold=CONF_THRESHOLD):
    """Converts model grid predictions into bounding boxes.
    
    Transforms raw logits and offsets into absolute normalized coordinates (x1, y1, x2, y2).
    
    Args:
        pred_tensor (Tensor): Model predictions (batch, channels, S, S)

    Returns:
        list: List with detections for each image in the batch
    """ 
    batch_detections = []
    batch_size = pred_tensor.size(0)

    batch_size, _, S_H, S_W = pred_tensor.shape

    for b in range(batch_size):
        detections = []
        pred = pred_tensor[b] 

        for row in range(S_H):
            for col in range(S_W):
                raw_conf = pred[0, row, col]    
                confidence = torch.sigmoid(raw_conf).item()

                if confidence < conf_threshold:
                    continue

                raw_x = torch.sigmoid(pred[1, row, col]).item()
                raw_y = torch.sigmoid(pred[2, row, col]).item()
                w = torch.exp(pred[3, row, col]).item()
                h = torch.exp(pred[4, row, col]).item()

                cx = (raw_x + col) / S_W
                cy = (raw_y + row) / S_H    

                x1 = cx - w / 2
                y1 = cy - h / 2
                x2 = cx + w / 2
                y2 = cy + h / 2

                raw_classes = pred[5:, row, col]
                class_probs = torch.softmax(raw_classes, dim=0)
                class_score, class_id = torch.max(class_probs, dim=0)
                class_score = class_score.item()
                class_id = class_id.item()

                detections.append({
                    "bbox": [x1, y1, x2, y2],
                    "confidence": confidence,
                    "class_id": class_id,
                    "class_score": class_score,
                })

        batch_detections.append(detections)

    return batch_detections


def iou(box1, box2):
    """Calculates Intersection over Union (IoU) for a single pair of boxes on the CPU.
    
    Used during post-processing (NMS) on standard Python lists of corner coordinates.
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter

    if union == 0:
        return 0.0
    return inter / union


def iou_vectorized(boxes_preds, boxes_labels):
    """Calculates Intersection over Union (IoU) for batches of PyTorch tensors.

    Used during training (loss.py) on GPU-bound midpoint tensors.

    Args:
        boxes_preds (Tensor): Absolute normalized coordinates of shape (..., 4) [x, y, w, h].
        boxes_labels (Tensor): Absolute normalized coordinates of shape (..., 4) [x, y, w, h].

    Returns:            
        Tensor: IoU values of shape (..., 1).
    """
    # Convert midpoint representation (x, y, w, h) to corner representation (x1, y1, x2, y2)
    box1_x1 = boxes_preds[..., 0:1] - boxes_preds[..., 2:3] / 2
    box1_y1 = boxes_preds[..., 1:2] - boxes_preds[..., 3:4] / 2
    box1_x2 = boxes_preds[..., 0:1] + boxes_preds[..., 2:3] / 2
    box1_y2 = boxes_preds[..., 1:2] + boxes_preds[..., 3:4] / 2

    box2_x1 = boxes_labels[..., 0:1] - boxes_labels[..., 2:3] / 2
    box2_y1 = boxes_labels[..., 1:2] - boxes_labels[..., 3:4] / 2
    box2_x2 = boxes_labels[..., 0:1] + boxes_labels[..., 2:3] / 2
    box2_y2 = boxes_labels[..., 1:2] + boxes_labels[..., 3:4] / 2

    # Find the coordinates of the intersection rectangle
    x1 = torch.max(box1_x1, box2_x1)
    y1 = torch.max(box1_y1, box2_y1)
    x2 = torch.min(box1_x2, box2_x2)
    y2 = torch.min(box1_y2, box2_y2)

    # Compute intersection area
    intersection = (x2 - x1).clamp(0) * (y2 - y1).clamp(0)

    # Compute individual box areas
    box1_area = (box1_x2 - box1_x1) * (box1_y2 - box1_y1)
    box2_area = (box2_x2 - box2_x1) * (box2_y2 - box2_y1)

    union = box1_area + box2_area - intersection

    return intersection / (union + 1e-6)


def non_max_suppression(detections, iou_threshold=NMS_IOU_THRESH):
    """Applies Non-Maximum Suppression (NMS) to a list of candidate detections."""
    if not detections:
        return []

    detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
    kept = []

    while detections:
        best = detections.pop(0)
        kept.append(best)
        detections = [
            d for d in detections
            if d["class_id"] != best["class_id"]
            or iou(d["bbox"], best["bbox"]) < iou_threshold
        ]

    return kept


def plot_predictions(image_tensor, detections, class_names, ax=None):
    """Plots bounding box predictions onto the image tensor (3, h, w).

    Args:
        image_tensor (Tensor): Image tensor (c, h, w).
        detections (list): Decoded and NMS-filtered detection dictionaries.
        class_names (list): List of class name strings.

    Returns:
        PIL.Image: The processed image with drawn bounding boxes and text.
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    import numpy as np

    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))

    # Un-normalize and permute the image tensor for plotting
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img = image_tensor.cpu() * std + mean
    img = torch.clamp(img, 0, 1).permute(1, 2, 0).numpy()
    ax.imshow(img)
    
    # Get image dimensions in pixels
    img_h, img_w, _ = img.shape
    colors = plt.cm.tab10(np.linspace(0, 1, len(class_names)))

    for det in detections:
        x1, y1, x2, y2 = det["bbox"]

        # Scale coordinates in [0,1] range to pixel values
        x1 *= img_w
        x2 *= img_w
        y1 *= img_h
        y2 *= img_h

        class_id = det["class_id"]
        conf = det["confidence"]
        color = colors[class_id]        

        rect = patches.Rectangle(
            (x1, y1), x2 - x1, y2 - y1,
            linewidth=2, edgecolor=color, facecolor="none",
        )
        ax.add_patch(rect)

        label = f"{class_names[class_id]} {conf:.2f}"
        ax.text(x1, y1 - 2, label, color="white", fontsize=10,
                bbox=dict(facecolor=color, alpha=0.8))

    ax.axis("off")
    return ax
