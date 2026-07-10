# predict.py
import os
import argparse
import torch
import torchvision.transforms as T
from PIL import Image, ImageDraw

from config import CONF_THRESHOLD, NMS_IOU_THRESH, DATASET_ROOT
from yolov1.model import YOLOv1
from yolov1.utils import decode_predictions, non_max_suppression, load_class_names, scale_box


def preprocess_image(image_path, target_scale=448):
    """Loads, resizes, and pads an image dynamically to target_scale (longest dimension), preserving aspect ratio."""
    image = Image.open(image_path).convert("RGB")
    orig_w, orig_h = image.size

    # Scale image proportionally so the longest side is exactly ``IMAGE_SIZE``
    r = target_scale / max(orig_w, orig_h)
    unpadded_w = round(int(orig_w * r))
    unpadded_h = round(int(orig_h * r))

    # Find the nearest larger multiple of 32 for both dimensions (stride-32 alignment)
    padded_w = int(math.ceil(unpadded_w / 32.0) * 32)
    padded_h = int(math.ceil(unpadded_h / 32.0) * 32)

    # Calculate the padding required
    pad_w = padded_w - unpadded_w
    pad_h = padded_h - unpadded_h

    pad_left = pad_w // 2
    pad_top = pad_h // 2
    pad_right = pad_w - pad_left
    pad_bottom = pad_h - pad_top

    transform = T.Compose([
        T.Resize((unpadded_h, unpadded_w)),
        T.Pad(padding=(pad_left, pad_top, pad_right, pad_bottom), fill=0),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(image).unsqueeze(0)

    # Package the metadata (used to reverse padding and scaling during postprocessing)
    metadata = {
        "orig_w": orig_w,
        "orig_h": orig_h,
        "unpadded_w": unpadded_w,
        "unpadded_h": unpadded_h,
        "padded_w": padded_w,
        "padded_h": padded_h,
        "pad_left": pad_left,
        "pad_top": pad_top
    }
    
    return input_tensor, metadata, image


def draw_boxes(image, detections, class_names, metadata):
    """Draws bounding boxes on a PIL image.

    Args:
        image (PIL.Image): A PIL image.
        detections (list): Decoded and NMS-filtered detection dictionaries.
        class_names (list): List of class name strings.

    Returns:
        PIL.Image: The processed image with drawn bounding boxes and text.
    """
    draw = ImageDraw.Draw(image)
    colors = [
        "#FF0000", "#00FF00", "#0000FF", "#FFFF00", "#FF00FF",
    ]
    
    for det in detections:
        # Scale coordinates in [0,1] range to pixel values
        x1, y1, x2, y2 = scale_box(det["bbox"], metadata)
        class_id = det["class_id"]
        conf = det["confidence"]
        color = colors[class_id % len(colors)]

        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        label = f"{class_names[class_id]} {conf:.2f}"
        draw.text((x1, y1 - 15), label, fill=color)

    return image


def main(args):
    """Executes the complete inference, post-processing, and visualization pipeline."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read class names
    data_yaml_path = os.path.join(DATASET_ROOT, "data.yaml")
    class_names = load_class_names(data_yaml_path)

    # Initialize model architecture and load trained weights
    model = YOLOv1().to(device)
    model.load_state_dict(torch.load(args.weights, map_location=device))
    model.eval()

    input_tensor, metadata, orig_image = preprocess_image(args.image, args.image_size)
    input_tensor = input_tensor.to(device)

    with torch.no_grad():
        predictions = model(input_tensor)

    # Post-process raw model outputs using confidence decoding and NMS
    detections = decode_predictions(predictions, conf_threshold=args.confidence)
    detections = [non_max_suppression(d, iou_threshold=args.iou) for d in detections]

    print(f"Found {len(detections[0])} objects:")
    for det in detections[0]:
        print(f"  {class_names[det['class_id']]}: {det['confidence']:.2f} at {det['bbox']}")

    # Generate and save final visualized output
    result_image = draw_boxes(orig_image.copy(), detections[0], class_names, metadata) 
    output_path = args.output or f"output_{os.path.basename(args.image)}"
    result_image.save(output_path)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--weights", type=str, default="checkpoints/best.pth", help="Path to model weights")
    parser.add_argument("--confidence", type=float, default=CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=NMS_IOU_THRESH, help="NMS IoU threshold")
    parser.add_argument("--output", type=str, default=None, help="Output image path")
    parser.add_argument("--image_size", type=int, default=448, help="Target max scale for inference")

    args = parser.parse_args()
    main(args)