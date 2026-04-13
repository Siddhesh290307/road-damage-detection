import argparse
import sys
from pathlib import Path

import cv2
from ultralytics import YOLO


ALLOWED_CLASSES = {
    "crack",
    "damage",
    "pothole",
    "pothole_water",
    "pothole_water_m",
    "garbage",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Test a custom YOLO11 model on a single image and save annotated output."
    )
    parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained .pt file, e.g. weights/best.pt"
    )
    parser.add_argument(
        "--image",
        type=str,
        required=True,
        help="Path to test image, e.g. images/test.jpeg"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="outputs/result.jpg",
        help="Path to save annotated result image"
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.25,
        help="Confidence threshold"
    )
    parser.add_argument(
        "--imgsz",
        type=int,
        default=640,
        help="Inference image size"
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Show result image in a window"
    )
    return parser.parse_args()


def draw_boxes(image, detections):
    """
    Draw filtered detections manually on the image.
    detections: list of dicts with keys:
        xmin, ymin, xmax, ymax, name, confidence
    """
    for row in detections:
        x1 = int(row["xmin"])
        y1 = int(row["ymin"])
        x2 = int(row["xmax"])
        y2 = int(row["ymax"])
        cls_name = str(row["name"])
        conf = float(row["confidence"])

        label = f"{cls_name} {conf:.2f}"

        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

        (tw, th), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2
        )

        text_y1 = max(y1 - th - baseline - 4, 0)
        text_y2 = text_y1 + th + baseline + 4

        cv2.rectangle(image, (x1, text_y1), (x1 + tw + 6, text_y2), (0, 255, 0), -1)
        cv2.putText(
            image,
            label,
            (x1 + 3, text_y2 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 0, 0),
            2,
            cv2.LINE_AA
        )

    return image


def main():
    args = parse_args()

    weights_path = Path(args.weights)
    image_path = Path(args.image)
    output_path = Path(args.output)

    if not weights_path.exists():
        print(f"[ERROR] Weights file not found: {weights_path}")
        sys.exit(1)

    if not image_path.exists():
        print(f"[ERROR] Image file not found: {image_path}")
        sys.exit(1)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("[INFO] Loading model...")
    try:
        model = YOLO(str(weights_path))
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    print("[INFO] Loaded model class names:")
    print(model.names)

    print("[INFO] Running inference...")
    try:
        results = model.predict(
            source=str(image_path),
            conf=args.conf,
            imgsz=args.imgsz,
            verbose=False
        )
    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")
        sys.exit(1)

    if not results:
        print("[INFO] No results returned by model.")
        sys.exit(0)

    result = results[0]
    boxes = result.boxes

    detections = []

    if boxes is None or len(boxes) == 0:
        print("[INFO] No detections from model.")
    else:
        print("\n[INFO] Raw detections:")
        for box in boxes:
            xyxy = box.xyxy[0].tolist()
            conf = float(box.conf[0].item())
            cls_id = int(box.cls[0].item())
            cls_name = model.names[cls_id]

            row = {
                "name": cls_name,
                "confidence": conf,
                "xmin": xyxy[0],
                "ymin": xyxy[1],
                "xmax": xyxy[2],
                "ymax": xyxy[3],
            }
            detections.append(row)

        for row in detections:
            print(
                f"{row['name']:15s} "
                f"{row['confidence']:.4f} "
                f"{row['xmin']:.1f} {row['ymin']:.1f} {row['xmax']:.1f} {row['ymax']:.1f}"
            )

    # Keep only your intended classes
    filtered_detections = [
        row for row in detections if row["name"] in ALLOWED_CLASSES
    ]

    if not filtered_detections:
        print("\n[INFO] No detections after filtering to custom classes.")
    else:
        print("\n[INFO] Filtered detections:")
        for row in filtered_detections:
            print(
                f"{row['name']:15s} "
                f"{row['confidence']:.4f} "
                f"{row['xmin']:.1f} {row['ymin']:.1f} {row['xmax']:.1f} {row['ymax']:.1f}"
            )

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print("[ERROR] Failed to read input image with OpenCV.")
        sys.exit(1)

    annotated = draw_boxes(image_bgr.copy(), filtered_detections)

    success = cv2.imwrite(str(output_path), annotated)
    if not success:
        print("[ERROR] Failed to save annotated output image.")
        sys.exit(1)

    print(f"\n[INFO] Saved annotated image to: {output_path}")

    if args.show:
        cv2.imshow("YOLO11 Result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()