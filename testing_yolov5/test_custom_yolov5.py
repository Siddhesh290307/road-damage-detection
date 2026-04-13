import argparse
import sys
import platform
import pathlib
from pathlib import Path

import cv2
import torch


# Fix for YOLOv5 custom weights saved on Linux and loaded on Windows
if platform.system() == "Windows":
    pathlib.PosixPath = pathlib.WindowsPath


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
        description="Test a custom YOLOv5 model on a single image and save annotated output."
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
    """
    for _, row in detections.iterrows():
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
        model = torch.hub.load(
            "ultralytics/yolov5",
            "custom",
            path=str(weights_path),
            force_reload=False
        )
    except Exception as e:
        print(f"[ERROR] Failed to load model: {e}")
        sys.exit(1)

    print("[INFO] Loaded model class names:")
    print(model.names)

    model.conf = args.conf

    print("[INFO] Running inference...")
    results = model(str(image_path), size=args.imgsz)

    df = results.pandas().xyxy[0]

    if df.empty:
        print("[INFO] No detections from model.")
    else:
        print("\n[INFO] Raw detections:")
        print(df[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]])

    # Keep only your intended classes
    filtered_df = df[df["name"].isin(ALLOWED_CLASSES)].copy()

    if filtered_df.empty:
        print("\n[INFO] No detections after filtering to custom classes.")
    else:
        print("\n[INFO] Filtered detections:")
        print(filtered_df[["name", "confidence", "xmin", "ymin", "xmax", "ymax"]])

    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        print("[ERROR] Failed to read input image with OpenCV.")
        sys.exit(1)

    annotated = draw_boxes(image_bgr.copy(), filtered_df)

    cv2.imwrite(str(output_path), annotated)
    print(f"\n[INFO] Saved annotated image to: {output_path}")

    if args.show:
        cv2.imshow("YOLOv5 Result", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()