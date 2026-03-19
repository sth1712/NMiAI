"""
NorgesGruppen Object Detection — run.py

FORBUDTE IMPORTS (gir BAN): os, sys, subprocess, socket
Bruk pathlib for filoperasjoner.
"""
import json
from pathlib import Path
from ultralytics import YOLO


def get_args():
    """Parse --input og --output fra /proc/self/cmdline (Linux).
    Unngår argparse som internt importerer sys."""
    with open("/proc/self/cmdline", "rb") as f:
        parts = f.read().split(b"\x00")
    parts = [p.decode() for p in parts if p]
    args = {}
    for i, p in enumerate(parts):
        if p == "--input" and i + 1 < len(parts):
            args["input"] = parts[i + 1]
        elif p == "--output" and i + 1 < len(parts):
            args["output"] = parts[i + 1]
    return args


def main():
    args = get_args()
    input_dir = Path(args["input"])
    output_path = Path(args["output"])

    model_path = Path(__file__).parent / "best.pt"
    model = YOLO(str(model_path))

    predictions = []

    for img_path in sorted(input_dir.glob("*")):
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        results = model(str(img_path), verbose=False)
        image_id = int(img_path.stem)

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                predictions.append({
                    "image_id": image_id,
                    "category_id": int(box.cls.item()),
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO: [x, y, w, h]
                    "score": float(box.conf.item()),
                })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
