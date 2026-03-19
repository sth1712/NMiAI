"""
Konverterer COCO-format annotations til YOLO-format.
Kjøres på GCP VM etter at data er lastet opp.

Bruk:
    python convert_coco_to_yolo.py --coco annotations.json --images data/images --out data/labels --yaml data.yaml
"""
import argparse
import json
import os
import shutil
from pathlib import Path
from collections import Counter


def coco_to_yolo(coco_path, images_dir, labels_dir, yaml_path, detection_only=False):
    with open(coco_path) as f:
        coco = json.load(f)

    images_dir = Path(images_dir)
    labels_dir = Path(labels_dir)
    labels_dir.mkdir(parents=True, exist_ok=True)

    # Bygg image-id → filnavn og dimensjoner
    img_info = {}
    for img in coco["images"]:
        img_info[img["id"]] = {
            "file_name": img["file_name"],
            "width": img["width"],
            "height": img["height"],
        }

    # Bygg category mapping
    categories = sorted(coco["categories"], key=lambda c: c["id"])
    if detection_only:
        cat_map = {c["id"]: 0 for c in categories}
        nc = 1
        names = ["object"]
    else:
        # Remap til 0-indexed
        cat_map = {}
        names = []
        for i, c in enumerate(categories):
            cat_map[c["id"]] = i
            names.append(c["name"])
        nc = len(names)

    print(f"Kategorier: {nc}")
    print(f"Bilder: {len(img_info)}")
    print(f"Annotasjoner: {len(coco['annotations'])}")

    # Grupper annotations per bilde
    ann_per_image = {}
    for ann in coco["annotations"]:
        img_id = ann["image_id"]
        if img_id not in ann_per_image:
            ann_per_image[img_id] = []
        ann_per_image[img_id].append(ann)

    # Skriv YOLO labels
    label_count = 0
    for img_id, info in img_info.items():
        w, h = info["width"], info["height"]
        stem = Path(info["file_name"]).stem
        label_file = labels_dir / f"{stem}.txt"

        lines = []
        if img_id in ann_per_image:
            for ann in ann_per_image[img_id]:
                cat_id = cat_map.get(ann["category_id"])
                if cat_id is None:
                    continue

                # COCO bbox: [x, y, width, height] (top-left)
                bx, by, bw, bh = ann["bbox"]

                # YOLO: x_center, y_center, width, height (normalized)
                x_center = (bx + bw / 2) / w
                y_center = (by + bh / 2) / h
                nw = bw / w
                nh = bh / h

                # Clamp til [0, 1]
                x_center = max(0, min(1, x_center))
                y_center = max(0, min(1, y_center))
                nw = max(0, min(1, nw))
                nh = max(0, min(1, nh))

                lines.append(f"{cat_id} {x_center:.6f} {y_center:.6f} {nw:.6f} {nh:.6f}")
                label_count += 1

        with open(label_file, "w") as f:
            f.write("\n".join(lines))

    print(f"Skrev {label_count} labels til {labels_dir}")

    # Lag train/val split (90/10)
    all_images = sorted(img_info.values(), key=lambda x: x["file_name"])
    split_idx = int(len(all_images) * 0.9)
    train_imgs = all_images[:split_idx]
    val_imgs = all_images[split_idx:]

    # Lag mappestuktur for YOLO
    dataset_dir = yaml_path.parent / "dataset"
    for split, imgs in [("train", train_imgs), ("val", val_imgs)]:
        img_dir = dataset_dir / split / "images"
        lbl_dir = dataset_dir / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for info in imgs:
            fname = info["file_name"]
            stem = Path(fname).stem

            src_img = images_dir / fname
            src_lbl = labels_dir / f"{stem}.txt"

            if src_img.exists():
                shutil.copy2(src_img, img_dir / fname)
            if src_lbl.exists():
                shutil.copy2(src_lbl, lbl_dir / f"{stem}.txt")

    print(f"Train: {len(train_imgs)} bilder, Val: {len(val_imgs)} bilder")

    # Skriv data.yaml
    yaml_content = f"""path: {dataset_dir.resolve()}
train: train/images
val: val/images

nc: {nc}
names: {names}
"""
    with open(yaml_path, "w") as f:
        f.write(yaml_content)

    print(f"Skrev {yaml_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco", required=True, help="Sti til COCO annotations.json")
    parser.add_argument("--images", required=True, help="Sti til bildemappen")
    parser.add_argument("--out", default="data/labels", help="Output-mappe for labels")
    parser.add_argument("--yaml", default="data.yaml", help="Output data.yaml")
    parser.add_argument("--detection-only", action="store_true", help="Kun deteksjon (nc=1)")
    args = parser.parse_args()

    coco_to_yolo(
        coco_path=args.coco,
        images_dir=args.images,
        labels_dir=Path(args.out),
        yaml_path=Path(args.yaml),
        detection_only=args.detection_only,
    )
