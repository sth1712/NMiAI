"""
NorgesGruppen Object Detection — run.py (ONNX, regelsikker)

Steg 1: YOLO (ONNX via ultralytics) detekterer bounding boxes
Steg 2: ResNet18 (ONNX via onnxruntime) matcher crops mot referanser

Ingen blokkerte imports. ONNX-basert. Regelsikker.
"""
# === KONFIGURASJON ===
# Sett DETECTION_ONLY = True for diagnostisk submission (category_id=0, gir 0.7 × det_mAP)
DETECTION_ONLY = False
import json
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from PIL import Image
import onnxruntime as ort


# Hardkodede paths fra docs: "python run.py --input /data/images --output /output/predictions.json"
# Fallback til /proc/self/cmdline hvis paths er annerledes
DEFAULT_INPUT = "/data/images"
DEFAULT_OUTPUT = "/output/predictions.json"


def get_args():
    """Parse args trygt — hardkodede defaults + /proc fallback."""
    args = {"input": DEFAULT_INPUT, "output": DEFAULT_OUTPUT}
    try:
        with open("/proc/self/cmdline", "rb") as f:
            parts = f.read().split(b"\x00")
        parts = [p.decode() for p in parts if p]
        for i, p in enumerate(parts):
            if p == "--input" and i + 1 < len(parts):
                args["input"] = parts[i + 1]
            elif p == "--output" and i + 1 < len(parts):
                args["output"] = parts[i + 1]
    except Exception:
        pass  # Bruk defaults
    return args


def load_resnet_onnx(onnx_path):
    """Last ResNet18 ONNX med GPU hvis tilgjengelig."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    session = ort.InferenceSession(str(onnx_path), providers=providers)
    return session


def resnet_embed_batch(session, crops, transform_mean, transform_std):
    """Batch-embed crops med ResNet18 ONNX."""
    tensors = []
    for crop in crops:
        img = crop.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - transform_mean) / transform_std
        arr = arr.transpose(2, 0, 1)  # HWC → CHW
        tensors.append(arr)

    batch = np.stack(tensors).astype(np.float32)
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: batch})
    embs = result[0]

    # L2-normaliser
    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    embs = embs / norms
    return embs


def main():
    args = get_args()
    input_dir = Path(args["input"])
    output_path = Path(args["output"])
    base_dir = Path(__file__).parent

    # Last YOLO (ONNX eller .pt — ultralytics håndterer begge)
    yolo_onnx = base_dir / "best.onnx"
    yolo_pt = base_dir / "best.pt"
    if yolo_onnx.exists():
        yolo = YOLO(str(yolo_onnx), task="detect")
    else:
        yolo = YOLO(str(yolo_pt), task="detect")

    # Sjekk om to-stegs filer finnes
    resnet_path = base_dir / "resnet18_features.onnx"
    ref_emb_path = base_dir / "ref_embeddings.npy"
    ref_ids_path = base_dir / "ref_cat_ids.json"
    use_twostage = resnet_path.exists() and ref_emb_path.exists() and ref_ids_path.exists()

    if use_twostage:
        resnet_session = load_resnet_onnx(resnet_path)
        ref_embs = np.load(str(ref_emb_path))  # (N, 512) L2-normalisert
        with open(str(ref_ids_path)) as f:
            ref_cat_ids = json.load(f)  # list of ints

        transform_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        transform_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    predictions = []
    image_paths = sorted(
        p for p in input_dir.glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in image_paths:
        results = yolo(str(img_path), verbose=False, augment=True)
        # Hent tall fra filnavn: "img_00042.jpg" → 42, "00042.jpg" → 42
        image_id = int("".join(c for c in img_path.stem if c.isdigit()))

        if DETECTION_ONLY or not use_twostage:
            for r in results:
                for box in r.boxes:
                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                    predictions.append({
                        "image_id": image_id,
                        "category_id": 0 if DETECTION_ONLY else int(box.cls.item()),
                        "bbox": [x1, y1, x2 - x1, y2 - y1],
                        "score": float(box.conf.item()),
                    })
            continue

        # To-stegs: YOLO deteksjon + ResNet klassifisering
        pil_img = Image.open(str(img_path)).convert("RGB")

        boxes_data = []
        crops = []
        yolo_cats = []
        yolo_confs = []

        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                conf = float(box.conf.item())
                yolo_cat = int(box.cls.item())

                crop = pil_img.crop((x1, y1, x2, y2))
                if crop.size[0] < 5 or crop.size[1] < 5:
                    continue

                boxes_data.append((x1, y1, x2, y2))
                crops.append(crop)
                yolo_cats.append(yolo_cat)
                yolo_confs.append(conf)

        if not crops:
            continue

        # Batch ResNet-inferens
        BATCH_SIZE = 128
        all_embs = []
        for i in range(0, len(crops), BATCH_SIZE):
            batch = crops[i:i + BATCH_SIZE]
            embs = resnet_embed_batch(resnet_session, batch, transform_mean, transform_std)
            all_embs.append(embs)

        all_embs = np.concatenate(all_embs, axis=0)  # (n_crops, 512)

        # Cosine similarity via matmul
        sims = all_embs @ ref_embs.T  # (n_crops, n_refs)
        best_ref_idx = sims.argmax(axis=1)
        best_sims = sims[np.arange(len(sims)), best_ref_idx]

        for j in range(len(crops)):
            x1, y1, x2, y2 = boxes_data[j]
            conf = yolo_confs[j]

            resnet_cat = ref_cat_ids[int(best_ref_idx[j])]
            resnet_sim = float(best_sims[j])

            if resnet_sim > 0.5:
                category_id = resnet_cat
            else:
                category_id = yolo_cats[j]

            predictions.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": conf,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
