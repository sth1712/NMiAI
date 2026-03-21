"""
NorgesGruppen Object Detection — run.py (Ensemble + ONNX)

Steg 1: Multiple YOLO-modeller detekterer bounding boxes
Steg 2: Weighted Boxes Fusion kombinerer prediksjoner
Steg 3: ResNet18 (ONNX) matcher crops mot referanser for klassifisering

Ingen blokkerte imports. ONNX-basert. Regelsikker.
"""
import json
import base64
from pathlib import Path

import numpy as np
from ultralytics import YOLO
from PIL import Image
import onnxruntime as ort
from ensemble_boxes import weighted_boxes_fusion


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
        pass
    return args


def load_resnet_onnx(onnx_path):
    """Last ResNet18 ONNX med GPU hvis tilgjengelig."""
    providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ort.InferenceSession(str(onnx_path), providers=providers)


def load_refs(base_dir):
    """Last referansedata — støtter både .npy og .json format."""
    refs_json = base_dir / "refs.json"
    ref_npy = base_dir / "ref_embeddings.npy"
    ref_ids_path = base_dir / "ref_cat_ids.json"

    if refs_json.exists():
        with open(str(refs_json)) as f:
            data = json.load(f)
        ref_embs = np.frombuffer(
            base64.b64decode(data["embeddings_b64"]), dtype=np.float32
        ).reshape(data["shape"])
        ref_cat_ids = data["cat_ids"]
    elif ref_npy.exists() and ref_ids_path.exists():
        ref_embs = np.load(str(ref_npy))
        with open(str(ref_ids_path)) as f:
            ref_cat_ids = json.load(f)
    else:
        return None, None
    return ref_embs, ref_cat_ids


def resnet_embed_batch(session, crops, transform_mean, transform_std):
    """Batch-embed crops med ResNet18 ONNX."""
    tensors = []
    for crop in crops:
        img = crop.resize((224, 224))
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - transform_mean) / transform_std
        arr = arr.transpose(2, 0, 1)
        tensors.append(arr)

    batch = np.stack(tensors).astype(np.float32)
    input_name = session.get_inputs()[0].name
    result = session.run(None, {input_name: batch})
    embs = result[0]

    norms = np.linalg.norm(embs, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    return embs / norms


def run_ensemble(models, img_path, iou_thr=0.55, skip_box_thr=0.001):
    """Kjør flere YOLO-modeller og fuser med WBF."""
    pil_img = Image.open(str(img_path))
    img_w, img_h = pil_img.size

    all_boxes = []
    all_scores = []
    all_labels = []

    for yolo, imgsz in models:
        results = yolo(str(img_path), verbose=False, imgsz=imgsz, conf=0.001, iou=0.7, max_det=1500, agnostic_nms=True)
        boxes, scores, labels = [], [], []
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                # WBF krever normaliserte koordinater [0, 1]
                boxes.append([x1 / img_w, y1 / img_h, x2 / img_w, y2 / img_h])
                scores.append(float(box.conf.item()))
                labels.append(int(box.cls.item()))
        all_boxes.append(boxes)
        all_scores.append(scores)
        all_labels.append(labels)

    # Weighted Boxes Fusion
    if any(len(b) > 0 for b in all_boxes):
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            all_boxes, all_scores, all_labels,
            iou_thr=iou_thr, skip_box_thr=skip_box_thr
        )
        # Denormaliser tilbake til pikselkoordinater
        fused_boxes[:, [0, 2]] *= img_w
        fused_boxes[:, [1, 3]] *= img_h
        return fused_boxes, fused_scores, fused_labels.astype(int), pil_img
    return np.array([]), np.array([]), np.array([]), pil_img


def main():
    args = get_args()
    input_dir = Path(args["input"])
    output_path = Path(args["output"])
    base_dir = Path(__file__).parent

    # Last YOLO-modeller (støtter ensemble med model1.onnx + model2.onnx)
    models = []
    model1 = base_dir / "best.onnx"
    model2 = base_dir / "model2.onnx"
    model1_pt = base_dir / "best.pt"

    if model1.exists():
        yolo1 = YOLO(str(model1), task="detect")
        # Hent imgsz fra ONNX input shape
        sess = ort.InferenceSession(str(model1), providers=["CPUExecutionProvider"])
        input_shape = sess.get_inputs()[0].shape
        imgsz1 = input_shape[2] if len(input_shape) >= 3 else 640
        del sess
        models.append((yolo1, imgsz1))
    elif model1_pt.exists():
        models.append((YOLO(str(model1_pt), task="detect"), 640))

    if model2.exists():
        yolo2 = YOLO(str(model2), task="detect")
        sess = ort.InferenceSession(str(model2), providers=["CPUExecutionProvider"])
        input_shape = sess.get_inputs()[0].shape
        imgsz2 = input_shape[2] if len(input_shape) >= 3 else 640
        del sess
        models.append((yolo2, imgsz2))

    use_ensemble = len(models) > 1

    # Last ResNet + referanser
    resnet_path = base_dir / "resnet18_features.onnx"
    ref_embs, ref_cat_ids = load_refs(base_dir)
    use_twostage = resnet_path.exists() and ref_embs is not None

    resnet_session = None
    if use_twostage:
        resnet_session = load_resnet_onnx(resnet_path)
        transform_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        transform_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)

    predictions = []
    image_paths = sorted(
        p for p in input_dir.glob("*")
        if p.suffix.lower() in (".jpg", ".jpeg", ".png")
    )

    for img_path in image_paths:
        image_id = int("".join(c for c in img_path.stem if c.isdigit()))

        if use_ensemble:
            fused_boxes, fused_scores, fused_labels, pil_img = run_ensemble(
                models, img_path
            )
            if len(fused_boxes) == 0:
                continue

            boxes_data = []
            crops = []
            yolo_cats = []
            yolo_confs = []

            for k in range(len(fused_boxes)):
                x1, y1, x2, y2 = fused_boxes[k]
                conf = float(fused_scores[k])
                yolo_cat = int(fused_labels[k])

                if not use_twostage:
                    predictions.append({
                        "image_id": image_id,
                        "category_id": yolo_cat,
                        "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                        "score": conf,
                    })
                    continue

                crop = pil_img.convert("RGB").crop((x1, y1, x2, y2))
                if crop.size[0] < 5 or crop.size[1] < 5:
                    continue
                boxes_data.append((x1, y1, x2, y2))
                crops.append(crop)
                yolo_cats.append(yolo_cat)
                yolo_confs.append(conf)

            if not use_twostage:
                continue

        else:
            # Single model
            yolo, imgsz = models[0]
            results = yolo(str(img_path), verbose=False, imgsz=imgsz, conf=0.001, iou=0.7, max_det=1500, agnostic_nms=True)
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

                    if not use_twostage:
                        predictions.append({
                            "image_id": image_id,
                            "category_id": yolo_cat,
                            "bbox": [x1, y1, x2 - x1, y2 - y1],
                            "score": conf,
                        })
                        continue

                    crop = pil_img.crop((x1, y1, x2, y2))
                    if crop.size[0] < 5 or crop.size[1] < 5:
                        continue
                    boxes_data.append((x1, y1, x2, y2))
                    crops.append(crop)
                    yolo_cats.append(yolo_cat)
                    yolo_confs.append(conf)

            if not use_twostage:
                continue

        if not crops:
            continue

        # Batch ResNet-inferens
        BATCH_SIZE = 128
        all_embs = []
        for i in range(0, len(crops), BATCH_SIZE):
            batch = crops[i:i + BATCH_SIZE]
            embs = resnet_embed_batch(resnet_session, batch, transform_mean, transform_std)
            all_embs.append(embs)

        all_embs = np.concatenate(all_embs, axis=0)

        # Cosine similarity
        sims = all_embs @ ref_embs.T
        best_ref_idx = sims.argmax(axis=1)
        best_sims = sims[np.arange(len(sims)), best_ref_idx]

        for j in range(len(crops)):
            x1, y1, x2, y2 = boxes_data[j]
            conf = yolo_confs[j]
            resnet_cat = ref_cat_ids[int(best_ref_idx[j])]
            resnet_sim = float(best_sims[j])

            category_id = resnet_cat if resnet_sim > 0.5 else yolo_cats[j]

            predictions.append({
                "image_id": image_id,
                "category_id": category_id,
                "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                "score": conf,
            })

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

    print(f"Wrote {len(predictions)} predictions to {output_path}")


if __name__ == "__main__":
    main()
