"""
Bygg referanse-embeddings fra treningsdata + produktbilder.
Kjøres LOKALT — output brukes i run.py for to-stegs klassifisering.

Endringer:
- Bruker ConvNeXt-tiny (768-dim, ImageNet-22k) i stedet for ResNet18 (512-dim)
- Bruker ALLE produktbilde-vinkler (main, front, back, left, right, top, bottom)
- Økt maks crops per kategori fra 10 til 30
- Genererer individuelle referanser for top-K voting
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import timm
from PIL import Image
from pathlib import Path
import numpy as np
import json
import base64
from collections import defaultdict

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ConvNeXt-tiny som feature extractor (768-dim, ImageNet-22k pretrained)
convnext = timm.create_model('convnext_tiny.fb_in22k_ft_in1k', pretrained=True, num_classes=0)
convnext.eval().to(DEVICE)
FEATURE_DIM = convnext.num_features
print(f"Feature extractor: ConvNeXt-tiny ({FEATURE_DIM}-dim)")

# timm's data config for riktig preprocessing
data_config = timm.data.resolve_model_data_config(convnext)
transform = timm.data.create_transform(**data_config, is_training=False)


def embed_batch(images):
    """Embed a list of PIL images."""
    tensors = torch.stack([transform(img) for img in images]).to(DEVICE)
    with torch.no_grad():
        embs = convnext(tensors)
    return embs.cpu().numpy()


# === 1. Crop-embeddings fra treningsdata ===
print("Steg 1: Crop-embeddings fra treningsdata...")
with open("train/annotations.json") as f:
    coco = json.load(f)

img_info = {im["id"]: im for im in coco["images"]}
cat_names = {c["id"]: c["name"] for c in coco["categories"]}

# Samle crops per kategori (maks 30 per)
MAX_CROPS_PER_CAT = 30
cat_anns = defaultdict(list)
for ann in coco["annotations"]:
    if len(cat_anns[ann["category_id"]]) < MAX_CROPS_PER_CAT:
        cat_anns[ann["category_id"]].append(ann)

total_crops = sum(len(v) for v in cat_anns.values())
done = 0

# Batch-prosesser per bilde
img_to_anns = defaultdict(list)
for cat_id, anns in cat_anns.items():
    for ann in anns:
        img_to_anns[ann["image_id"]].append((cat_id, ann))

cat_crop_embs = defaultdict(list)

for img_id, cat_ann_list in img_to_anns.items():
    im = img_info[img_id]
    img_path = Path("train/images") / im["file_name"]
    if not img_path.exists():
        continue

    pil_img = Image.open(img_path).convert("RGB")

    crops = []
    cat_ids = []
    for cat_id, ann in cat_ann_list:
        x, y, w, h = ann["bbox"]
        crop = pil_img.crop((x, y, x + w, y + h))
        if crop.size[0] < 10 or crop.size[1] < 10:
            continue
        crops.append(crop)
        cat_ids.append(cat_id)

    if crops:
        embs = embed_batch(crops)
        for cat_id, emb in zip(cat_ids, embs):
            cat_crop_embs[cat_id].append(emb)
        done += len(crops)
        if done % 500 == 0:
            print(f"  {done}/{total_crops} crops...")

print(f"  Ferdig: {len(cat_crop_embs)} kategorier fra {done} crops")


# === 2. Produktbilde-embeddings (ALLE vinkler) ===
print("Steg 2: Produktbilde-embeddings (alle vinkler)...")
prod_dir = Path("NM_NGD_product_images")

ALL_ANGLES = ["main.jpg", "front.jpg", "back.jpg", "left.jpg", "right.jpg", "top.jpg", "bottom.jpg"]

batch_imgs = []
batch_eans = []
batch_angles = []
for ean_folder in sorted(prod_dir.iterdir()):
    if not ean_folder.is_dir():
        continue
    for angle in ALL_ANGLES:
        img_path = ean_folder / angle
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            batch_imgs.append(img)
            batch_eans.append(ean_folder.name)
            batch_angles.append(angle)

print(f"  Embedder {len(batch_imgs)} produktbilder ({len(set(batch_eans))} produkter, {len(ALL_ANGLES)} vinkler)...")

BATCH = 64
all_prod_embs = []
for i in range(0, len(batch_imgs), BATCH):
    embs = embed_batch(batch_imgs[i:i + BATCH])
    all_prod_embs.append(embs)
    if (i // BATCH) % 5 == 0:
        print(f"  Batch {i//BATCH + 1}/{(len(batch_imgs) + BATCH - 1)//BATCH}...")
all_prod_embs = np.vstack(all_prod_embs)

# Bygg gjennomsnittlige kategori-embeddings fra crops
cat_avg_embs = {}
for cat_id, embs in cat_crop_embs.items():
    avg = np.mean(embs, axis=0)
    avg = avg / np.linalg.norm(avg)
    cat_avg_embs[cat_id] = avg

cat_avg_matrix = np.stack([cat_avg_embs[k] for k in sorted(cat_avg_embs.keys())])
cat_avg_ids = sorted(cat_avg_embs.keys())

# Match hvert produktbilde til nærmeste kategori
prod_norms = all_prod_embs / np.linalg.norm(all_prod_embs, axis=1, keepdims=True)
sims = prod_norms @ cat_avg_matrix.T
best_idx = sims.argmax(axis=1)
best_sims = sims[np.arange(len(sims)), best_idx]

matched_high = 0
for i, (ean, sim) in enumerate(zip(batch_eans, best_sims)):
    cat_id = cat_avg_ids[best_idx[i]]
    if sim > 0.6:
        cat_crop_embs[cat_id].append(all_prod_embs[i])
        matched_high += 1

print(f"  Matched {matched_high}/{len(batch_eans)} produktbilder (sim>0.6)")


# === 3. Bygg individuelle referanse-embeddings ===
print("Steg 3: Bygger individuelle referanse-embeddings...")
all_cat_ids = sorted(cat_names.keys())

ref_embs_indiv = []
ref_labels_indiv = []
for cat_id in all_cat_ids:
    if cat_id in cat_crop_embs and cat_crop_embs[cat_id]:
        for emb in cat_crop_embs[cat_id]:
            norm_emb = emb / np.linalg.norm(emb)
            ref_embs_indiv.append(norm_emb)
            ref_labels_indiv.append(cat_id)

ref_embs_indiv = np.stack(ref_embs_indiv).astype(np.float32)
ref_labels_indiv = np.array(ref_labels_indiv, dtype=np.int32)
print(f"  Individuelle referanser: {ref_embs_indiv.shape} ({len(set(ref_labels_indiv))} kategorier)")

# Lagre som JSON (teller IKKE som weight-fil)
refs_data = {
    "embeddings_b64": base64.b64encode(ref_embs_indiv.tobytes()).decode(),
    "labels": ref_labels_indiv.tolist(),
    "shape": list(ref_embs_indiv.shape),
    "feature_dim": FEATURE_DIM,
}
with open("refs_individual.json", "w") as f:
    json.dump(refs_data, f)
print(f"  Lagret refs_individual.json ({Path('refs_individual.json').stat().st_size / 1024 / 1024:.1f} MB)")

# === 4. Eksporter ConvNeXt til ONNX ===
print("Steg 4: Eksporterer ConvNeXt-tiny til ONNX...")
convnext.cpu()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(convnext, dummy, 'convnext_features.onnx',
                  opset_version=17,
                  input_names=['input'], output_names=['embedding'],
                  dynamic_axes={'input': {0: 'batch'}, 'embedding': {0: 'batch'}},
                  export_params=True,
                  dynamo=False)
print(f"  Lagret convnext_features.onnx ({Path('convnext_features.onnx').stat().st_size / 1e6:.1f} MB)")

print("\nFerdig! Filer for submission:")
print("  - best.onnx (YOLO)")
print("  - convnext_features.onnx (ConvNeXt-tiny feature extractor)")
print("  - refs_individual.json (individuelle referanser)")
print("  - run.py")
