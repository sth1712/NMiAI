"""
Bygg referanse-embeddings med ResNet18 + ALLE produktbilde-vinkler + 30 crops.
Kjøres LOKALT. Output: refs_individual.json for run.py.
"""
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
from pathlib import Path
import numpy as np
import json
import base64
from collections import defaultdict

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ResNet18 som feature extractor (512-dim)
resnet = models.resnet18(weights="DEFAULT")
resnet.fc = torch.nn.Identity()
resnet.eval().to(DEVICE)
FEATURE_DIM = 512
print(f"Feature extractor: ResNet18 ({FEATURE_DIM}-dim)")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def embed_batch(images):
    tensors = torch.stack([transform(img) for img in images]).to(DEVICE)
    with torch.no_grad():
        embs = resnet(tensors)
    return embs.cpu().numpy()


# === 1. Crop-embeddings fra treningsdata (maks 30 per kategori) ===
print("Steg 1: Crop-embeddings fra treningsdata...")
with open("train/annotations.json") as f:
    coco = json.load(f)

img_info = {im["id"]: im for im in coco["images"]}
cat_names = {c["id"]: c["name"] for c in coco["categories"]}

MAX_CROPS = 30
cat_anns = defaultdict(list)
for ann in coco["annotations"]:
    if len(cat_anns[ann["category_id"]]) < MAX_CROPS:
        cat_anns[ann["category_id"]].append(ann)

img_to_anns = defaultdict(list)
for cat_id, anns in cat_anns.items():
    for ann in anns:
        img_to_anns[ann["image_id"]].append((cat_id, ann))

cat_crop_embs = defaultdict(list)
done = 0
total_crops = sum(len(v) for v in cat_anns.values())

for img_id, cat_ann_list in img_to_anns.items():
    im = img_info[img_id]
    img_path = Path("train/images") / im["file_name"]
    if not img_path.exists():
        continue
    pil_img = Image.open(img_path).convert("RGB")
    crops, cat_ids = [], []
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

batch_imgs, batch_eans = [], []
for ean_folder in sorted(prod_dir.iterdir()):
    if not ean_folder.is_dir():
        continue
    for angle in ALL_ANGLES:
        img_path = ean_folder / angle
        if img_path.exists():
            batch_imgs.append(Image.open(img_path).convert("RGB"))
            batch_eans.append(ean_folder.name)

print(f"  Embedder {len(batch_imgs)} produktbilder ({len(set(batch_eans))} produkter)...")
BATCH = 64
all_prod_embs = []
for i in range(0, len(batch_imgs), BATCH):
    all_prod_embs.append(embed_batch(batch_imgs[i:i + BATCH]))
all_prod_embs = np.vstack(all_prod_embs)

# Match produktbilder til kategorier
cat_avg_embs = {}
for cat_id, embs in cat_crop_embs.items():
    avg = np.mean(embs, axis=0)
    cat_avg_embs[cat_id] = avg / np.linalg.norm(avg)

cat_avg_matrix = np.stack([cat_avg_embs[k] for k in sorted(cat_avg_embs.keys())])
cat_avg_ids = sorted(cat_avg_embs.keys())

prod_norms = all_prod_embs / np.linalg.norm(all_prod_embs, axis=1, keepdims=True)
sims = prod_norms @ cat_avg_matrix.T
best_idx = sims.argmax(axis=1)
best_sims = sims[np.arange(len(sims)), best_idx]

matched = 0
for i, sim in enumerate(best_sims):
    if sim > 0.6:
        cat_crop_embs[cat_avg_ids[best_idx[i]]].append(all_prod_embs[i])
        matched += 1
print(f"  Matched {matched}/{len(batch_eans)} produktbilder (sim>0.6)")


# === 3. Bygg individuelle referanser ===
print("Steg 3: Bygger individuelle referanser...")
all_cat_ids = sorted(cat_names.keys())
ref_embs, ref_labels = [], []
for cat_id in all_cat_ids:
    if cat_id in cat_crop_embs:
        for emb in cat_crop_embs[cat_id]:
            norm = emb / np.linalg.norm(emb)
            ref_embs.append(norm)
            ref_labels.append(cat_id)

ref_embs = np.stack(ref_embs).astype(np.float32)
ref_labels = np.array(ref_labels, dtype=np.int32)
print(f"  {ref_embs.shape} ({len(set(ref_labels))} kategorier)")

# Lagre som JSON
data = {
    "embeddings_b64": base64.b64encode(ref_embs.tobytes()).decode(),
    "labels": ref_labels.tolist(),
    "shape": list(ref_embs.shape),
}
with open("refs_individual.json", "w") as f:
    json.dump(data, f)

size = Path("refs_individual.json").stat().st_size / 1e6
print(f"  refs_individual.json: {size:.1f} MB")
print("Ferdig!")
