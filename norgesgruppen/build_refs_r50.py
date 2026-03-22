"""
Bygg referanse-embeddings med ResNet50 V2 (2048-dim) + alle vinkler + 30 crops.
Refs lagres som float16 for å spare plass.
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

resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
resnet.fc = torch.nn.Identity()
resnet.eval().to(DEVICE)
FEATURE_DIM = 2048
print(f"Feature extractor: ResNet50 V2 ({FEATURE_DIM}-dim)")

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

# === 1. Crops ===
print("Steg 1: Crop-embeddings (maks 30 per kategori)...")
with open("train/annotations.json") as f:
    coco = json.load(f)
img_info = {im["id"]: im for im in coco["images"]}
cat_names = {c["id"]: c["name"] for c in coco["categories"]}

cat_anns = defaultdict(list)
for ann in coco["annotations"]:
    if len(cat_anns[ann["category_id"]]) < 30:
        cat_anns[ann["category_id"]].append(ann)

img_to_anns = defaultdict(list)
for cat_id, anns in cat_anns.items():
    for ann in anns:
        img_to_anns[ann["image_id"]].append((cat_id, ann))

cat_crop_embs = defaultdict(list)
done = 0
for img_id, cat_ann_list in img_to_anns.items():
    im = img_info[img_id]
    img_path = Path("train/images") / im["file_name"]
    if not img_path.exists(): continue
    pil_img = Image.open(img_path).convert("RGB")
    crops, cat_ids = [], []
    for cat_id, ann in cat_ann_list:
        x, y, w, h = ann["bbox"]
        crop = pil_img.crop((x, y, x+w, y+h))
        if crop.size[0] < 10 or crop.size[1] < 10: continue
        crops.append(crop)
        cat_ids.append(cat_id)
    if crops:
        embs = embed_batch(crops)
        for cid, emb in zip(cat_ids, embs):
            cat_crop_embs[cid].append(emb)
        done += len(crops)
        if done % 500 == 0: print(f"  {done} crops...")
print(f"  {len(cat_crop_embs)} kategorier, {done} crops")

# === 2. Produktbilder (alle vinkler) ===
print("Steg 2: Produktbilder (alle vinkler)...")
ALL_ANGLES = ["main.jpg","front.jpg","back.jpg","left.jpg","right.jpg","top.jpg","bottom.jpg"]
batch_imgs, batch_eans = [], []
for ean in sorted(Path("NM_NGD_product_images").iterdir()):
    if not ean.is_dir(): continue
    for angle in ALL_ANGLES:
        p = ean / angle
        if p.exists():
            batch_imgs.append(Image.open(p).convert("RGB"))
            batch_eans.append(ean.name)
print(f"  {len(batch_imgs)} bilder fra {len(set(batch_eans))} produkter")

all_prod = []
for i in range(0, len(batch_imgs), 64):
    all_prod.append(embed_batch(batch_imgs[i:i+64]))
all_prod = np.vstack(all_prod)

# Match til kategorier
cat_avg = {}
for cid, embs in cat_crop_embs.items():
    avg = np.mean(embs, axis=0)
    cat_avg[cid] = avg / np.linalg.norm(avg)
cat_matrix = np.stack([cat_avg[k] for k in sorted(cat_avg)])
cat_ids_sorted = sorted(cat_avg)

prod_norm = all_prod / np.linalg.norm(all_prod, axis=1, keepdims=True)
sims = prod_norm @ cat_matrix.T
best_idx = sims.argmax(axis=1)
best_sims = sims[np.arange(len(sims)), best_idx]
matched = 0
for i, sim in enumerate(best_sims):
    if sim > 0.6:
        cat_crop_embs[cat_ids_sorted[best_idx[i]]].append(all_prod[i])
        matched += 1
print(f"  Matched {matched}/{len(batch_eans)}")

# === 3. Bygg refs (float16 for å spare plass) ===
print("Steg 3: Bygger refs...")
ref_embs, ref_labels = [], []
for cid in sorted(cat_names):
    if cid in cat_crop_embs:
        for emb in cat_crop_embs[cid]:
            ref_embs.append(emb / np.linalg.norm(emb))
            ref_labels.append(cid)

ref_embs = np.stack(ref_embs).astype(np.float16)  # float16!
ref_labels = np.array(ref_labels, dtype=np.int32)
print(f"  {ref_embs.shape} ({len(set(ref_labels))} kat), float16")

data = {
    "embeddings_b64": base64.b64encode(ref_embs.tobytes()).decode(),
    "labels": ref_labels.tolist(),
    "shape": list(ref_embs.shape),
    "dtype": "float16",
}
with open("refs_individual.json", "w") as f:
    json.dump(data, f)
print(f"  refs_individual.json: {Path('refs_individual.json').stat().st_size/1e6:.1f} MB")
print("Ferdig!")
