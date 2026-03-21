"""
Bygg referanse-embeddings fra treningsdata + produktbilder.
Kjøres LOKALT — output brukes i run.py for to-stegs klassifisering.

Output: ref_embeddings.pt (category_id → normalized embedding)
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
from collections import defaultdict

DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
print(f"Device: {DEVICE}")

# ResNet18 som feature extractor
resnet = models.resnet18(weights="DEFAULT")
resnet.fc = torch.nn.Identity()
resnet.eval().to(DEVICE)

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def embed_batch(images):
    """Embed a list of PIL images."""
    tensors = torch.stack([transform(img) for img in images]).to(DEVICE)
    with torch.no_grad():
        embs = resnet(tensors)
    return embs.cpu().numpy()


def embed_single(img):
    return embed_batch([img])[0]


# === 1. Crop-embeddings fra treningsdata ===
print("Steg 1: Crop-embeddings fra treningsdata...")
with open("train/annotations.json") as f:
    coco = json.load(f)

img_info = {im["id"]: im for im in coco["images"]}
cat_names = {c["id"]: c["name"] for c in coco["categories"]}

# Samle crops per kategori (maks 10 per)
cat_anns = defaultdict(list)
for ann in coco["annotations"]:
    if len(cat_anns[ann["category_id"]]) < 10:
        cat_anns[ann["category_id"]].append(ann)

cat_crop_embs = defaultdict(list)
total_crops = sum(len(v) for v in cat_anns.values())
done = 0

# Batch-prosesser per bilde for hastighet
img_to_anns = defaultdict(list)
for cat_id, anns in cat_anns.items():
    for ann in anns:
        img_to_anns[ann["image_id"]].append((cat_id, ann))

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
        if done % 200 == 0:
            print(f"  {done}/{total_crops} crops...")

print(f"  Ferdig: {len(cat_crop_embs)} kategorier fra {done} crops")


# === 2. Produktbilde-embeddings ===
print("Steg 2: Produktbilde-embeddings...")
prod_dir = Path("NM_NGD_product_images")

# Først: map EAN → category_id via cosine similarity
ean_embs = {}
prod_images_for_cat = defaultdict(list)

batch_imgs = []
batch_eans = []
for ean_folder in sorted(prod_dir.iterdir()):
    if not ean_folder.is_dir():
        continue
    # Bruk alle tilgjengelige vinkler
    for img_name in ["main.jpg", "front.jpg"]:
        img_path = ean_folder / img_name
        if img_path.exists():
            img = Image.open(img_path).convert("RGB")
            batch_imgs.append(img)
            batch_eans.append(ean_folder.name)
            break

# Batch embed alle produktbilder
print(f"  Embedder {len(batch_imgs)} produktbilder...")
BATCH = 64
all_prod_embs = []
for i in range(0, len(batch_imgs), BATCH):
    embs = embed_batch(batch_imgs[i : i + BATCH])
    all_prod_embs.append(embs)
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
sims = prod_norms @ cat_avg_matrix.T  # (n_prods, n_cats)
best_idx = sims.argmax(axis=1)
best_sims = sims[np.arange(len(sims)), best_idx]

matched_high = 0
for i, (ean, sim) in enumerate(zip(batch_eans, best_sims)):
    cat_id = cat_avg_ids[best_idx[i]]
    if sim > 0.6:  # Bare inkluder gode matches
        cat_crop_embs[cat_id].append(all_prod_embs[i])
        matched_high += 1

print(f"  Matched {matched_high}/{len(batch_eans)} produktbilder (sim>0.6)")


# === 3. Bygg finale referanse-embeddings ===
print("Steg 3: Bygger finale referanse-embeddings...")
all_cat_ids = sorted(cat_names.keys())

# === 3a. GJENNOMSNITTLIGE embeddings (legacy, for backward compat) ===
ref_embs_avg = []
valid_cat_ids_avg = []
for cat_id in all_cat_ids:
    if cat_id in cat_crop_embs and cat_crop_embs[cat_id]:
        avg = np.mean(cat_crop_embs[cat_id], axis=0)
        avg = avg / np.linalg.norm(avg)
        ref_embs_avg.append(avg)
        valid_cat_ids_avg.append(cat_id)

ref_tensor_avg = torch.tensor(np.stack(ref_embs_avg), dtype=torch.float32)
cat_id_tensor_avg = torch.tensor(valid_cat_ids_avg, dtype=torch.long)
print(f"  Gjennomsnittlige referanser: {ref_tensor_avg.shape} ({len(valid_cat_ids_avg)} kategorier)")

# === 3b. INDIVIDUELLE embeddings (ny, for top-K voting) ===
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

# Lagre begge formater
# Legacy: gjennomsnitt
torch.save({
    "embeddings": ref_tensor_avg,
    "cat_ids": cat_id_tensor_avg,
    "cat_names": cat_names,
}, "ref_embeddings.pt")
print(f"Lagret ref_embeddings.pt ({ref_tensor_avg.shape[0]} kategorier, {ref_tensor_avg.element_size() * ref_tensor_avg.nelement() / 1024:.0f} KB)")

# Ny: individuelle (for top-K voting i run.py)
np.save("ref_embeddings_individual.npy", ref_embs_indiv)
np.save("ref_labels_individual.npy", ref_labels_indiv)
print(f"Lagret ref_embeddings_individual.npy ({ref_embs_indiv.shape}, {ref_embs_indiv.nbytes / 1024:.0f} KB)")
print(f"Lagret ref_labels_individual.npy ({ref_labels_indiv.shape}, {ref_labels_indiv.nbytes / 1024:.0f} KB)")

# Lagre også som JSON for sandbox (ikke weight-fil)
import base64
refs_data = {
    "embeddings_b64": base64.b64encode(ref_embs_indiv.tobytes()).decode(),
    "labels": ref_labels_indiv.tolist(),
    "shape": list(ref_embs_indiv.shape),
    "cat_ids_avg": valid_cat_ids_avg,
    "embeddings_avg_b64": base64.b64encode(ref_tensor_avg.numpy().tobytes()).decode(),
    "shape_avg": list(ref_tensor_avg.shape),
}
with open("refs_individual.json", "w") as f:
    json.dump(refs_data, f)
print(f"Lagret refs_individual.json ({Path('refs_individual.json').stat().st_size / 1024:.0f} KB)")

# Lagre ResNet18 vekter separat (for submission)
torch.save(resnet.state_dict(), "resnet18_features.pt")
print(f"Lagret resnet18_features.pt")

print("\nFerdig! Filer for submission:")
print("  - best.onnx (YOLO)")
print("  - resnet18_features.onnx (ResNet18 feature extractor)")
print("  - ref_embeddings.npy (gjennomsnittlige, legacy)")
print("  - ref_embeddings_individual.npy (individuelle, for top-K)")
print("  - ref_labels_individual.npy (labels for individuelle)")
print("  - refs_individual.json (JSON-format, teller ikke som weight)")
print("  - run.py")
