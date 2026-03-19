#!/bin/bash
# ============================================
# NorgesGruppen YOLOv8 — Treningsskript for GCP VM
# Kjøres etter at data er lastet opp og pakket ut
# ============================================

set -e

echo "=== Steg 1: Installer avhengigheter ==="
pip install ultralytics==8.1.0

echo "=== Steg 2: Pakk ut data ==="
mkdir -p ~/ng_train/data
cd ~/ng_train

# Juster filnavnet hvis det heter noe annet
if [ -f ~/NM_NGD_coco_dataset.zip ]; then
    unzip -o ~/NM_NGD_coco_dataset.zip -d data/
    echo "Data pakket ut til data/"
    ls -la data/
else
    echo "FEIL: Fant ikke ~/NM_NGD_coco_dataset.zip"
    echo "Last opp dataene først!"
    exit 1
fi

echo ""
echo "=== Steg 3: Finn annotations og bilder ==="
echo "Innhold i data/:"
find data/ -maxdepth 3 -type f | head -20
echo "..."
echo ""
echo "STOPP HER og sjekk output over."
echo "Kjør deretter manuelt:"
echo ""
echo "  # Konverter COCO → YOLO (juster stier basert på output over):"
echo "  python convert_coco_to_yolo.py --coco data/annotations.json --images data/images --out data/labels --yaml data.yaml"
echo ""
echo "  # For detection-only (raskest, gir 70%):"
echo "  python convert_coco_to_yolo.py --coco data/annotations.json --images data/images --out data/labels --yaml data.yaml --detection-only"
echo ""
echo "  # Tren:"
echo "  yolo detect train data=data.yaml model=yolov8m.pt epochs=100 imgsz=640 batch=16 device=0"
echo ""
echo "  # Når ferdig, last ned best.pt:"
echo "  # (fra din Mac:)"
echo "  # gcloud compute scp yolo-trainer:~/ng_train/runs/detect/train/weights/best.pt ./norgesgruppen/ --zone=europe-west4-a"
