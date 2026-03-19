# Agent 02: NorgesGruppen — Object Detection

## Rolle
Detekter og klassifiser dagligvareprodukter på butikkhyller. Upload .zip med run.py + modellvekter.

## Filer
- `norgesgruppen/` — arbeidsmappe for trening og pakking
- Treningsdata lastes ned fra app.ainm.no (864 MB COCO + 60 MB produktbilder)

## Scoring
- **0.7 x detection_mAP + 0.3 x classification_mAP** (mAP@0.5)
- Detection-only (category_id: 0 for alt) gir maks 70%
- Full klassifisering (356 kategorier) gir resterende 30%
- 3 submissions/dag, maks 2 in-flight

## Sandbox-begrensninger (KRITISK)
- NVIDIA L4 GPU, 24GB VRAM, 8GB RAM
- Python 3.11, PyTorch 2.6, **ultralytics 8.1.0** (pin denne lokalt!)
- **INGEN nettverkstilgang** i sandbox
- 300 sek timeout, 420MB maks zip
- Forbudte imports: `os`, `subprocess`, `socket`. Bruk `pathlib` i stedet.

## Treningspipeline

### Steg 1: Last ned data
Fra app.ainm.no → `norgesgruppen/data/images/` (248 bilder) + `annotations.json` (COCO-format)

### Steg 2: Analyser og konverter
```bash
# Sjekk data, konverter COCO → YOLO txt-format
# Lag data.yaml med nc=356 (eller nc=1 for detection-only)
```

### Steg 4: Tren modell
```bash
# PÅ GCP Compute Engine med GPU!
pip install ultralytics==8.1.0

# Detection-only (rask, gir 70%):
yolo detect train data=data.yaml model=yolov8m.pt epochs=50 imgsz=640 batch=16 device=0

# Full klassifisering (gir opptil 100%):
yolo detect train data=data.yaml model=yolov8m.pt epochs=100 imgsz=640 batch=16 device=0 nc=356
```

### Steg 5: Eksporter
```bash
yolo export model=runs/detect/train/weights/best.pt format=torchscript
# Alternativt: format=onnx (kan være raskere i sandbox)
```

### Steg 6: Pakk og submit
```bash
cd norgesgruppen/submission
# VIKTIG: run.py MÅ ligge i roten av zip-filen!
zip -r submission.zip run.py best.torchscript
# Sjekk størrelse: maks 420MB
ls -lh submission.zip
```

## Modellvalg-strategi
| Modell | Størrelse | Hastighet | Presisjon | Anbefaling |
|--------|-----------|-----------|-----------|------------|
| YOLOv8n | ~6MB | Raskest | Lavest | Første test |
| YOLOv8s | ~22MB | Rask | OK | God balanse |
| YOLOv8m | ~50MB | Medium | God | **Anbefalt** |
| YOLOv8l | ~83MB | Treg | Best | Hvis tid tillater |

**Strategi:** Start med YOLOv8m detection-only (70%). Hvis det fungerer, tren med nc=356.

## Detection-only vs full klassifisering
- **Detection-only:** Tren med nc=1, sett category_id=0 i output. Maks 70%.
- **Full:** Tren med nc=356. Krever mer data og tid, men gir 30% ekstra.
- **Hybrid:** Tren detection-only først, submit, deretter tren full mens du venter.

## run.py mal
```python
import argparse
import json
from pathlib import Path
from ultralytics import YOLO

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    model = YOLO(str(Path(__file__).parent / "best.pt"))
    input_dir = Path(args.input)
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
                    "category_id": int(box.cls.item()),  # 0 for detection-only
                    "bbox": [x1, y1, x2 - x1, y2 - y1],  # COCO: [x,y,w,h]
                    "score": float(box.conf.item())
                })

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(predictions, f)

if __name__ == "__main__":
    main()
```

## Vanlige feil og fixes
| Feil | Årsak | Fix |
|------|-------|-----|
| `ModuleNotFoundError: ultralytics` | Feil versjon | Pin ultralytics==8.1.0 |
| `import os` blokkert | Sikkerhetsbegrensning | Bruk `pathlib` |
| Timeout (300s) | For stor modell | Bruk YOLOv8n/s eller ONNX |
| Zip > 420MB | Modell for stor | Bruk mindre modell eller komprimer |
| Feil output-format | Ikke COCO-format | bbox=[x,y,w,h], ikke [x1,y1,x2,y2] |
| run.py ikke i rot | Feil zip-struktur | `zip` fra mappen der run.py ligger |
| `image_id` feil type | Streng i stedet for int | `int(img_path.stem)` |

## GCP Compute Engine for trening
```bash
gcloud compute instances create yolo-trainer \
  --zone=europe-west4-a --machine-type=g2-standard-8 \
  --accelerator=type=nvidia-l4,count=1 \
  --image-family=pytorch-latest-gpu \
  --image-project=deeplearning-platform-release --boot-disk-size=100GB
```
