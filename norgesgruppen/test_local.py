"""
Lokal test av NorgesGruppen-modellen.
Tester at modellen laster og produserer riktig output-format.
"""
import json
from pathlib import Path

# Test 1: Sjekk at modellen laster med YOLO
print("=== Test 1: Laster modell ===")
try:
    from ultralytics import YOLO
    model = YOLO(str(Path(__file__).parent / "best.pt"))
    print(f"  OK — modell lastet ({model.task})")
    print(f"  Antall klasser: {len(model.names)}")
    print(f"  Første 5 klasser: {list(model.names.values())[:5]}")
except Exception as e:
    print(f"  FEIL: {e}")
    print("  Dette betyr at modellen kanskje ikke fungerer i sandboxen!")
    exit(1)

# Test 2: Kjør prediksjon på et testbilde
print("\n=== Test 2: Kjør prediksjon ===")
test_images = list(Path(__file__).parent.glob("train/images/*.jpg"))[:1]
if not test_images:
    test_images = list(Path(__file__).parent.glob("train/images/*.jpeg"))[:1]

if test_images:
    img_path = test_images[0]
    print(f"  Tester på: {img_path.name}")
    results = model(str(img_path), verbose=False)

    predictions = []
    for r in results:
        for box in r.boxes:
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            predictions.append({
                "image_id": int(img_path.stem.replace("img_", "")),
                "category_id": int(box.cls.item()),
                "bbox": [x1, y1, x2 - x1, y2 - y1],
                "score": float(box.conf.item()),
            })

    print(f"  Fant {len(predictions)} objekter")
    if predictions:
        print(f"  Eksempel: {json.dumps(predictions[0], indent=2)}")

        # Sjekk format
        p = predictions[0]
        assert "image_id" in p and isinstance(p["image_id"], int), "image_id skal være int"
        assert "category_id" in p and isinstance(p["category_id"], int), "category_id skal være int"
        assert "bbox" in p and len(p["bbox"]) == 4, "bbox skal ha 4 elementer [x,y,w,h]"
        assert "score" in p and isinstance(p["score"], float), "score skal være float"
        assert p["bbox"][2] > 0 and p["bbox"][3] > 0, "width og height skal være positive"
        print("  Format: OK (COCO-kompatibel)")
    else:
        print("  ADVARSEL: Ingen deteksjoner!")
else:
    print("  Ingen testbilder funnet i train/images/ — hopper over")

# Test 3: Sjekk zip-størrelse
print("\n=== Test 3: Estimert zip-størrelse ===")
best_size = Path(__file__).parent / "best.pt"
run_size = Path(__file__).parent / "run.py"
total = best_size.stat().st_size + run_size.stat().st_size
print(f"  run.py + best.pt = {total / 1024 / 1024:.1f} MB")
print(f"  Maks tillatt: 420 MB")
print(f"  {'OK' if total < 420 * 1024 * 1024 else 'FOR STOR!'}")

print("\n=== FERDIG ===")
print("Hvis alle tester bestod: klar for zip og submission!")
