"""
Nattens treningsplan — kjøres på VM med nohup.
Trener 3 modeller sekvensielt og eksporterer til ONNX.
"""
from ultralytics import YOLO

# === EKSPERIMENT A: YOLOv8l imgsz=1024, tung augmentation ===
print("=" * 60)
print("STARTER EKSPERIMENT A: YOLOv8l 1024px")
print("=" * 60)
model_a = YOLO("yolov8l.pt")
model_a.train(
    data="data.yaml",
    epochs=1500,
    imgsz=1024,
    batch=2,
    patience=200,
    augment=True,
    mosaic=1.0,
    mixup=0.5,
    copy_paste=0.5,
    degrees=15,
    scale=0.5,
    fliplr=0.5,
    translate=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    name="exp_a_l_1024",
)
# Eksporter
print("Eksporterer A til ONNX...")
best_a = YOLO("runs/detect/exp_a_l_1024/weights/best.pt")
best_a.export(format="onnx", imgsz=1024, opset=12)
print("EKSPERIMENT A FERDIG!")

# === EKSPERIMENT B: YOLOv8l imgsz=640, veldig lang trening ===
print("=" * 60)
print("STARTER EKSPERIMENT B: YOLOv8l 640px lang")
print("=" * 60)
model_b = YOLO("yolov8l.pt")
model_b.train(
    data="data.yaml",
    epochs=2000,
    imgsz=640,
    batch=8,
    patience=200,
    augment=True,
    mosaic=1.0,
    mixup=0.5,
    copy_paste=0.5,
    degrees=15,
    scale=0.5,
    fliplr=0.5,
    translate=0.2,
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    name="exp_b_l_640_long",
)
print("Eksporterer B til ONNX...")
best_b = YOLO("runs/detect/exp_b_l_640_long/weights/best.pt")
best_b.export(format="onnx", imgsz=640, opset=12)
print("EKSPERIMENT B FERDIG!")

# === EKSPERIMENT C: YOLOv8l imgsz=800, fine-tune fra beste modell ===
print("=" * 60)
print("STARTER EKSPERIMENT C: Fine-tune fra exp_a")
print("=" * 60)
model_c = YOLO("runs/detect/exp_a_l_1024/weights/best.pt")
model_c.train(
    data="data.yaml",
    epochs=500,
    imgsz=800,
    batch=4,
    patience=100,
    augment=True,
    mosaic=0.5,
    mixup=0.3,
    copy_paste=0.3,
    lr0=0.001,
    lrf=0.01,
    name="exp_c_finetune_800",
)
print("Eksporterer C til ONNX...")
best_c = YOLO("runs/detect/exp_c_finetune_800/weights/best.pt")
best_c.export(format="onnx", imgsz=800, opset=12)
print("EKSPERIMENT C FERDIG!")

print("=" * 60)
print("ALLE EKSPERIMENTER FERDIGE!")
print("Sjekk runs/detect/exp_*/results.csv for resultater")
print("ONNX-filer ligger i runs/detect/exp_*/weights/best.onnx")
print("=" * 60)
