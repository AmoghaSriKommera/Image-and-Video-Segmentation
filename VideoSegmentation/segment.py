from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n-seg.pt")

input_frames = Path("frames1")
output_dir = Path("segmented_frames1")
output_dir.mkdir(exist_ok=True)

for img in input_frames.glob("*.png"):
    results = model(img)
    results[0].save(filename=str(output_dir / f"{img.stem}_seg.jpg"))
model.predict(source="frames1", save=True)