from ultralytics import YOLO
from pathlib import Path

model = YOLO("yolov8n-seg.pt")  # Or yolov8s-seg.pt for better accuracy

input_folder = Path("images")
output_folder = Path("segmented_images")
output_folder.mkdir(exist_ok=True)
for image_file in input_folder.glob("*.[jp][pn]g"):
    results = model(image_file)
    results[0].save(filename=str(output_folder / f"{image_file.stem}_seg.jpeg"))