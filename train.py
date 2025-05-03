

from ultralytics import YOLO

model = YOLO("OptiSAR-Net.yaml")

model.train(data="seaship.yaml", epochs=100, batch=8, imgsz=416)

