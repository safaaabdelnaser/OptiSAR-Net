

from ultralytics import YOLO

model = YOLO("OptiSAR-Net.yaml")

model.train(data="opensar.yaml", epochs=200, batch=8, imgsz=416)

