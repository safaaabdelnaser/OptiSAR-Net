

from ultralytics import YOLO

model = YOLO("OptiSAR-Net.yaml")

model.train(data="lastYaml.yaml", epochs=50, batch=32, imgsz=640)

