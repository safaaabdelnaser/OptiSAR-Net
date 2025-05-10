

from ultralytics import YOLO

model = YOLO("OptiSAR-Net.yaml")

# train of seaship
# model.train(data="seaship.yaml", epochs=50, imgsz=416,plots=True ,val=True ,save=True, batch=8, close_mosaic=0)

# # opensar dataset
model = YOLO('OptiSAR-Net.yaml').load('last125epochOpensar.pt')
model.train(data="opensar.yaml", epochs=15 ,imgsz=416 ,plots=True ,val=True ,save=True, batch=8, close_mosaic=0)

