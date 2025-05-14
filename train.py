

from ultralytics import YOLO

model = YOLO("OptiSAR-Net.yaml")

# train of seaship
# model.train(data="seaship.yaml", epochs=50, imgsz=416,plots=True ,val=True ,save=True, batch=8, close_mosaic=0)

# # opensar dataset
# model = YOLO('OptiSAR-Net.yaml').load('last125epochOpensar.pt')
# model.train(data="opensar.yaml", epochs=25 ,imgsz=416 ,plots=True ,val=True ,save=True, batch=8, close_mosaic=0)

# train of shipRIS
model.train(data="shipRIS.yaml", epochs=50, imgsz=416,plots=True ,val=True ,save=True, batch=8, close_mosaic=0)
