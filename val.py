from ultralytics import YOLO

model = YOLO('/kaggle/working/OptiSAR-Net/runs/detect/train/weights/best.pt')

model.val(data='opensar.yaml', batch=1)