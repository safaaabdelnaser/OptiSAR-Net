from ultralytics import YOLO

model = YOLO('/kaggle/working/seashipDataset-2/runs/detect/train/weights/best.pt')

model.val(data='seaship.yaml', batch=1)