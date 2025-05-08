from ultralytics import YOLO

# val seaship 
model = YOLO('best-seaship.pt')

model.val(data='seaship.yaml', batch=1)

# val opensar dataset
# model = YOLO('/kaggle/working/open-4/runs/detect/train/weights/best.pt')
# model.val(data='opensar.yaml', batch=8)