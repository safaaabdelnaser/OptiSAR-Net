from ultralytics import YOLO

# val seaship 
# model = YOLO('best-seaship.pt')

# model.val(data='seaship.yaml', batch=1)

# val opensar dataset
model = YOLO('best-OPENSAR.pt')
model.val(data='opensar.yaml', batch=8)