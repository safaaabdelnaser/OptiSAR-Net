from ultralytics import YOLO

# val seaship 
# model = YOLO('best-seaship.pt')

# model.val(data='seaship.yaml', batch=1)

# val opensar dataset
model = YOLO('best_shipRSI.pt')
model.val(data='shipRIS.yaml', batch=8)