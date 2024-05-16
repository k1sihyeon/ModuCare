from ultralytics import YOLO

# Load a YOLOv8n PyTorch model
#model = YOLO('yolov8n.pt')

# Export the model
# model.export(format='engine')  # creates 'yolov8n.engine'

# Load the exported TensorRT model
trt_model = YOLO('yolov8n.engine')

# Run inference
results = trt_model('home01.jpg', show=True, save=True)

# results = trt_model.predict(source='0', show=True)

detected_labels = []
danger = [34, 42, 43]

for result in results:
    for box in result.boxes:
        detected_labels.append(trt_model.names[int(box.cls)])
        if int(box.cls) in danger:
            print("!!!!!!! danger detected !!!!!!!!")

print(detected_labels)

