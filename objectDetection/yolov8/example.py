from ultralytics import YOLO
import requests

fcm = {
    'token' : 'eb9PJybxQ66VOlakaltS42:APA91bHEQpdDaUS1LRxZoMl701oYkN4ntF7uNdDfN0C_mSfe1CO4TnR-wEpa19ofi_RhmkG_Ew90FfRoBTMoW9jEgxvlev3DT0iC2D-x1ZzwjJEKlJYbzpdp4TaRvRPbLPtMhUAWHoav',
    'title' : 'on Jetson TX2 Title',
    'body' : 'TX2 Body'
}

headers = {
    "Content-Type": "application/json"
}

url = "http://118.219.42.214:8080/api/fcm/send"


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
            response = requests.post(url, json=fcm, headers=headers)

            print("Status Code:", response.status_code)

            try:
                response_json = response.json()
                print("Response JSON:", response_json)
            except ValueError:
                print("Response Content:", response.text)

print(detected_labels)

