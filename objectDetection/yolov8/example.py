from ultralytics import YOLO
import requests
import json
from datetime import datetime

fcm = {
    'token' : 'eb9PJybxQ66VOlakaltS42:APA91bHEQpdDaUS1LRxZoMl701oYkN4ntF7uNdDfN0C_mSfe1CO4TnR-wEpa19ofi_RhmkG_Ew90FfRoBTMoW9jEgxvlev3DT0iC2D-x1ZzwjJEKlJYbzpdp4TaRvRPbLPtMhUAWHoav',
    'title' : 'on Jetson TX2 Title',
    'body' : 'TX2 Body'
}

log = {
  "camId": 1,
  "content": "Danger Detected!!",
  "imagePath": "/path/to/image.jpg", 
  "createdAt": "2024-05-16T16:26:00",
  "isChecked": False
}

headers = {
    "Content-Type": "application/json"
}

fcm_url = "http://118.219.42.214:8080/api/fcm/send"
log_url = "http://118.219.42.214:8080/api/logs"


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

            current = datetime.now().strftime('%Y-%m-%dT%H:%M:%S')

            log = {
                "camId": 1,
                "content": "on TX2 이미지 감지됨",
                "imagePath": "/path/to/image.jpg", 
                "createdAt": current,
                "isChecked": False
            }

            fcm_response = requests.post(fcm_url, json=fcm, headers=headers)    #fcm post
            log_response = requests.post(log_url, data=json.dumps(log), headers=headers)    #log post
            
            print("FCM Status Code:", fcm_response.status_code)
            print("Log Status Code:", log_response.status_code)

            try:
                response_json = fcm_response.json()
                print("Response JSON:", response_json)
            except ValueError:
                print("Response Content:", fcm_response.text)

            try:
                response_json = log_response.json()
                print("Response JSON:", response_json)
            except ValueError:
                print("Response Content:", log_response.text)
            


print(detected_labels)

