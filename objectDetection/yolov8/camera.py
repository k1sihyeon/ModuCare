from ultralytics import YOLO
import requests
import json
from datetime import datetime
import threading

### HTTP ###

def getFcm(title, body):
    fcm = {
        'token' : 'eb9PJybxQ66VOlakaltS42:APA91bHEQpdDaUS1LRxZoMl701oYkN4ntF7uNdDfN0C_mSfe1CO4TnR-wEpa19ofi_RhmkG_Ew90FfRoBTMoW9jEgxvlev3DT0iC2D-x1ZzwjJEKlJYbzpdp4TaRvRPbLPtMhUAWHoav',
        'title' : title,
        'body' : body
    }
    
    return fcm

def sendFcm(fcm):
    headers = {
        "Content-Type": "application/json"
    }
    
    fcm_url = "http://118.219.42.214:8080/api/fcm/send"
    
    response = requests.post(fcm_url, data=json.dumps(fcm), headers=headers)    #fcm post
    
    print("FCM Status Code:", response.status_code)
    try:
        response_json = response.json()
        print("Response JSON:", response_json)
    except ValueError:
        print("Response Content:", response.text)
    
def getLog(content):
    log = {
        "camId": 1,
        "content": content,
        "imagePath": "/path/to/image.jpg", 
        "createdAt": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        "isChecked": False
    }
    
    return log

def sendLog(log):
    headers = {
        "Content-Type": "application/json"
    }
    
    log_url = "http://118.219.42.214:8080/api/logs"
    
    response = requests.post(log_url, data=json.dumps(log), headers=headers)    #log post
    
    print("Log Status Code:", response.status_code)
    try:
        response_json = response.json()
        print("Response JSON:", response_json)
    except ValueError:
        print("Response Content:", response.text)
    

### HTTP ###

detection_active = True
danger = [34, 42, 43]

def reset_flag():
    global detection_active
    detection_active = True
    print("Detection re-enabled")

# 1. 모든 사물 확인 
#results = trt_model.predict(source='0', show=True, stream=True)

# 2. 이미 정의된 위험한 사물만 확인
# results = trt_model.predict(source='0', show=True, stream=True, classes=danger)


def main():
    global detection_active
    global danger
    
    trt_model = YOLO('yolov8n.engine')
    results = trt_model.predict(source='0', show=True, stream=True, classes=danger)

    for result in results:
        
        if not detection_active:
            continue
        
        detected_labels = []
        
        for box in result.boxes:
            detected_labels.append(trt_model.names[int(box.cls)])
            
            if int(box.cls) in danger:
                if detection_active:
                    # 현재 해당 indent의 코드는 30초 간격으로 실행됨
                    print("!!!!!!! danger detected !!!!!!!!")
                    
                    obj = trt_model.names[int(box.cls)]
                    msg = str(obj) + " detected on Jetson TX2"
                    fcm = getFcm("danger detected", msg)
                    sendFcm(fcm)
                    
                    log = getLog(msg)
                    sendLog(log)

                    detection_active = False
                    # Set a timer to reset the flag after 30 seconds
                    threading.Timer(30, reset_flag).start()
                    print("Detection paused for 30 seconds")
                    break

if __name__ == "__main__":
    main()