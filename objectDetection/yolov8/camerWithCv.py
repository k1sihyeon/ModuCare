import cv2
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
    
def getLog(content, imagePath="/error.jpg"):
    log = {
        "camId": 1,
        "content": content,
        "imagePath": imagePath, 
        "createdAt": datetime.now().strftime('%Y-%m-%dT%H:%M:%S'),
        "isChecked": False
    }
    
    return log

def printResponse(response):
    print("Status Code:", response.status_code)
    try:
        response_json = response.json()
        print("Response JSON:", response_json)
    except ValueError:
        print("Response Content:", response.text)

def sendHttpPost(url, data):
    headers = {
        "Content-Type": "application/json"
    }
    
    response = requests.post(url, data=json.dumps(data), headers=headers)
    printResponse(response)

def sendFcm(fcm):
    sendHttpPost("http://118.219.42.214:8080/api/fcm/send", fcm)

def sendLog(log):
    sendHttpPost("http://118.219.42.214:8080/api/logs", log)

def sendImage(file_path):
    with open(file_path, 'rb') as file:
        files = {'file': file}
        response = requests.post("http://118.219.42.214:8080/api/image/upload", files=files)
        printResponse(response)
    
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
    
    model = YOLO("yolov8n.pt")
    model.export(format='engine')
    trt_model = YOLO('yolov8n.engine')

    video_path = "dev/video1"
    cap = cv2.VideoCapture(video_path)

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()

        if success:
            # Run YOLOv8 inference on the frame
            results = trt_model.predict(source=frame, show=True, classes=danger)

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            
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
                            
                            # image post
                            path = "/" + datetime.now().strftime('%Y_%m_%dT%H:%M:%S') + ".jpg"
                            cv2.imwrite("detected" + path, frame)
                            sendImage("detected" + path)
                            
                            obj = trt_model.names[int(box.cls)]
                            msg = str(obj) + " detected on Jetson TX2"
                            fcm = getFcm("danger detected", msg)
                            sendFcm(fcm)
                            
                            log = getLog(msg, path)
                            sendLog(log)

                            detection_active = False
                            # Set a timer to reset the flag after 30 seconds
                            threading.Timer(30, reset_flag).start()
                            print("Detection paused for 30 seconds")
                            break
            
            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()
    
    
if __name__ == "__main__":
    main()    
