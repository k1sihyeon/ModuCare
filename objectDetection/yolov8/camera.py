from ultralytics import YOLO
import requests
import json
from datetime import datetime
import threading

detection_active = False
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
                    print("!!!!!!! danger detected !!!!!!!!")

                    detection_active = False
                    # Set a timer to reset the flag after 10 seconds
                    threading.Timer(10, reset_flag).start()
                    print("Detection paused for 10 seconds")
                    break

if __name__ == "__main__":
    main()